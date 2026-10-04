"""Orchestrator: Omnigent-primitive session, policies and the main loop.

Mapping to Omnigent primitives
  * agents      -> `AgentDef` (agents/), run through an Omnigent `Executor` (llm.py)
  * policies    -> `FunctionPolicy` ALLOW/ASK/DENY objects evaluated on every tool call (policies.py)
  * tools       -> agents only get a `FunctionTool` that echoes their proposal; the *only* scoring path is
                   the `oracle_evaluate` step below, which is itself policy-gated
  * session     -> `LabSession` (beam, oracle_calls_used, round history) + Omnigent `SessionState` lifecycle
Not used: the Omnigent server/CLI runtime (its executors need provider credentials); see README.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field

from omnigent.inner.datamodel import SessionState

from rdkit.Chem import Descriptors

import chem_core
from chem_core import Candidate, Gatekeeper, ScaffoldBeam, SEEDS
from agents import REGISTRY
from agents.coordinator import Coordinator
import telemetry
from agents.base import run_agent
from llm import TokenLedger, llm_mode, make_executor
from chatlog import ChatLog
from domain import DATA
from policies import build_gate
from surrogate import Surrogate

BRANCHES = {"a": "branch_a", "b": "branch_b", "c": "branch_c"}


@dataclass
class LabSession:
    oracle: object
    budget: int
    beam: ScaffoldBeam = field(default_factory=ScaffoldBeam)
    gatekeeper: Gatekeeper = field(default_factory=Gatekeeper)
    history: list = field(default_factory=list)
    state: SessionState = SessionState.CREATED
    ledger: TokenLedger = field(default_factory=TokenLedger)
    all_candidates: list = field(default_factory=list)
    last_rejections: dict = field(default_factory=dict)
    chatlog: object = None
    adversary_enabled: bool = True
    stalled: bool = False
    stall_reason: str | None = None
    stall_is_error: bool = False  # True when agent calls themselves failed (API error, policy deny), not just rejected proposals
    round_agent_error: str | None = None  # first agent failure seen in the current round
    labels: list = field(default_factory=list)  # (smiles, oracle score) of everything scored so far: the surrogate's data
    recent: list = field(default_factory=list)  # per round, the oracle's verdicts on that round's scored proposals
    last_rejects: dict = field(default_factory=dict)  # branch -> its own fixable rejections from last round
    surrogate: Surrogate = field(default_factory=Surrogate)
    coordinator: object = None
    trigger_log: list = field(default_factory=list)
    adversary_calls: int = 0
    round: int = 0

    @property
    def oracle_calls_used(self) -> int:
        return self.oracle.calls


def cold_seeds(n: int = 5, seed: int = 0) -> list[str]:
    """Random ZINC molecules (valid, pass the general Gatekeeper limits): a cold start comparable to PMO methods."""
    from tdc.generation import MolGen
    import compat  # noqa: F401
    from domain import DATA
    pool = MolGen(name="ZINC", path=str(DATA)).get_data()["smiles"].tolist()
    gk, out, rng = Gatekeeper(), [], random.Random(1000 + seed)
    while len(out) < n:
        smi = rng.choice(pool)
        mol, _ = gk.check(smi, "seed")
        if mol is not None:
            out.append(smi)
    return out


def seed_beam(oracle, beam: ScaffoldBeam | None = None, gatekeeper: Gatekeeper | None = None,
              mode: str = "known", seed: int = 0) -> ScaffoldBeam:
    """Generation 0, 5 molecules scored via the oracle (they count against the budget).
    known = the spec's 5 DRD2 ligands of mixed strength (warm start; oracle already ~1.0 on 3 of them)
    cold  = 5 random ZINC molecules (fair comparison with cold-start methods such as Graph GA)"""
    beam = ScaffoldBeam() if beam is None else beam
    smiles = list(SEEDS.values()) if mode == "known" else cold_seeds(5, seed)
    for smi in smiles:
        if gatekeeper is not None:
            gatekeeper.register_seen(smi)
        c = Candidate(smiles=smi, origin_branch="seed", generation=0)
        oracle.score_candidate(c)
        beam.add(c)
    return beam


def beam_payload(beam, n, extra=None, limit=20):
    def row(c):
        mol = chem_core.mol_from_smiles(c.smiles)
        return {"id": c.id, "smiles": c.smiles, "score": round(c.score, 3), "scaffold": c.core_scaffold,
                "mw": round(Descriptors.MolWt(mol), 1),  # lets agents respect the MW cap ...
                "logp": round(Descriptors.MolLogP(mol), 1)}  # ... and the logP cap
    return {"beam": [row(c) for c in sorted(beam.members, key=lambda c: -c.score)[:limit]], "n": n, **(extra or {})}


RECENT_ROUNDS, RECENT_MAX, REJECT_MAX, HOT_MAX = 2, 16, 4, 20
FIXABLE = {"invalid_smiles", "logp", "mw", "ring_count_change", "scaffold_change", "scaffold_unchanged"}


def recent_results(sess) -> list:
    """The oracle's verdicts on the last RECENT_ROUNDS rounds of proposals, newest first."""
    return [r for rnd in reversed(sess.recent[-RECENT_ROUNDS:]) for r in rnd][:RECENT_MAX]


async def oracle_step(sess: LabSession, gate, cand: Candidate) -> bool:
    """The ONLY path to a score. Policies (budget cap, approval gate) vet it first."""
    mol = chem_core.mol_from_smiles(cand.smiles)
    verdict = await gate.check("tool_call", {
        "tool": "oracle_evaluate", "agent": "oracle", "branch": cand.origin_branch,
        "arguments": {"smiles": cand.smiles, "alerts": chem_core.get_alerts(mol)}})
    if not verdict.allowed:
        return False
    sess.oracle.score_candidate(cand)
    sess.labels.append((cand.smiles, cand.score))
    return True


def _log_outcome(sess, out, **kw):
    if sess.chatlog:
        sess.chatlog.proposal_outcome(**out, gate_reason=kw.pop("gate_reason", None),
                                      oracle_score=kw.pop("oracle_score", None), **kw)


async def run_round(sess: LabSession, gate, executor, rnd: int, quotas: dict[str, int], instructions=None,
                    oversample: int = 3) -> dict:
    instructions = instructions or {}
    sess.round_agent_error = None
    memory = {"recent_results": recent_results(sess)}
    # the do-not-repeat list skips molecules the payload already shows (beam, recent results): no need to say it twice
    shown = {c.smiles for c in sess.beam.members} | {r[0] for r in memory["recent_results"]}
    hot = [m for m in sess.gatekeeper.hot_duplicates(HOT_MAX + len(shown)) if m not in shown][:HOT_MAX]
    brief = await run_agent(REGISTRY["scout"], executor, beam_payload(sess.beam, 1, memory), gate, sess.ledger,
                            chatlog=sess.chatlog, rnd=rnd)
    brief_text = brief.args.get("brief", "")
    if brief.error or brief.denied:
        sess.round_agent_error = f"scout: {brief.error or brief.denied}"

    async def branch(name, quota):
        extra = {"brief": brief_text, "feedback_last_round": sess.last_rejections.get(name, {}),
                 "instruction": instructions.get(name, ""), **memory, "do_not_repeat": hot,
                 "your_recent_rejections": sess.last_rejects.get(name, [])}
        r = await run_agent(REGISTRY[name], executor, beam_payload(sess.beam, quota * oversample, extra),
                            gate, sess.ledger, chatlog=sess.chatlog, rnd=rnd)
        return name, r

    results = await asyncio.gather(*(branch(n, q) for n, q in quotas.items() if q > 0))
    by_id = {c.id: c for c in sess.beam.members}
    sess.surrogate.fit(sess.labels)  # once per round: fitted on the oracle results so far, never on proposals
    scored, verdicts = [], []
    for name, r in results:
        if r.denied or r.error:
            sess.last_rejections[name] = {"policy_denied_or_error": 1}
            sess.round_agent_error = sess.round_agent_error or f"{name}: {r.error or r.denied}"
            continue
        before = dict(sess.gatekeeper.rejections.get(name, {}))
        await _score_ranked(sess, gate, name, r, quotas[name], by_id, rnd, scored, verdicts)
        after = sess.gatekeeper.rejections.get(name, {})
        sess.last_rejections[name] = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}
    sess.recent.append(verdicts)
    top = sess.beam.top(10)
    rec = {**telemetry.snapshot(sess.beam), "round": rnd, "oracle_calls_used": sess.oracle_calls_used, "scored": len(scored),
           "top10_mean": round(sum(c.score for c in top) / len(top), 4), "best": round(top[0].score, 4),
           "scaffolds_in_beam": len(sess.beam.scaffold_counts())}
    sess.history.append(rec)
    return rec


def _outcome(r, rnd, name, p):
    return {"call_id": r.call_id, "round": rnd, "agent": name, "parent_id": p.get("parent_id"),
            "smiles": p.get("smiles"), "agent_generated": True}


async def _score_ranked(sess, gate, name, r, quota, by_id, rnd, scored, verdicts):
    """Validate every proposal first, rank the valid ones with the surrogate, spend the quota on the best.
    Valid proposals that are not selected are logged as `not_selected` and are NOT marked seen, so they can be
    proposed again later."""
    valid, fixable = [], []
    for p in r.args.get("proposals", []):
        out = _outcome(r, rnd, name, p)
        parent = by_id.get(p.get("parent_id"))
        if parent is None:
            sess.gatekeeper._reject(name, "unknown_parent")
            _log_outcome(sess, out, gate_reason="unknown_parent")
            continue
        mol, reason = sess.gatekeeper.check(p.get("smiles", ""), name, parent.smiles, commit=False)
        if mol is None:
            _log_outcome(sess, out, gate_reason=reason)
            if reason in FIXABLE and len(fixable) < REJECT_MAX:
                fixable.append([p.get("smiles"), reason.replace("_", " ")])
            continue
        valid.append((p, parent, mol, out))
    preds = sess.surrogate.predict([chem_core.Chem.MolToSmiles(v[2]) for v in valid])
    order = sess.surrogate.rank([chem_core.Chem.MolToSmiles(v[2]) for v in valid])
    taken = 0
    for k, i in enumerate(order):
        p, parent, mol, out = valid[i]
        pred = {"surrogate_rank": k + 1, "surrogate_mean": round(preds[i][0], 3) if preds[i] else None}
        if taken >= quota or sess.oracle.calls >= sess.budget:
            _log_outcome(sess, out, gate_reason="not_selected", **pred)
            continue
        if not sess.gatekeeper.commit(mol, name):  # another branch got the same molecule this round
            _log_outcome(sess, out, gate_reason="duplicate", **pred)
            continue
        cand = Candidate(smiles=chem_core.Chem.MolToSmiles(mol), origin_branch=name, parent_id=parent.id,
                         generation=rnd)
        if await oracle_step(sess, gate, cand):
            sess.beam.add(cand)
            sess.all_candidates.append(cand)
            scored.append(cand)
            taken += 1
            verdicts.append([cand.smiles, round(cand.score, 3), round(cand.score - parent.score, 3)])  # smiles, score, delta
            _log_outcome(sess, out, oracle_score=round(cand.score, 4), ad_similarity=cand.ad_similarity,
                         sa_score=cand.sa_score, alerts=cand.alerts, scaffold=cand.core_scaffold, **pred)
        else:
            _log_outcome(sess, out, gate_reason="policy_rejected", **pred)
    sess.last_rejects[name] = fixable


async def adversary_step(sess, gate, executor, rnd):
    """Deterministic trigger (free). Sonnet adversary + coordinator only when it fires."""
    digest = telemetry.build_digest(sess.beam, sess.history, rnd, sess.oracle_calls_used)
    fired = telemetry.trigger(digest)
    blame = telemetry.flagged_branch(sess.beam) if fired else None
    rec = {"round": rnd, "fired": fired, "flagged": blame, "acted": False, "stats": digest["stats"]}
    if fired and blame and sess.coordinator.in_cooldown(blame):
        rec["skipped"] = "cooldown"
    elif fired and blame and sess.adversary_enabled:
        r = await run_agent(REGISTRY["adversary"], executor, digest, gate, sess.ledger, temperature=0.0,
                            chatlog=sess.chatlog, rnd=rnd)
        sess.adversary_calls += 1
        if r.args:
            rec["diagnosis"], rec["instruction"] = r.args.get("diagnosis"), r.args.get("instruction")
            rec["acted"] = sess.coordinator.flag(blame, r.args.get("instruction", ""))
    sess.trigger_log.append(rec)
    if sess.chatlog and fired:
        sess.chatlog.event("adversary_trigger", **{k: v for k, v in rec.items() if k != "stats"}, stats=rec["stats"])
    return rec


async def run_lab(budget=100, branches="ab", quota=4, llm=None, seed=0, traj_path=None, verbose=True,
                  noise=1.0, max_rounds=1000, adversary=True, exploit=False, run_id=None, seed_mode="known",
                  approver=None, should_stop=None):
    from oracle import DRD2Oracle
    random.seed(seed)
    run_id = run_id or f"{'adv' if adversary else 'noadv'}_{branches}_{seed_mode}_seed{seed}"
    oracle = DRD2Oracle(traj_path=traj_path or str(DATA / "trajectories" / f"ours_{run_id}.csv"),
                        report_at_exit=False)
    sess = LabSession(oracle=oracle, budget=budget)
    sess.adversary_enabled = adversary
    sess.chatlog = ChatLog(run_id)
    gate = build_gate(oracle, budget, approver=approver)
    sess.gate = gate

    def _policy_event(rec):  # DENY / ASK records, attributed to this run's chat log
        extra = {k: v for k, v in rec.items() if k not in ("event", "t")}
        sess.chatlog.event("policy", round=sess.round, verdict=rec["event"], **extra)
    gate.on_event = _policy_event
    executor = make_executor(llm or llm_mode(), seed)
    if hasattr(executor, "noise"):
        executor.noise, executor.exploit = noise, exploit
    seed_beam(oracle, sess.beam, sess.gatekeeper, seed_mode, seed)
    sess.labels = [(c.smiles, c.score) for c in sess.beam.members]  # the seeds are the surrogate's first labels
    sess.history.append({**telemetry.snapshot(sess.beam), "round": 0})  # history[r] == round r
    sess.chatlog.event("round", **sess.history[0], oracle_calls_used=oracle.calls,
                       best=round(sess.beam.top(1)[0].score, 4))
    sess.state = SessionState.RUNNING
    sess.coordinator = Coordinator([BRANCHES[b] for b in branches], default=quota)
    rnd, stalled = 0, 0
    while oracle.calls < budget and rnd < max_rounds:  # hard stop at the ceiling
        if should_stop and should_stop():  # cooperative stop from the UI
            break
        rnd += 1
        sess.round = rnd
        before = oracle.calls
        quotas, instr = sess.coordinator.quotas(), sess.coordinator.instructions()
        rec = await run_round(sess, gate, executor, rnd, quotas, instr)
        sess.coordinator.end_round()  # decrement cooldowns BEFORE a fresh flag
        trig = await adversary_step(sess, gate, executor, rnd)
        rec["quotas"], rec["trigger"] = quotas, trig["fired"]
        sess.chatlog.event("round", **rec)
        if verbose:
            print(f"round {rnd:3d} calls={rec['oracle_calls_used']:4d} top10={rec['top10_mean']:.3f} "
                  f"best={rec['best']:.3f} scaffolds={rec['scaffolds_in_beam']} quotas={list(quotas.values())}"
                  + (f" TRIGGER={trig['fired']} -> {trig['flagged']}" if trig["fired"] else ""))
        stalled = stalled + 1 if oracle.calls == before else 0
        if stalled >= 5:  # gatekeeper/policies rejecting everything: stop rather than spin
            sess.stalled = True  # surfaced in the summary; never a silent truncation
            sess.stall_is_error = sess.round_agent_error is not None
            sess.stall_reason = (f"agent calls are failing ({sess.round_agent_error})" if sess.stall_is_error
                                 else "the Gatekeeper and policies rejected every proposal")
            break
    sess.state = SessionState.COMPLETED
    oracle.report()
    oracle._fh.flush()
    return sess


def summarize(sess: LabSession) -> dict:
    out = {"llm_mode": llm_mode(), "oracle_calls": sess.oracle.calls, "failures": sess.oracle.failures,
           "rejections": sess.gatekeeper.rejection_report(), "accepted": dict(sess.gatekeeper.accepted),
           "policy_counts": sess.gate.counts, "tokens": sess.ledger.as_dict(),
           "trigger_rounds": sum(1 for t in sess.trigger_log if t["fired"]),
           "adversary_calls": sess.adversary_calls, "stalled": sess.stalled, "stall_reason": sess.stall_reason,
           "rounds": len(sess.history) - 1}
    by_branch = {}
    for c in sess.all_candidates:
        by_branch.setdefault(c.origin_branch, set()).add(c.core_scaffold)
    out["scaffold_sets"] = {k: len(v) for k, v in by_branch.items()}
    names = list(by_branch)
    if len(names) >= 2:
        a, b = by_branch[names[0]], by_branch[names[1]]
        out["scaffold_jaccard_" + "_".join(names[:2])] = round(len(a & b) / len(a | b), 3) if a | b else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--branches", default="ab")
    ap.add_argument("--quota", type=int, default=4)
    ap.add_argument("--llm", choices=["offline", "anthropic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=1.0, help="offline mock: rate of rule-violating proposals")
    ap.add_argument("--seed-mode", choices=["known", "cold"], default="known")
    ap.add_argument("--no-adversary", action="store_true")
    ap.add_argument("--exploit", action="store_true", help="offline mock: Explorer reward-hacks (demo only)")
    a = ap.parse_args()
    s = asyncio.run(run_lab(a.budget, a.branches, a.quota, a.llm, a.seed, noise=a.noise,
                         adversary=not a.no_adversary, exploit=a.exploit, seed_mode=a.seed_mode))
    print(json.dumps(summarize(s), indent=2))
