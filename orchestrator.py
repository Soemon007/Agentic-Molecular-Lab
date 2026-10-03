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

import chem_core
from chem_core import Candidate, Gatekeeper, ScaffoldBeam, SEEDS
from agents import REGISTRY
from agents.base import run_agent
from llm import TokenLedger, llm_mode, make_executor
from policies import build_gate

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

    @property
    def oracle_calls_used(self) -> int:
        return self.oracle.calls


def seed_beam(oracle, beam: ScaffoldBeam | None = None, gatekeeper: Gatekeeper | None = None) -> ScaffoldBeam:
    """Generation 0: 5 known DRD2 ligands of mixed strength, scored via the oracle (they count)."""
    beam = ScaffoldBeam() if beam is None else beam
    for name, smi in SEEDS.items():
        if gatekeeper is not None:
            gatekeeper.register_seen(smi)
        c = Candidate(smiles=smi, origin_branch="seed", generation=0)
        oracle.score_candidate(c)
        beam.add(c)
    return beam


def beam_payload(beam, n, extra=None, limit=20):
    rows = [{"id": c.id, "smiles": c.smiles, "score": round(c.score, 3), "scaffold": c.core_scaffold}
            for c in sorted(beam.members, key=lambda c: -c.score)[:limit]]
    return {"beam": rows, "n": n, **(extra or {})}


async def oracle_step(sess: LabSession, gate, cand: Candidate) -> bool:
    """The ONLY path to a score. Policies (budget cap, approval gate) vet it first."""
    mol = chem_core.mol_from_smiles(cand.smiles)
    verdict = await gate.check("tool_call", {
        "tool": "oracle_evaluate", "agent": "oracle",
        "arguments": {"smiles": cand.smiles, "alerts": chem_core.get_alerts(mol)}})
    if not verdict.allowed:
        return False
    sess.oracle.score_candidate(cand)
    return True


async def run_round(sess: LabSession, gate, executor, rnd: int, quotas: dict[str, int], instructions=None,
                    oversample: int = 3) -> dict:
    instructions = instructions or {}
    brief = await run_agent(REGISTRY["scout"], executor, beam_payload(sess.beam, 1), gate, sess.ledger)
    brief_text = brief.args.get("brief", "")

    async def branch(name, quota):
        extra = {"brief": brief_text, "feedback_last_round": sess.last_rejections.get(name, {}),
                 "instruction": instructions.get(name, "")}
        r = await run_agent(REGISTRY[name], executor, beam_payload(sess.beam, quota * oversample, extra),
                            gate, sess.ledger)
        return name, r

    results = await asyncio.gather(*(branch(n, q) for n, q in quotas.items() if q > 0))
    by_id = {c.id: c for c in sess.beam.members}
    scored = []
    for name, r in results:
        if r.denied or r.error:
            sess.last_rejections[name] = {"policy_denied_or_error": 1}
            continue
        taken, before = 0, dict(sess.gatekeeper.rejections.get(name, {}))
        for p in r.args.get("proposals", []):
            if taken >= quotas[name] or sess.oracle.calls >= sess.budget:
                break
            parent = by_id.get(p.get("parent_id"))
            if parent is None:
                sess.gatekeeper._reject(name, "unknown_parent")
                continue
            mol, reason = sess.gatekeeper.check(p.get("smiles", ""), name, parent.smiles)
            if mol is None:
                continue
            cand = Candidate(smiles=chem_core.Chem.MolToSmiles(mol), origin_branch=name, parent_id=parent.id,
                             generation=rnd)
            if await oracle_step(sess, gate, cand):
                sess.beam.add(cand)
                sess.all_candidates.append(cand)
                scored.append(cand)
                taken += 1
        after = sess.gatekeeper.rejections.get(name, {})
        sess.last_rejections[name] = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}
    top = sess.beam.top(10)
    rec = {"round": rnd, "oracle_calls_used": sess.oracle_calls_used, "scored": len(scored),
           "top10_mean": round(sum(c.score for c in top) / len(top), 4), "best": round(top[0].score, 4),
           "scaffolds_in_beam": len(sess.beam.scaffold_counts())}
    sess.history.append(rec)
    return rec


async def run_lab(budget=100, branches="ab", quota=4, llm=None, seed=0, traj_path=None, verbose=True,
                  noise=1.0, max_rounds=1000):
    from oracle import DRD2Oracle
    random.seed(seed)
    oracle = DRD2Oracle(traj_path=traj_path or f"data/trajectories/ours_{branches}_seed{seed}.csv",
                        report_at_exit=False)
    sess = LabSession(oracle=oracle, budget=budget)
    gate = build_gate(oracle, budget)
    sess.gate = gate
    executor = make_executor(llm or llm_mode(), seed)
    if hasattr(executor, "noise"):
        executor.noise = noise
    seed_beam(oracle, sess.beam, sess.gatekeeper)
    sess.state = SessionState.RUNNING
    quotas = {BRANCHES[b]: quota for b in branches}
    rnd, stalled = 0, 0
    while oracle.calls < budget and rnd < max_rounds:  # hard stop at the ceiling
        rnd += 1
        before = oracle.calls
        rec = await run_round(sess, gate, executor, rnd, quotas)
        if verbose:
            print(f"round {rnd:3d} calls={rec['oracle_calls_used']:4d} top10={rec['top10_mean']:.3f} "
                  f"best={rec['best']:.3f} scaffolds={rec['scaffolds_in_beam']}")
        stalled = stalled + 1 if oracle.calls == before else 0
        if stalled >= 5:  # gatekeeper/policies rejecting everything: stop rather than spin
            break
    sess.state = SessionState.COMPLETED
    oracle.report()
    oracle._fh.flush()
    return sess


def summarize(sess: LabSession) -> dict:
    out = {"llm_mode": llm_mode(), "oracle_calls": sess.oracle.calls, "failures": sess.oracle.failures,
           "rejections": sess.gatekeeper.rejection_report(), "accepted": dict(sess.gatekeeper.accepted),
           "policy_counts": sess.gate.counts, "tokens": sess.ledger.as_dict()}
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
    a = ap.parse_args()
    s = asyncio.run(run_lab(a.budget, a.branches, a.quota, a.llm, a.seed, noise=a.noise))
    print(json.dumps(summarize(s), indent=2))
