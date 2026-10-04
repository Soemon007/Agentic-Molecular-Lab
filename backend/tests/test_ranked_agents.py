"""Memory in the payload, limits in the prompts, and surrogate-ranked selection."""
import asyncio
import csv
import json

import pytest
from rdkit import Chem

import policies
from agents import REGISTRY
from chem_core import MAX_LOGP, MAX_MW, Gatekeeper
from orchestrator import HOT_MAX, run_lab
from surrogate import MIN_LABELS, Surrogate

CAN = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))  # noqa: E731


def _labels():
    high = [(f"{sub}c1ccc(CC2CCN(C)CC2)cc1", 0.9) for sub in ("F", "Cl", "C", "O", "N", "OC", "CC", "Br", "S", "C#N", "CF")]
    low = [("C" * n, 0.0) for n in range(2, 24)]
    return high + low


# --------------------------------------------------------------------------------------------- surrogate
def test_surrogate_is_a_noop_until_it_has_enough_labels():
    s = Surrogate().fit(_labels()[: MIN_LABELS - 1])
    assert not s.ready and s.rank(["CCO", "CCN", "CCC"]) == [0, 1, 2] and s.predict(["CCO"]) == [None]


def test_surrogate_ranks_the_candidate_that_resembles_high_scorers_first():
    s = Surrogate().fit(_labels())
    assert s.ready
    cands = ["CCCCCCCCCC", "Nc1ccc(CC2CCN(CC)CC2)cc1", "not a smiles", "CCCCCCCCCCCCCCC"]
    order = s.rank(cands)
    assert order[0] == 1 and order[-1] == 2  # the aryl-piperidine first, the unparsable one last
    mean, std = s.predict(["Nc1ccc(CC2CCN(CC)CC2)cc1"])[0]
    assert mean > 0.5 and 0.0 <= std <= 1.0
    assert s.rank(cands) == order  # deterministic


def test_surrogate_is_less_sure_about_molecules_unlike_anything_it_has_seen():
    s = Surrogate().fit(_labels())
    (_, near), (_, far) = s.predict(["Fc1ccc(CC2CCN(C)CC2)cc1", "O=S(=O)(Nc1ncccn1)c1ccc(N)cc1"])
    assert far > near


def test_surrogate_default_exploration_weight_is_the_small_one_the_replay_supports():
    assert Surrogate().beta <= 0.1  # beta = 1 was no better than random on the replayed live runs


# --------------------------------------------------------------------------------------------- gatekeeper
def test_gatekeeper_commit_false_leaves_the_molecule_proposable_and_commit_is_once_only():
    gk = Gatekeeper()
    mol, why = gk.check("Fc1ccccc1", "branch_a", "c1ccccc1", commit=False)
    assert mol is not None and why is None and not gk.seen and gk.accepted["branch_a"] == 0
    assert gk.check("Fc1ccccc1", "branch_a", "c1ccccc1", commit=False)[0] is not None  # still not a duplicate
    assert gk.commit(mol, "branch_a") and gk.accepted["branch_a"] == 1
    assert not gk.commit(mol, "branch_b")  # e.g. two branches picked the same molecule
    assert gk.rejections["branch_b"]["duplicate"] == 1
    assert gk.check("Fc1ccccc1", "branch_a", "c1ccccc1")[1] == "duplicate"


def test_gatekeeper_counts_duplicate_hits_and_lists_the_most_reproposed_first():
    gk = Gatekeeper()
    for smi in ("Fc1ccccc1", "Clc1ccccc1"):
        gk.check(smi, "branch_a", "c1ccccc1")  # scored once each
    for _ in range(3):
        assert gk.check("Fc1ccccc1", "branch_a", "c1ccccc1")[1] == "duplicate"
    assert gk.check("Clc1ccccc1", "branch_b", "c1ccccc1")[1] == "duplicate"
    assert gk.hot_duplicates(5) == [CAN("Fc1ccccc1"), CAN("Clc1ccccc1")] and gk.hot_duplicates(1) == [CAN("Fc1ccccc1")]
    assert gk.dup_hits[CAN("Fc1ccccc1")] == 3


# --------------------------------------------------------------------------------------------- prompts
def test_prompts_state_the_limits_and_describe_the_payload_fields():
    for name in ("branch_a", "branch_b", "branch_c"):
        p = REGISTRY[name].prompt
        assert f"MW <= {MAX_MW:.0f}" in p and f"logP <= {MAX_LOGP:.0f}" in p
        assert all(k in p for k in ("recent_results", "do_not_repeat", "your_recent_rejections"))
    assert "recent_results" in REGISTRY["scout"].prompt
    assert "logP" not in REGISTRY["adversary"].prompt  # the adversary's brief is unchanged: it must not learn the agents' limits


# --------------------------------------------------------------------------------------------- end to end (mock)
def _run(tmp_path, budget=200, seed=0, name="ranked_test"):
    policies.AUTOPILOT = True
    sess = asyncio.run(run_lab(budget=budget, branches="abc", llm="offline", seed=seed, verbose=False, seed_mode="known",
                               traj_path=tmp_path / f"{name}.csv", run_id=name))
    log = [json.loads(line) for line in sess.chatlog.path.read_text().splitlines()]
    sess.chatlog.path.unlink()
    return sess, log


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def test_run_gives_agents_the_oracles_verdicts_and_keeps_the_budget_exact(tmp_path):
    sess, log = _run(tmp_path)
    calls = [r for r in log if r["kind"] == "agent_call"]
    assert sess.oracle.calls == 200 and len(sess.labels) == 200 and not sess.stalled

    first = [c for c in calls if c["round"] == 1 and c["agent"] == "branch_a"][0]["input"]
    assert first["recent_results"] == [] and "your_recent_rejections" in first and "logp" in first["beam"][0]
    later = [c for c in calls if c["round"] == 3 and c["agent"] == "branch_b"][0]["input"]
    assert later["recent_results"], "round 3 should see the oracle's verdicts on rounds 1-2"
    assert all(len(r) == 3 and isinstance(r[0], str) for r in later["recent_results"])  # [smiles, score, delta]
    by_smiles = {CAN(r["smiles"]): round(float(r["score"]), 3) for r in csv_rows(tmp_path / "ranked_test.csv")}
    assert all(by_smiles[r[0]] == r[1] for r in later["recent_results"])  # real oracle numbers, nothing invented
    assert len({r[0] for r in later["recent_results"]}) == len(later["recent_results"])
    scout = [c for c in calls if c["round"] == 3 and c["agent"] == "scout"][0]["input"]
    assert scout["recent_results"] == later["recent_results"]


def test_branches_get_a_bounded_do_not_repeat_list_that_does_not_repeat_the_payload(tmp_path):
    _, log = _run(tmp_path, budget=250)
    branch_inputs = [c["input"] for c in log if c["kind"] == "agent_call" and c["agent"].startswith("branch_")]
    assert all(len(i["do_not_repeat"]) <= HOT_MAX for i in branch_inputs)
    assert any(i["do_not_repeat"] for i in branch_inputs), "the mock re-proposes molecules, so the list should fill up"
    for i in branch_inputs:
        shown = {b["smiles"] for b in i["beam"]} | {r[0] for r in i["recent_results"]}
        assert not shown & set(i["do_not_repeat"])
    assert all("do_not_repeat" not in c["input"] for c in log if c["kind"] == "agent_call" and c["agent"] == "scout")


def test_each_branch_quota_is_spent_on_ranked_proposals_and_the_rest_are_not_burned(tmp_path):
    sess, log = _run(tmp_path)
    outs = [r for r in log if r["kind"] == "proposal_outcome"]
    quotas = {h["round"]: h["quotas"] for h in sess.history[1:]}
    per_call = {}
    for o in outs:
        if o.get("oracle_score") is not None:
            per_call[(o["round"], o["agent"])] = per_call.get((o["round"], o["agent"]), 0) + 1
    assert per_call and all(n <= quotas[r][a] for (r, a), n in per_call.items())

    ranked = [o for o in outs if o.get("surrogate_rank")]
    assert ranked and any(o["surrogate_mean"] is not None for o in ranked), "the surrogate never engaged"
    skipped = [o for o in outs if o.get("gate_reason") == "not_selected"]
    assert skipped, "the mock over-proposes, so some valid proposals must fall below the quota"
    scored_smiles = {CAN(o["smiles"]) for o in outs if o.get("oracle_score") is not None}
    # a valid proposal that was not selected must not have been marked seen (it may be proposed again)
    assert all(CAN(o["smiles"]) not in sess.gatekeeper.seen for o in skipped if CAN(o["smiles"]) not in scored_smiles)


def test_selection_follows_the_rankers_order_not_the_order_the_agent_wrote_proposals(tmp_path, monkeypatch):
    """Plant a ranker that prefers short SMILES; within every call the scored proposals must be the shortest valid ones."""
    import orchestrator
    monkeypatch.setattr(orchestrator.Surrogate, "rank", lambda self, smi: sorted(range(len(smi)), key=lambda i: len(smi[i])))
    _, log = _run(tmp_path, budget=150, name="ranker_plant")
    outs = [r for r in log if r["kind"] == "proposal_outcome"]
    checked = 0
    for call in {o["call_id"] for o in outs if o.get("gate_reason") == "not_selected"}:
        chosen = [len(CAN(o["smiles"])) for o in outs if o["call_id"] == call and o.get("oracle_score") is not None]
        left = [len(CAN(o["smiles"])) for o in outs if o["call_id"] == call and o.get("gate_reason") == "not_selected"]
        if chosen and left:
            assert max(chosen) <= min(left)
            checked += 1
    assert checked >= 5
