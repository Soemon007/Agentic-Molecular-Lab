"""The planner (exploit vs explore), the ChEMBL evidence tool, and the policy that guards it."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

import chembl
import orchestrator
import planner
from agents import REGISTRY
from chem_core import Candidate, ScaffoldBeam
from policies import build_gate

PREDS = [(0.9, 0.05), (0.1, 0.6), (0.8, 0.05), (0.2, 0.7)]  # (predicted mean, std) for four valid proposals


def _rank(preds):
    return lambda beta: sorted(range(len(preds)), key=lambda i: -(preds[i][0] + beta * preds[i][1]) if preds[i] else float("inf"))


def test_planner_exploits_while_there_is_predicted_gain():
    choice, options = planner.plan(PREDS, 2, _rank(PREDS), floor_score=0.5, remaining_fraction=1.0)
    assert [o.name for o in options] == ["exploit", "explore"] and choice.name == "exploit"
    assert choice.chosen == [0, 2] and choice.cost == 2 and choice.expected_gain == pytest.approx(0.35)


def test_planner_explores_when_the_beam_is_saturated_and_learning_is_still_worth_something():
    choice, options = planner.plan(PREDS, 2, _rank(PREDS), floor_score=0.99, remaining_fraction=1.0)
    assert choice.name == "explore" and choice.expected_learning > options[0].expected_learning
    late, _ = planner.plan(PREDS, 2, _rank(PREDS), floor_score=0.99, remaining_fraction=0.0)
    assert late.name == "exploit"  # nothing left to learn for: a tie goes to exploit


def test_planner_keeps_the_agents_order_when_the_surrogate_cannot_score():
    choice, options = planner.plan([None, None, None], 2, _rank([None, None, None]), 0.5, 1.0)
    assert choice.name == "exploit" and not choice.feasible and choice.order == [0, 1, 2]
    assert planner.summary(choice)["feasible"] is False


def test_run_logs_a_plan_with_both_options_for_batches_it_can_score(tmp_path):
    import json
    import policies
    policies.AUTOPILOT = True
    sess = asyncio.run(orchestrator.run_lab(budget=150, branches="abc", llm="offline", seed=0, verbose=False,
                                            traj_path=tmp_path / "t.csv", run_id="plan_test"))
    log = [json.loads(line) for line in sess.chatlog.path.read_text().splitlines()]
    sess.chatlog.path.unlink()
    plans = [e for e in log if e.get("event") == "plan"]
    assert plans and all([o["name"] for o in e["options"]] == ["exploit", "explore"] for e in plans)
    assert any(e["decided"] for e in plans), "once the surrogate has labels the planner should decide"


# ------------------------------------------------------------------------------------------------ ChEMBL tool
def _chembl(handler):
    return httpx.MockTransport(handler)


def _ok(request):
    if "/similarity/" in request.url.path:
        return httpx.Response(200, json={"molecules": [
            {"molecule_chembl_id": "CHEMBL54", "pref_name": "HALOPERIDOL", "similarity": "88.5",
             "molecule_structures": {"canonical_smiles": "O=C(CCCN1CCC(O)(CC1)c1ccc(Cl)cc1)c1ccc(F)cc1"}},
            {"molecule_chembl_id": "CHEMBL999", "pref_name": None, "similarity": "72", "molecule_structures": None}]})
    return httpx.Response(200, json={"activities": [
        {"molecule_chembl_id": "CHEMBL54", "pchembl_value": "8.25", "document_chembl_id": "CHEMBL1133667"},
        {"molecule_chembl_id": "CHEMBL54", "pchembl_value": None, "document_chembl_id": "CHEMBL1127438"}]})


def test_chembl_lookup_reports_an_active_analogue_with_citations():
    r = asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(_ok)))
    assert r["verdict"] == "analogue_active" and r["error"] is None and r["target"].startswith("CHEMBL217")
    n = {x["chembl_id"]: x for x in r["neighbours"]}
    assert n["CHEMBL54"]["drd2_pchembl_max"] == 8.25 and n["CHEMBL54"]["drd2_activities"] == 2
    assert n["CHEMBL54"]["similarity"] == 88.5 and n["CHEMBL54"]["url"].endswith("/CHEMBL54/")
    assert n["CHEMBL999"]["drd2_activities"] == 0 and n["CHEMBL999"]["smiles"] is None
    assert [d["chembl_id"] for d in r["documents"]] == ["CHEMBL1133667"]  # only the potent measurement is cited


def test_chembl_verdicts_for_no_neighbours_weak_neighbours_and_untested_neighbours():
    none = lambda req: httpx.Response(200, json={"molecules": []})  # noqa: E731
    assert asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(none)))["verdict"] == "no_analogue"

    def weak(req):
        if "/similarity/" in req.url.path:
            return _ok(req)
        return httpx.Response(200, json={"activities": [{"molecule_chembl_id": "CHEMBL54", "pchembl_value": "4.5"}]})
    assert asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(weak)))["verdict"] == "analogue_inactive"

    def untested(req):
        return _ok(req) if "/similarity/" in req.url.path else httpx.Response(200, json={"activities": []})
    assert asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(untested)))["verdict"] == "analogue_untested"


def test_chembl_failures_become_an_unavailable_verdict_not_an_exception():
    def down(req):
        raise httpx.ConnectError("no network")
    r = asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(down)))
    assert r["verdict"] == "unavailable" and "ConnectError" in r["error"]
    r = asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(lambda req: httpx.Response(429))))
    assert r["verdict"] == "unavailable" and "429" in r["error"]
    r = asyncio.run(chembl.lookup_chembl("CCO", transport=_chembl(lambda req: httpx.Response(200, json=["not", "a", "dict"]))))
    assert r["verdict"] == "unavailable"


# ------------------------------------------------------------------------------------------------ policy + step
def _gate(tmp_path):
    return build_gate(SimpleNamespace(calls=0), 100, log_path=tmp_path / "p.jsonl", autopilot=True)


def test_only_the_evidence_agent_may_call_lookup_chembl_and_only_with_one_smiles(tmp_path):
    gate = _gate(tmp_path)
    call = lambda agent, smiles: asyncio.run(gate.check("tool_call", {  # noqa: E731
        "tool": "lookup_chembl", "agent": agent, "arguments": {"smiles": smiles}}))
    assert call("evidence", "CCO").allowed
    assert not call("branch_a", "CCO").allowed  # not its tool
    assert not call("evidence", "").allowed and not call("evidence", "C" * 501).allowed


def test_evidence_lookups_are_capped_per_run(tmp_path):
    gate = _gate(tmp_path)
    results = [asyncio.run(gate.check("tool_call", {"tool": "lookup_chembl", "agent": "evidence",
                                                    "arguments": {"smiles": "CCO"}})).allowed for _ in range(25)]
    assert results.count(True) == 20 and results[-1] is False


def test_evidence_step_checks_the_top_hits_through_the_gate_and_logs_them(tmp_path, monkeypatch):
    async def fake(smiles):
        return {"smiles": smiles, "verdict": "no_analogue", "neighbours": [], "documents": [], "error": None}
    monkeypatch.setattr(REGISTRY["evidence"].tools["lookup_chembl"], "callable", fake)
    beam = ScaffoldBeam()
    beam.add([Candidate(smiles=s, score=0.9 - i * 0.1, core_scaffold=f"s{i}", origin_branch="branch_a", ad_similarity=0.4)
              for i, s in enumerate(["CCO", "CCN", "CCC", "CCCl"])])
    sess = SimpleNamespace(beam=beam, chatlog=None, round=7)
    out = asyncio.run(orchestrator.evidence_step(sess, _gate(tmp_path), top_k=3))
    assert [r["smiles"] for r in out] == ["CCO", "CCN", "CCC"]  # best first
    assert all(r["verdict"] == "no_analogue" and r["origin"] == "branch_a" and r["ad_similarity"] == 0.4 for r in out)
    assert out[0]["oracle_score"] == 0.9
