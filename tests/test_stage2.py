import asyncio

import pytest

import chem_core
from eval.pmo_auc import top10_auc, top10_curve
from policies import build_gate
from baselines.single_call_llm import parse_smiles
from orchestrator import run_lab, summarize


class FakeOracle:
    calls = 0


def run(coro):
    return asyncio.run(coro)


def test_write_permission_blocks_self_scoring_and_foreign_tools(tmp_path):
    gate = build_gate(FakeOracle(), 100, log_path=tmp_path / "p.jsonl", autopilot=True)
    ok = run(gate.check("tool_call", {"tool": "submit_proposals", "agent": "branch_a",
                                      "arguments": {"proposals": [{"parent_id": "x", "smiles": "CC"}]}}))
    assert ok.allowed
    cheat = run(gate.check("tool_call", {"tool": "submit_proposals", "agent": "branch_a",
                                         "arguments": {"proposals": [{"parent_id": "x", "smiles": "CC", "score": 0.99}]}}))
    assert not cheat.allowed and "score" in cheat.reason
    foreign = run(gate.check("tool_call", {"tool": "oracle_evaluate", "agent": "branch_b", "arguments": {}}))
    assert not foreign.allowed


def test_budget_cap_is_a_hard_stop(tmp_path):
    o = FakeOracle(); o.calls = 100
    gate = build_gate(o, 100, log_path=tmp_path / "p.jsonl", autopilot=True)
    v = run(gate.check("tool_call", {"tool": "oracle_evaluate", "agent": "oracle", "arguments": {"smiles": "CC", "alerts": []}}))
    assert not v.allowed and "budget" in v.reason


def test_electrophile_gate_autopilot_rejects_but_brenk_telemetry_does_not_pause(tmp_path):
    gate = build_gate(FakeOracle(), 100, log_path=tmp_path / "p.jsonl", autopilot=True)
    elec = chem_core.get_alerts(chem_core.mol_from_smiles("C=CC(=O)c1ccccc1"))
    v = run(gate.check("tool_call", {"tool": "oracle_evaluate", "agent": "oracle", "arguments": {"smiles": "x", "alerts": elec}}))
    assert not v.allowed and v.action == "ASK" and gate.counts["ASK_REJECTED"] == 1
    tele = chem_core.get_alerts(chem_core.mol_from_smiles("Nc1ccccc1"))  # aniline: BRENK, not electrophile
    v = run(gate.check("tool_call", {"tool": "oracle_evaluate", "agent": "oracle", "arguments": {"smiles": "x", "alerts": tele}}))
    assert v.allowed
    assert "electrophile" in (tmp_path / "p.jsonl").read_text()


def test_pmo_auc_basics():
    assert top10_auc(["C"] * 0, [], 100) == 0.0
    # perfect constant 1.0 -> AUC close to 1 (first 10 calls ramp from 0)
    smiles = [f"{'C' * i}" for i in range(1, 101)]
    assert 0.9 < top10_auc(smiles, [1.0] * 100, 100) <= 1.0
    assert top10_curve(smiles, [1.0] * 100, 100)[-1][0] == 100


def test_parse_smiles_strips_numbering():
    assert parse_smiles("1. CCO\n- c1ccccc1\n`CCN`\n") == ["CCO", "c1ccccc1", "CCN"]


@pytest.fixture(scope="module")
def gate_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("g")
    import os
    os.environ["AUTOPILOT"] = "true"
    import policies
    policies.AUTOPILOT = True
    return run(run_lab(budget=100, branches="ab", llm="offline", seed=0, traj_path=d / "t.csv", verbose=False))


def test_stage2_gate_budget_exact_and_no_leaks(gate_run):
    assert gate_run.oracle.calls == 100  # hard stop, exact
    assert gate_run.oracle.failure_rate <= 0.02


def test_stage2_gate_rejections_nonzero_both_branches_and_a_is_mixed(gate_run):
    rep = gate_run.gatekeeper.rejection_report()
    assert sum(rep["branch_a"].values()) > 0 and sum(rep["branch_b"].values()) > 0
    assert len(rep["branch_a"]) >= 3, rep  # a MIX of reasons...
    assert "ring_count_change" in rep["branch_a"]  # ...and the ring-count check demonstrably fires


def test_stage2_gate_branches_are_structurally_different(gate_run):
    s = summarize(gate_run)
    assert s["scaffold_jaccard_branch_a_branch_b"] < 1.0
    a = {c.core_scaffold for c in gate_run.all_candidates if c.origin_branch == "branch_a"}
    b = {c.core_scaffold for c in gate_run.all_candidates if c.origin_branch == "branch_b"}
    assert a != b and b - a
