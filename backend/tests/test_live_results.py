"""eval/live_results.py: live runs -> results_<tag>.json in the shape /api/results serves."""
import json

import pytest

from eval import live_results
from eval.pmo_auc import top10_auc


def _write_run(root, run_id, scores, fired=(), acted=()):
    (root / "trajectories").mkdir(exist_ok=True)
    (root / "chats").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    rows = ["call_n,smiles,score,origin_branch,sa,ad_similarity,alerts"]
    rows += [f"{i},{'C' * (i + 1)},{s},branch_a,2.0,0.5," for i, s in enumerate(scores, 1)]
    (root / "trajectories" / f"ours_{run_id}.csv").write_text("\n".join(rows) + "\n")
    ev = [{"kind": "event", "event": "adversary_trigger", "round": r, "acted": r in acted} for r in fired]
    (root / "chats" / f"{run_id}.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + ("\n" if ev else ""))
    (root / "runs" / f"{run_id}.json").write_text(json.dumps(
        {"summary": {"adversary_calls": len(acted), "tokens": {"total": 1000 + len(scores)}}}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(live_results, "DATA", tmp_path)
    return tmp_path


def test_build_uses_each_runs_own_trajectory_and_reports_trigger_activity(root):
    a, b = [0.1 * (i % 5) for i in range(100)], [0.05 * (i % 7) for i in range(100)]
    _write_run(root, "r_on_0", a, fired=(3, 9), acted=(3,))
    _write_run(root, "r_on_1", b)
    _write_run(root, "r_off_0", b)
    _write_run(root, "r_off_1", a)
    res = live_results.build("t", ["r_on_0", "r_on_1"], ["r_off_0", "r_off_1"], 100, None, ["a note"])

    expect = top10_auc([("C" * (i + 1)) for i in range(100)], a, 100)
    assert res["ours"]["100"]["values"][0] == pytest.approx(expect, abs=1e-4)
    assert set(res["ours"]) == {"50", "100"} and res["llm_mode"] == "anthropic" and res["seed_mode"] == "cold"
    assert res["trigger_rounds"] == [2, 0] and res["acted_rounds"] == [1, 0] and res["adversary_calls"] == [1, 0]
    assert res["tokens"] == [1100, 1100] and res["notes"] == ["a note"]
    assert json.loads((root / "results_t.json").read_text())["run_ids"]["ablated"] == ["r_off_0", "r_off_1"]


def test_single_call_is_reported_at_50_only_and_ablated_may_be_absent(root):
    _write_run(root, "r0", [0.2] * 60)
    (root / "trajectories" / "sc.csv").write_text(
        "call_n,smiles,score,origin_branch,sa,ad_similarity,alerts\n" + "\n".join(f"{i},{'N' * i},0.9,x,,," for i in range(1, 51)) + "\n")
    res = live_results.build("t2", ["r0"], [], 100, str(root / "trajectories" / "sc.csv"), [])
    assert res["ablated"] == {} and list(res["single_call"]) == ["50"] and res["single_call"]["50"] > 0.8


def test_arms_may_be_absent_and_the_adversary_on_arm_drives_the_trigger_columns(root):
    a, b = [0.1 * (i % 5) for i in range(100)], [0.3 * (i % 3) for i in range(100)]
    for rid in ("o0", "o1"):
        _write_run(root, rid, a)
    for rid in ("n0", "n1"):
        _write_run(root, rid, b, fired=(4,), acted=(4,))
    res = live_results.build("t3", ["n0", "n1"], [], 100, None, [])
    assert res["ablated"] == {} and set(res["ours"]) == {"50", "100"}
    assert res["trigger_rounds"] == [1, 1] and res["acted_rounds"] == [1, 1] and res["seeds"] == [0, 1]
    only_off = live_results.build("t4", [], ["o0", "o1"], 100, None, [])  # e.g. runs where the adversary was inoperative
    assert only_off["ours"] == {} and only_off["seeds"] == [0, 1] and only_off["run_ids"] == {"ours": [], "ablated": ["o0", "o1"]}
