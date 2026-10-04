"""API tests against a synthetic run (no oracle, no LLM): derived views, filters, approvals plumbing."""
import json

import pytest
from fastapi.testclient import TestClient

import server

SMILES = ["CCO", "CCN", "c1ccccc1O", "CC(=O)O"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    chats, traj = tmp_path / "chats", tmp_path / "traj"
    chats.mkdir(); traj.mkdir()
    monkeypatch.setattr(server, "CHATS", chats)
    monkeypatch.setattr(server, "TRAJ", traj)
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(server, "DATA", tmp_path)
    server._log_cache.clear()

    rows = ["call_n,smiles,score,origin_branch,sa,ad_similarity,alerts"]
    for i, smi in enumerate(SMILES, 1):
        rows.append(f"{i},{smi},{0.2 * i:.6f},{'seed' if i == 1 else 'branch_a'},2.0,0.5,")
    (traj / "ours_ours_abc_known_seed0.csv").write_text("\n".join(rows) + "\n")

    run = "ours_abc_known_seed0"
    recs = [
        {"ts": "2026-10-04T10:00:00", "kind": "event", "event": "round", "round": 0, "oracle_calls_used": 1, "best": 0.2,
         "top10_mean": 0.2, "sa_top10": 2.0, "ad_top10": 0.5},
        {"ts": "2026-10-04T10:00:01", "kind": "agent_call", "call_id": "c1", "round": 1, "agent": "branch_a", "model": "m",
         "system_prompt": "sp", "input": {"beam": []}, "output": {"tool": "submit_proposals", "args": {"proposals": []}},
         "usage": {"input_tokens": 10, "output_tokens": 5}, "latency_ms": 3, "verdict": {"allowed": True, "reason": None}},
        {"ts": "2026-10-04T10:00:01", "kind": "proposal_outcome", "call_id": "c1", "round": 1, "agent": "branch_a", "smiles": "CCN",
         "oracle_score": 0.4, "ad_similarity": 0.5, "sa_score": 2.0, "alerts": [], "gate_reason": None},
        {"ts": "2026-10-04T10:00:01", "kind": "proposal_outcome", "call_id": "c1", "round": 1, "agent": "branch_a", "smiles": "CCCl",
         "oracle_score": None, "gate_reason": "duplicate"},
        {"ts": "2026-10-04T10:00:02", "kind": "agent_call", "call_id": "c2", "round": 1, "agent": "scout", "model": "m",
         "system_prompt": "sp", "input": {}, "output": {"tool": None, "args": {}}, "usage": None, "latency_ms": 1,
         "verdict": {"allowed": False, "reason": "scout may not call x"}},
        {"ts": "2026-10-04T10:00:03", "kind": "event", "event": "adversary_trigger", "round": 1, "fired": ["sa_creep"],
         "flagged": "branch_a", "acted": False, "stats": {"unique_scaffolds_top10": 5, "sa_trend_3r": 1.5}},
    ]
    (chats / f"{run}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return TestClient(server.app), run


def test_run_listing_recovers_config_from_name(client):
    c, run = client
    row = next(r for r in c.get("/api/runs").json() if r["id"] == run)
    assert row["calls"] == 4 and row["adversary"] is True and row["seed_mode"] == "known" and row["branches"] == "abc"


def test_run_state_is_derived_from_files(client):
    c, run = client
    s = c.get(f"/api/runs/{run}").json()
    assert s["used"] == 4 and s["tokens"]["total"] == 15 and s["status"] == "finished"
    assert [m["smiles"] for m in s["beam"]][:2] == ["CC(=O)O", "c1ccccc1O"]  # best first
    assert s["gatekeeper"] == {"duplicate": 1} and s["policy"]["agent_denied"] == 1
    assert s["triggers"][0]["fired"] == ["sa_creep"] and len(s["rounds"]) == 1
    assert s["curve"][-1][1] == pytest.approx(0.8)


def test_call_filters_and_detail(client):
    c, run = client
    assert c.get(f"/api/runs/{run}/calls").json()["total"] == 2
    assert c.get(f"/api/runs/{run}/calls", params={"verdict": "denied"}).json()["items"][0]["agent"] == "scout"
    assert c.get(f"/api/runs/{run}/calls", params={"agent": "branch_a"}).json()["items"][0]["n_scored"] == 1
    d = c.get(f"/api/runs/{run}/calls/c1").json()
    assert d["system_prompt"] == "sp" and len(d["outcomes"]) == 2


def test_unknown_run_and_bad_input(client):
    c, _ = client
    assert c.get("/api/runs/nope").status_code == 404
    assert c.get("/api/runs/..%2Fetc").status_code == 404
    assert c.post("/api/runs", json={"branches": "xyz"}).status_code == 422
    assert c.post("/api/runs", json={"llm": "anthropic"}).status_code == 400  # no ANTHROPIC_API_KEY in tests


def test_molecule_svg(client):
    c, _ = client
    r = c.get("/api/mol.svg", params={"smiles": "c1ccccc1O"})
    assert r.status_code == 200 and "currentColor" in r.text
    assert c.get("/api/mol.svg", params={"smiles": "not a smiles"}).status_code == 422
