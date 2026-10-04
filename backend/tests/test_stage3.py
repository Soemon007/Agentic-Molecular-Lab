import asyncio
import json

import pytest

import telemetry
from agents.coordinator import Coordinator
from chem_core import Candidate, ScaffoldBeam
from orchestrator import run_lab


def _cand(i, scaf, score, branch, sa=2.0, ad=0.6):
    return Candidate(smiles=f"C{i}", score=score, core_scaffold=scaf, origin_branch=branch, sa_score=sa, ad_similarity=ad)


def _hist(vals):  # vals: list of (sa, ad) for rounds 0..n
    return [{"round": i, "sa_top10": sa, "ad_top10": ad} for i, (sa, ad) in enumerate(vals)]


def test_trend_is_zero_before_round_3_and_never_crashes():
    h = _hist([(2, .6), (9, .1)])
    assert telemetry.trend(h, "sa_top10", 1) == 0.0 and telemetry.trend(h, "ad_top10", 2) == 0.0
    beam = ScaffoldBeam(); beam.add([_cand(i, f"s{i}", 0.9 - i * .01, "branch_c") for i in range(12)])
    d = telemetry.build_digest(beam, _hist([(2, .6)]), 1, 20)  # round 1, short history
    assert d["stats"]["sa_trend_3r"] == 0.0


def test_trend_over_three_rounds():
    h = _hist([(2.0, .6), (2.1, .6), (2.5, .5), (3.4, .35)])
    assert telemetry.trend(h, "sa_top10", 3) == pytest.approx(1.4)
    assert telemetry.trend(h, "ad_top10", 3) == pytest.approx(-0.25)


@pytest.mark.parametrize("stats,expected", [
    ({"unique_scaffolds_top10": 2, "sa_trend_3r": 0, "ad_similarity_trend": 0}, ["low_scaffold_diversity"]),
    ({"unique_scaffolds_top10": 5, "sa_trend_3r": 1.3, "ad_similarity_trend": 0}, ["sa_creep"]),
    ({"unique_scaffolds_top10": 5, "sa_trend_3r": 0, "ad_similarity_trend": -0.2}, ["ad_similarity_drop"]),
    ({"unique_scaffolds_top10": 3, "sa_trend_3r": 1.2, "ad_similarity_trend": -0.15}, []),  # boundaries don't fire
])
def test_trigger_conditions(stats, expected):
    assert telemetry.trigger({"stats": stats}) == expected


def test_digest_has_no_raw_smiles_and_tags_agent_generated():
    beam = ScaffoldBeam()
    beam.add([_cand(i, f"s{i%4}", 0.9 - i * .01, "branch_c" if i % 2 else "seed") for i in range(12)])
    d = telemetry.build_digest(beam, _hist([(2, .6)] * 5), 4, 99)
    assert all("smiles" not in row for row in d["top_k"])
    assert all("agent_generated" in row for row in d["top_k"])
    assert d["stats"]["dominant_branch"].split()[0] in ("explorer", "seed")


def test_coordinator_rules():
    c = Coordinator(["branch_a", "branch_b", "branch_c"])
    assert c.quotas() == {"branch_a": 4, "branch_b": 4, "branch_c": 4}
    assert c.flag("branch_c", "retain biaryl")
    q = c.quotas()
    assert q["branch_c"] == 1 and sum(q.values()) == 12 and min(q.values()) >= 1  # floor, total preserved
    assert not c.flag("branch_c", "again")  # no re-flag during cooldown
    assert c.instructions() == {"branch_c": "retain biaryl"}
    c.end_round(); assert c.quotas()["branch_c"] == 1
    c.end_round(); assert c.quotas() == {"branch_a": 4, "branch_b": 4, "branch_c": 4} and not c.instructions()
    assert Coordinator(["branch_a"]).quotas() == {"branch_a": 4}
    solo = Coordinator(["branch_a"]); solo.flag("branch_a", "x"); assert solo.quotas()["branch_a"] == 1


def test_end_to_end_adversary_trigger_flags_coordinator_and_injects_instruction(tmp_path, monkeypatch):
    """Plumbing test with an INJECTED trigger at round 3 (mock runs never drift on their own)."""
    import policies
    policies.AUTOPILOT = True
    real = telemetry.trigger
    monkeypatch.setattr(telemetry, "trigger", lambda d: ["ad_similarity_drop"] if d["stats"]["round"] == 3 else real(d))
    sess = asyncio.run(run_lab(budget=150, branches="abc", llm="offline", seed=1, verbose=False,
                               traj_path=tmp_path / "t.csv", run_id="e2e_test"))
    assert sess.adversary_calls == 1
    fired = [t for t in sess.trigger_log if t["fired"]]
    assert len(fired) == 1 and fired[0]["acted"] and fired[0]["round"] == 3
    flagged = fired[0]["flagged"]
    qs = {h["round"]: h["quotas"] for h in sess.history[1:]}
    assert qs[3][flagged] == 4 and qs[4][flagged] == 1 and qs[5][flagged] == 1 and qs[6][flagged] == 4
    log = [json.loads(l) for l in sess.chatlog.path.read_text().splitlines()]
    calls = [r for r in log if r["kind"] == "agent_call"]
    adv = [r for r in calls if r["agent"] == "adversary"]
    assert len(adv) == 1 and "smiles" not in json.dumps(adv[0]["input"]["top_k"])
    inj = {r["round"]: r["input"].get("instruction") for r in calls if r["agent"] == flagged}
    assert inj[4] and inj[5] and not inj[6] and not inj[3]  # instruction lives exactly the cooldown window
    outcomes = [r for r in log if r["kind"] == "proposal_outcome"]
    assert outcomes and all(o["agent_generated"] for o in outcomes)
    assert {o["call_id"] for o in outcomes} <= {c["call_id"] for c in calls}  # outcomes join to calls
    assert sess.oracle.calls == 150
    sess.chatlog.path.unlink()
