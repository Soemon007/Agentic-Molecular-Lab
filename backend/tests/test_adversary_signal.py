"""The adversary's detector: plausibility filter, the level-type trigger, the digest, and the call rate limit."""
import asyncio
import json
from types import SimpleNamespace

import pytest

import policies
import telemetry
from chem_core import Candidate, Gatekeeper, ScaffoldBeam
from orchestrator import ADVERSARY_MIN_GAP, run_lab


def test_gatekeeper_rejects_free_thiols_and_acyclic_aminals_for_agent_branches_only():
    gk = Gatekeeper()
    assert gk.check("Sc1ccccc1", "branch_a", "c1ccccc1")[1] == "implausible_thiol"
    assert gk.check("CN(C)C(N(C)C)c1ccccc1", "branch_c", "c1ccccc1")[1] == "implausible_aminal"
    assert gk.check("Fc1ccccc1", "branch_a", "c1ccccc1")[0] is not None  # a plain substituent still passes
    # seeds and the baselines are not subject to the agents' filter
    assert Gatekeeper().check("Sc1ccccc1", "seed")[0] is not None
    assert Gatekeeper().check("Sc1ccccc1", "single_call_llm")[0] is not None
    assert gk.rejections["branch_a"]["implausible_thiol"] == 1


def test_aminal_pattern_leaves_amides_ureas_and_ring_aminals_alone():
    gk = Gatekeeper()
    for smi in ("O=C(NCN1CCCC1)c1ccccc1", "CC(=O)NCNC(C)=O", "C1CN(C)CN1C"):  # amide-N, bis-amide, ring aminal
        assert gk.check(smi, "branch_c")[1] != "implausible_aminal", smi


def _stats(**kw):
    base = {"unique_scaffolds_top10": 5, "sa_trend_3r": 0.0, "ad_similarity_trend": 0.0,
            "top10_mean_score": 0.95, "top10_mean_ad": 0.40, "ad_known_ligand_floor": 0.51}
    return {"stats": {**base, **kw}}


def test_high_score_at_low_domain_fires_the_level_trigger():
    assert telemetry.trigger(_stats()) == ["high_score_low_domain"]
    assert telemetry.trigger(_stats(top10_mean_ad=0.50)) == []  # within the margin of the known-ligand floor
    assert telemetry.trigger(_stats(top10_mean_score=0.5)) == []  # a low score at low AD is just a weak search
    assert telemetry.trigger(_stats(ad_known_ligand_floor=None)) == []  # no floor, no condition (old digests, unit tests)
    assert telemetry.trigger(_stats(top10_mean_ad=0.1, sa_trend_3r=2.0)) == ["sa_creep", "high_score_low_domain"]


def test_digest_carries_the_floor_the_top_ten_means_and_the_new_motifs():
    beam = ScaffoldBeam()
    beam.add([Candidate(smiles="Sc1ccccc1" if i == 0 else "CCO", score=0.9 - i * 0.01, core_scaffold=f"s{i}",
                        origin_branch="branch_a", sa_score=2.0, ad_similarity=0.4) for i in range(12)])
    d = telemetry.build_digest(beam, [{"round": 0, "sa_top10": 2.0, "ad_top10": 0.4}], 1, 20, ad_floor=0.51)
    assert d["stats"]["ad_known_ligand_floor"] == 0.51 and d["stats"]["top10_mean_ad"] == 0.4
    assert 0.8 < d["stats"]["top10_mean_score"] < 0.9
    assert any(row["motifs"].get("thiol") == 1 for row in d["top_k"]) and all("aminal" in r["motifs"] for r in d["top_k"])
    assert telemetry.build_digest(beam, [], 1, 20)["stats"]["ad_known_ligand_floor"] is None


def test_known_ligand_floor_is_the_lowest_ad_of_the_five_known_drugs():
    assert 0.40 < telemetry.known_ligand_ad_floor() < 0.60  # measured 0.51 (chlorpromazine) on this oracle build


def test_a_persistent_trigger_calls_the_adversary_at_most_once_per_min_gap(tmp_path, monkeypatch):
    policies.AUTOPILOT = True
    monkeypatch.setattr(telemetry, "trigger", lambda d: ["high_score_low_domain"])  # always on, like the real level trigger
    sess = asyncio.run(run_lab(budget=200, branches="abc", llm="offline", seed=1, verbose=False,
                               traj_path=tmp_path / "t.csv", run_id="gap_test"))
    log = [json.loads(line) for line in sess.chatlog.path.read_text().splitlines()]
    sess.chatlog.path.unlink()
    rounds = [r["round"] for r in log if r["kind"] == "agent_call" and r["agent"] == "adversary"]
    assert rounds, "the adversary should have been called at least once"
    assert all(b - a >= ADVERSARY_MIN_GAP for a, b in zip(rounds, rounds[1:]))
    assert any(t.get("skipped") == "rate_limit" for t in sess.trigger_log)
