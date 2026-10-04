import csv
import random

import pytest
from rdkit import Chem

import chem_core
import domain
from chem_core import Candidate, Gatekeeper, ScaffoldBeam
from orchestrator import seed_beam


def test_branch_a_accepts_fluoro_rejects_morpholine():
    gk = Gatekeeper()
    mol, reason = gk.check("Fc1ccccc1", "branch_a", "c1ccccc1")
    assert mol is not None and reason is None
    mol, reason = gk.check("c1ccc(N2CCOCC2)cc1", "branch_a", "c1ccccc1")
    assert mol is None and reason == "ring_count_change"
    # direct rule functions too
    b, p = Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("c1ccc(N2CCOCC2)cc1")
    assert chem_core.is_valid_branch_a(Chem.MolFromSmiles("Fc1ccccc1"), b)
    assert not chem_core.is_valid_branch_a(p, b)
    assert chem_core.is_valid_branch_b(p, b)  # new ring system = Branch B territory


def test_gatekeeper_reasons_are_broken_out():
    gk = Gatekeeper()
    gk.check("not_a_smiles", "branch_a", "c1ccccc1")
    gk.check("C" * 80, "branch_a", "c1ccccc1")  # logP/MW
    gk.check("c1ccccc1", "branch_b", "c1ccccc1")  # same scaffold
    rep = gk.rejection_report()
    assert rep["branch_a"]["invalid_smiles"] == 1
    assert rep["branch_b"]["scaffold_unchanged"] == 1
    assert sum(rep["branch_a"].values()) == 2


def test_alerts_use_all_matches_and_electrophile_subset():
    m = Chem.MolFromSmiles("C=CC(=O)c1ccccc1")  # Michael acceptor
    alerts = chem_core.get_alerts(m)
    assert any("Michael_acceptor" in a for a in alerts)
    assert chem_core.electrophile_alerts(alerts)
    assert not chem_core.electrophile_alerts(chem_core.get_alerts(Chem.MolFromSmiles("Nc1ccccc1")))  # aniline: telemetry only


def _fake(i, scaf, score):
    return Candidate(smiles=f"C{i}", score=score, core_scaffold=scaf, origin_branch="branch_c")


def test_beam_30_candidates_3_scaffolds_prunes_to_20_keeping_all_3():
    rng = random.Random(0)
    cands = [_fake(i, ["s1", "s2", "s3"][i % 3], rng.random()) for i in range(30)]
    cands[0].score = -1.0  # worst member of s1 must not become its "last member" problem
    beam = ScaffoldBeam(size=20, max_per_family=2)
    beam.add(cands)
    assert len(beam) == 20
    assert set(beam.scaffold_counts()) == {"s1", "s2", "s3"}
    assert max(beam.scaffold_counts().values()) - min(beam.scaffold_counts().values()) <= 1  # evicted from largest


def test_beam_stays_bounded_and_enforces_family_cap_when_diverse():
    rng = random.Random(1)
    beam = ScaffoldBeam(size=20, max_per_family=2)
    for r in range(10):  # repeated rounds of new scaffolds must not grow the beam
        beam.add([_fake(r * 100 + i, f"s{rng.randrange(40)}", rng.random()) for i in range(15)])
        assert len(beam) <= 20
    assert max(beam.scaffold_counts().values()) <= 2
    assert len(beam.top(10)) == 10


def test_beam_never_evicts_last_family_member_in_favour_of_dupes():
    beam = ScaffoldBeam(size=4, max_per_family=2)
    beam.add([_fake(i, "big", 0.9 - i * 0.01) for i in range(6)] + [_fake(10, "rare", 0.01), _fake(11, "rare2", 0.02)])
    assert {"rare", "rare2"} <= set(beam.scaffold_counts())


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    from oracle import DRD2Oracle
    d = tmp_path_factory.mktemp("o")
    o = DRD2Oracle(traj_path=d / "t.csv", fail_log=d / "f.log", report_at_exit=False)
    yield o
    o._fh.close()


def test_oracle_scores_and_counts_seeding(oracle):
    assert 0.0 <= oracle("CC(=O)Nc1ccc(O)cc1", "test") <= 1.0
    assert oracle.calls == 1
    assert oracle("this is not smiles", "test") == 0.0
    assert oracle.calls == 2  # invalid SMILES still costs budget


def test_seeding_counter_and_trajectory(tmp_path):
    from oracle import DRD2Oracle
    o = DRD2Oracle(traj_path=tmp_path / "t.csv", fail_log=tmp_path / "f.log", report_at_exit=False)
    beam = seed_beam(o)
    o._fh.flush()
    assert o.calls == 5
    rows = list(csv.DictReader((tmp_path / "t.csv").open()))
    assert len(rows) == 5 and [r["call_n"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert all(r["origin_branch"] == "seed" for r in rows)
    assert len(beam) == 5
    scores = {c.smiles: c.score for c in beam}
    assert scores[chem_core.SEEDS["dopamine"]] < scores[chem_core.SEEDS["haloperidol"]]
    assert o.failures == 0
    o._fh.close()


def test_failures_count_and_log(tmp_path):
    from oracle import DRD2Oracle
    o = DRD2Oracle(traj_path=tmp_path / "t.csv", fail_log=tmp_path / "f.log", report_at_exit=False)
    o._oracle = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    assert o("CCO", "test") == 0.0
    assert o.calls == 1 and o.failures == 1 and o.failure_rate == 1.0
    assert "boom" in (tmp_path / "f.log").read_text()
    o._fh.close()


def test_domain_reference_size_and_direction():
    assert 1_000 < len(domain.reference()) < 50_000
    hal = domain.ad_similarity(Chem.MolFromSmiles(chem_core.SEEDS["haloperidol"]))
    junk = domain.ad_similarity(Chem.MolFromSmiles("C" * 10))
    assert hal > junk  # HIGH = in-domain


def test_cold_seeds_are_counted_valid_and_not_the_known_drugs(tmp_path):
    from oracle import DRD2Oracle
    o = DRD2Oracle(traj_path=tmp_path / "t.csv", fail_log=tmp_path / "f.log", report_at_exit=False)
    beam = seed_beam(o, mode="cold", seed=0)
    assert o.calls == 5 and len(beam) >= 1 and o.failures == 0
    assert not set(c.smiles for c in beam) & set(chem_core.SEEDS.values())
    o._fh.close()
