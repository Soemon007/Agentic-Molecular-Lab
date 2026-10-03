"""Orchestrator. Stage 1 only provides generation-0 seeding; the Omnigent session,
policies and main loop arrive in Stage 2."""
from __future__ import annotations

import chem_core
from chem_core import Candidate, Gatekeeper, ScaffoldBeam, SEEDS


def seed_beam(oracle, beam: ScaffoldBeam | None = None, gatekeeper: Gatekeeper | None = None) -> ScaffoldBeam:
    """Generation 0: 5 known DRD2 ligands of mixed strength, scored via the oracle (they count)."""
    beam = beam or ScaffoldBeam()
    for name, smi in SEEDS.items():
        if gatekeeper is not None:
            gatekeeper.register_seen(smi)
        c = Candidate(smiles=smi, origin_branch="seed", generation=0)
        oracle.score_candidate(c)
        beam.add(c)
    return beam
