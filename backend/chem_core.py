"""Deterministic chemistry core: Candidate, Gatekeeper, scaffold-niched beam, alerts.

Nothing in this module calls an LLM or the oracle.
"""
from __future__ import annotations

import itertools
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

MAX_MW = 600.0
MAX_LOGP = 6.0

# generation-0 reference ligands (they go through the oracle and count against budget)
SEEDS = {
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "chlorpromazine": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
    "aripiprazole": "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
    "haloperidol": "O=C(CCCN1CCC(O)(CC1)c1ccc(Cl)cc1)c1ccc(F)cc1",
    "risperidone": "Cc1nc2n(c(=O)c1CCN1CCC(CC1)c1noc3cc(F)ccc13)CCCC2",
}

_id_counter = itertools.count()


def _new_id() -> str:
    return f"c{next(_id_counter):06d}"


@dataclass
class Candidate:
    smiles: str
    score: float | None = None  # only the oracle wrapper may write this
    origin_branch: str = "unknown"  # "seed" | "branch_a" | ... ; anything but seed is agent-generated
    core_scaffold: str = ""
    parent_id: str | None = None
    generation: int = 0
    sa_score: float | None = None
    ad_similarity: float | None = None
    alerts: list[str] = field(default_factory=list)
    id: str = field(default_factory=_new_id)

    @property
    def agent_generated(self) -> bool:
        return self.origin_branch != "seed"


# --------------------------------------------------------------------------- chemistry helpers

def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical(smiles: str) -> str | None:
    m = mol_from_smiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def scaffold(mol) -> str:
    """Murcko scaffold SMILES ('' for acyclic molecules)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def sa_score(mol) -> float:
    return float(sascorer.calculateScore(mol))


# --------------------------------------------------------------------------- alerts (PAINS + BRENK)

def _build_catalogs():
    out = {}
    for name in ("PAINS", "BRENK"):
        p = FilterCatalogParams()
        p.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
        out[name] = FilterCatalog(p)
    return out


_CATALOGS = _build_catalogs()  # initialised once

# BRENK entries that are genuine electrophiles; ONLY these gate human approval.
# Everything else (PAINS, other BRENK) is telemetry for the adversary.
ELECTROPHILE_ALERTS = frozenset({
    "Michael_acceptor_1", "Michael_acceptor_2", "Michael_acceptor_3", "Michael_acceptor_4",
    "Michael_acceptor_5", "acid_halide", "aldehyde", "alkyl_halide", "isocyanate",
    "Three-membered_heterocycle", "N-halo", "N-C-halo", "sulfonyl_cyanide", "acyl_cyanide",
    "ketene", "diazo_group", "triflate", "chinone_1", "chinone_2", "cyanate_/aminonitrile_/thiocyanate",
    "four_member_lactones", "silicon_halogen",
})


def get_alerts(mol) -> list[str]:
    """All PAINS/BRENK matches, as 'PAINS:name' / 'BRENK:name'. Uses GetMatches, so
    repeated hits count: alert count is signal for the adversary."""
    out = []
    for src, cat in _CATALOGS.items():
        for m in cat.GetMatches(mol):
            out.append(f"{src}:{m.GetDescription()}")
    return out


def electrophile_alerts(alerts: list[str]) -> list[str]:
    return [a for a in alerts if a.startswith("BRENK:") and a.split(":", 1)[1] in ELECTROPHILE_ALERTS]


# --------------------------------------------------------------------------- branch rules

def is_valid_branch_a(mol, parent_mol) -> bool:
    # ring-count check FIRST: Murcko counts every ring as scaffold, so appending
    # morpholine changes the scaffold. Without this, Branch A is silently reduced
    # to halogen edits only.
    if rdMolDescriptors.CalcNumRings(mol) != rdMolDescriptors.CalcNumRings(parent_mol):
        return False
    return scaffold(mol) == scaffold(parent_mol)


def is_valid_branch_b(mol, parent_mol) -> bool:
    return scaffold(mol) != scaffold(parent_mol)


# --------------------------------------------------------------------------- Gatekeeper

class Gatekeeper:
    """Deterministic, zero-LLM validity filter. Rejections are logged by branch AND reason."""

    def __init__(self, max_mw: float = MAX_MW, max_logp: float = MAX_LOGP):
        self.max_mw = max_mw
        self.max_logp = max_logp
        self.rejections: dict[str, Counter] = defaultdict(Counter)
        self.accepted: Counter = Counter()
        self.seen: set[str] = set()  # canonical SMILES already sent to (or queued for) the oracle

    def register_seen(self, smiles: str) -> None:
        c = canonical(smiles)
        if c:
            self.seen.add(c)

    def check(self, smiles: str, branch: str, parent_smiles: str | None = None):
        """Return (mol_or_None, reason_or_None). On accept the canonical SMILES is marked seen."""
        mol = mol_from_smiles(smiles)
        if mol is None:
            return self._reject(branch, "invalid_smiles")
        if Descriptors.MolWt(mol) > self.max_mw:
            return self._reject(branch, "mw")
        if Descriptors.MolLogP(mol) > self.max_logp:
            return self._reject(branch, "logp")
        can = Chem.MolToSmiles(mol)
        if can in self.seen:
            return self._reject(branch, "duplicate")

        if branch in ("branch_a", "branch_b"):
            parent = mol_from_smiles(parent_smiles) if parent_smiles else None
            if parent is None:
                return self._reject(branch, "missing_parent")
            if branch == "branch_a":
                if rdMolDescriptors.CalcNumRings(mol) != rdMolDescriptors.CalcNumRings(parent):
                    return self._reject(branch, "ring_count_change")
                if not is_valid_branch_a(mol, parent):
                    return self._reject(branch, "scaffold_change")
            elif not is_valid_branch_b(mol, parent):
                return self._reject(branch, "scaffold_unchanged")

        self.seen.add(can)
        self.accepted[branch] += 1
        return mol, None

    def _reject(self, branch, reason):
        self.rejections[branch][reason] += 1
        return None, reason

    def rejection_report(self) -> dict[str, dict[str, int]]:
        return {b: dict(c) for b, c in self.rejections.items()}


# --------------------------------------------------------------------------- scaffold-niched beam

class ScaffoldBeam:
    """Top-`size` candidates, niched by Murcko family.

    Rule: at most `max_per_family` per family *while diversity can fill the beam*.
    Overflow members only backfill empty slots (smallest family first), so the beam never
    starves when few scaffolds exist. Pruning is global (len <= size always) and evicts from
    the LARGEST family, never its last member (unless every family is a singleton, where
    niching is moot and the globally lowest score goes).
    """

    def __init__(self, size: int = 20, max_per_family: int = 2):
        self.size = size
        self.max_per_family = max_per_family
        self.members: list[Candidate] = []

    # -- helpers
    @staticmethod
    def _key(c: Candidate) -> float:
        return c.score if c.score is not None else float("-inf")

    def _families(self, cands):
        fam = defaultdict(list)
        for c in cands:
            fam[c.core_scaffold].append(c)
        for v in fam.values():
            v.sort(key=self._key, reverse=True)
        return fam

    def add(self, cands) -> None:
        if isinstance(cands, Candidate):
            cands = [cands]
        pool = {c.id: c for c in self.members}
        for c in cands:
            pool[c.id] = c
        fam = self._families(pool.values())

        core, overflow = [], defaultdict(list)
        for f, ms in fam.items():
            core += ms[: self.max_per_family]
            if len(ms) > self.max_per_family:
                overflow[f] = ms[self.max_per_family:]

        if len(core) > self.size:
            core = self._evict_to_size(core)
        else:
            counts = Counter(c.core_scaffold for c in core)
            while len(core) < self.size and overflow:
                # smallest current family first; ties -> highest-scoring candidate
                f = min(overflow, key=lambda k: (counts[k], -self._key(overflow[k][0])))
                c = overflow[f].pop(0)
                core.append(c)
                counts[f] += 1
                if not overflow[f]:
                    del overflow[f]
        self.members = sorted(core, key=self._key, reverse=True)

    def _evict_to_size(self, cands):
        cands = list(cands)
        while len(cands) > self.size:
            fam = self._families(cands)
            biggest = max(len(v) for v in fam.values())
            if biggest == 1:  # all singletons
                victim = min(cands, key=self._key)
            else:
                victim = min((v[-1] for v in fam.values() if len(v) == biggest), key=self._key)
            cands.remove(victim)
        return cands

    # -- views
    def top(self, k: int = 10) -> list[Candidate]:
        return sorted(self.members, key=self._key, reverse=True)[:k]

    def bottom_percentile(self, frac: float = 0.5) -> list[Candidate]:
        ranked = sorted(self.members, key=self._key, reverse=True)
        return ranked[int(len(ranked) * (1 - frac)):]

    def scaffold_counts(self) -> Counter:
        return Counter(c.core_scaffold for c in self.members)

    def __len__(self):
        return len(self.members)

    def __iter__(self):
        return iter(self.members)
