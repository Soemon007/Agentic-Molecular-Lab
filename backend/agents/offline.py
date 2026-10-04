"""Offline stand-in for the LLM branches: RDKit edits + a configurable rate of rule-violating
proposals (as real LLMs also make). Used only when no API key is available. Plumbing test, not evidence."""
from __future__ import annotations

import random

from rdkit import Chem
from rdkit.Chem import AllChem

SUBS_A = ["F", "Cl", "C(F)(F)F", "O", "OC", "N(C)C", "C", "CC", "C(C)C"]
RING_SUBS = ["N1CCOCC1", "c1ccccc1", "C1CC1", "N1CCCC1"]  # violate "no new rings"
BIOISOSTERES = [  # (reaction SMARTS)
    "[c:1][OH:2]>>[c:1][NH2:2]", "[c:1]Cl>>[c:1]C(F)(F)F", "[c:1]F>>[c:1]Cl", "[C:1][OH:2]>>[C:1][NH2:2]",
    "[c:1]C(F)(F)F>>[c:1]S(C)(=O)=O", "[CH3:1][c:2]>>[CH3:1]O[c:2]", "[c:1]O[CH3:2]>>[c:1]SC",
    "[C:1](=O)[N:2]>>[C:1](=S)[N:2]", "[c:1]Cl>>[c:1]C#N", "[c:1]F>>[c:1]OC(F)(F)F",
]


def _attach(mol, idx, frag_smiles):
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    n = mol.GetNumAtoms()
    em = Chem.RWMol(Chem.CombineMols(mol, frag))
    em.AddBond(idx, n, Chem.BondType.SINGLE)
    try:
        m = em.GetMol(); Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _h_sites(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]


def _ring_swap(mol, rng, aromatic=True):
    sites = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and a.GetTotalNumHs() > 0
             and a.GetIsAromatic() == aromatic and a.GetAtomicNum() == 6]
    if not sites:
        return None
    em = Chem.RWMol(mol)
    a = em.GetAtomWithIdx(rng.choice(sites))
    a.SetAtomicNum(rng.choice([7, 8, 16]) if not aromatic else 7)
    a.SetNoImplicit(False); a.SetNumExplicitHs(0)
    try:
        m = em.GetMol(); Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _trim(mol, rng):
    """Remove a terminal, non-ring heavy atom (keeps ring system/scaffold): what an LLM does when told it is over the MW cap."""
    ends = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1 and not a.IsInRing()]
    if not ends:
        return None
    em = Chem.RWMol(mol)
    em.RemoveAtom(rng.choice(ends))
    try:
        m = em.GetMol(); Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _a(mol, rng, noise):
    r = rng.random()
    sites = _h_sites(mol)
    if r < 0.05 * noise:
        return "C(" + Chem.MolToSmiles(mol)
    if r < 0.30 * noise:
        return _attach(mol, rng.choice(sites), rng.choice(RING_SUBS)) if sites else None
    if r < 0.45 * noise:
        return _ring_swap(mol, rng)
    return _attach(mol, rng.choice(sites), rng.choice(SUBS_A)) if sites else None


def _b(mol, rng, noise):
    r = rng.random()
    if r < 0.20 * noise:  # scaffold-preserving edit: a Branch B mistake
        sites = _h_sites(mol)
        return _attach(mol, rng.choice(sites), "F") if sites else None
    return _ring_swap(mol, rng, aromatic=rng.random() < 0.6)


def _c(mol, rng, noise):
    rxns = BIOISOSTERES[:]
    rng.shuffle(rxns)
    for s in rxns:
        prods = AllChem.ReactionFromSmarts(s).RunReactants((mol,))
        for p in prods:
            try:
                m = p[0]; Chem.SanitizeMol(m)
                return Chem.MolToSmiles(m)
            except Exception:
                continue
    return None


EXPLOIT_TAILS = ["CCN(C)C", "C[N+](C)(C)C", "CCCN1CCN(CC1)C", "CCN(CC)CCN(C)C"]  # charged/basic amine stacking


def _c_exploit(mol, rng, noise):
    """Demo-only reward hacker: bolts amine motifs onto the molecule, growing it out of domain."""
    sites = _h_sites(mol)
    return _attach(mol, rng.choice(sites), rng.choice(EXPLOIT_TAILS)) if sites else None


def respond(agent: str, payload: dict, seed: int, noise: float = 1.0, exploit: bool = False) -> dict:
    rng = random.Random(seed)
    if agent == "scout":
        top = payload.get("beam", [])[:3]
        return {"brief": "Retain basic amine + aryl pharmacophore. Top: " + "; ".join(b["smiles"] for b in top)}
    if agent == "adversary":
        st = payload["stats"]
        return {"diagnosis": f"[offline mock] {st['dominant_branch']} dominates the top-10 while "
                             f"ad_similarity moved {st['ad_similarity_trend']:+.2f} and SA {st['sa_trend_3r']:+.2f} over 3 rounds.",
                "instruction": "Retain the core scaffold; drop appended charged amine motifs; vary the linker instead."}
    fn = {"branch_a": _a, "branch_b": _b, "branch_c": _c_exploit if exploit else _c}[agent]
    if "retain the core" in (payload.get("instruction") or "").lower():  # mock obeys the corrective instruction
        fn = {"branch_a": _a, "branch_b": _b, "branch_c": _c}[agent]
    beam = payload["beam"]
    if agent == "branch_c":  # explorer reads the LOWER half of the beam
        beam = beam[len(beam) // 2:] or beam
    out, tries = [], 0
    while len(out) < payload["n"] and tries < payload["n"] * 6:
        tries += 1
        parent = rng.choice(beam)
        mol = Chem.MolFromSmiles(parent["smiles"])
        if mol and parent.get("mw", 0) > 520 and agent in ("branch_a", "branch_c") and rng.random() < 0.8:
            smi = _trim(mol, rng)  # heavy parent: shrink instead of grow
        else:
            smi = fn(mol, rng, noise) if mol else None
        if smi:
            out.append({"parent_id": parent["id"], "smiles": smi})
    return {"proposals": out}
