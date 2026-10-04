"""Telemetry digest + deterministic adversary trigger. Free, code-only, no LLM.

The digest is what the adversary sees. It deliberately carries NO raw SMILES: structures are
summarised as Murcko scaffolds and functional-motif counts, so the LLM judges population-level
behaviour (score vs. domain drift, diversity, SA creep), not individual molecules.
"""
from __future__ import annotations

from collections import Counter

from rdkit import Chem

import chem_core

BRANCH_LABEL = {"branch_a": "local", "branch_b": "hopper", "branch_c": "explorer", "seed": "seed"}
TREND_WINDOW = 3
TRIGGER = {"unique_scaffolds_top10_lt": 3, "sa_trend_3r_gt": 1.2, "ad_similarity_trend_lt": -0.15}

_MOTIFS = {
    "basic_amine": "[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]",
    "charged_N": "[N+,n+;!$([N+][O-]);!$([n+][O-])]",
    "carboxylic_acid": "C(=O)[OH]",
    "halogen": "[F,Cl,Br,I]",
    "aryl_ring": "a1aaaaa1",
}
_MOTIF_Q = {k: Chem.MolFromSmarts(v) for k, v in _MOTIFS.items()}


def motifs(mol) -> dict:
    return {k: len(mol.GetSubstructMatches(q)) for k, q in _MOTIF_Q.items()}


def snapshot(beam, k: int = 10) -> dict:
    """Per-round top-k means used for trend computation."""
    top = beam.top(k)
    n = max(len(top), 1)
    return {"sa_top10": sum(c.sa_score or 0 for c in top) / n, "ad_top10": sum(c.ad_similarity or 0 for c in top) / n}


def trend(history: list[dict], key: str, rnd: int) -> float:
    """Change of `key` over the last 3 rounds. 0.0 when rnd < 3 (history[0] is the post-seeding snapshot)."""
    if rnd < TREND_WINDOW or len(history) <= rnd:
        return 0.0
    return float(history[rnd][key] - history[rnd - TREND_WINDOW][key])


def build_digest(beam, history: list[dict], rnd: int, oracle_calls_used: int, k: int = 10) -> dict:
    top = beam.top(k)
    top_k = []
    for c in top:
        mol = chem_core.mol_from_smiles(c.smiles)
        top_k.append({"id": c.id, "score": round(c.score, 3), "scaffold": c.core_scaffold,
                      "sa_score": round(c.sa_score or 0, 2), "ad_similarity": round(c.ad_similarity or 0, 3),
                      "alerts": c.alerts, "origin": BRANCH_LABEL.get(c.origin_branch, c.origin_branch),
                      "agent_generated": c.agent_generated, "motifs": motifs(mol) if mol else {}})
    by_branch = Counter(c.origin_branch for c in top)
    dom, n_dom = by_branch.most_common(1)[0]
    return {"top_k": top_k, "stats": {
        "unique_scaffolds_top10": len({c.core_scaffold for c in top}),
        "dominant_branch": f"{BRANCH_LABEL.get(dom, dom)} {n_dom}/{len(top)}",
        "sa_trend_3r": round(trend(history, "sa_top10", rnd), 3),
        "ad_similarity_trend": round(trend(history, "ad_top10", rnd), 3),
        "oracle_calls_used": oracle_calls_used, "round": rnd}}


def trigger(digest: dict) -> list[str]:
    """Deterministic: returns the list of fired conditions (empty = no trigger).
    unique_scaffolds_top10 < 3 OR sa_trend_3r > 1.2 OR ad_similarity_trend < -0.15"""
    s, fired = digest["stats"], []
    if s["unique_scaffolds_top10"] < TRIGGER["unique_scaffolds_top10_lt"]:
        fired.append("low_scaffold_diversity")
    if s["sa_trend_3r"] > TRIGGER["sa_trend_3r_gt"]:
        fired.append("sa_creep")
    if s["ad_similarity_trend"] < TRIGGER["ad_similarity_trend_lt"]:
        fired.append("ad_similarity_drop")
    return fired


def flagged_branch(beam, k: int = 10) -> str | None:
    """Branch to blame: the non-seed branch with the most members in the top-k (None if all seeds)."""
    c = Counter(m.origin_branch for m in beam.top(k) if m.origin_branch != "seed")
    return c.most_common(1)[0][0] if c else None
