"""Independent check of a molecule against measured data: ChEMBL (EMBL-EBI), target CHEMBL217 = D(2) dopamine receptor.

The lab's DRD2 score is a classifier's opinion. This looks the molecule up in ChEMBL by structural similarity and
reports whether its nearest neighbours have measured DRD2 activity, with a link to every compound and document it relies
on. It does not measure the molecule: an "analogue_active" verdict means a similar compound is active, nothing more, and
"no_analogue" means ChEMBL has nothing within the similarity cutoff (novel, or an artifact of the classifier).

Read-only, one fixed host, no credentials. Any failure (offline, rate limit, a changed payload) returns verdict
"unavailable" with the error text and never raises, so a lookup cannot break a run. Endpoint and field names were read from
the live API (similarity -> {"molecules": [...]}, activity -> {"activities": [...]}); the code has not been run end to end.
"""
from __future__ import annotations

import time
import urllib.parse

import httpx

BASE = "https://www.ebi.ac.uk/chembl/api/data"
DRD2_TARGET = "CHEMBL217"
SIMILARITY_CUTOFF = 70  # percent; ChEMBL accepts 40-100
MAX_NEIGHBOURS = 10
ACTIVE_PCHEMBL = 6.0  # pChEMBL >= 6 means a potency of 1 uM or better
TIMEOUT = 60.0
COMPOUND_URL = "https://www.ebi.ac.uk/chembl/compound_report_card/{}/"
DOCUMENT_URL = "https://www.ebi.ac.uk/chembl/document_report_card/{}/"
LABEL = "Agent-generated check against ChEMBL, not a measurement of this molecule."


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


async def lookup_chembl(smiles: str, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Nearest ChEMBL compounds to `smiles` and their measured DRD2 activity, with citations."""
    out = {"smiles": smiles, "verdict": "unavailable", "label": LABEL, "source": "ChEMBL REST API (EMBL-EBI)",
           "target": f"{DRD2_TARGET} (D(2) dopamine receptor)", "cutoff_percent": SIMILARITY_CUTOFF,
           "neighbours": [], "documents": [], "error": None, "checked_at": time.strftime("%FT%T")}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport, headers={"Accept": "application/json"}) as c:
            r = await c.get(f"{BASE}/similarity/{urllib.parse.quote(smiles, safe='')}/{SIMILARITY_CUTOFF}.json",
                            params={"limit": MAX_NEIGHBOURS})
            r.raise_for_status()
            mols = r.json().get("molecules", [])
            activities = []
            if mols:
                r2 = await c.get(f"{BASE}/activity.json", params={
                    "target_chembl_id": DRD2_TARGET, "limit": 200,
                    "molecule_chembl_id__in": ",".join(m["molecule_chembl_id"] for m in mols)})
                r2.raise_for_status()
                activities = r2.json().get("activities", [])
    except Exception as e:  # never let a lookup break a run
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    per_mol: dict[str, dict] = {}
    docs: list[str] = []
    for a in activities:
        m = per_mol.setdefault(a.get("molecule_chembl_id"), {"n": 0, "best": None})
        m["n"] += 1
        p = _float(a.get("pchembl_value"))
        if p is not None and (m["best"] is None or p > m["best"]):
            m["best"] = p
        if p is not None and p >= ACTIVE_PCHEMBL and a.get("document_chembl_id") and a["document_chembl_id"] not in docs:
            docs.append(a["document_chembl_id"])
    for m in mols:
        cid = m["molecule_chembl_id"]
        info = per_mol.get(cid, {"n": 0, "best": None})
        out["neighbours"].append({
            "chembl_id": cid, "name": m.get("pref_name"), "similarity": _float(m.get("similarity")),
            "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles"),
            "drd2_activities": info["n"], "drd2_pchembl_max": info["best"], "url": COMPOUND_URL.format(cid)})
    out["documents"] = [{"chembl_id": d, "url": DOCUMENT_URL.format(d)} for d in docs[:5]]

    nb = out["neighbours"]
    if not nb:
        out["verdict"] = "no_analogue"
    elif any((n["drd2_pchembl_max"] or 0) >= ACTIVE_PCHEMBL for n in nb):
        out["verdict"] = "analogue_active"
    elif any(n["drd2_activities"] for n in nb):
        out["verdict"] = "analogue_inactive"
    else:
        out["verdict"] = "analogue_untested"
    return out
