"""Applicability-domain reference for the DRD2 oracle.

NOTE (deviation from the original spec): PyTDC has no `HTS('DRD2')` dataset. The DRD2
oracle is an sklearn SVC (Olivecrona et al.) whose 2,159 *support vectors* are the
training points that define its decision boundary. We use those, in the oracle's own
feature space (count-based Morgan r=3 features-invariants, folded to 2048), as the
domain reference. That is the right yardstick for "is this molecule inside what the
classifier was trained on".

ad_similarity: HIGH = in-domain, LOW = out-of-domain exploit. The adversary trigger
fires when it DROPS. (Never call it a distance.)
Similarity is count-Tanimoto (sum(min)/sum(max)) to the nearest support vector.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
ORACLE_PKL = DATA / "oracle" / "drd2_current.pkl"
CACHE = DATA / "drd2_support_vectors.npy"


def _fingerprint(mol) -> np.ndarray:
    from tdc.chem_utils.oracle.oracle import fingerprints_from_mol  # the oracle's own featuriser
    return fingerprints_from_mol(mol)[0].astype(np.int32)


def _load_reference() -> np.ndarray:
    if not CACHE.exists():
        from oracle import ensure_tdc_assets  # downloads the pickle into data/oracle if absent
        ensure_tdc_assets()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ORACLE_PKL.open("rb") as f:
                model = pickle.load(f)
        sv = np.asarray(model.support_vectors_, dtype=np.int32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, sv)
    ref = np.load(CACHE)
    assert 1_000 < len(ref) < 50_000, f"bad training set: {len(ref)}"
    return ref


_REF: np.ndarray | None = None


def reference() -> np.ndarray:
    global _REF
    if _REF is None:
        _REF = _load_reference()
    return _REF


def ad_similarity(mol) -> float:
    """HIGH = in-domain, LOW = out-of-domain exploit."""
    ref = reference()
    fp = _fingerprint(mol)
    num = np.minimum(ref, fp).sum(axis=1)
    den = np.maximum(ref, fp).sum(axis=1)
    den[den == 0] = 1
    return float((num / den).max())
