"""Free, local ranker for proposals: a Tanimoto-kernel Gaussian process over molecules the oracle has already scored.

Why it exists. Each branch is asked for ~3x its quota of proposals, but the quota's worth of *valid* ones used to be
scored in the order the LLM wrote them. This ranks the valid proposals by mean + beta * std
and spends the quota on the best, so the oracle budget goes to the most promising candidates.

Rules it keeps.
  * It makes no oracle call and writes no score: `Candidate.score` still comes only from the oracle wrapper.
  * It is trained only on molecules the oracle has already scored, so it uses information the lab already paid for.
  * It does nothing until MIN_LABELS molecules are scored (before that the order is the LLM's own), and it never
    sees the oracle's internals (no support vectors), so it cannot become a route for gaming the domain boundary.
beta = 0.05 comes from an offline replay of two live 500-call runs (each round's scored molecules predicted from
earlier rounds only; how often the true top-4 of a round were also the predicted top-4): beta 0 to 0.1 picked about
70% of them against about 40% for random, 0.5 about 55-65%, and the textbook beta = 1 was no better than random
because the std term swamps the mean. That replay measures ranking among molecules the earlier agents scored in the
order they were written, not yet the choice among ~10 valid proposals that this module makes; a live run tests that.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MIN_LABELS = 30  # below this the sample is too small to rank on
MAX_TRAIN = 1500  # O(n^3) solve; keep the best and the most recent
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return None if mol is None else _GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def _tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


class Surrogate:
    def __init__(self, beta: float = 0.05, noise: float = 0.05):
        self.beta, self.noise = beta, noise
        self._X: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        self._mean = 0.0

    @property
    def ready(self) -> bool:
        return self._X is not None

    def fit(self, labels: list[tuple[str, float]]) -> "Surrogate":
        """labels: (smiles, oracle score) for every molecule scored so far."""
        pts = [(fp, s) for smi, s in labels if (fp := _fp(smi)) is not None]
        self._X = self._alpha = self._chol = None
        if len(pts) < MIN_LABELS:
            return self
        if len(pts) > MAX_TRAIN:  # best half + most recent half
            order = sorted(range(len(pts)), key=lambda i: -pts[i][1])[: MAX_TRAIN // 2]
            keep = set(order) | set(range(len(pts) - MAX_TRAIN // 2, len(pts)))
            pts = [pts[i] for i in sorted(keep)]
        X = np.stack([p[0] for p in pts])
        y = np.array([p[1] for p in pts], dtype=np.float64)
        self._mean = float(y.mean())
        K = _tanimoto(X, X) + self.noise * np.eye(len(X))
        self._chol = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, y - self._mean))
        self._X = X
        return self

    def predict(self, smiles: list[str]) -> list[tuple[float, float] | None]:
        """(mean, std) per SMILES; None where the SMILES does not parse or the model is not ready."""
        if not self.ready:
            return [None] * len(smiles)
        fps = [_fp(s) for s in smiles]
        ok = [i for i, f in enumerate(fps) if f is not None]
        out: list[tuple[float, float] | None] = [None] * len(smiles)
        if not ok:
            return out
        Ks = _tanimoto(np.stack([fps[i] for i in ok]), self._X)
        mu = Ks @ self._alpha + self._mean
        v = np.linalg.solve(self._chol, Ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 0.0, None)
        for j, i in enumerate(ok):
            out[i] = (float(mu[j]), float(np.sqrt(var[j])))
        return out

    def rank(self, smiles: list[str], beta: float | None = None) -> list[int]:
        """Indices of `smiles`, best first by mean + beta * std (beta defaults to self.beta). Original order when the
        model is not ready; unparsable last."""
        if not self.ready:
            return list(range(len(smiles)))
        beta = self.beta if beta is None else beta
        preds = self.predict(smiles)
        ucb = [(-(p[0] + beta * p[1]) if p else float("inf"), i) for i, p in enumerate(preds)]
        return [i for _, i in sorted(ucb)]
