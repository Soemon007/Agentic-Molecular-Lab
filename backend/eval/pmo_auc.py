"""PMO-style top-10 AUC, normalised to [0,1].

Top-10 mean (over unique molecules, best score each) is sampled every `freq` calls; AUC is the
trapezoid over those samples divided by `max_calls`. Runs shorter than `max_calls` are padded with
their last top-10 value (PMO convention). Calls beyond `max_calls` are ignored.
"""
from __future__ import annotations

import heapq

import pandas as pd

import chem_core

CHECKPOINTS = (50, 100, 250, 500)


def top10_curve(smiles, scores, max_calls: int, freq: int = 10):
    best, curve = {}, []
    for i, (s, v) in enumerate(zip(smiles, scores), 1):
        if i > max_calls:
            break
        key = chem_core.canonical(s) or s
        if v > best.get(key, -1):
            best[key] = v
        if i % freq == 0 or i == min(len(scores), max_calls):
            curve.append((i, sum(heapq.nlargest(10, best.values())) / min(10, len(best))))
    return curve


def top10_auc(smiles, scores, max_calls: int, freq: int = 10) -> float:
    curve = top10_curve(smiles, scores, max_calls, freq)
    if not curve:
        return 0.0
    xs, ys = [0] + [c[0] for c in curve], [0.0] + [c[1] for c in curve]
    if xs[-1] < max_calls:  # pad with last value
        xs.append(max_calls); ys.append(ys[-1])
    area = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2 for i in range(len(xs) - 1))
    return area / max_calls


def report(traj_csv: str, checkpoints=CHECKPOINTS) -> dict:
    df = pd.read_csv(traj_csv)
    return {n: round(top10_auc(df.smiles.tolist(), df.score.tolist(), n), 4) for n in checkpoints}


def calls_to_reach(traj_csv: str, target: float, freq: int = 10):
    """First call count at which the running top-10 mean reaches `target` (None if never)."""
    df = pd.read_csv(traj_csv)
    for n, v in top10_curve(df.smiles.tolist(), df.score.tolist(), len(df), freq):
        if v >= target:
            return n
    return None


if __name__ == "__main__":
    import sys
    print(report(sys.argv[1]))
