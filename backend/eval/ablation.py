"""Ablation: with vs without the adversary at an IDENTICAL oracle budget (asserted)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from eval.pmo_auc import report  # noqa: E402
from orchestrator import run_lab, summarize  # noqa: E402

CHECKPOINTS = (50, 100, 250, 500, 1000)


def run_pair(seed: int, budget: int, branches: str = "abc", llm=None, noise: float = 1.0, exploit: bool = False, seed_mode: str = "known"):
    out = {}
    for name, adv in (("ours", True), ("ablated", False)):
        sess = asyncio.run(run_lab(budget=budget, branches=branches, llm=llm, seed=seed, verbose=False,
                                   adversary=adv, noise=noise, exploit=exploit, seed_mode=seed_mode,
                                   run_id=f"{name}_{branches}_{seed_mode}_seed{seed}"))
        csv = sess.oracle.traj_path
        n_rows = len(pd.read_csv(csv))
        # identical-budget assertions: same ceiling, fully spent, and the CSV agrees with the counter
        assert sess.budget == budget and sess.oracle.calls == budget == n_rows, (name, sess.oracle.calls, n_rows, budget)
        out[name] = {"auc": report(csv, tuple(c for c in CHECKPOINTS if c <= budget)), "summary": summarize(sess)}
    assert out["ours"]["summary"]["oracle_calls"] == out["ablated"]["summary"]["oracle_calls"]
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_pair(int(sys.argv[1]) if len(sys.argv) > 1 else 0, 500), indent=1, default=str))
