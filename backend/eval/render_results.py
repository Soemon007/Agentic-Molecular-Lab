"""Render the README results table from data/results_<tag>.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLS = ["50", "100", "250", "500", "1000"]


def cell(d, k):
    return f"{d[k]['mean']:.3f} (var {d[k]['var']:.1e})" if k in d else "n/a"


def main(tag="main"):
    r = json.loads((ROOT / "data" / f"results_{tag}.json").read_text())
    rand = list(r["random"].values())
    rc = {c: f"{np.mean([x[c] for x in rand]):.3f} (var {np.var([x[c] for x in rand], ddof=1):.1e})" for c in COLS + ["10000"]}
    rows = [
        ("ours (adversary on)", [cell(r["ours"], c) for c in COLS] + ["N/A — budget-capped by design"]),
        ("ours-ablated (adversary off)", [cell(r["ablated"], c) for c in COLS] + ["N/A — budget-capped by design"]),
        ("single-call LLM", ["not run (no API key)"] * 5 + ["N/A (50 molecules total)"]),
        ("random (ZINC)", [rc[c] for c in COLS] + [rc["10000"]]),
        ("Graph GA (published, PMO)", ["not published"] * 5 + ["0.964 ± 0.012"]),
    ]
    out = ["| method | @50 | @100 | @250 | @500 | @1,000 | @10,000 |", "|---|---|---|---|---|---|---|"]
    out += [f"| {n} | " + " | ".join(v) + " |" for n, v in rows]
    return "\n".join(out)


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
