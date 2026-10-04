"""3 seeds x {ours, ours-ablated} at an identical budget, AUTOPILOT=true. Writes data/results.json."""
import json
import os
import sys
from pathlib import Path

os.environ["AUTOPILOT"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import policies  # noqa: E402
from eval.ablation import CHECKPOINTS, run_pair  # noqa: E402
from eval.pmo_auc import report  # noqa: E402

policies.AUTOPILOT = True


def agg(runs, cond):
    cps = sorted({k for r in runs for k in r[cond]["auc"]})
    return {str(c): {"mean": float(np.mean([r[cond]["auc"][c] for r in runs])),
                     "var": float(np.var([r[cond]["auc"][c] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                     "values": [r[cond]["auc"][c] for r in runs]} for c in cps}


def main(budget=1000, seeds=(0, 1, 2), branches="abc", llm=None, exploit=False, tag="main", seed_mode="known"):
    runs = [run_pair(s, budget, branches, llm=llm, exploit=exploit, seed_mode=seed_mode) for s in seeds]
    res = {"budget": budget, "seeds": list(seeds), "llm_mode": runs[0]["ours"]["summary"]["llm_mode"],
           "exploit_mock": exploit, "seed_mode": seed_mode, "ours": agg(runs, "ours"), "ablated": agg(runs, "ablated"),
           "trigger_rounds": [r["ours"]["summary"]["trigger_rounds"] for r in runs],
           "adversary_calls": [r["ours"]["summary"]["adversary_calls"] for r in runs],
           "tokens": [r["ours"]["summary"]["tokens"]["total"] for r in runs],
           "rejections_seed0": runs[0]["ours"]["summary"]["rejections"]}
    rand = {}
    for s in (0, 1, 2):
        f = ROOT / "data" / "trajectories" / f"random_seed{s}.csv"
        if f.exists():
            rand[s] = report(str(f), tuple(CHECKPOINTS) + (10000,))
    res["random"] = rand
    (ROOT / "data" / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "known"
    print(json.dumps(main(seed_mode=mode, tag="main" if mode == "known" else mode), indent=1))
