"""Aggregate live-LLM runs (started from the UI or the API) into data/results_<tag>.json.

`run_seeds.py` runs both arms itself, which doubles the token bill under a live LLM. Live runs here are started
one at a time, so this script collects their trajectories afterwards into the same file shape `/api/results` serves.

    .venv/bin/python backend/eval/live_results.py cold_live \
        --ours <run ids with the adversary on> --ablated <run ids with the adversary off> \
        [--single-call backend/data/trajectories/single_call_llm.csv] --note "free text shown above the results"

Per-seed AUCs come from each run's own trajectory CSV, so nothing here is estimated. `--single-call` points at the
CSV written by `baselines/single_call_llm.py` (a 50-molecule one-shot, so it is reported at @50 only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from domain import DATA  # noqa: E402
from eval.pmo_auc import CHECKPOINTS, report  # noqa: E402


def _traj(run_id: str) -> Path:
    return DATA / "trajectories" / f"ours_{run_id}.csv"


def _meta(run_id: str) -> dict:
    f = DATA / "runs" / f"{run_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _trigger_rounds(run_id: str) -> tuple[int, int]:
    """(rounds where the trigger fired, rounds where the coordinator acted on it), from the run's chat log."""
    fired = acted = 0
    f = DATA / "chats" / f"{run_id}.jsonl"
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "event" and r.get("event") == "adversary_trigger":
            fired += 1
            acted += bool(r.get("acted"))
    return fired, acted


def _arm(run_ids: list[str], budget: int) -> dict:
    cps = tuple(c for c in CHECKPOINTS + (1000,) if c <= budget)
    per_run = [report(str(_traj(i)), cps) for i in run_ids]
    return {str(c): {"mean": float(np.mean([r[c] for r in per_run])),
                     "var": float(np.var([r[c] for r in per_run], ddof=1)) if len(per_run) > 1 else 0.0,
                     "values": [r[c] for r in per_run]} for c in cps}


def build(tag: str, ours: list[str], ablated: list[str], budget: int, single_call: str | None, notes: list[str]) -> dict:
    head = ours or ablated  # the trigger / token columns describe the adversary-on arm when there is one
    metas = [_meta(i) for i in head]
    rand = {}
    for s in (0, 1, 2):
        f = DATA / "trajectories" / f"random_seed{s}.csv"
        if f.exists():
            rand[s] = report(str(f), tuple(c for c in CHECKPOINTS + (1000,) if c <= budget) + (10000,))
    fired = [_trigger_rounds(i) for i in head]
    res = {
        "budget": budget, "llm_mode": "anthropic", "exploit_mock": False,
        "seed_mode": "cold", "ours": _arm(ours, budget) if ours else {}, "ablated": _arm(ablated, budget) if ablated else {},
        "trigger_rounds": [f for f, _ in fired], "acted_rounds": [a for _, a in fired],
        "adversary_calls": [(m.get("summary") or {}).get("adversary_calls") for m in metas],
        "tokens": [((m.get("summary") or {}).get("tokens") or {}).get("total") for m in metas],
        "seeds": list(range(len(head))), "random": rand, "run_ids": {"ours": ours, "ablated": ablated},
        "notes": notes,
    }
    if single_call:
        # the single-call baseline sees 50 molecules, so only its @50 AUC is comparable
        res["single_call"] = {"50": report(single_call, (50,))[50]}
    (DATA / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--ours", nargs="*", default=[], help="run ids with the adversary on")
    ap.add_argument("--ablated", nargs="*", default=[], help="run ids with the adversary off")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--single-call", help="trajectory CSV from baselines/single_call_llm.py")
    ap.add_argument("--note", action="append", default=[], help="free-text caveat shown with the results (repeatable)")
    a = ap.parse_args()
    print(json.dumps(build(a.tag, a.ours, a.ablated, a.budget, a.single_call, a.note), indent=1))
