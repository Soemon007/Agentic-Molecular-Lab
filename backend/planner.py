"""Planner: which of two tests to send to the oracle this batch, chosen by expected gain, expected learning and cost.

After a branch's proposals are validated, the quota can be spent two ways, and the surrogate predicts what each would
return before any oracle call is paid:

  exploit  rank by predicted mean (beta = 0.05, the weight an offline replay of two live runs supported)
  explore  rank by mean + 1.0 * std: favours molecules the model is unsure about

For each option the planner computes, over the molecules it would send:
  expected gain      mean of max(0, predicted score - the weakest top-10 score): how much it should improve the beam
  expected learning  mean predictive std: how much the oracle's answer should teach the surrogate
  cost               oracle calls spent (equal for the two options, which is why per-call means are compared; the field is
                     kept so batches of different size would compare fairly)
and picks the larger of gain + w * learning, with w = LEARNING_WEIGHT * (budget left / budget): learning is worth more
early than late. Ties go to exploit, and so does a batch the surrogate cannot score (too few labels yet), which keeps the
agent's own order.

The decision and both options' numbers are written to the chat log (event "plan") so each choice can be audited.
LEARNING_WEIGHT is untuned, and "expected learning = predictive std" is a proxy, not a measured information gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EXPLOIT_BETA = 0.05
EXPLORE_BETA = 1.0
LEARNING_WEIGHT = 0.25


@dataclass
class Option:
    name: str
    beta: float
    order: list = field(repr=False)  # indices of the candidates, best first under this option
    chosen: list = field(repr=False)  # the first `quota` of them the surrogate could score
    expected_gain: float = 0.0
    expected_learning: float = 0.0
    cost: int = 0
    utility: float = 0.0
    feasible: bool = False


def _option(name, beta, rank, preds, quota, floor_score, weight) -> Option:
    order = rank(beta)
    chosen = [i for i in order if i < len(preds) and preds[i] is not None][:quota]
    if not chosen:
        return Option(name, beta, order, [])
    n = len(chosen)
    gain = sum(max(0.0, preds[i][0] - floor_score) for i in chosen) / n
    learning = sum(preds[i][1] for i in chosen) / n
    return Option(name, beta, order, chosen, gain, learning, n, gain + weight * learning, True)


def plan(preds: list, quota: int, rank, floor_score: float, remaining_fraction: float):
    """preds: surrogate (mean, std) or None per candidate. rank(beta) -> candidate indices, best first.
    Returns (chosen option, [exploit, explore])."""
    weight = LEARNING_WEIGHT * max(0.0, min(1.0, remaining_fraction))
    options = [_option("exploit", EXPLOIT_BETA, rank, preds, quota, floor_score, weight),
               _option("explore", EXPLORE_BETA, rank, preds, quota, floor_score, weight)]
    feasible = [o for o in options if o.feasible]
    return (max(feasible, key=lambda o: o.utility) if feasible else options[0]), options


def summary(o: Option) -> dict:
    return {"name": o.name, "beta": o.beta, "expected_gain": round(o.expected_gain, 4),
            "expected_learning": round(o.expected_learning, 4), "cost": o.cost, "utility": round(o.utility, 4),
            "feasible": o.feasible}
