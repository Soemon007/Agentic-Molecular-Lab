"""Coordinator: plain if/else quota logic, no bandit math.

Default quotas 4 per branch. A flagged branch drops to 1 (floor, never 0) for a two-round cooldown;
the freed quota is redistributed to the other branches; the adversary's corrective instruction is
injected into that branch's next prompts and removed when the cooldown ends.
The registered AgentDef is a spec holder only — the logic below is deterministic code.
"""
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "coordinator",
    "Deterministic coordinator (no LLM calls): default quotas 4/4/4; a flagged branch drops to 1 with a "
    "two-round cooldown, the remainder is redistributed, and the adversary's instruction is injected.",
    HAIKU, "submit_plan", "Submit quota plan.", {"type": "object", "properties": {"quotas": {"type": "object"}}})

COOLDOWN_ROUNDS = 2


class Coordinator:
    def __init__(self, branches: list[str], default: int = 4):
        self.branches, self.default = list(branches), default
        self.cooldown: dict[str, int] = {}
        self.instruction: dict[str, str] = {}

    def quotas(self) -> dict[str, int]:
        flagged = [b for b in self.branches if self.cooldown.get(b, 0) > 0]
        free = [b for b in self.branches if b not in flagged]
        q = {b: 1 for b in flagged}  # floor of 1, never 0
        remainder = self.default * len(self.branches) - len(flagged)
        for i, b in enumerate(free):  # even split; leftover goes to the earliest branches
            q[b] = remainder // len(free) + (1 if i < remainder % len(free) else 0)
        return {b: q[b] for b in self.branches}

    def instructions(self) -> dict[str, str]:
        return {b: t for b, t in self.instruction.items() if self.cooldown.get(b, 0) > 0}

    def in_cooldown(self, branch: str) -> bool:
        return self.cooldown.get(branch, 0) > 0

    def end_round(self):  # call BEFORE flag() in the same round so a fresh flag keeps its full cooldown
        for b in list(self.cooldown):
            self.cooldown[b] = max(0, self.cooldown[b] - 1)

    def flag(self, branch: str, instruction: str):
        if branch in self.branches and not self.in_cooldown(branch):
            self.cooldown[branch] = COOLDOWN_ROUNDS
            self.instruction[branch] = instruction
            return True
        return False
