"""Coordinator: plain if/else quota logic (Stage 3). Registered now as the plan-holder agent."""
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "coordinator",
    "Deterministic coordinator: default quotas 4/4/4; a flagged branch drops to 1 (never 0) with a two-round "
    "cooldown, the remainder is redistributed, and the adversary's instruction is injected into that branch.",
    HAIKU, "submit_plan", "Submit quota plan.",
    {"type": "object", "properties": {"quotas": {"type": "object"}}})
