"""Registered agents. Each module exposes AGENT, an Omnigent AgentDef."""
from agents import adversary, branch_a, branch_b, branch_c, coordinator, scout  # noqa: F401

REGISTRY = {m.AGENT.name: m.AGENT for m in (scout, branch_a, branch_b, branch_c, adversary, coordinator)}
