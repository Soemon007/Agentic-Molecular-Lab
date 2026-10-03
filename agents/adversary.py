"""Adversary (Sonnet). Stage 3 wires the trigger + telemetry digest; the spec is registered now."""
from agents.base import make_agent
from llm import SONNET

AGENT = make_agent(
    "adversary",
    "You are the Adversary. You receive a telemetry digest (never raw SMILES) of an optimisation run against "
    "the DRD2 classifier oracle. The exploit signature is score climbing while ad_similarity falls: DRD2 is a "
    "classifier, so gaming means finding adversarial examples outside its training domain. Reply with ONE "
    "sentence diagnosing the exploit and ONE corrective med-chem instruction for the flagged branch.",
    SONNET, "submit_diagnosis", "Submit diagnosis.",
    {"type": "object", "required": ["diagnosis", "instruction"],
     "properties": {"diagnosis": {"type": "string"}, "instruction": {"type": "string"}}})
