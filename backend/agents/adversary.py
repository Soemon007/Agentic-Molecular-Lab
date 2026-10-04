"""Adversary (Sonnet). Stage 3 wires the trigger + telemetry digest; the spec is registered now."""
from agents.base import make_agent
from llm import SONNET

AGENT = make_agent(
    "adversary",
    "You are the Adversary. You receive a telemetry digest (never raw SMILES) of an optimisation run against "
    "the DRD2 classifier oracle. The exploit signature is score climbing while ad_similarity falls, or a high "
    "top-10 score that sits at an ad_similarity below the floor of known DRD2 ligands (stats.ad_known_ligand_floor): "
    "DRD2 is a classifier, so gaming means finding adversarial examples outside its training domain. Look at the "
    "motif counts too: free thiols and acyclic N,N-aminals are chemically implausible and often mark an exploit. Reply "
    "with ONE sentence diagnosing the exploit and ONE corrective med-chem instruction for the flagged branch.",
    SONNET, "submit_diagnosis", "Submit diagnosis.",
    {"type": "object", "required": ["diagnosis", "instruction"],
     "properties": {"diagnosis": {"type": "string"}, "instruction": {"type": "string"}}})
