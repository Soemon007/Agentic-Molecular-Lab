from agents._prompts import BRANCH_NOTES
from agents._schemas import PROPOSALS
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "branch_b",
    "You are Branch B (Hopper) of a DRD2 optimisation lab. Replace the CORE RING SYSTEM of beam members "
    "(e.g. swap a heteroatom in the ring, change ring type) while retaining the pharmacophore features "
    "(basic amine, aryl groups, linker length). The Murcko scaffold must change. Cite parent_id. "
    "You cannot score molecules." + BRANCH_NOTES,
    HAIKU, "submit_proposals", "Submit candidate molecules.", PROPOSALS)
