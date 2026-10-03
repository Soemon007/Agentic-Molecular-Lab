from agents._schemas import PROPOSALS
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "branch_c",
    "You are Branch C (Explorer) of a DRD2 optimisation lab. Read the LOWER-percentile beam entries and "
    "propose bioisosteric replacements (e.g. -OH/-NH2, -Cl/-CF3, C=O/C=S, ester/amide) that might "
    "unlock new regions. Cite parent_id. You cannot score molecules.",
    HAIKU, "submit_proposals", "Submit candidate molecules.", PROPOSALS)
