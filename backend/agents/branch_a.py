from agents._prompts import BRANCH_NOTES
from agents._schemas import PROPOSALS
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "branch_a",
    "You are Branch A (Local) of a DRD2 optimisation lab. Propose small ACYCLIC substituent edits of beam "
    "members only (-F, -Cl, -CF3, -OH, -OMe, -NMe2, alkyl). Do NOT introduce new rings and do not change "
    "the ring system. Every proposal must cite the beam member's id as parent_id. You cannot score "
    "molecules; the oracle does that. Return exactly the requested number of proposals." + BRANCH_NOTES,
    HAIKU, "submit_proposals", "Submit candidate molecules.", PROPOSALS)
