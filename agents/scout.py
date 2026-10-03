from agents._schemas import PROPOSALS  # noqa: F401
from agents.base import make_agent
from llm import HAIKU

AGENT = make_agent(
    "scout",
    "You are the Scout for a DRD2 molecular-optimisation lab. Read the shared beam (SMILES, oracle scores, "
    "scaffolds) and write a 3-sentence pharmacophore/SAR brief that the branch agents will read. "
    "Do not propose molecules and never state scores you were not given.",
    HAIKU, "submit_brief", "Submit the brief.",
    {"type": "object", "required": ["brief"], "properties": {"brief": {"type": "string"}}})
