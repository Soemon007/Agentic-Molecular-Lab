"""Prompt text the branches and the Scout share: what the filter enforces and what the extra payload fields mean."""
from chem_core import MAX_LOGP, MAX_MW

BRANCH_NOTES = (
    f"\n\nA filter silently discards proposals that break these limits: MW <= {MAX_MW:.0f}, logP <= {MAX_LOGP:.0f} (each beam "
    "row shows mw and logp), an unparsable SMILES, or any molecule that was already scored. So never re-propose a beam "
    "member or anything in `recent_results` or `do_not_repeat` (molecules proposed again and again), or trivial variants. "
    "`recent_results` holds [smiles, score, delta vs parent] for recent proposals from all branches: avoid the edits that "
    "lowered the score and build on the ones that raised it. `your_recent_rejections` holds your own proposals the filter "
    "discarded last round, with the reason."
)
SCOUT_NOTES = (
    "\n\n`recent_results` holds [smiles, score, delta vs parent] for the latest proposals: use it as evidence for what to "
    "keep and what to avoid, including edits that made things worse."
)
