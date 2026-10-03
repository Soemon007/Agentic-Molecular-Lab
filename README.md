# Agentic Molecular Discovery Lab

Multi-agent molecular optimization on PMO/DRD2 under a tight oracle-call budget, with an
adversary agent that flags reward-hacking of the surrogate. **Status: Stages 1–2 built (deterministic foundation; agents, policies, loop, baselines, AUC). Adversary, coordinator logic and the results table land in Stage 3.**

## Setup (Python 3.13)
Omnigent needs Python >=3.12, and PyTDC 1.1.15 pins an old scikit-learn/rdkit that do not build on 3.13, so PyTDC is installed without its pins and a one-line shim (`compat.py`) covers the removed `rdkit.six`. numpy must stay <2.4: numpy 2.5 makes TDC's `float(array([x]))` raise, which TDC swallows into a **silent 0.0 score** (`oracle.py` runs an install self-test for this).
```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install --no-deps PyTDC==1.1.15
AUTOPILOT=true .venv/bin/python -m pytest -q
AUTOPILOT=true .venv/bin/python orchestrator.py --budget 100 --branches ab   # Stage 2 gate run
```
Without `ANTHROPIC_API_KEY` the agents run on an **offline mock** (`agents/offline.py`: RDKit edits plus a configurable rate of deliberately rule-violating proposals). That validates plumbing only, not LLM behaviour.
Run scripts from the repo root. TDC oracle weights are cached in `data/oracle/` (gitignored).

## Stage 1 notes
- `oracle.py` counts every call (failures and invalid SMILES included), logs one CSV row per call, and prints the failure rate at exit (>2% means the Gatekeeper is leaking).
- `chem_core.py` Gatekeeper is deterministic/LLM-free; rejections are logged per branch and per reason (`ring_count_change` vs `scaffold_change` for Branch A).
- **Deviation:** PyTDC 1.1.15 has no `HTS('DRD2')` dataset. `domain.py` instead uses the DRD2 oracle's own 2,159 SVM support vectors (its training points that define the boundary, in its own count-Morgan feature space) as the applicability-domain reference. `ad_similarity` is count-Tanimoto to the nearest support vector; HIGH = in-domain.
- **Beam:** the "max 2 per family" cap is soft — overflow members only backfill empty slots, otherwise a beam with fewer than 10 scaffolds could never reach `beam_size`.
