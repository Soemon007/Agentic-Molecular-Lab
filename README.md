# Agentic Molecular Discovery Lab

Multi-agent molecular optimization on PMO/DRD2 under a tight oracle-call budget, with an
adversary agent that flags reward-hacking of the surrogate. **Status: Stage 1 (deterministic
foundation) complete; agents, adversary and results table land in Stages 2–3.**

## Setup
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```
Run scripts from the repo root. TDC oracle weights are cached in `data/oracle/` (gitignored).

## Stage 1 notes
- `oracle.py` counts every call (failures and invalid SMILES included), logs one CSV row per call, and prints the failure rate at exit (>2% means the Gatekeeper is leaking).
- `chem_core.py` Gatekeeper is deterministic/LLM-free; rejections are logged per branch and per reason (`ring_count_change` vs `scaffold_change` for Branch A).
- **Deviation:** PyTDC 1.1.15 has no `HTS('DRD2')` dataset. `domain.py` instead uses the DRD2 oracle's own 2,159 SVM support vectors (its training points that define the boundary, in its own count-Morgan feature space) as the applicability-domain reference. `ad_similarity` is count-Tanimoto to the nearest support vector; HIGH = in-domain.
- **Beam:** the "max 2 per family" cap is soft — overflow members only backfill empty slots, otherwise a beam with fewer than 10 scaffolds could never reach `beam_size`.
