# Agentic Molecular Discovery Lab

A multi-agent molecular optimiser for the PMO **DRD2** oracle under a tight oracle-call budget, with an
adversary agent that watches for the lab gaming the surrogate (score rising while the population drifts out
of the oracle's training domain).

> **Read this first — what the numbers below are and are not.**
> No `ANTHROPIC_API_KEY` was available when this was built, so **every "ours" number comes from an offline
> mock** (`backend/agents/offline.py`: RDKit edits plus a configurable rate of rule-violating proposals), not
> from Haiku/Sonnet. The mock validates plumbing, policies, logging and evaluation; it is **not evidence about
> LLM behaviour**. The Anthropic executor and the single-call baseline are written but have not been run live.
> Also, the seed beam contains known DRD2 drugs the oracle already scores ≈1.0, which flatters every
> comparison against a cold-start method like Graph GA (see *Acceleration claim*).

## Layout
```
backend/
  chem_core.py      Candidate, Gatekeeper, scaffold-niched beam, PAINS/BRENK alerts
  oracle.py         DRD2 wrapper: global call counter, failure log, trajectory CSV
  domain.py         applicability-domain reference + ad_similarity
  telemetry.py      adversary digest (no raw SMILES) + deterministic trigger
  policies.py       Omnigent FunctionPolicy objects + PolicyGate
  llm.py            Omnigent Executors: AnthropicExecutor (Haiku/Sonnet), OfflineExecutor (mock)
  chatlog.py        agent chat / proposal-outcome logging
  orchestrator.py   session, main loop, adversary step
  agents/           scout, branch_a/b/c, adversary, coordinator (+ offline mock)
  baselines/        random_stub.py, single_call_llm.py
  eval/             pmo_auc.py, ablation.py, run_seeds.py, render_results.py
  tests/            27 tests
```
Everything runs from the repo root; data lands in `backend/data/` (gitignored except the folder skeleton).

## Setup (Python 3.13)
Omnigent needs Python ≥3.12. PyTDC 1.1.15 pins an old scikit-learn/rdkit that do not build on 3.13, so it is
installed without its pins, plus a one-line shim (`backend/compat.py`) for the removed `rdkit.six`. numpy must
stay <2.4: numpy 2.5 makes TDC's `float(array([x]))` raise, which TDC swallows into a **silent 0.0 score**
(`oracle.py` runs an install self-test for exactly this).
```bash
python3.13 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt && .venv/bin/pip install --no-deps PyTDC==1.1.15
AUTOPILOT=true .venv/bin/python -m pytest -q
AUTOPILOT=true .venv/bin/python backend/orchestrator.py --budget 100 --branches ab     # Stage 2 gate run
AUTOPILOT=true .venv/bin/python backend/eval/run_seeds.py                               # 3 seeds x {ours, ablated}
```
Set `ANTHROPIC_API_KEY` (or `LLM_MODE=anthropic`) to use real Haiku/Sonnet; without it the mock is used.
`AUTOPILOT=true` logs approval-gate events and auto-rejects; leave it unset for the interactive `input()` demo.

## Results (offline mock; budget 1,000; seeds 0–2; mean and variance across seeds)
Metric: PMO top-10 AUC, trapezoid over top-10 mean sampled every 10 calls, normalised by the budget at each column.

| method | @50 | @100 | @250 | @500 | @10,000 |
|---|---|---|---|---|---|
| ours (adversary on) | 0.853 (var 3.9e-04) | 0.924 (var 1.0e-04) | 0.968 (var 1.6e-05) | 0.984 (var 3.5e-06) | N/A — budget-capped by design |
| ours-ablated (adversary off) | 0.853 (var 3.9e-04) | 0.924 (var 1.0e-04) | 0.968 (var 1.6e-05) | 0.984 (var 3.5e-06) | N/A — budget-capped by design |
| single-call LLM | not run (no API key) | not run (no API key) | not run (no API key) | not run (no API key) | N/A (50 molecules total) |
| random (ZINC) | 0.026 (var 3.6e-04) | 0.037 (var 6.5e-04) | 0.064 (var 1.5e-03) | 0.113 (var 6.2e-04) | 0.534 (var 4.7e-03) |
| Graph GA (published, PMO) | not published | not published | not published | not published | 0.964 ± 0.012 |

**Ablation is a null result, and by construction.** With and without the adversary, the runs are identical
because the trigger (`unique_scaffolds_top10 < 3 OR sa_trend_3r > 1.2 OR ad_similarity_trend < -0.15`) fired in
**0 of 3 seeds** — the mock never reward-hacks, and a synthetic "exploiter" mode (bolting amine stacks onto
molecules) did not reach the top-10 either, so no exploit signature appeared. I did not tune thresholds to make
it fire. The harness asserts identical budgets (`ceiling == oracle.calls == CSV rows` for both arms). The
adversary → coordinator → prompt-injection path is covered by a plumbing test with an *injected* trigger
(`tests/test_stage3.py`), not by evidence that it catches real exploits. Showing that needs a live-LLM run.

### Acceleration claim (derived only from the curves)
- Our running top-10 mean passes Graph GA's published 0.964 after **15–19 oracle calls** (3 seeds), and our
  AUC computed over N calls first exceeds 0.964 between **N = 200 (0.961) and N = 250 (0.968)**.
- **This is not a defensible speed-up claim.** Seeding with haloperidol/aripiprazole/risperidone (oracle ≈ 1.0)
  means one fluorine on haloperidol already scores ≈1.0; Graph GA's 0.964 is a cold-start 10,000-call AUC, and
  AUC at different budgets is not like-for-like. I am not stating a yield ratio. A fair test needs cold-start
  seeds (or weak seeds only) and the real LLM branches.
- **Crossover where the advantage reverses: not determinable.** PMO publishes only Graph GA's 10,000-call
  endpoint, not its AUC@50/100/250/500, so there is no published curve to cross. Our budget cap means our curve
  also stops at 1,000.

## Disclosures
1. **Versions and reference table.** PyTDC **1.1.15** (rdkit 2026.3.6, scikit-learn 1.9.1, numpy 2.3.5). The
   Graph GA figure (0.964 ± 0.012) is the DRD2 top-10 AUC from the PMO benchmark paper (Gao et al., NeurIPS
   2022 Datasets & Benchmarks), not recomputed here. PMO/TDC scores have shifted across releases: TDC loads a
   different DRD2 pickle depending on the scikit-learn version (`drd2.pkl` for <0.24, `drd2_current.pkl`
   otherwise), and a newer sklearn/rdkit than the paper's means our oracle may not score identically to the
   one Graph GA was measured on. Seeds scored identically to a Python 3.11 / older-sklearn run to 6 decimals,
   but that is four molecules, not a proof.
2. **Oracle calls are counted, LLM calls are not.** PMO counts oracle calls only. This lab spends
   LLM tokens (scout + 3 branches per round, adversary when triggered) to save oracle calls — it trades compute
   for sample efficiency, which is the right trade when the oracle is a wet-lab assay and the wrong one when it
   is a function like DRD2. Token totals are tracked per agent (`TokenLedger`, in each run summary); the mock
   run reports 0 tokens and ~450 agent calls per 1,000-call run, so real token totals are **unmeasured**.
3. **One oracle.** Everything is DRD2, a single SVM classifier. Nothing here shows the method generalises to
   other PMO tasks, to regression oracles, or to real assays. The "applicability domain" is that SVM's own
   2,159 support vectors, since PyTDC 1.1.15 exposes no `HTS('DRD2')` training set.

## Agents
| agent | model | job | tool (the only one it may call) |
|---|---|---|---|
| scout | Haiku | writes a short SAR brief from the beam | `submit_brief` |
| branch_a (Local) | Haiku | acyclic substituent edits; **no new rings** | `submit_proposals` |
| branch_b (Hopper) | Haiku | replace the core ring system, keep pharmacophore | `submit_proposals` |
| branch_c (Explorer) | Haiku | bioisosteres from lower-percentile beam entries | `submit_proposals` |
| adversary | Sonnet | only when triggered: one-sentence exploit diagnosis + corrective instruction, from a telemetry digest (no raw SMILES) | `submit_diagnosis` |
| coordinator | none (code) | quotas 4/4/4; flagged branch → 1 (never 0), remainder redistributed, instruction injected, two-round cooldown | — |

Agents are Omnigent `AgentDef`s run through Omnigent `Executor`s. **Not used:** Omnigent's server/CLI runtime —
its executors need provider credentials, so the session is a `LabSession` dataclass plus Omnigent's
`SessionState` enum. The Gatekeeper makes no LLM calls.

## Policies (enforced in code at the orchestration layer, not in prompts)
1. **write_permission** — only the oracle wrapper writes scores; an agent calling any tool but its own, or
   submitting a `score`-like field, is DENIED.
2. **budget_cap** — hard stop at the oracle-call ceiling (`oracle_evaluate` is DENIED once reached).
3. **electrophile_approval** — a BRENK *electrophile* alert (Michael acceptors, alkyl/N-halides, aldehydes,
   etc.) ASKs for human approval; `AUTOPILOT=true` logs and auto-rejects. Other BRENK alerts and PAINS are
   telemetry for the adversary only and never pause the run.
Gatekeeper rejections are logged by branch **and reason** (e.g. `ring_count_change` vs `scaffold_change`).

## Logging for tuning agent behaviour
`backend/data/chats/<run_id>.jsonl`: one `agent_call` record per call (system prompt, input payload, output,
tokens, latency, policy verdict) and one `proposal_outcome` per proposal (parent, SMILES, Gatekeeper reason or
oracle score, AD similarity, SA, alerts), joined on `call_id`; plus `adversary_trigger` events. Every
agent-generated molecule is tagged (`origin_branch`, `agent_generated`). Policy events:
`backend/data/policy_events.jsonl`. Oracle trajectories: `backend/data/trajectories/*.csv`.

## Validation still needed before real-world use
- Run with real Haiku/Sonnet; re-check the Stage 2 rejection mix (the current mix is injected by the mock).
- Cold-start (or weak-seed) comparison against Graph GA run on *this* oracle build; measure token cost.
- Show the adversary catches a real exploit: needs a run where score climbs while `ad_similarity` falls; confirm
  trigger thresholds on real data and measure false-positive cost.
- Check that high-scoring molecules are actually in-domain and chemically sensible (SA, medchem review),
  that PAINS/BRENK + SA are adequate filters, and test on other oracles.
- Confirm DRD2 scores are reproducible across rdkit/sklearn/PyTDC versions, and validate any hit
  experimentally — a classifier score is not binding affinity.
