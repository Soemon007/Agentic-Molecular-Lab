# Agentic Molecular Discovery Lab

A multi-agent molecular optimiser for the PMO **DRD2** oracle under a tight oracle-call budget, with an
adversary agent that watches for the lab gaming the surrogate (score rising while the population drifts out
of the oracle's training domain).

> **Read this first — what each number is.**
> The two result tables (A, B) come from an **offline mock** (`backend/agents/offline.py`: RDKit edits plus a configurable
> rate of rule-violating proposals), not from Haiku/Sonnet. They validate plumbing, policies, logging and evaluation, not LLM
> behaviour. The live Claude path has been run for **2 seeds x 500 calls, with an earlier version of the agents** (no memory, no
> ranked selection; git `5fe8058` plus the live-path fixes), and those runs are in "Live Claude runs". The agents in this repo
> now (recent-results memory, limits in the prompts, surrogate-ranked selection) are checked offline but **have not been run
> live**. Also, the default seed beam contains known DRD2 drugs the oracle already scores ≈1.0, which flatters every comparison
> against a cold-start method like Graph GA; see the cold-start numbers for the fair view.

## Layout
```
backend/
  chem_core.py      Candidate, Gatekeeper, scaffold-niched beam, PAINS/BRENK alerts
  oracle.py         DRD2 wrapper: global call counter, failure log, trajectory CSV
  domain.py         applicability-domain reference + ad_similarity
  telemetry.py      adversary digest (no raw SMILES) + deterministic trigger (incl. the known-ligand AD floor)
  planner.py        per batch: exploit vs explore, by expected gain, expected learning and cost
  chembl.py         read-only ChEMBL lookup of a molecule's nearest neighbours' measured DRD2 activity, with citations
  surrogate.py      free Tanimoto-GP that ranks valid proposals before the quota is spent
  policies.py       Omnigent FunctionPolicy objects + PolicyGate
  llm.py            Omnigent Executors: AnthropicExecutor (Haiku/Sonnet), OfflineExecutor (mock)
  chatlog.py        agent chat / proposal-outcome logging
  server.py         FastAPI: reads the lab's files for the UI, starts/stops runs, relays approvals
  orchestrator.py   session, main loop, adversary step
  agents/           scout, branch_a/b/c, adversary, coordinator (+ offline mock)
  baselines/        random_stub.py, single_call_llm.py
  eval/             pmo_auc.py, ablation.py, run_seeds.py, live_results.py, render_results.py
  tests/            71 tests (55 were run at commit 3fa6ad8; the 16 added since have NOT been run)
frontend/           React + Vite + Tailwind UI (see "Frontend")
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
AUTOPILOT=true .venv/bin/python backend/eval/run_seeds.py [known|cold]                  # 3 seeds x {ours, ablated}
```
Set `ANTHROPIC_API_KEY` in the environment of the process that runs the lab (or `LLM_MODE=anthropic`) to use real
Haiku/Sonnet; without it the mock is used. Nothing here reads a `.env` file. `anthropic` >= 1.11 no longer accepts
`temperature` in `messages.create()`, and `claude-sonnet-5-5` rejects a forced `tool_choice`; `backend/llm.py` handles both
(`backend/tests/test_live_path.py` pins them to the installed SDK).
`AUTOPILOT=true` logs approval-gate events and auto-rejects; leave it unset for the interactive `input()` demo.

## Frontend
React + TypeScript + Tailwind, implementing the Claude Design file `Agentic Molecular Lab.dc.html` (tokens, type, light/dark).
Every screen reads real data; nothing is placeholder.
```bash
cd frontend && npm install && npm run build     # once; the API server then serves frontend/dist
.venv/bin/python backend/server.py              # http://127.0.0.1:8000
# hot reload: keep the server running, then `npm run dev` in frontend/ (http://localhost:5173, /api is proxied)
```
| page | what it shows | source |
|---|---|---|
| Home | hero, one real rejected and one real scored proposal, headline AUC numbers with their caveats | latest run, `results_*.json` |
| Setup | budget, branches, adversary, mock/live LLM, known/cold seeds, electrophile approval (ask me / auto-reject) | `POST /api/runs` → `run_lab` in a worker thread |
| Run | live oracle-call meter, beam (RDKit depictions), best-score and AD/SA charts, agent feed, policy gate with human approve/deny, Gatekeeper rejection mix, adversary trigger banner, Stop | trajectory CSV + chat log |
| Inspector | every agent call: prompt, input, tool output, tokens, latency, verdict, and the outcome of each proposal | chat log |
| Results | per-seed AUC-10 by method and budget checkpoint, with the offline-mock caveats; CSV export | `results_known.json` / `results_cold.json` |

Deliberately **not** built, because the backend has no such thing: pause/resume (there is Stop), dollar cost (no pricing in
the repo; tokens are shown, and are 0 for the mock), a fixed "round N of 20" (rounds run until the budget is spent),
"surrogate best" and "exploit rate" columns (the lab does not measure them), a design-system page, and the design's
invented policies (checkpoint overwrite) and sample data. The AD/SA-per-round chart and electrophile decisions only exist
for runs started after the `round` / `policy` chat-log events were added; older runs say so instead of faking it.

## Results (offline mock, current agents; budget 1,000; seeds 0–2; mean and variance across seeds)
Metric: PMO top-10 AUC (trapezoid over the top-10 mean sampled every 10 calls, normalised by the budget at each
column). Two seed modes, both with the 5 seed molecules counted against the budget.

**A. Warm start — the spec's 5 known DRD2 ligands** (`--seed-mode known`; oracle already ≈1.0 on 3 of them)

| method | @50 | @100 | @250 | @500 | @1,000 | @10,000 |
|---|---|---|---|---|---|---|
| ours (adversary on) | 0.853 (var 3.9e-04) | 0.925 (var 9.4e-05) | 0.969 (var 1.5e-05) | 0.984 (var 3.7e-06) | 0.992 (var 8.9e-07) | N/A — budget-capped by design |
| ours-ablated (adversary off) | 0.853 (var 3.9e-04) | 0.925 (var 9.4e-05) | 0.969 (var 1.5e-05) | 0.984 (var 3.7e-06) | 0.992 (var 8.9e-07) | N/A — budget-capped by design |
| single-call LLM | not run yet | not run yet | not run yet | not run yet | not run yet | N/A (50 molecules total) |
| random (ZINC) | 0.026 (var 3.6e-04) | 0.037 (var 6.5e-04) | 0.064 (var 1.5e-03) | 0.113 (var 6.2e-04) | 0.182 (var 1.1e-03) | 0.534 (var 4.7e-03) |
| Graph GA (published, PMO) | not published | not published | not published | not published | not published | 0.964 ± 0.012 |

**B. Cold start — 5 random ZINC molecules** (`--seed-mode cold`; the fair comparison with Graph GA / random)

| method | @50 | @100 | @250 | @500 | @1,000 | @10,000 |
|---|---|---|---|---|---|---|
| ours (adversary on) | 0.019 (var 3.9e-05) | 0.038 (var 2.0e-04) | 0.091 (var 3.9e-03) | 0.204 (var 2.7e-02) | 0.327 (var 5.6e-02) | N/A — budget-capped by design |
| ours-ablated (adversary off) | 0.019 (var 3.9e-05) | 0.038 (var 2.0e-04) | 0.091 (var 3.9e-03) | 0.204 (var 2.7e-02) | 0.327 (var 5.6e-02) | N/A — budget-capped by design |
| single-call LLM | not run yet | not run yet | not run yet | not run yet | not run yet | N/A (50 molecules total) |
| random (ZINC) | 0.026 (var 3.6e-04) | 0.037 (var 6.5e-04) | 0.064 (var 1.5e-03) | 0.113 (var 6.2e-04) | 0.182 (var 1.1e-03) | 0.534 (var 4.7e-03) |
| Graph GA (published, PMO) | not published | not published | not published | not published | not published | 0.964 ± 0.012 |

**How to read these.**
- **Warm start (A) is not evidence of a speed-up.** One fluorine on haloperidol already scores ≈1.0, so the
  running top-10 mean passes Graph GA's published 0.964 after only **15–19 calls**, and our AUC over N calls first
  exceeds 0.964 between N = 200 (0.962) and N = 250 (0.969). Graph GA's 0.964 is a cold-start 10,000-call AUC and
  AUC at different budgets is not like-for-like, so I am not stating a yield ratio.
- **Cold start (B): any gain over random comes from the ranking, and three seeds do not establish it.** AUC@1,000 = 0.327
  against random 0.182, but the seeds give 0.290 / 0.112 / 0.581 (var 5.6e-02): one is below random's mean, one is three times it
  (best DRD2 score reached: 0.65 / 0.21 / 0.87). The mock proposes edits without looking at anything in its payload, so the only
  part of the loop that uses scores is the surrogate-ranked selection; the earlier agents, which scored proposals in the order
  written, got 0.223. That is consistent with the ranking helping, nothing more, and it says nothing for or against the LLM
  branches. **The "match Graph GA in ~10× fewer oracle calls" claim is therefore currently unsupported.** Testing it needs the
  real Haiku/Sonnet branches under cold start, with more than two seeds.
- **Crossover where the advantage reverses: not determinable.** PMO publishes only Graph GA's 10,000-call
  endpoint, not its AUC@50/100/250/500/1,000, so there is no published curve to cross; our curve also stops at
  1,000 by design.

**Ablation is a null result, and by construction.** With and without the adversary the runs are identical in
both modes, because the trigger (`unique_scaffolds_top10 < 3 OR sa_trend_3r > 1.2 OR ad_similarity_trend <
-0.15`) fired in **0 of 3 seeds** per mode — the mock never reward-hacks, and a synthetic "exploiter" mode did not
reach the top-10 either. I did not tune thresholds to make it fire. The harness asserts identical budgets
(`ceiling == oracle.calls == CSV rows` for both arms) — an assertion that caught a real dead-end: a cold-start
run filled the beam with ~600 Da molecules and stalled at 500 calls (every add-on edit failed the MW cap). Fixed
by giving agents each parent's MW and teaching the mock to trim; runs now flag `stalled` in their summary instead
of truncating silently. The adversary → coordinator → prompt-injection path is covered by a plumbing test with an
*injected* trigger (`tests/test_stage3.py`), not by evidence that it catches real exploits.

## Live Claude runs (preliminary: earlier agents, cold start, 2 seeds)
Haiku 4.5 for the Scout and the three branches, Sonnet 5.5 for the adversary, 5 random ZINC seed molecules, budget 500. These
runs used the **earlier agents** (git `5fe8058` plus the live-path fixes below): no recent-results memory and proposals scored in
the order written, not ranked. The adversary was switched on but did **not** work (its Sonnet call was rejected by the API, see
"Bugs the first live run found"), so they are effectively the adversary-off arm. A third seed was started and then stopped at
185 calls by a stop request, and is excluded. The code that produced them is no longer in the tree; check out `5fe8058` to rerun.

| run | AUC@50 | @100 | @250 | @500 | best DRD2 | calls to first score > 0.5 | tokens |
|---|---|---|---|---|---|---|---|
| live, seed 0 | 0.038 | 0.065 | 0.130 | 0.428 | 0.934 | 305 | 928k |
| live, seed 1 | 0.122 | 0.343 | 0.671 | 0.828 | 0.993 | 53 | 791k |
| random ZINC (3 seeds: 0.088, 0.138, 0.112 at @500) | 0.026 | 0.037 | 0.064 | 0.112 | – | – | – |
| offline mock, cold, earlier agents (mean of 3) | 0.019 | 0.039 | 0.075 | 0.119 | – | – | – |

**How to read this.** Both live runs are above every random seed and far above the mock, but they differ by 0.4 AUC at @500 and
took 305 vs 53 calls to reach a score of 0.5, so two seeds support "it can optimise from a cold start" and nothing about a
typical run or a mean. The single-call LLM control has not been run, so there is still no evidence that the multi-agent
structure beats one prompt. This is not a prior-free cold start: the models know DRD2 pharmacology (the Scout's briefs converge on
an indole-piperazine series), so the numbers are not comparable with Graph GA's. A live run used 0.8-0.9M tokens (roughly $1.5
at Haiku 4.5 list prices, which I took from memory).

**Where the earlier agents lost efficiency** (both seeds, from the chat logs; seed 0 / seed 1):
- 28% / 35% of proposals reached the oracle: duplicates 39% / 37%, unparsable SMILES 22% / 20%, over the logP cap 8% / 3%.
  Rejected proposals cost no oracle calls, so this costs tokens and time, not benchmark score: the higher-yield seed used
  1.6k tokens per scored molecule against 1.9k. Yield fell as the runs went on (37% to 20% and 50% to 27% between the first and
  last 15 rounds), partly because later rounds have more scored molecules to duplicate. Seed 0's logP rejections rose from 2% to
  12% as its molecules got more lipophilic, seed 1's from 0% to 3%; no prompt stated the cap.
- 67% / 71% of oracle results were never shown to any agent. They saw the top-20 beam and a count of last round's rejection
  reasons, nothing else. 95% / 92% of duplicates re-propose a molecule scored in an earlier round, and 40% / 40% of
  duplicates come from a different branch.
- Duplicates reach back up to 20 rounds (only 32% / 36% are within 2 rounds) but concentrate: 30 molecules make up about half of
  them, 100 make up 84%, and one molecule was re-proposed 58 / 26 times.
- Children rarely beat their parent, and the mean delta is negative for every branch in both seeds: A 40% / 25%, B 13% / 18%,
  C 32% / 28% beat it. Branch B (ring hopping) has the lowest hit rate in both seeds but produced the most large gains in seed 1.
  Which branch found the best molecules differs by seed (top-10: A 4, B 3, C 3 vs A 3, B 1, C 6), so these runs give no basis for
  dropping or favouring a branch.
- Time to take off differs a lot: seed 0 stayed below 0.3 for 273 calls and then reached 0.89 by call 317; seed 1 passed 0.5
  at call 53 and 0.98 at call 213. Roughly 60% of all proposals came from the 5 best beam members in both.
- The Scout is 20% / 19% of tokens and about 3.4 s of every round (it runs before the branches); its value has not been ablated.

**What the live scores do not establish.** The molecules the live runs ended with look like classifier artifacts as much as
hits, and the project's own gaming detector cannot tell. All ten top molecules of seed 1 score 0.99 and are near-identical
indole-piperidine diamines that differ by F/Cl swaps, several carrying an aryl thiol and an N,N-aminal that a medicinal chemist
would question. Their AD similarity is 0.35-0.39. The five known DRD2 ligands score 0.51-0.70 on the same measure (dopamine 0.58,
chlorpromazine 0.51, aripiprazole 0.58, haloperidol 0.70, risperidone 0.58), and the random ZINC starting molecules, which
score 0, sit at 0.39-0.47. Across both runs the mean AD of molecules scoring 0.9 or more is 0.41 / 0.37, below every known
ligand and flat against score (rank correlation of score and AD: +0.23 / -0.07). So either the nearest-support-vector measure has
little dynamic range, or the optimiser reached 0.99 without moving toward the oracle's training domain; the data cannot say which.
The original trigger only watched for AD *falling* over three rounds, so a run that reached a high score at persistently low AD never tripped it; see "Adversary signal, planner and evidence step" for the change.
No claim here should be read as "these are DRD2 binders".

**What the agents do now** (`backend/agents/`, `_score_ranked` in `orchestrator.py`):
1. *Memory*: branches and the Scout get `recent_results` (the oracle's verdict, `[smiles, score, delta vs parent]`, on the last two
   rounds of proposals from all branches); branches also get `your_recent_rejections` (their own fixable rejections) and a short
   `do_not_repeat` list (the most re-proposed molecules, skipping ones the payload already shows).
2. *Limits in the prompt*: MW and logP caps, "already scored means discarded", and logP in each beam row.
3. *Ranked selection*: the valid proposals are ranked by a Tanimoto-kernel GP (`surrogate.py`) trained on already-scored molecules
   and the quota is spent on the best, instead of on the first valid ones. No oracle call, no score written. Valid proposals not
   selected are logged as `not_selected` and can be proposed again.

**What was checked offline** (replaying the two live runs, free; both seeds agree):
- The surrogate has signal. Trained only on earlier rounds, its mean picks 69% / 69% of each round's true top-4 (random 40% /
  38%; the parent's score alone 55% / 58%). The textbook exploration weight beta = 1 was no better than random (43% / 49%)
  because the std term swamps the mean, so the default is beta = 0.05 (beta 0 to 0.1 were equivalent; 0.5 and above degrade).
  This measures ranking among molecules the earlier agents scored, not yet the choice among ~10 valid proposals that
  `_score_ranked` makes.
- Memory does not remove duplicates by itself (a 2-round window covers about a third of them); `do_not_repeat` targets the
  concentration instead and is the first thing to ablate if it does not pay for its tokens.
- Cost: the new payload adds about 3.7k characters (+80%) to each branch call at real SMILES lengths, so expect roughly 1.5-1.7x the
  tokens of an earlier-agents run unless higher yield shortens it. Not measured live.

**Bugs the first live run found** (all fixed; the tests for the first three fail on the old code):
- `messages.create()` no longer accepts `temperature` (anthropic >= 1.11): every agent call failed and the run ended after the 5 seed
  molecules. It also showed as "Budget spent" with a misleading reason; a run whose agent calls fail now ends as an error with the
  real message.
- `claude-sonnet-5-5` rejects `tool_choice` of type `tool`, so the adversary could never return a diagnosis. The executor now falls
  back to the default `tool_choice` plus a prompt instruction and remembers it per model.
- `DRD2Oracle()` silently downloaded 35 MB into whatever directory it was started from (TDC resolves its cache relative to the
  CWD); it now constructs the oracle where the weights live.

**Still to run live:** the current agents with the adversary actually working (and the new trigger), on the same seeds, the evidence step, and the single-call baseline
(`backend/baselines/single_call_llm.py`). `backend/eval/live_results.py` turns live runs into `results_cold_live.json`, which the
Results page shows next to the mock sets.

## Adversary signal, planner and evidence step (written after the live runs; NOT run)
Everything in this section was written without running the test suite or any run, and the 16 tests added with it have not been
run. Treat it as unverified until `pytest` and a live run have been through it.

**1. A trigger the live data would have tripped** (`telemetry.py`). New condition `high_score_low_domain`: the top-10 mean score
is at least 0.8 *and* the top-10 mean AD similarity is below the lowest AD of the five known DRD2 ligands (0.51, chlorpromazine)
minus 0.05. It is a level, not a three-round trend, because the live runs stayed at a low AD from the first round. The 0.8 and
0.05 were chosen by looking at those two runs (top-10 mean AD 0.41 and 0.38), so they are calibrated on the data they would have
caught, not validated. Because the condition stays on once true, the adversary is called at most once every 5 rounds. The digest
now carries the floor, the top-10 mean score and AD, and thiol / aminal motif counts, and the adversary's prompt describes them.
Known weakness: the coordinator only cuts the flagged branch's quota for two rounds, so the other branches keep going.

**2. A stricter plausibility filter** (`chem_core.IMPLAUSIBLE`). The Gatekeeper now rejects a free thiol (`implausible_thiol`) and
an acyclic N,N-aminal (`implausible_aminal`) in agent proposals; seeds and the baselines are exempt. **The mock tables above
were generated before this filter**, so they will not reproduce exactly; the live results predate it as well.

**3. A planner** (`planner.py`, used in `_score_ranked`). For each batch of validated proposals it compares two tests the
surrogate can price before any oracle call: *exploit* (rank by predicted mean) and *explore* (mean + 1.0 x std). It scores each by
expected gain (predicted improvement over the weakest top-10 score) plus a learning weight (0.25 x budget remaining) times expected
learning (mean predictive std), per oracle call, and picks the larger. Both options spend the same number of oracle calls, so cost
does not separate them here. The choice and both options' numbers are logged (`plan` events) and the Run page shows the split.
The learning weight is untuned and std is a proxy for information gain, not a measurement of it. With the surrogate unable to
score yet (fewer than 30 labels) the agent's own order is kept.

**4. An evidence step** (`chembl.py`, `agents/evidence.py`, `evidence_step` in `orchestrator.py`). At the end of a live run the
final top 5 molecules are looked up in ChEMBL by similarity (>= 70%), and the nearest neighbours' measured activity at DRD2
(`CHEMBL217`) is reported with links to every compound and document used. Verdicts: `analogue_active` (a neighbour with pChEMBL >= 6),
`analogue_inactive`, `analogue_untested`, `no_analogue`, `unavailable`. This is a check of nearby compounds, not a measurement of the
molecule, and `no_analogue` does not distinguish a novel chemotype from a classifier artifact. The lookup is a real Omnigent
`FunctionTool` on an `evidence` specialist, called through the same policy gate as everything else; a new `evidence_cap` policy
allows it only for that agent, with one SMILES of at most 500 characters, at most 20 times per run. Failures return `unavailable`
and never stop a run. It is on for live runs and off for the mock (`--evidence` / `--no-evidence`, `POST /api/runs`). The ChEMBL
endpoints and field names were read from the live API; the code was not run end to end.

**What is still not done: Omnigent does not orchestrate.** Omnigent's real runtime is a server and runner stack (processes,
sessions, authentication), not a library call, and it could not be wired in without running it. The specialists remain Omnigent
`AgentDef`s with Omnigent tools and policies, driven by this repo's own loop. The brief's "choose between competing tests" is
covered only in the narrow sense of item 3.

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
| evidence | none (tool only) | end of a live run: ChEMBL check of the final top 5 hits, with citations | `lookup_chembl` (a real HTTP lookup) |

Agents are Omnigent `AgentDef`s run through Omnigent `Executor`s. Every agent's tool except the evidence specialist's just echoes its structured output back; none of them can read or compute anything. **Not used:** Omnigent's server/CLI runtime —
its executors need provider credentials, so the session is a `LabSession` dataclass plus Omnigent's
`SessionState` enum. The Gatekeeper makes no LLM calls.

## Policies (enforced in code at the orchestration layer, not in prompts)
1. **write_permission** — only the oracle wrapper writes scores; an agent calling any tool but its own, or
   submitting a `score`-like field, is DENIED.
2. **budget_cap** — hard stop at the oracle-call ceiling (`oracle_evaluate` is DENIED once reached).
3. **electrophile_approval** — a BRENK *electrophile* alert (Michael acceptors, alkyl/N-halides, aldehydes,
   etc.) ASKs for human approval; `AUTOPILOT=true` logs and auto-rejects. Other BRENK alerts and PAINS are
   telemetry for the adversary only and never pause the run.
4. **evidence_cap** — `lookup_chembl`, the one tool that reaches the network, takes one SMILES of at most 500 characters
   and runs at most 20 times per run; only the `evidence` agent may call it.
Gatekeeper rejections are logged by branch **and reason** (e.g. `ring_count_change`, `scaffold_change`, `implausible_thiol`).

## Logging for tuning agent behaviour
`backend/data/chats/<run_id>.jsonl`: one `agent_call` record per call (system prompt, input payload, output,
tokens, latency, policy verdict) and one `proposal_outcome` per proposal (parent, SMILES, Gatekeeper reason or
oracle score, AD similarity, SA, alerts), joined on `call_id`; plus `adversary_trigger` events. Every
agent-generated molecule is tagged (`origin_branch`, `agent_generated`). Policy events:
`backend/data/policy_events.jsonl`. Oracle trajectories: `backend/data/trajectories/*.csv`.

## Validation still needed before real-world use
- Partly done: 2 seeds with real Haiku using the earlier agents and no working adversary. Still needed: more seeds, the current agents, the adversary live, and the single-call control.
- Cold-start (or weak-seed) comparison against Graph GA run on *this* oracle build; measure token cost.
- Show the adversary catches a real exploit: needs a run where score climbs while `ad_similarity` falls; confirm
  trigger thresholds on real data and measure false-positive cost. In live seed 0 the trigger fired once at round 4 (a 3-round
  AD change of -0.155 against a -0.15 threshold) in a healthy run, so early rounds look prone to false positives; a mock run
  also drifted slowly (SA +1.7 over 35 rounds) without ever tripping a 3-round trigger.
- Check that high-scoring molecules are actually in-domain and chemically sensible (SA, medchem review),
  that PAINS/BRENK + SA are adequate filters, and test on other oracles.
- Confirm DRD2 scores are reproducible across rdkit/sklearn/PyTDC versions, and validate any hit
  experimentally — a classifier score is not binding affinity.
