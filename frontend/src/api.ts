// Typed client for backend/server.py. Every shape mirrors what that module returns.
export type RunStatus = 'running' | 'finished' | 'stopped' | 'error'

export interface RunRow {
  id: string; source: 'ui' | 'eval'; status: RunStatus; calls: number; budget: number
  branches: string | null; adversary: boolean | null; seed_mode: string | null; llm: string | null
  seed: number | null; updated: number
}
export interface Config { llm_available: boolean; active_run: string | null }
export interface StartBody { budget: number; branches: string; adversary: boolean; llm: 'offline' | 'anthropic'; seed_mode: 'known' | 'cold'; seed?: number; ask_human?: boolean }

export interface BeamMol { rank: number; smiles: string; score: number; ad: number | null; sa: number | null; branch: string; alerts: string[]; call_n: number }
export interface RoundRec { round: number; oracle_calls_used: number; best: number; top10_mean: number; sa_top10: number; ad_top10: number; scaffolds_in_beam: number; quotas?: Record<string, number>; trigger?: string[] }
export interface Trigger {
  round: number; fired: string[]; flagged: string | null; acted: boolean; diagnosis?: string; instruction?: string; skipped?: string
  stats: { unique_scaffolds_top10: number; dominant_branch: string; sa_trend_3r: number; ad_similarity_trend: number; oracle_calls_used: number; round: number; top10_mean_score?: number; top10_mean_ad?: number; ad_known_ligand_floor?: number | null }
}
export interface FeedItem { agent: string; round: number | null; ts: string; text: string; smiles: string | null; allowed: boolean }
export interface EvidenceNeighbour { chembl_id: string; name: string | null; similarity: number | null; smiles: string | null; drd2_activities: number; drd2_pchembl_max: number | null; url: string }
export interface Evidence {
  id: string; smiles: string; oracle_score: number; ad_similarity: number | null; origin: string
  verdict: 'analogue_active' | 'analogue_inactive' | 'analogue_untested' | 'no_analogue' | 'unavailable' | 'blocked'
  label?: string; source?: string; target?: string; cutoff_percent?: number; error: string | null
  neighbours: EvidenceNeighbour[]; documents: { chembl_id: string; url: string }[]
}
export interface Approval { id: string; status: 'pending' | 'approved' | 'rejected'; reason: string; branch: string | null; smiles: string | null; alerts: string[] }
export interface PolicyEvent { round: number; verdict: 'ASK' | 'DENY'; policy?: string; agent?: string; tool?: string; reason: string; approved?: boolean; smiles?: string; branch?: string; ts?: string }

export interface RunState {
  id: string; status: RunStatus; error: string | null; summary: { stalled?: boolean; stall_reason?: string | null; trigger_rounds?: number } | null
  cfg: { budget?: number; branches?: string; adversary?: boolean; seed_mode?: string; llm?: string; seed?: number; ask_human?: boolean }
  used: number; budget: number; round: number; llm: string | null
  tokens: { input: number; output: number; total: number }
  beam: BeamMol[]; curve: [number, number, number][]; auc: number | null
  rounds: RoundRec[]; triggers: Trigger[]
  agents: Record<string, { calls: number; denied: number; tokens: number }>
  gatekeeper: Record<string, number>; feed: FeedItem[]; planner: Record<string, number>; evidence: Evidence[]
  policy: { events: PolicyEvent[]; ask: number; ask_approved: number; deny: number; agent_denied: number; oracle_rejected: number; recorded: boolean; pending: Approval[] }
}

export interface CallRow {
  call_id: string; round: number | null; agent: string; model: string; allowed: boolean; reason: string | null; title: string
  latency_ms: number; tokens_in: number; tokens_out: number; n_outcomes: number; n_scored: number; rejections: Record<string, number>; best: number | null
}
export interface CallPage { total: number; of: number; items: CallRow[]; rounds: number[]; agents: string[] }
export interface Outcome { parent_id: string; smiles: string; gate_reason: string | null; oracle_score: number | null; ad_similarity: number | null; sa_score: number | null; alerts: string[] | null }
export interface CallDetail extends CallRow { system_prompt: string; input: unknown; output: { tool: string | null; args: unknown }; outcomes: Outcome[] }

export interface AucCell { mean: number; var: number; values: number[] }
export interface ResultsFile {
  budget: number; seeds: number[]; llm_mode: string; exploit_mock: boolean; seed_mode: string
  ours: Record<string, AucCell>; ablated: Record<string, AucCell>
  trigger_rounds: number[]; adversary_calls: number[]; tokens: ({ input_tokens: number; output_tokens: number } | number | null)[]
  random: Record<string, Record<string, number>>
  // live sets only (eval/live_results.py)
  acted_rounds?: number[]; single_call?: Record<string, number>; notes?: string[]
}
export type Results = Partial<Record<'known' | 'cold' | 'live', ResultsFile>>

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail ?? msg } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return r.json()
}
const post = (url: string, body?: unknown) =>
  req<{ id?: string; ok?: boolean }>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined })

export const api = {
  config: () => req<Config>('/api/config'),
  runs: () => req<RunRow[]>('/api/runs'),
  run: (id: string) => req<RunState>(`/api/runs/${id}`),
  start: (b: StartBody) => post('/api/runs', b),
  stop: (id: string) => post(`/api/runs/${id}/stop`),
  decide: (id: string, approval: string, approve: boolean) => post(`/api/runs/${id}/approvals/${approval}`, { approve }),
  calls: (id: string, q: Record<string, string | number | undefined>, signal?: AbortSignal) => {
    const p = new URLSearchParams()
    Object.entries(q).forEach(([k, v]) => v !== undefined && v !== '' && p.set(k, String(v)))
    return req<CallPage>(`/api/runs/${id}/calls?${p}`, { signal })
  },
  call: (id: string, callId: string) => req<CallDetail>(`/api/runs/${id}/calls/${callId}`),
  results: () => req<Results>('/api/results'),
  molSvg: (smiles: string, w = 240, h = 160) => `/api/mol.svg?smiles=${encodeURIComponent(smiles)}&w=${w}&h=${h}`,
}
