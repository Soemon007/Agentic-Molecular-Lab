import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, type CallDetail, type CallPage, type CallRow } from '../api'
import { rememberRun } from '../App'
import { agentMeta, f2, fmt, humanReason, usePoll } from '../lib'
import { AgentDot, Badge, Chip, Empty, PageHero } from '../components/ui'

const PAGE = 50
const VERDICTS: [string, string][] = [['', 'All'], ['allowed', 'Allowed'], ['denied', 'Denied']]

function FilterRow({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="flex flex-wrap items-center gap-2.5"><span className="w-[62px] text-[13px] text-mute">{label}</span>{children}</div>
}
function Pill({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button onClick={onClick} aria-pressed={on} className={`cursor-pointer rounded-full border px-3.5 py-1.5 text-[13px] font-medium transition hover:-translate-y-px active:scale-95 ${on ? 'border-ink bg-ink text-canvas' : 'border-line bg-transparent text-ink'}`}>{children}</button>
}

export default function Inspector() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const [sp, setSp] = useSearchParams()
  const round = sp.get('round') ?? '', agent = sp.get('agent') ?? '', verdict = sp.get('verdict') ?? ''
  const [items, setItems] = useState<CallRow[]>([])
  const [meta, setMeta] = useState<CallPage | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const filterKey = `${id}|${round}|${agent}|${verdict}`
  const { data: runs } = usePoll(api.runs, null, [])
  const facets = useRef<{ rounds: number[]; agents: string[] }>({ rounds: [], agents: [] })

  useEffect(() => { rememberRun(id) }, [id])
  useEffect(() => {
    const ac = new AbortController()
    setLoading(true); setErr(null)
    api.calls(id, { round: round || undefined, agent: agent || undefined, verdict: verdict || undefined, offset: 0, limit: PAGE }, ac.signal)
      .then(p => { setItems(p.items); setMeta(p); if (!round && !agent && !verdict) facets.current = { rounds: p.rounds, agents: p.agents }; setLoading(false) })
      .catch(e => { if (e.name !== 'AbortError') { setErr(e.message); setLoading(false) } })
    return () => ac.abort()
  }, [filterKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); v ? n.set(k, v) : n.delete(k); setSp(n, { replace: true }); setOpen(null) }
  const more = async () => {
    const p = await api.calls(id, { round: round || undefined, agent: agent || undefined, verdict: verdict || undefined, offset: items.length, limit: PAGE })
    setItems([...items, ...p.items])
  }
  const f = facets.current.rounds.length ? facets.current : { rounds: meta?.rounds ?? [], agents: meta?.agents ?? [] }
  const filtered = !!(round || agent || verdict)

  return (
    <div>
      <PageHero bg="#6F6A62" decor="right" width="max-w-[980px]" title="Read the" accent="conversation." sub="Every call, in order. Open one to see exactly what the agent saw and what happened to its proposals." />
      <div className="mx-auto max-w-[980px] px-4 pb-30 pt-14 sm:px-8">
        <div className="mb-5 flex flex-col gap-3.5 rounded-[28px] border border-line bg-surface px-6 py-5">
          <FilterRow label="Run">
            <select aria-label="Run" value={id} onChange={e => nav(`/inspector/${e.target.value}`)} className="max-w-full cursor-pointer rounded-full border border-line bg-surface px-3 py-1.5 text-[13px] text-ink">
              {(runs ?? [{ id }]).map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
            </select>
          </FilterRow>
          <FilterRow label="Round">
            <Pill on={!round} onClick={() => set('round', '')}>All</Pill>
            {f.rounds.length > 24
              ? <input type="number" min={1} max={f.rounds.at(-1)} placeholder="round #" aria-label="Round number" value={round} onChange={e => set('round', e.target.value)} className="w-24 rounded-full border border-line bg-transparent px-3 py-1.5 text-[13px] text-ink" />
              : f.rounds.map(r => <Pill key={r} on={round === String(r)} onClick={() => set('round', String(r))}>R{r}</Pill>)}
          </FilterRow>
          <FilterRow label="Agent">
            <Pill on={!agent} onClick={() => set('agent', '')}>All</Pill>
            {f.agents.map(a => <Pill key={a} on={agent === a} onClick={() => set('agent', a)}>{agentMeta(a).name}</Pill>)}
          </FilterRow>
          <FilterRow label="Verdict">{VERDICTS.map(([v, l]) => <Pill key={l} on={verdict === v} onClick={() => set('verdict', v)}>{l}</Pill>)}</FilterRow>
        </div>

        <div className="mx-2 mb-3 text-sm text-mute" aria-live="polite">{meta ? `${fmt(meta.total)} of ${fmt(meta.of)} calls` : ' '}</div>
        {err ? <Empty title="Couldn’t load calls." body={err} />
          : loading ? <div className="flex flex-col gap-2.5">{[0, 1, 2, 3].map(i => <div key={i} className="skeleton h-[68px] rounded-3xl" />)}</div>
          : items.length === 0 ? (
            <Empty title="Nothing matches." body={filtered ? 'Try loosening a filter.' : 'This run has no logged agent calls.'}>
              {filtered && <button onClick={() => setSp({}, { replace: true })} className="btn btn-ghost h-10 px-5 text-sm">Clear filters</button>}
            </Empty>
          ) : (
            <div className="flex flex-col gap-2.5">
              {items.map(c => <CallItem key={c.call_id} run={id} c={c} open={open === c.call_id} onToggle={() => setOpen(open === c.call_id ? null : c.call_id)} />)}
              {meta && items.length < meta.total && <button onClick={more} className="btn btn-ghost mx-auto mt-3 h-11 px-6 text-sm">Show {Math.min(PAGE, meta.total - items.length)} more</button>}
            </div>
          )}
      </div>
    </div>
  )
}

const json = (v: unknown) => JSON.stringify(v, null, 2)

function CallItem({ run, c, open, onToggle }: { run: string; c: CallRow; open: boolean; onToggle: () => void }) {
  const [d, setD] = useState<CallDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { if (open && !d) api.call(run, c.call_id).then(setD).catch(e => setErr(e.message)) }, [open]) // eslint-disable-line react-hooks/exhaustive-deps
  const a = agentMeta(c.agent)
  return (
    <div className="overflow-hidden rounded-3xl border bg-surface transition-colors" style={{ borderColor: open ? 'var(--ink)' : 'var(--line)' }}>
      <button onClick={onToggle} aria-expanded={open} className="grid w-full cursor-pointer grid-cols-[auto_auto_1fr_auto_auto] items-center gap-3.5 border-0 bg-transparent px-5 py-4 text-left text-ink transition-colors hover:bg-soft">
        <span className="mono text-xs text-mute">R{c.round}</span>
        <AgentDot agent={c.agent} size={30} />
        <span className="min-w-0"><span className="block truncate text-[15px] font-medium">{c.title}</span><span className="block text-[13px] text-mute">{a.name} · {c.latency_ms} ms · {fmt(c.tokens_in + c.tokens_out)} tok</span></span>
        <Badge tone={c.allowed ? 'ok' : 'deny'}>{c.allowed ? 'ALLOW' : 'DENY'}</Badge>
        <span className="text-mute transition-transform duration-300" style={{ transform: `rotate(${open ? 180 : 0}deg)` }} aria-hidden>▾</span>
      </button>
      {open && (
        <div className="anim-rise flex flex-col gap-3.5 px-5 pb-[22px] pt-1">
          {err && <div className="text-sm text-deny">{err}</div>}
          {!d && !err && <div className="skeleton h-24 rounded-2xl" />}
          {d && <>
            {!d.allowed && <div className="rounded-2xl bg-denybg px-4 py-3 text-sm text-deny">Policy verdict: {d.reason}</div>}
            <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))]">
              <Block label="System prompt" text={d.system_prompt} />
              <Block label="Input" text={json(d.input)} scroll />
            </div>
            <Block label={`Tool output · ${d.output?.tool ?? 'none'}`} text={json(d.output?.args)} scroll />
            <div className="flex flex-wrap gap-2"><Chip>in {fmt(d.tokens_in)} tok</Chip><Chip>out {fmt(d.tokens_out)} tok</Chip><Chip>{d.latency_ms} ms</Chip><Chip>{d.model}</Chip></div>
            {d.outcomes.length > 0 && (
              <div className="rounded-[20px] border border-line px-[18px] py-4">
                <div className="mb-2.5 text-sm font-bold">Proposal outcomes · {d.n_scored} scored, {d.outcomes.length - d.n_scored} rejected</div>
                <div className="flex max-h-72 flex-col gap-2 overflow-auto">
                  {d.outcomes.map((o, i) => (
                    <div key={i} className="grid items-center gap-x-3 gap-y-1 border-t border-line pt-2 text-sm [grid-template-columns:minmax(0,1fr)_auto] first:border-0 first:pt-0">
                      <span className="mono break-all text-xs">{o.smiles}</span>
                      {o.oracle_score != null
                        ? <span className="flex flex-wrap justify-end gap-1.5"><Chip strong>{f2(o.oracle_score)}</Chip><Chip>AD {f2(o.ad_similarity)}</Chip><Chip>SA {f2(o.sa_score)}</Chip>{o.alerts?.length ? <Chip>{o.alerts.length} alert{o.alerts.length > 1 ? 's' : ''}</Chip> : null}</span>
                        : <Badge tone="deny">{humanReason(o.gate_reason ?? 'rejected')}</Badge>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>}
        </div>
      )}
    </div>
  )
}

function Block({ label, text, scroll }: { label: string; text: string; scroll?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="mb-1.5 text-xs uppercase tracking-[.07em] text-mute">{label}</div>
      <pre className={`mono m-0 whitespace-pre-wrap rounded-2xl [overflow-wrap:anywhere] bg-soft p-3.5 text-xs leading-relaxed ${scroll ? 'max-h-64 overflow-auto' : ''}`}>{text}</pre>
    </div>
  )
}
