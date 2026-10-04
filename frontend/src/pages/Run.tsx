import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type Approval, type RunState, type Trigger } from '../api'
import { rememberRun } from '../App'
import { AGENT_KEYS, TRIGGER_TEXT, agentMeta, f2, fmt, humanReason, usePoll } from '../lib'
import { AgentDot, Badge, Chip, Empty, Mol, Reveal } from '../components/ui'
import { LineChart, niceAxis } from '../components/Chart'

const TILE: Record<string, [string, string]> = {
  branch_a: ['#2F7D52', '#fff'], branch_b: ['#6F6A62', '#fff'], branch_c: ['#D8D3C8', '#14130F'], seed: ['var(--sec)', 'var(--ink)'],
}
const STATUS: Record<string, string> = { running: 'Running', finished: 'Budget spent', stopped: 'Stopped', error: 'Failed' }

export default function Run({ id: forced }: { id: string | null | undefined }) {
  const params = useParams()
  const id = forced === null ? null : params.id!
  if (!id) return <NoRun />
  return <RunView id={id} />
}

function NoRun() {
  const nav = useNavigate()
  return (
    <div className="mx-auto max-w-[760px] px-8 py-24">
      <Empty title="No runs yet." body="Start a run and the beam, charts and agent chatter will appear here as oracle calls are spent.">
        <button onClick={() => nav('/setup')} className="btn btn-primary px-5 py-[9px] text-sm">Start a run</button>
      </Empty>
    </div>
  )
}

function RunView({ id }: { id: string }) {
  const nav = useNavigate()
  const [running, setRunning] = useState(true)
  const { data: s, error, refresh } = usePoll(() => api.run(id), running ? 1500 : null, [id])
  const { data: runs } = usePoll(api.runs, running ? 6000 : null, [running])
  const [dismissed, setDismissed] = useState<string | null>(null)
  useEffect(() => { rememberRun(id) }, [id])
  useEffect(() => { if (s) setRunning(s.status === 'running') }, [s])

  if (error && !s) return <div className="mx-auto max-w-[760px] px-8 py-24"><Empty title="Couldn’t load this run." body={error}><button onClick={() => nav('/run')} className="btn btn-ghost px-5 py-[9px] text-sm">Back</button></Empty></div>
  if (!s) return <div className="mx-auto grid max-w-[1240px] gap-4 px-8 py-10"><div className="skeleton h-24 rounded-[26px]" /><div className="skeleton h-64 rounded-[26px]" /></div>

  const alert = [...s.triggers].reverse().find(t => t.fired.length > 0)
  const alertKey = alert ? `${id}:${alert.round}` : null
  const pct = Math.min(100, (s.used / Math.max(1, s.budget)) * 100)

  return (
    <div>
      <div className="z-[15] border-b border-line backdrop-blur-[14px] sm:sticky sm:top-16" style={{ background: 'var(--navbg)' }}>
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center justify-between gap-x-7 gap-y-2 px-4 pt-3.5 sm:px-8">
          <div className="flex flex-wrap items-baseline gap-x-7 gap-y-1">
            <div><span className="num text-[26px] font-bold tracking-[-.03em]">{fmt(s.used)}</span><span className="text-[15px] text-mute"> / {fmt(s.budget)} oracle calls</span></div>
            <div className="text-[15px] text-mute">Round <b className="num text-ink">{s.round}</b></div>
            <div className="text-[15px] text-mute" title="Offline-mock runs make no token-metered calls, so they report 0.">Tokens <b className="num text-ink">{fmt(s.tokens.total)}</b></div>
          </div>
          <div className="flex items-center gap-2.5 pb-2">
            <select aria-label="Switch run" value={id} onChange={e => nav(`/run/${e.target.value}`)} className="max-w-[210px] cursor-pointer truncate rounded-full border border-line bg-surface px-3 py-1.5 text-[13px] text-ink">
              {(runs ?? [{ id }]).map(r => <option key={r.id} value={r.id}>{r.id}</option>)}
            </select>
            <span className="flex items-center gap-2 text-sm text-mute">
              {s.status === 'running' && <span className="anim-blink h-[7px] w-[7px] rounded-full bg-allow" />}{STATUS[s.status]}
            </span>
            {s.status === 'running' && <button onClick={async () => { await api.stop(id); refresh() }} className="btn btn-ghost h-[38px] px-5 text-sm">Stop</button>}
          </div>
        </div>
        <div className="mt-1 h-[3px] bg-line"><div className="h-[3px] bg-ink transition-[width] duration-[800ms]" style={{ width: `${pct}%` }} /></div>
      </div>

      <div className="mx-auto flex max-w-[1240px] flex-col gap-11 px-4 pb-30 pt-9 sm:px-8">
        {s.status === 'error' && <Banner tone="deny" title="This run failed." body={s.error ?? 'Unknown error'} />}
        {s.summary?.stalled && s.status !== 'error' && <Banner tone="ask" title="The run stalled." body={`For five rounds in a row no proposal reached the oracle (${s.summary.stall_reason ?? 'the Gatekeeper and policies rejected every proposal'}), so the run stopped early instead of spinning.`} />}
        {s.policy.pending.filter(a => a.status === 'pending').map(a => <ApprovalBanner key={a.id} a={a} onDecide={async ok => { await api.decide(id, a.id, ok); refresh() }} />)}
        {alert && alertKey !== dismissed && <AdversaryBanner t={alert} adversaryOn={s.cfg.adversary !== false} onDismiss={() => setDismissed(alertKey)} inspect={() => nav(`/inspector/${id}?agent=adversary&round=${alert.round}`)} />}

        <BeamSection s={s} />
        <Reveal><Charts s={s} /></Reveal>
        <Reveal><div className="grid items-start gap-5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr))]"><Agents s={s} /><Policy s={s} /></div></Reveal>
        <div className="text-center text-sm text-mute">Every call, prompt and verdict of this run is in the <Link to={`/inspector/${id}`} className="text-ink">Inspector</Link>.</div>
      </div>
    </div>
  )
}

function Banner({ tone, title, body }: { tone: 'ask' | 'deny'; title: string; body: string }) {
  return (
    <div role="alert" className="anim-rise flex flex-wrap items-center gap-5 rounded-[28px] border px-[26px] py-[22px]" style={{ borderColor: `var(--${tone})`, background: `var(--${tone}bg)` }}>
      <span className="h-3 w-3 flex-none rounded-full" style={{ background: `var(--${tone})` }} />
      <div className="min-w-[260px] flex-1"><div className="text-lg font-bold tracking-[-.01em]">{title}</div><div className="mt-1 text-[15px] leading-normal text-mute">{body}</div></div>
    </div>
  )
}

function ApprovalBanner({ a, onDecide }: { a: Approval; onDecide: (ok: boolean) => void }) {
  return (
    <div role="alert" className="anim-rise flex flex-wrap items-center gap-5 rounded-[28px] border border-ask bg-askbg px-[26px] py-[22px]">
      <span className="anim-pulse h-3 w-3 flex-none rounded-full bg-ask" />
      <div className="min-w-[260px] flex-1">
        <div className="text-lg font-bold tracking-[-.01em]">Approval needed before the oracle sees this molecule.</div>
        <div className="mt-1 text-[15px] leading-normal text-mute">{a.reason}{a.branch ? ` · proposed by ${agentMeta(a.branch).name}` : ''}. The run is paused until you decide.</div>
        <div className="mono mt-2 break-all text-xs text-mute">{a.smiles}</div>
      </div>
      {a.smiles && <div className="h-[84px] w-[120px] flex-none rounded-2xl bg-surface p-1"><Mol smiles={a.smiles} w={120} h={84} /></div>}
      <div className="flex gap-2.5">
        <button onClick={() => onDecide(true)} className="btn btn-ink h-[42px] px-5 text-sm">Approve</button>
        <button onClick={() => onDecide(false)} className="btn btn-ghost h-[42px] px-5 text-sm">Deny</button>
      </div>
    </div>
  )
}

function AdversaryBanner({ t, adversaryOn, onDismiss, inspect }: { t: Trigger; adversaryOn: boolean; onDismiss: () => void; inspect: () => void }) {
  const st = t.stats
  const chips: string[] = []
  if (t.fired.includes('low_scaffold_diversity')) chips.push(`Unique top-10 scaffolds ${st.unique_scaffolds_top10} · fires below 3`)
  if (t.fired.includes('sa_creep')) chips.push(`SA ${st.sa_trend_3r >= 0 ? '+' : ''}${st.sa_trend_3r.toFixed(2)} over 3 rounds · fires above +1.2`)
  if (t.fired.includes('ad_similarity_drop')) chips.push(`AD similarity ${st.ad_similarity_trend.toFixed(2)} over 3 rounds · fires below −0.15`)
  const why = t.fired.map(f => TRIGGER_TEXT[f]).join('; ')
  return (
    <div className="anim-rise flex flex-wrap items-center gap-5 rounded-[28px] border border-ask bg-askbg px-[26px] py-[22px]">
      <span className="anim-pulse h-3 w-3 flex-none rounded-full bg-ask" />
      <div className="min-w-[260px] flex-1">
        <div className="text-lg font-bold tracking-[-.01em]">{adversaryOn ? 'Adversary' : 'Trigger fired (adversary off)'} · round {t.round}: {why}.</div>
        <div className="mb-3 mt-1 text-[15px] leading-normal text-mute">
          {!adversaryOn ? 'The adversary is disabled for this run, so nothing acted on it.'
            : t.acted ? <>{t.diagnosis} <b className="text-ink">Instruction to {agentMeta(t.flagged ?? '').name}:</b> {t.instruction}</>
            : t.skipped === 'cooldown' ? `${agentMeta(t.flagged ?? '').name} is already in cooldown from an earlier flag.` : 'No branch was flagged.'}
        </div>
        <div className="flex flex-wrap gap-2">{chips.map(c => <span key={c} className="rounded-full border border-line bg-surface px-[13px] py-[5px] text-[13px]">{c}</span>)}</div>
      </div>
      <div className="flex gap-2.5">
        <button onClick={inspect} className="btn btn-ink h-[42px] px-5 text-sm">Review evidence</button>
        <button onClick={onDismiss} className="btn btn-ghost h-[42px] px-5 text-sm">Dismiss</button>
      </div>
    </div>
  )
}

function BeamSection({ s }: { s: RunState }) {
  return (
    <Reveal>
      <div className="mb-[18px] flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 text-[32px] font-bold tracking-[-.03em]">The beam <em className="font-medium">right now</em></h2>
        <span className="text-[15px] text-mute">Top 10 unique molecules by oracle-confirmed DRD2</span>
      </div>
      {s.beam.length === 0 ? <Empty title="The beam is empty." body="No molecule has been scored yet." /> : (
        <div className="grid gap-3.5 [grid-template-columns:repeat(auto-fill,minmax(210px,1fr))]">
          {s.beam.map(m => {
            const [bg, fg] = TILE[m.branch] ?? TILE.seed
            return (
              <div key={m.smiles} className="flex flex-col gap-3 rounded-[26px] border border-line bg-surface p-3.5 transition duration-300 hover:-translate-y-1 hover:border-ink">
                <div className="relative grid h-[104px] place-items-center rounded-2xl p-2" style={{ background: bg, color: fg }}>
                  <span className="num absolute left-3 top-2 text-xs opacity-85">#{m.rank}</span>
                  <div className="h-full w-full"><Mol smiles={m.smiles} w={190} h={100} className="mx-auto !h-full !w-auto" /></div>
                </div>
                <div className="mono h-[50px] overflow-hidden break-all text-[11px] leading-normal text-mute" title={m.smiles}>{m.smiles}</div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Chip strong className="!px-[11px] !py-1 font-semibold">{f2(m.score)}</Chip>
                  <Chip className="!px-[9px] !py-1 !text-xs">AD {f2(m.ad)}</Chip>
                  <Chip className="!px-[9px] !py-1 !text-xs">SA {f2(m.sa)}</Chip>
                </div>
                <div className="flex items-center gap-1.5 text-xs text-mute">
                  <span className="grid h-[18px] w-[18px] place-items-center rounded-full border border-ink text-[10px] font-bold text-ink">{agentMeta(m.branch).ini}</span>
                  {agentMeta(m.branch).name} <span className="num ml-auto">call {m.call_n}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </Reveal>
  )
}

function Charts({ s }: { s: RunState }) {
  const callsOfRound = (r: number) => s.rounds.find(x => x.round === r)?.oracle_calls_used
  const fired = s.triggers.filter(t => t.fired.length)
  const markers = fired.map(t => callsOfRound(t.round)).filter((x): x is number => x != null).map((x, i) => ({ x, label: i === 0 ? 'trigger fired' : undefined }))
  const xs = [0, 0.25, 0.5, 0.75, 1].map(f => Math.round(f * s.budget))
  const rounds = s.rounds
  const ad = niceAxis(Math.min(...rounds.map(r => r.ad_top10)), Math.max(...rounds.map(r => r.ad_top10)), 4)
  const sa = niceAxis(Math.min(...rounds.map(r => r.sa_top10)), Math.max(...rounds.map(r => r.sa_top10)), 4)
  const last = Math.max(1, rounds.length ? rounds[rounds.length - 1].round : 1)
  const rt = [...new Set([0, Math.round(last / 3), Math.round((2 * last) / 3), last])]
  return (
    <div className="grid gap-5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))]">
      <div className="card">
        <div className="text-lg font-bold">Best score vs. oracle calls</div>
        <div className="mb-4 mt-[3px] flex flex-wrap gap-x-4 text-sm text-mute"><span>── best so far <b className="num text-ink">{f2(s.curve.at(-1)?.[1])}</b></span><span>╌╌ top-10 mean <b className="num text-ink">{f2(s.curve.at(-1)?.[2])}</b></span><span>PMO AUC-10 so far <b className="num text-ink">{s.auc?.toFixed(3) ?? '–'}</b></span></div>
        <LineChart ariaLabel="Best score and top-10 mean against oracle calls" x={[0, s.budget]} yl={[0, 1]} yTicks={[0, 0.25, 0.5, 0.75, 1]} yFmt={v => String(v)} xTicks={xs} xFmt={v => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v))} markers={markers}
          series={[{ points: s.curve.map(p => [p[0], p[2]] as [number, number]), color: 'var(--mute)', dashed: true }, { points: s.curve.map(p => [p[0], p[1]] as [number, number]), color: 'var(--ink)', dot: true, area: true }]} />
      </div>
      <div className="card">
        <div className="text-lg font-bold">Applicability domain &amp; SA per round</div>
        <div className="mb-4 mt-[3px] flex gap-4 text-sm text-mute"><span>── AD similarity (left)</span><span>╌╌ SA score (right)</span><span className="ml-auto">top-10 of the beam</span></div>
        {rounds.length < 2 ? (s.status === 'running'
          ? <Empty title="Collecting." body="The chart needs two finished rounds." />
          : <Empty title="Not recorded." body="Per-round telemetry is only logged by runs started from this UI (or after it was added). Start a new run to see it." />) : (
          <LineChart ariaLabel="AD similarity and SA score per round" x={[0, last]} yl={ad.dom} yr={sa.dom} yTicks={ad.ticks} yrTicks={sa.ticks} yFmt={v => v.toFixed(2)} yrFmt={v => v.toFixed(1)} xTicks={rt} xFmt={v => `R${v}`}
            markers={fired.map((t, i) => ({ x: t.round, label: i === 0 ? 'trigger fired' : undefined }))}
            series={[{ points: rounds.map(r => [r.round, r.ad_top10] as [number, number]), color: 'var(--ink)' }, { points: rounds.map(r => [r.round, r.sa_top10] as [number, number]), color: 'var(--mute)', dashed: true, axis: 'r' }]} />
        )}
      </div>
    </div>
  )
}

function Agents({ s }: { s: RunState }) {
  const newest = s.feed.at(-1)?.agent
  const feed = [...s.feed].reverse()
  return (
    <div className="card">
      <h2 className="m-0 mb-1 text-[26px] font-bold tracking-[-.03em]">Watch the agents <em className="font-medium">think</em></h2>
      <div className="mb-[22px] mt-[18px] flex flex-wrap gap-3.5">
        {AGENT_KEYS.map(k => (
          <div key={k} className="flex flex-col items-center gap-1.5 text-xs text-mute" title={k === 'coordinator' ? 'Deterministic code: no LLM calls' : `${s.agents[k]?.calls ?? 0} calls`}>
            <AgentDot agent={k} size={42} pulse={s.status === 'running' && k === newest} />{agentMeta(k).name.split(' · ').pop()}
            <span className="num -mt-1 text-[11px]">{k === 'coordinator' ? 'code' : s.agents[k]?.calls ?? 0}</span>
          </div>
        ))}
      </div>
      {feed.length === 0 ? <Empty title="Quiet so far." body="Agent messages appear here once round 1 starts." /> : (
        <div className="flex flex-col gap-4">
          {feed.map((f, i) => (
            <div key={`${f.ts}-${f.agent}-${i}`} className="anim-rise flex gap-3">
              <AgentDot agent={f.agent} />
              <div className="min-w-0 flex-1">
                <div className="mb-[5px] text-[13px] text-mute"><b className="text-ink">{agentMeta(f.agent).name}</b> · Round {f.round} · {f.ts}</div>
                <div className={`text-pretty rounded-[4px_20px_20px_20px] px-4 py-3 text-[15px] leading-[1.55] ${f.allowed ? 'bg-soft' : 'bg-denybg text-deny'}`}>{f.text}</div>
                {f.smiles && <div className="mono mt-1.5 break-all text-[11.5px] text-mute">{f.smiles}</div>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Policy({ s }: { s: RunState }) {
  const p = s.policy
  const pending = p.pending.filter(a => a.status === 'pending').length
  const askN = p.recorded ? p.ask : p.oracle_rejected
  const rows: { name: string; agent: string; text: string; tone: 'ok' | 'ask' | 'deny'; label: string }[] = [
    { name: 'write-permission', agent: 'every agent', text: 'Agents may call only their own tool and may never write a score.', tone: p.agent_denied ? 'deny' : 'ok', label: p.agent_denied ? `${p.agent_denied} BLOCKED` : 'CLEAR' },
    { name: 'budget-cap', agent: 'oracle step', text: `Hard stop at the ceiling: ${fmt(s.used)} of ${fmt(s.budget)} calls spent.`, tone: s.used >= s.budget ? 'deny' : 'ok', label: s.used >= s.budget ? 'REACHED' : 'OPEN' },
    { name: 'electrophile-approval', agent: 'oracle step', text: p.recorded ? (s.cfg.ask_human ? `${p.ask} electrophile${p.ask === 1 ? '' : 's'} asked a human, ${p.ask_approved} approved.` : `${p.ask} electrophile${p.ask === 1 ? '' : 's'} auto-rejected (autopilot, no human asked).`) : `${p.oracle_rejected} proposals were stopped at the oracle step (older run: reasons not recorded).`, tone: askN ? 'ask' : 'ok', label: askN ? `ASK ×${askN}` : 'CLEAR' },
  ]
  const rej = Object.entries(s.gatekeeper).filter(([k]) => k !== 'policy_rejected' && k !== 'not_selected').sort((a, b) => b[1] - a[1])
  const notSelected = s.gatekeeper.not_selected ?? 0  // valid proposals the surrogate ranked below the quota; not a rejection
  const maxRej = Math.max(1, ...rej.map(r => r[1]))
  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <h2 className="m-0 mb-1 text-[26px] font-bold tracking-[-.03em]">Policy <em className="font-medium">gate</em></h2>
        <div className="mb-[18px] text-[15px] text-mute">{pending ? `${pending} waiting on you.` : 'Nothing waiting on you.'}</div>
        <div className="flex flex-col gap-3">
          {rows.map(r => (
            <div key={r.name} className="flex flex-col gap-2.5 rounded-[22px] border border-line px-[18px] py-4 transition-colors hover:border-ink">
              <div className="flex items-center justify-between gap-2.5"><span className="text-xs uppercase tracking-[.07em] text-mute">{r.name} · {r.agent}</span><Badge tone={r.tone}>{r.label}</Badge></div>
              <div className="text-[15px] leading-normal">{r.text}</div>
            </div>
          ))}
        </div>
        {p.events.length > 0 && (
          <div className="mt-5">
            <div className="mb-2 text-xs uppercase tracking-[.07em] text-mute">Recent decisions</div>
            <div className="flex flex-col gap-2">
              {[...p.events].reverse().slice(0, 5).map((e, i) => (
                <div key={i} className="flex items-start gap-2.5 text-sm">
                  <Badge tone={e.verdict === 'DENY' ? 'deny' : e.approved ? 'ok' : 'ask'}>{e.verdict === 'DENY' ? 'DENY' : e.approved ? 'APPROVED' : 'REJECTED'}</Badge>
                  <div className="min-w-0"><div>{e.reason}</div>{e.smiles && <div className="mono truncate text-[11px] text-mute">{e.smiles}</div>}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="card">
        <div className="text-lg font-bold">Gatekeeper rejections</div>
        <div className="mb-4 mt-[3px] text-sm text-mute">Proposals stopped on chemistry before they could cost a call.</div>
        {notSelected > 0 && <div className="mb-3 text-sm text-mute">{fmt(notSelected)} further proposals were valid but ranked below the quota by the surrogate, so they were not scored (and can be proposed again).</div>}
        {rej.length === 0 ? <div className="text-sm text-mute">None yet.</div> : (
          <div className="flex flex-col gap-2.5">
            {rej.map(([k, v]) => (
              <div key={k} className="grid grid-cols-[minmax(110px,150px)_1fr_44px] items-center gap-3 text-sm">
                <span>{humanReason(k)}</span>
                <span className="h-2.5 rounded-full bg-soft"><span className="block h-2.5 rounded-full bg-ink" style={{ width: `${(v / maxRej) * 100}%` }} /></span>
                <span className="num text-right font-semibold">{fmt(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
