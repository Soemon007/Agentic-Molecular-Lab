import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { fmt, usePoll } from '../lib'
import { PageHero } from '../components/ui'

const BRANCHES = [
  { k: 'a', n: 'Local', d: 'Acyclic substituent edits on beam members. No new rings.' },
  { k: 'b', n: 'Hopper', d: 'Replaces the core ring system, keeping the pharmacophore.' },
  { k: 'c', n: 'Explorer', d: 'Bioisosteres from lower-percentile beam entries.' },
]

function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { v: T; label: string; disabled?: boolean }[] }) {
  return (
    <div className="flex flex-none rounded-full bg-soft p-1">
      {options.map(o => (
        <button key={o.v} disabled={o.disabled} onClick={() => onChange(o.v)} className={`cursor-pointer rounded-full border-0 px-4 py-2 text-sm font-semibold transition ${value === o.v ? 'bg-ink text-canvas' : 'bg-transparent text-ink'}`}>{o.label}</button>
      ))}
    </div>
  )
}

export default function Setup() {
  const nav = useNavigate()
  const { data: cfg } = usePoll(api.config, 4000, [])
  const [budget, setBudget] = useState(500)
  const [br, setBr] = useState({ a: true, b: true, c: true })
  const [adv, setAdv] = useState(true)
  const [live, setLive] = useState(false)
  const [cold, setCold] = useState(false)
  const [ask, setAsk] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const branches = (['a', 'b', 'c'] as const).filter(k => br[k]).join('')
  const nb = branches.length
  const liveOk = !!cfg?.llm_available
  const active = cfg?.active_run

  const start = async () => {
    setBusy(true); setErr(null)
    try {
      const r = await api.start({ budget, branches, adversary: adv, llm: live ? 'anthropic' : 'offline', seed_mode: cold ? 'cold' : 'known', ask_human: ask })
      nav(`/run/${r.id}`)
    } catch (e) { setErr((e as Error).message); setBusy(false) }
  }

  const card = 'rounded-[28px] border border-line bg-surface p-7'
  return (
    <div>
      <PageHero bg="#2F7D52" decor="left" width="max-w-[720px]" title="Set up a" accent="run." sub="A handful of choices. Each one maps to a flag of the orchestrator." />
      <div className="mx-auto max-w-[720px] px-8 pb-30 pt-14">
        <div className="anim-rise flex flex-col gap-4 [animation-delay:.1s]">
          <div className={card}>
            <div className="flex items-baseline justify-between"><span className="text-lg font-semibold">Budget</span>
              <span className="num text-[32px] font-bold tracking-[-.03em]">{fmt(budget)} <span className="text-[15px] font-normal text-mute">oracle calls</span></span></div>
            <input type="range" min={100} max={5000} step={100} value={budget} onChange={e => setBudget(+e.target.value)} aria-label="Oracle-call budget" className="mb-1.5 mt-[18px]" />
            <div className="text-sm text-mute">PMO standard is 10,000. The mock runs a few hundred calls in seconds; a live LLM run spends tokens on every round.</div>
          </div>

          <div className={card}>
            <div className="mb-4 text-lg font-semibold">Branches</div>
            <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr))]">
              {BRANCHES.map(b => {
                const on = br[b.k as 'a']
                return (
                  <button key={b.k} aria-pressed={on} onClick={() => setBr({ ...br, [b.k]: !on })}
                    className={`flex cursor-pointer flex-col gap-1.5 rounded-[20px] border p-[18px] text-left text-ink transition hover:-translate-y-[3px] active:scale-[.97] ${on ? 'border-ink bg-sec' : 'border-line bg-transparent'}`}>
                    <span className="flex items-center justify-between"><span className="text-[17px] font-bold">Branch {b.k.toUpperCase()}</span>
                      <span className="grid h-5 w-5 place-items-center rounded-full border-[1.5px] border-ink text-xs text-canvas transition" style={{ background: on ? 'var(--ink)' : 'transparent' }}>{on ? '✓' : ''}</span></span>
                    <span className="text-[15px] font-medium">{b.n}</span>
                    <span className="text-[13px] leading-snug text-mute">{b.d}</span>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="rounded-[28px] border border-line bg-surface px-7 py-2">
            <Row title="Adversary" hint="Watches the beam’s diversity, SA and domain drift, and tells the flagged branch to correct course.">
              <button role="switch" aria-checked={adv} aria-label="Adversary" onClick={() => setAdv(!adv)} className="h-8 w-14 flex-none cursor-pointer rounded-full border-0 p-[3px] transition-colors" style={{ background: adv ? 'var(--ink)' : 'var(--line)' }}>
                <span className="block h-[26px] w-[26px] rounded-full bg-canvas transition-transform duration-300" style={{ transform: `translateX(${adv ? 24 : 0}px)` }} />
              </button>
            </Row>
            <Row title="Mode" hint={live ? 'Real Haiku / Sonnet calls. Costs tokens and takes longer.' : liveOk ? 'Offline mock. Free and fast, but it is a plumbing test, not an LLM.' : 'Offline mock. Live needs ANTHROPIC_API_KEY set where the server runs.'}>
              <Seg value={live ? 'live' : 'mock'} onChange={v => setLive(v === 'live')} options={[{ v: 'mock', label: 'Offline mock' }, { v: 'live', label: 'Live LLM', disabled: !liveOk }]} />
            </Row>
            <Row title="Seed molecules" hint={cold ? '5 random ZINC molecules: the fair comparison with cold-start methods.' : '5 known DRD2 ligands. The oracle already scores three of them near 1.0, which flatters every comparison.'}>
              <Seg value={cold ? 'cold' : 'warm'} onChange={v => setCold(v === 'cold')} options={[{ v: 'warm', label: 'Known' }, { v: 'cold', label: 'Cold' }]} />
            </Row>
            <Row title="Electrophiles" hint={ask ? 'The run pauses and asks you before a molecule with an electrophile alert is scored. The mock proposes these often.' : 'Electrophile alerts are logged and auto-rejected, so the run never waits on you.'} last>
              <Seg value={ask ? 'ask' : 'auto'} onChange={v => setAsk(v === 'ask')} options={[{ v: 'auto', label: 'Auto-reject' }, { v: 'ask', label: 'Ask me' }]} />
            </Row>
          </div>

          <button onClick={start} disabled={busy || nb === 0 || !!active} className="btn btn-primary mt-3 h-[60px] text-lg">{busy ? 'Starting…' : 'Start run'}</button>
          <div className="text-center text-sm text-mute" role="status">
            {active ? <>A run is already in progress. <Link to={`/run/${active}`} className="text-ink">Open it</Link> or stop it first.</>
              : nb === 0 ? 'Pick at least one branch.'
              : `${fmt(budget)} calls · ${nb} branch${nb === 1 ? '' : 'es'} · adversary ${adv ? 'on' : 'off'} · ${live ? 'live LLM' : 'mock mode'} · ${cold ? 'cold' : 'known'} seeds`}
          </div>
          {err && <div className="rounded-2xl border border-deny bg-denybg px-4 py-3 text-sm text-deny" role="alert">{err}</div>}
        </div>
      </div>
    </div>
  )
}

function Row({ title, hint, children, last }: { title: string; hint: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div className={`flex items-center justify-between gap-4 py-5 ${last ? '' : 'border-b border-line'}`}>
      <div><div className="text-lg font-semibold">{title}</div><div className="mt-[3px] max-w-[380px] text-sm text-mute">{hint}</div></div>
      {children}
    </div>
  )
}
