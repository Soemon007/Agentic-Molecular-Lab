import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type Outcome, type ResultsFile } from '../api'
import { f2, f3, humanReason, usePoll } from '../lib'
import { Badge, Band, Chip, Decor, Mol, Reveal } from '../components/ui'

function useCountUp() {
  const [k, setK] = useState(0)
  useEffect(() => {
    const t0 = performance.now()
    let raf = 0
    const f = (n: number) => {
      const p = Math.min(1, Math.max(0, (n - t0 - 400) / 1600))
      setK(1 - Math.pow(1 - p, 3))
      if (p < 1) raf = requestAnimationFrame(f)
    }
    raf = requestAnimationFrame(f)
    return () => cancelAnimationFrame(raf)
  }, [])
  return k
}

/** A real rejected proposal and a real scored one, from the newest run that has both. */
function useShowcase() {
  const { data: runs } = usePoll(api.runs, null, [])
  const [s, setS] = useState<{ run: string; bad: Outcome; good: Outcome } | null>(null)
  const done = useRef(false)
  useEffect(() => {
    if (!runs || done.current) return
    done.current = true
    ;(async () => {
      for (const r of runs.slice(0, 3)) {
        try {
          const page = await api.calls(r.id, { agent: 'branch_a', limit: 60 })
          const hit = page.items.find(c => c.n_scored > 0 && Object.keys(c.rejections).length > 0)
          if (!hit) continue
          const d = await api.call(r.id, hit.call_id)
          const bad = d.outcomes.find(o => o.gate_reason)
          const good = d.outcomes.filter(o => o.oracle_score != null).sort((a, b) => b.oracle_score! - a.oracle_score!)[0]
          if (bad && good) return setS({ run: r.id, bad, good })
        } catch { /* try the next run */ }
      }
    })()
  }, [runs])
  return { showcase: s, latest: runs?.[0]?.id ?? null }
}

const AUC_AT = '1000'
const auc = (r: ResultsFile | undefined, arm: 'ours' | 'ablated') => r?.[arm]?.[AUC_AT]?.mean
const randAuc = (r: ResultsFile | undefined) => {
  const v = Object.values(r?.random ?? {}).map(x => x[AUC_AT]).filter(x => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined
}

export default function Home() {
  const nav = useNavigate()
  const k = useCountUp()
  const { showcase, latest } = useShowcase()
  const { data: results } = usePoll(api.results, null, [])
  const warm = results?.known, cold = results?.cold
  const stats = [
    { c: '#2F7D52', v: auc(warm, 'ours'), l: 'AUC-10 @1,000 · warm start' },
    { c: '#6F6A62', v: auc(cold, 'ours'), l: 'AUC-10 @1,000 · cold start' },
    { c: '#2F7D52', v: randAuc(cold), l: 'Random ZINC · same budget' },
    { c: '#6F6A62', v: warm ? warm.trigger_rounds.filter(t => t > 0).length : undefined, l: `Seeds where the adversary fired (of ${warm?.seeds.length ?? 3})`, int: true },
  ]

  return (
    <div>
      <section className="relative overflow-hidden">
        <Decor onDark={false} dir="left" />
        <div className="relative mx-auto max-w-[1240px] px-8">
          <div className="flex flex-col items-center gap-[30px] pb-28 pt-26 text-center">
            <div className="anim-rise inline-flex max-w-full items-center gap-2.5 whitespace-nowrap rounded-full border border-line bg-surface px-4 py-[7px] text-sm text-mute">
              <span className="anim-blink h-[7px] w-[7px] rounded-full bg-allow" />DRD2 · PMO benchmark · oracle-call budget
            </div>
            <h1 className="anim-rise m-0 max-w-[980px] text-balance text-[clamp(46px,8.4vw,104px)] font-bold leading-[.98] tracking-[-.045em] [animation-delay:.1s]">
              Discover molecules, <em className="font-medium tracking-[-.04em]">not shortcuts.</em>
            </h1>
            <p className="anim-rise m-0 max-w-[600px] text-pretty text-[clamp(17px,2vw,21px)] leading-normal text-mute [animation-delay:.2s]">
              A small crew of agents proposes drug-like molecules. Policies gate every oracle spend. One adversary watches for anyone gaming the scoring model.
            </p>
            <div className="anim-rise flex flex-wrap justify-center gap-3.5 [animation-delay:.3s]">
              <button onClick={() => nav('/setup')} className="btn btn-primary h-14 px-[34px] text-[17px]">Start a run</button>
              <button onClick={() => nav(latest ? `/run/${latest}` : '/run')} disabled={!latest} className="btn btn-ghost h-14 px-[34px] text-[17px]">See an example run</button>
            </div>
          </div>
        </div>
      </section>

      <Band bg="#2F7D52" decor="right" className="max-w-[1240px] px-8 py-28">
        <Reveal className="mb-11 text-center">
          <h2 className="m-0 text-[clamp(34px,5vw,60px)] font-bold leading-[1.05] tracking-[-.04em]">Messy in. <em className="font-medium">Polished out.</em></h2>
          <p className="mt-3 text-white/90">{showcase ? <>Two real proposals from run <span className="mono text-sm">{showcase.run}</span>.</> : 'Run the lab to see real proposals here.'}</p>
        </Reveal>
        <Reveal>
          <div className="grid gap-5 [grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))]">
            <div className="flex flex-col gap-[18px] rounded-[28px] border border-line bg-surface p-7 text-[#14130F]">
              <div className="flex items-center justify-between"><span className="text-[13px] uppercase tracking-[.08em] text-[#6F6A62]">Before · raw proposal</span><Badge tone="deny">REJECTED</Badge></div>
              {showcase ? <>
                <div className="mono break-all rounded-[18px] bg-[#EAE6DE] p-4 text-sm leading-[1.7] text-[#6F6A62]">{showcase.bad.smiles}</div>
                <div className="flex flex-wrap gap-2"><span className="rounded-full bg-[rgba(182,61,50,.09)] px-3 py-[5px] text-[13px] text-[#B63D32]">Gatekeeper: {humanReason(showcase.bad.gate_reason!)}</span></div>
              </> : <div className="skeleton h-24 rounded-[18px]" />}
              <p className="m-0 text-[15px] leading-normal text-[#6F6A62]">Caught by the Gatekeeper before it could cost an oracle call.</p>
            </div>
            <div className="flex flex-col gap-[18px] rounded-[28px] border border-line bg-surface p-7 text-[#14130F]">
              <div className="flex items-center justify-between"><span className="text-[13px] uppercase tracking-[.08em] text-[#6F6A62]">After · scored by the oracle</span><Badge tone="ok">ALLOWED</Badge></div>
              {showcase ? <>
                <div className="flex items-center gap-4">
                  <div className="grid h-[104px] w-[120px] flex-none place-items-center rounded-[18px] border border-line bg-[#E4DFD5] p-1.5"><Mol smiles={showcase.good.smiles} w={150} h={110} /></div>
                  <div className="mono break-all text-[13px] leading-relaxed">{showcase.good.smiles}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Chip strong className="!bg-[#14130F] !text-[#F3F0EA]">DRD2 {f2(showcase.good.oracle_score)}</Chip>
                  <Chip className="!bg-[#EAE6DE]">AD {f2(showcase.good.ad_similarity)}</Chip>
                  <Chip className="!bg-[#EAE6DE]">SA {f2(showcase.good.sa_score)}</Chip>
                </div>
              </> : <div className="skeleton h-24 rounded-[18px]" />}
              <p className="m-0 text-[15px] leading-normal text-[#6F6A62]">Scored by the real oracle. One call spent from the budget.</p>
            </div>
          </div>
        </Reveal>
      </Band>

      <Band bg="#6F6A62" className="max-w-[1240px] px-8 py-30">
        <Reveal className="text-center">
          <h2 className="m-0 text-[clamp(34px,5vw,60px)] font-bold leading-[1.05] tracking-[-.04em]">Spend your budget <em className="font-medium">wisely.</em></h2>
          <p className="mb-12 mt-4 text-[19px] opacity-90">Three steps, repeated until the calls run out.</p>
          <div className="grid gap-5 text-left [grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))]">
            {[
              { c: '#14130F', fg: '#fff', t: 'Agents propose', d: 'The Scout writes an SAR brief; branches A, B and C each propose edits, hops and bioisosteres. A coordinator splits the quota between them.' },
              { c: '#F3F0EA', fg: '#14130F', t: 'Gates decide', d: 'A Gatekeeper checks the chemistry. Policies check every write and spend; an electrophile stops and asks a human before it can be scored.' },
              { c: '#2F7D52', fg: '#fff', t: 'Oracle scores', d: 'Only approved molecules reach the real DRD2 oracle. Every call counts against the budget, so each call is earned.' },
            ].map((s, i) => (
              <div key={s.t} className="flex min-h-60 flex-col gap-3.5 rounded-[32px] p-8 transition-transform duration-300 hover:-translate-y-1.5" style={{ background: s.c, color: s.fg }}>
                <span className="grid h-10 w-10 place-items-center rounded-full bg-surface font-semibold text-[#14130F]">{i + 1}</span>
                <div className="text-[26px] font-bold tracking-[-.025em]">{s.t}</div>
                <div className="text-pretty text-base leading-[1.55] opacity-90">{s.d}</div>
              </div>
            ))}
          </div>
        </Reveal>
      </Band>

      <Band bg="#14130F" decor="right" className="max-w-[1240px] px-8 py-24">
        <Reveal>
          <div className="grid gap-4 text-center [grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr))]">
            {stats.map(s => (
              <div key={s.l} className="rounded-[28px] px-3 py-8 text-white" style={{ background: s.c }}>
                <div className="num text-[clamp(40px,5vw,64px)] font-bold tracking-[-.04em]">{s.v == null ? '–' : s.int ? Math.round(s.v * k) : f3(s.v * k)}</div>
                <div className="mt-1.5 text-[15px] opacity-90">{s.l}</div>
              </div>
            ))}
          </div>
          <p className="mx-auto mt-6 max-w-[760px] text-center text-[13px] leading-relaxed text-[#A39D91]">
            {warm ? `Mean of ${warm.seeds.length} seeds, ${warm.budget.toLocaleString()}-call budget, ${warm.llm_mode} LLM mode. ` : ''}
            These come from an offline mock, not from Haiku or Sonnet, and the warm start begins with known DRD2 drugs. They test the plumbing, not the method. <Link to="/results" className="text-white">Read the caveats →</Link>
          </p>
        </Reveal>
      </Band>
    </div>
  )
}
