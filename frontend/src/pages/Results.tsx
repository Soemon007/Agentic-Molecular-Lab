import { useMemo, useState } from 'react'
import { api, type ResultsFile } from '../api'
import { downloadCsv, f3, usePoll } from '../lib'
import { Empty, PageHero, Reveal } from '../components/ui'

interface Row { name: string; adv: string; mean?: number; sd?: number; values: number[]; bar: string; note?: string; triggers?: string }
const CPS = ['50', '100', '250', '500', '1000']
const sd = (v: number) => Math.sqrt(v)
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const stdev = (xs: number[]) => { const m = mean(xs); return xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0 }

function rowsFor(r: ResultsFile, cp: string): Row[] {
  const rand = Object.values(r.random ?? {}).map(x => x[cp]).filter(v => v != null)
  return [
    { name: 'Random (ZINC)', adv: 'n/a', mean: rand.length ? mean(rand) : undefined, sd: rand.length ? stdev(rand) : undefined, values: rand, bar: '#C4BEB1' },
    { name: 'Single-call LLM', adv: 'n/a', values: [], bar: 'var(--sec)', note: 'Not run (no API key)' },
    { name: 'Multi-agent', adv: 'Off', mean: r.ablated[cp]?.mean, sd: r.ablated[cp] && sd(r.ablated[cp].var), values: r.ablated[cp]?.values ?? [], bar: '#A39D91' },
    { name: 'Multi-agent', adv: 'On', mean: r.ours[cp]?.mean, sd: r.ours[cp] && sd(r.ours[cp].var), values: r.ours[cp]?.values ?? [], bar: '#2F7D52', triggers: r.trigger_rounds.join(' · ') },
  ]
}

export default function Results() {
  const { data, error } = usePoll(api.results, null, [])
  const [start, setStart] = useState<'known' | 'cold'>('known')
  const [cp, setCp] = useState('1000')
  const avail = (['known', 'cold'] as const).filter(k => data?.[k])
  const mode = data?.[start] ? start : avail[0]
  const r = mode ? data?.[mode] : undefined
  const rows = useMemo(() => (r ? rowsFor(r, cp) : []), [r, cp])

  const exportCsv = () => {
    if (!data) return
    const out: (string | number)[][] = [['start', 'method', 'adversary', 'oracle_calls', 'auc10_mean', 'auc10_sd', 'per_seed']]
    for (const k of avail) for (const c of CPS) for (const row of rowsFor(data[k]!, c)) if (row.mean != null) out.push([k, row.name, row.adv, c, row.mean.toFixed(4), (row.sd ?? 0).toFixed(4), row.values.map(v => v.toFixed(4)).join(' ')])
    downloadCsv('results.csv', out)
  }

  const identical = r ? CPS.every(c => r.ours[c] && r.ablated[c] && Math.abs(r.ours[c].mean - r.ablated[c].mean) < 1e-9) : false
  const fired = r ? r.trigger_rounds.filter(t => t > 0).length : 0

  return (
    <div>
      <PageHero bg="#2F7D52" decor="left" title="Does it" accent="actually help?" sub={r ? `${r.seeds.length} seeds, DRD2, up to ${r.budget.toLocaleString()} oracle calls each. Scores below are oracle-verified, not surrogate-claimed.` : 'Multi-seed evaluation results.'}>
        <button onClick={exportCsv} disabled={!data || !avail.length} className="btn btn-ink mt-7 h-12 px-7 text-[15px]">Export results.csv</button>
      </PageHero>
      <div className="mx-auto max-w-[1040px] px-4 pb-30 pt-14 sm:px-8">
        {error ? <Empty title="Couldn’t load results." body={error} />
          : !data ? <div className="skeleton h-72 rounded-[32px]" />
          : !r ? <Empty title="No results yet." body={<>Run <code className="mono">.venv/bin/python backend/eval/run_seeds.py</code> to write <code className="mono">backend/data/results_*.json</code>.</>} />
          : <>
            <div className="mb-5 rounded-[28px] border border-ask bg-askbg px-6 py-4 text-[15px] leading-normal">
              <b>Read this first.</b> {r.llm_mode === 'offline' ? 'These runs used the offline mock (RDKit edits plus a rate of rule-violating proposals), not Haiku or Sonnet. They validate plumbing and evaluation, not LLM behaviour.' : `LLM mode: ${r.llm_mode}.`}
              {mode === 'known' && ' The warm start begins with known DRD2 drugs the oracle already scores near 1.0, so every number here is flattered. Switch to Cold for the fair view.'}
            </div>

            <div className="mb-5 flex flex-wrap items-center gap-x-8 gap-y-3">
              <Toggle label="Seed molecules" value={mode!} onChange={v => setStart(v as 'known')} options={[['known', 'Known ligands', !data.known], ['cold', 'Cold (ZINC)', !data.cold]]} />
              <Toggle label="Budget checkpoint" value={cp} onChange={setCp} options={CPS.map(c => [c, `@${Number(c).toLocaleString()}`, !r.ours[c]] as [string, string, boolean])} />
            </div>

            <Reveal>
              <div className="card">
                <div className="mb-5 text-lg font-bold">Oracle AUC top-10 <span className="text-sm font-normal text-mute">· bar = mean, dots = seeds · @{Number(cp).toLocaleString()} calls</span></div>
                <div className="flex flex-col gap-5">
                  {rows.map((x, i) => (
                    <div key={i} className="grid items-center gap-4 [grid-template-columns:minmax(110px,210px)_1fr_64px]">
                      <div className="text-[15px] font-medium leading-tight">{x.name}{x.adv !== 'n/a' && <span className="text-mute"> · adversary {x.adv.toLowerCase()}</span>}</div>
                      <div className="relative h-[30px] rounded-full bg-soft">
                        {x.mean != null ? <>
                          <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-1000" style={{ background: x.bar, width: `${Math.max(1, x.mean * 100)}%` }} />
                          {x.values.map((v, j) => <span key={j} className="absolute top-[11px] h-2 w-2 rounded-full border-[1.5px] border-ink bg-canvas" style={{ left: `calc(${v * 100}% - 4px)` }} title={`seed value ${f3(v)}`} />)}
                        </> : <span className="absolute inset-0 grid place-items-center text-[13px] text-mute">{x.note ?? 'No data'}</span>}
                      </div>
                      <div className="num text-right font-bold">{x.mean != null ? f3(x.mean) : '–'}</div>
                    </div>
                  ))}
                  <div className="grid items-center gap-4 [grid-template-columns:minmax(110px,210px)_1fr_64px]">
                    <div className="text-[15px] font-medium leading-tight">Graph GA <span className="text-mute">(published)</span></div>
                    <div className="rounded-full bg-soft px-4 py-[5px] text-[13px] text-mute">0.964 ± 0.012 at 10,000 calls, from a cold start. PMO publishes no curve at this budget, so there is nothing to compare here.</div>
                    <div className="num text-right font-bold">–</div>
                  </div>
                </div>
              </div>
            </Reveal>

            <Reveal className="mt-5">
              <div className="overflow-x-auto rounded-[32px] border border-line bg-surface px-7 py-3">
                <table className="w-full min-w-[620px] border-collapse text-[15px]">
                  <thead><tr className="text-left text-[13px] text-mute">
                    <th className="px-2 py-4 font-medium">Method</th><th className="px-2 py-4 font-medium">Adversary</th><th className="px-2 py-4 text-right font-medium">Seeds</th>
                    <th className="px-2 py-4 text-right font-medium">Oracle AUC-10</th><th className="px-2 py-4 text-right font-medium">Trigger fired (rounds per seed)</th></tr></thead>
                  <tbody>{rows.map((x, i) => (
                    <tr key={i} className="border-t border-line transition-colors hover:bg-soft">
                      <td className="px-2 py-4 font-semibold">{x.name}</td><td className="px-2 py-4 text-mute">{x.adv}</td>
                      <td className="num px-2 py-4 text-right">{x.values.length || '–'}</td>
                      <td className="num px-2 py-4 text-right font-semibold">{x.mean != null ? <>{f3(x.mean)} <span className="font-normal text-mute">± {f3(x.sd)}</span></> : <span className="font-normal text-mute">{x.note ?? '–'}</span>}</td>
                      <td className="num px-2 py-4 text-right text-mute">{x.triggers ?? '–'}</td></tr>))}</tbody>
                </table>
              </div>
            </Reveal>

            <div className="mx-auto mt-8 max-w-[760px] text-[15px] leading-relaxed text-mute">
              <p className="m-0 mb-3"><b className="text-ink">The ablation is a null result{identical ? ', by construction' : ''}.</b>{' '}
                {identical
                  ? `With and without the adversary the runs are identical at every checkpoint, because the deterministic trigger fired in ${fired} of ${r.seeds.length} seeds. The mock never reward-hacks, so the adversary was never called.`
                  : `The trigger fired in ${fired} of ${r.seeds.length} seeds.`}{' '}
                Nothing here shows that catching shortcuts helps; that needs a run where score climbs while AD similarity falls.</p>
              {mode === 'cold' && <p className="m-0 mb-3"><b className="text-ink">Cold start shows the mock is not an optimiser.</b> Its AUC sits near random’s, so this table says nothing for or against the LLM branches.</p>}
              <p className="m-0"><b className="text-ink">One oracle.</b> Everything is DRD2, a single SVM classifier. A classifier score is not binding affinity.</p>
            </div>
          </>}
      </div>
    </div>
  )
}

function Toggle({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: [string, string, boolean?][] }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-[13px] text-mute">{label}</span>
      <div className="flex flex-wrap rounded-full bg-soft p-1">
        {options.map(([v, l, off]) => (
          <button key={v} disabled={off} onClick={() => onChange(v)} aria-pressed={value === v} className={`cursor-pointer rounded-full border-0 px-4 py-[7px] text-sm font-semibold transition ${value === v ? 'bg-ink text-canvas' : 'bg-transparent text-ink'}`}>{l}</button>
        ))}
      </div>
    </div>
  )
}
