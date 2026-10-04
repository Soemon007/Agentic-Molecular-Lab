import { useCallback, useEffect, useRef, useState } from 'react'

export const fmt = (n: number) => n.toLocaleString('en-US')
export const f2 = (n: number | null | undefined) => (n == null ? '–' : n.toFixed(2))
export const f3 = (n: number | null | undefined) => (n == null ? '–' : n.toFixed(3))

export interface AgentMeta { key: string; name: string; ini: string; bg: string; fg: string; outlined?: boolean }
const AGENTS: Record<string, AgentMeta> = {
  scout: { key: 'scout', name: 'Scout', ini: 'S', bg: '#14130F', fg: '#F3F0EA' },
  branch_a: { key: 'branch_a', name: 'Branch A · local', ini: 'A', bg: '#2F7D52', fg: '#fff' },
  branch_b: { key: 'branch_b', name: 'Branch B · hopper', ini: 'B', bg: '#6F6A62', fg: '#fff' },
  branch_c: { key: 'branch_c', name: 'Branch C · explorer', ini: 'C', bg: '#D8D3C8', fg: '#14130F' },
  coordinator: { key: 'coordinator', name: 'Coordinator', ini: 'Co', bg: '#4A463F', fg: '#fff' },
  adversary: { key: 'adversary', name: 'Adversary', ini: 'Ad', bg: 'transparent', fg: 'var(--deny)', outlined: true },
  seed: { key: 'seed', name: 'Seed molecule', ini: '·', bg: 'var(--sec)', fg: 'var(--ink)' },
}
export const agentMeta = (k: string): AgentMeta => AGENTS[k] ?? { key: k, name: k, ini: k.slice(0, 1).toUpperCase(), bg: 'var(--soft)', fg: 'var(--ink)' }
export const AGENT_KEYS = ['scout', 'branch_a', 'branch_b', 'branch_c', 'coordinator', 'adversary']

const REASONS: Record<string, string> = { not_selected: 'valid, ranked below the quota', policy_rejected: 'stopped by a policy' }
export const humanReason = (r: string) => REASONS[r] ?? r.replace(/_/g, ' ')

export const TRIGGER_TEXT: Record<string, string> = {
  low_scaffold_diversity: 'fewer than 3 distinct scaffolds in the top 10',
  sa_creep: 'synthetic accessibility is creeping up',
  ad_similarity_drop: 'molecules are drifting out of the oracle’s domain',
}

/** Poll an async loader. Keeps the previous value on error/refresh so the UI never flashes empty. */
export function usePoll<T>(load: () => Promise<T>, ms: number | null, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const loadRef = useRef(load)
  loadRef.current = load
  const refresh = useCallback(async () => {
    try { setData(await loadRef.current()); setError(null) } catch (e) { setError((e as Error).message) }
  }, [])
  useEffect(() => {
    setData(null); setError(null)
    let stop = false
    refresh()
    if (!ms) return
    const t = setInterval(() => { if (!stop && !document.hidden) refresh() }, ms)
    return () => { stop = true; clearInterval(t) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, ...deps])
  return { data, error, refresh }
}

export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    const next = !dark
    document.documentElement.classList.toggle('dark', next)
    try { localStorage.setItem('aml-theme', next ? 'dark' : 'light') } catch { /* private mode */ }
    setDark(next)
  }
  return { dark, toggle }
}

export function downloadCsv(name: string, rows: (string | number)[][]) {
  const esc = (v: string | number) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  const blob = new Blob([rows.map(r => r.map(esc).join(',')).join('\n')], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob); a.download = name; a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
