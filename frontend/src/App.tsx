import { useEffect } from 'react'
import { BrowserRouter, Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { api } from './api'
import { useTheme, usePoll } from './lib'
import Home from './pages/Home'
import Setup from './pages/Setup'
import Run from './pages/Run'
import Inspector from './pages/Inspector'
import Results from './pages/Results'

const LAST_RUN = 'aml-last-run'
export const rememberRun = (id: string) => { try { localStorage.setItem(LAST_RUN, id) } catch { /* ignore */ } }

/** Run / Inspector without an id: go to the active run, else the one you last viewed, else the newest. */
export function useDefaultRun() {
  const { data } = usePoll(api.runs, null, [])
  if (!data) return { loading: true as const, id: null }
  let last: string | null = null
  try { last = localStorage.getItem(LAST_RUN) } catch { /* ignore */ }
  const id = data.find(r => r.status === 'running')?.id ?? (data.some(r => r.id === last) ? last : data[0]?.id ?? null)
  return { loading: false as const, id }
}

function Nav() {
  const { dark, toggle } = useTheme()
  const nav = useNavigate()
  const items = [['Home', '/'], ['Run', '/run'], ['Inspector', '/inspector'], ['Results', '/results']] as const
  return (
    <header className="sticky top-0 z-20 border-b border-line backdrop-blur-[14px]" style={{ background: 'var(--navbg)' }}>
      <div className="mx-auto flex h-16 max-w-[1240px] items-center justify-between gap-4 px-4 sm:px-8">
        <button onClick={() => nav('/')} className="flex cursor-pointer items-center gap-2 whitespace-nowrap border-0 bg-transparent p-0 text-[15px] sm:gap-2.5 sm:text-[17px] font-bold tracking-[-.01em] text-ink">
          <span className="inline-block h-3 w-3 rounded-full bg-ink" />Agentic Molecular Lab
        </button>
        <nav className="hidden gap-1 sm:flex">
          {items.map(([label, to]) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `rounded-full px-[15px] py-2 text-sm font-medium no-underline transition hover:-translate-y-px hover:bg-soft ${isActive ? 'bg-soft text-ink' : 'text-mute'}`}>{label}</NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-2 sm:gap-2.5">
          <button onClick={toggle} title="Toggle theme" aria-label="Toggle theme" className="grid h-[38px] w-[38px] cursor-pointer place-items-center rounded-full border border-line bg-transparent text-[15px] text-ink transition hover:rotate-[30deg] hover:bg-soft active:scale-90">{dark ? '☀' : '☾'}</button>
          <button onClick={() => nav('/setup')} className="btn btn-primary h-[38px] px-4 text-sm sm:px-5">Start a run</button>
        </div>
      </div>
      <nav className="flex justify-center gap-1 border-t border-line px-2 py-1.5 sm:hidden">
        {items.map(([label, to]) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `rounded-full px-3 py-1.5 text-[13px] font-medium no-underline ${isActive ? 'bg-soft text-ink' : 'text-mute'}`}>{label}</NavLink>
        ))}
      </nav>
    </header>
  )
}

function ScrollTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function ToRun({ to }: { to: 'run' | 'inspector' }) {
  const { loading, id } = useDefaultRun()
  if (loading) return <div className="mx-auto max-w-[1240px] p-8"><div className="skeleton h-40 rounded-[26px]" /></div>
  return id ? <Navigate to={`/${to}/${id}`} replace /> : <Run id={null} />
}

export default function App() {
  return (
    <BrowserRouter>
      <ScrollTop />
      <div className="min-h-screen bg-canvas text-ink">
        <Nav />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/setup" element={<Setup />} />
          <Route path="/run" element={<ToRun to="run" />} />
          <Route path="/run/:id" element={<Run id={undefined} />} />
          <Route path="/inspector" element={<ToRun to="inspector" />} />
          <Route path="/inspector/:id" element={<Inspector />} />
          <Route path="/results" element={<Results />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <footer className="border-t border-line p-8 text-center text-sm text-mute">Agentic Molecular Lab · DRD2 · PMO benchmark</footer>
      </div>
    </BrowserRouter>
  )
}
