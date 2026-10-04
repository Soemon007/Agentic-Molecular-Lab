import { useEffect, useRef, useState, type ReactNode } from 'react'
import { agentMeta } from '../lib'
import { api } from '../api'

const HEX_A = '20,6 34,14 34,30 20,38 6,30 6,14'
const HEX_B = '48,22 62,30 62,46 48,54 34,46 34,30'

export function Hexes({ stroke, className = '', double = false, style }: { stroke: string; className?: string; double?: boolean; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 70 62" fill="none" stroke={stroke} strokeWidth="1" className={`absolute ${className}`} style={style} aria-hidden>
      <polygon vectorEffect="non-scaling-stroke" points={HEX_A} />
      {double && <polygon vectorEffect="non-scaling-stroke" points={HEX_B} />}
    </svg>
  )
}

/** Decorative dot grid + hexagons, as in the design. `dir` picks which side the dots fade from. */
export function Decor({ onDark, dir = 'left' }: { onDark: boolean; dir?: 'left' | 'right' }) {
  const dot = onDark ? 'rgba(255,255,255,.16)' : 'color-mix(in srgb, var(--ink) 14%, transparent)'
  const line = onDark ? 'rgba(255,255,255,.18)' : 'color-mix(in srgb, var(--ink) 10%, transparent)'
  const mask = `linear-gradient(${dir === 'left' ? 270 : 90}deg,transparent 30%,#000 100%)`
  const left = dir === 'left'
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <div className="absolute inset-0 opacity-70" style={{ backgroundImage: `radial-gradient(${dot} 1.4px,transparent 1.5px)`, backgroundSize: '26px 26px', WebkitMaskImage: mask, maskImage: mask }} />
      <Hexes stroke={line} double className="h-[370px] w-[420px]" style={{ [left ? 'left' : 'right']: -70, top: -50 }} />
      <Hexes stroke={line} className="h-[165px] w-[190px]" style={{ [left ? 'right' : 'left']: '6%', bottom: -40 }} />
    </div>
  )
}

/** Full-bleed coloured section/hero. */
export function Band({ bg, children, className = '', decor = 'left', onDark = true }: { bg: string; children: ReactNode; className?: string; decor?: 'left' | 'right'; onDark?: boolean }) {
  return (
    <section className="relative overflow-hidden text-white" style={{ background: bg }}>
      <Decor onDark={onDark} dir={decor} />
      <div className={`relative mx-auto ${className}`}>{children}</div>
    </section>
  )
}

export function PageHero({ bg, title, accent, sub, width = 'max-w-[1040px]', decor = 'left', children }: { bg: string; title: string; accent: string; sub: ReactNode; width?: string; decor?: 'left' | 'right'; children?: ReactNode }) {
  return (
    <Band bg={bg} decor={decor} className={`${width} px-8 pb-16 pt-20`}>
      <div className="anim-rise text-center">
        <h1 className="m-0 text-[clamp(40px,6vw,68px)] font-bold leading-none tracking-[-.04em]">{title} <em className="font-medium">{accent}</em></h1>
        <p className="mx-auto mt-4 max-w-[640px] text-lg text-white/90">{sub}</p>
        {children}
      </div>
    </Band>
  )
}

export function Reveal({ children, className = '' }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(es => es.some(e => e.isIntersecting) && (setSeen(true), io.disconnect()), { threshold: 0.12 })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return <div ref={ref} className={className} style={{ opacity: seen ? 1 : 0, transform: seen ? 'none' : 'translateY(28px)', transition: 'opacity .9s cubic-bezier(.2,.7,.2,1),transform .9s cubic-bezier(.2,.7,.2,1)' }}>{children}</div>
}

export type Tone = 'ok' | 'ask' | 'deny' | 'soft'
const TONE: Record<Tone, string> = {
  ok: 'text-allow bg-allowbg', ask: 'text-ask bg-askbg', deny: 'text-deny bg-denybg', soft: 'bg-soft text-ink',
}
export function Badge({ tone, children, className = '' }: { tone: Tone; children: ReactNode; className?: string }) {
  return <span className={`whitespace-nowrap rounded-full px-3 py-1 text-xs font-bold tracking-wide ${TONE[tone]} ${className}`}>{children}</span>
}
export function Chip({ children, className = '', strong = false }: { children: ReactNode; className?: string; strong?: boolean }) {
  return <span className={`num whitespace-nowrap rounded-full px-3 py-[5px] text-[13px] ${strong ? 'bg-ink font-semibold text-canvas' : 'bg-soft'} ${className}`}>{children}</span>
}

export function AgentDot({ agent, size = 32, pulse = false }: { agent: string; size?: number; pulse?: boolean }) {
  const a = agentMeta(agent)
  return (
    <span className={`grid flex-none place-items-center rounded-full font-bold ${pulse ? 'anim-pulse' : ''}`}
      style={{ width: size, height: size, background: a.bg, color: a.fg, fontSize: size * 0.36, border: a.outlined ? '2px solid var(--deny)' : 'none' }}>{a.ini}</span>
  )
}

const svgCache = new Map<string, Promise<string | null>>()
function loadSvg(url: string) {
  let p = svgCache.get(url)
  if (!p) {
    p = fetch(url).then(r => (r.ok ? r.text() : null)).then(t => (t ? t.replace(/<\?xml[^>]*\?>/, '').replace(/<script[\s\S]*?<\/script>/gi, '') : null)).catch(() => null)
    svgCache.set(url, p)
  }
  return p
}

/** Real structure depiction rendered by RDKit on the server and inlined, so strokes follow the text colour. */
export function Mol({ smiles, w = 240, h = 160, className = '' }: { smiles: string; w?: number; h?: number; className?: string }) {
  const [svg, setSvg] = useState<string | null | undefined>(undefined)
  useEffect(() => {
    let live = true
    setSvg(undefined)
    loadSvg(api.molSvg(smiles, w, h)).then(t => live && setSvg(t))
    return () => { live = false }
  }, [smiles, w, h])
  if (svg === null) return <span className="mono text-xs opacity-70">no depiction</span>
  if (svg === undefined) return <span className="skeleton block rounded-xl" style={{ width: w * 0.6, height: h * 0.6 }} />
  return <span role="img" aria-label={smiles} className={`block [&>svg]:h-auto [&>svg]:max-h-full [&>svg]:w-full ${className}`} dangerouslySetInnerHTML={{ __html: svg }} />
}

export function Empty({ title, body, children }: { title: string; body: ReactNode; children?: ReactNode }) {
  return (
    <div className="rounded-[26px] border border-dashed border-line px-5 py-12 text-center">
      <div className="text-xl font-bold tracking-tight">{title}</div>
      <div className="mx-0 mb-4 mt-1.5 text-sm text-mute">{body}</div>
      {children}
    </div>
  )
}
