export interface Series { points: [number, number][]; color: string; dashed?: boolean; axis?: 'l' | 'r'; area?: boolean; dot?: boolean; label?: string }
export interface Marker { x: number; label?: string }
interface Props {
  series: Series[]; x: [number, number]; yl: [number, number]; yr?: [number, number]
  xTicks: number[]; yTicks: number[]; yrTicks?: number[]; xFmt?: (v: number) => string; yFmt?: (v: number) => string; yrFmt?: (v: number) => string
  markers?: Marker[]; ariaLabel: string
}
const L = 40, R = 480, T = 18, B = 200

export function LineChart({ series, x, yl, yr, xTicks, yTicks, yrTicks, xFmt = String, yFmt = String, yrFmt = String, markers = [], ariaLabel }: Props) {
  const X = (v: number) => L + ((v - x[0]) / (x[1] - x[0] || 1)) * (R - L)
  const Y = (v: number, a: 'l' | 'r' = 'l') => { const d = a === 'r' && yr ? yr : yl; return B - ((v - d[0]) / (d[1] - d[0] || 1)) * (B - T) }
  const path = (s: Series) => s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)} ${Y(p[1], s.axis).toFixed(1)}`).join(' ')
  return (
    <svg viewBox="0 0 540 232" role="img" aria-label={ariaLabel} className="block w-full overflow-visible">
      <g stroke="var(--line)" strokeWidth="1">{yTicks.map(t => <line key={t} x1={L} x2={R} y1={Y(t)} y2={Y(t)} />)}</g>
      <g fill="var(--mute)" fontSize="11">
        {yTicks.map(t => <text key={t} x={0} y={Y(t) + 4}>{yFmt(t)}</text>)}
        {yr && yrTicks?.map(t => <text key={t} x={540} y={Y(t, 'r') + 4} textAnchor="end">{yrFmt(t)}</text>)}
        {xTicks.map(t => <text key={t} x={X(t)} y={222} textAnchor="middle">{xFmt(t)}</text>)}
      </g>
      {markers.map((m, i) => (
        <g key={i}>
          <line x1={X(m.x)} x2={X(m.x)} y1={T} y2={B} stroke="var(--deny)" strokeDasharray="3 4" opacity=".55" />
          {m.label && i === 0 && <text x={X(m.x)} y={T - 5} fontSize="11" fill="var(--deny)" textAnchor="middle">{m.label}</text>}
        </g>
      ))}
      {series.filter(s => s.points.length > 0).map((s, i) => {
        const last = s.points[s.points.length - 1]
        return (
          <g key={i}>
            {s.area && <path d={`${path(s)} L${X(last[0]).toFixed(1)} ${B} L${X(s.points[0][0]).toFixed(1)} ${B} Z`} fill="var(--sec)" />}
            <path d={path(s)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeDasharray={s.dashed ? '5 5' : undefined} />
            {s.dot && <>
              <circle cx={X(last[0])} cy={Y(last[1], s.axis)} r="5" fill={s.color} />
              <circle cx={X(last[0])} cy={Y(last[1], s.axis)} r="10" fill="none" stroke={s.color} opacity=".25" className="anim-blink" />
            </>}
          </g>
        )
      })}
    </svg>
  )
}

/** Round a [min,max] data range outwards onto a tidy axis with `n` ticks. */
export function niceAxis(lo: number, hi: number, n = 4): { dom: [number, number]; ticks: number[] } {
  if (!isFinite(lo) || !isFinite(hi)) return { dom: [0, 1], ticks: [0, 0.5, 1] }
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5 }
  const raw = (hi - lo) / n
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) ?? raw
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step
  const ticks: number[] = []
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(6))
  return { dom: [a, b], ticks }
}
