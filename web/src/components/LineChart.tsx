/**
 * A small SVG chart for the training view: lines (2px) and dots (8px with a surface ring) on one
 * y-axis, recessive hairline grid, and a crosshair that snaps to the nearest x and lists every
 * series there. Arrow keys move the crosshair when the chart has focus.
 */

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

export interface Series {
  id: string
  label: string
  color: string // a CSS colour, e.g. var(--chart-1)
  kind: 'line' | 'dots'
  points: { x: number; y: number }[]
}

export interface ReferenceLine {
  label: string
  points: { x: number; y: number }[]
}

const M = { top: 10, right: 14, bottom: 42, left: 44 }

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min]
  const raw = (max - min) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const ticks = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(+v.toFixed(10))
  return ticks
}

export function LineChart({
  series, reference, height = 232, yMin, yMax, xLabel, yFormat = (v) => String(v), xFormat = (v) => String(v),
  integerX = false, label,
}: {
  series: Series[]
  reference?: ReferenceLine
  height?: number
  yMin?: number
  yMax?: number
  xLabel: string
  yFormat?: (v: number) => string
  xFormat?: (v: number) => string
  integerX?: boolean
  label: string
}) {
  const wrap = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(480)
  const [hover, setHover] = useState<number | null>(null) // index into xs

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const all = series.flatMap((s) => s.points).concat(reference?.points ?? [])
  const xs = useMemo(() => [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [series])
  const x0 = Math.min(...all.map((p) => p.x), integerX ? 1 : 0)
  const x1 = Math.max(...all.map((p) => p.x), x0 + 1)
  const lo = yMin ?? Math.min(0, ...all.map((p) => p.y))
  const hi = yMax ?? Math.max(...all.map((p) => p.y), lo + 1e-6) * 1.08
  const yTicks = niceTicks(lo, hi)
  const top = yMax ?? Math.max(hi, yTicks[yTicks.length - 1])
  const xTicks = niceTicks(x0, x1, Math.max(2, Math.min(6, Math.floor(width / 90)))).filter((t) => !integerX || Number.isInteger(t))

  const w = width - M.left - M.right
  const h = height - M.top - M.bottom
  const sx = (x: number) => M.left + ((x - x0) / (x1 - x0 || 1)) * w
  const sy = (y: number) => M.top + h - ((y - lo) / (top - lo || 1)) * h
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join('')

  const nearest = (px: number) => {
    let best = 0
    for (let i = 1; i < xs.length; i++) if (Math.abs(sx(xs[i]) - px) < Math.abs(sx(xs[best]) - px)) best = i
    return best
  }
  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    if (!xs.length) return
    const r = e.currentTarget.getBoundingClientRect()
    setHover(nearest(e.clientX - r.left))
  }
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (!xs.length) return
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault()
      const d = e.key === 'ArrowLeft' ? -1 : 1
      setHover((i) => Math.min(xs.length - 1, Math.max(0, (i ?? (d > 0 ? -1 : xs.length)) + d)))
    }
  }

  const hx = hover !== null ? xs[hover] : null
  // Each series' value at the hovered x: the point there, or (for a dense line) the closest one.
  const readout = hx === null ? [] : series.map((s) => {
    let p = null as { x: number; y: number } | null
    for (const q of s.points) if (!p || Math.abs(q.x - hx) < Math.abs(p.x - hx)) p = q
    const close = p && (s.kind === 'line' || p.x === hx) && Math.abs(sx(p.x) - sx(hx)) < 12
    return { s, p: close ? p : null }
  })
  const tipLeft = hx === null ? 0 : sx(hx)
  const flip = tipLeft > width * 0.6

  return (
    <div ref={wrap} className="relative w-full select-none">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={label}
        tabIndex={0}
        className="block overflow-visible rounded-md focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + w} y1={sy(t)} y2={sy(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={M.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="fill-ink-faint text-[11px] tabular-nums">{yFormat(t)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={M.top + h + 18} textAnchor="middle" className="fill-ink-faint text-[11px] tabular-nums">{xFormat(t)}</text>
        ))}
        <text x={M.left + w / 2} y={height - 4} textAnchor="middle" className="fill-ink-faint text-[11px]">{xLabel}</text>

        {reference && reference.points.length > 1 && (
          <g>
            <path d={path(reference.points)} fill="none" stroke="var(--ink-faint)" strokeWidth={1} strokeDasharray="4 4" />
            <text x={sx(reference.points[reference.points.length - 1].x)} y={sy(reference.points[reference.points.length - 1].y) - 6}
              textAnchor="end" className="fill-ink-faint text-[11px]">{reference.label}</text>
          </g>
        )}

        {series.map((s) => s.kind === 'line'
          ? <path key={s.id} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          : null)}
        {hx !== null && <line x1={sx(hx)} x2={sx(hx)} y1={M.top} y2={M.top + h} stroke="var(--line-strong)" strokeWidth={1} />}
        {series.map((s) => (s.kind === 'dots' || s.points.length === 1) && (
          <g key={`${s.id}-dots`}>
            {s.points.map((p) => (
              <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
            ))}
          </g>
        ))}
        {readout.map(({ s, p }) => p && s.kind === 'line' && (
          <circle key={`${s.id}-hover`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
        ))}
      </svg>

      {hx !== null && readout.some((r) => r.p) && (
        <div
          className="pointer-events-none absolute top-1 z-[var(--z-popover)] min-w-[9rem] rounded-lg border border-line-strong bg-surface px-3 py-2 text-[12px] shadow-[0_10px_30px_-12px_rgb(0_0_0/0.5)]"
          style={{ left: flip ? undefined : tipLeft + 12, right: flip ? width - tipLeft + 12 : undefined }}
        >
          <p className="mb-1 text-ink-faint">{xLabel} {xFormat(hx)}</p>
          {readout.map(({ s, p }) => p && (
            <p key={s.id} className="flex items-center gap-2 whitespace-nowrap">
              <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: s.color }} />
              <span className="font-semibold text-ink tabular-nums">{yFormat(p.y)}</span>
              <span className="text-ink-muted">{s.label}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

export function Legend({ items }: { items: { label: string; color: string; kind: 'line' | 'dots' | 'ref' }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink-muted">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          {i.kind === 'dots'
            ? <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: i.color }} />
            : <span aria-hidden className={`w-3.5 ${i.kind === 'ref' ? 'border-t border-dashed' : 'h-0.5 rounded-full'}`}
              style={i.kind === 'ref' ? { borderColor: i.color } : { background: i.color }} />}
          {i.label}
        </li>
      ))}
    </ul>
  )
}
