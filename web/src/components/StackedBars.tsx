/**
 * Stacked columns of shares (0–100%), one per x: thin columns on one baseline, a 2px surface gap
 * between segments, a hairline grid, and a readout of every segment when a column is hovered or
 * focused (arrow keys move between columns).
 */

import { useEffect, useRef, useState, type KeyboardEvent } from 'react'

export interface StackKey {
  id: string
  label: string
  color: string
}

const M = { top: 10, right: 14, bottom: 42, left: 44 }

export function StackedBars({ bars, keys, label, xLabel, height = 232 }: {
  bars: { x: number; values: Record<string, number> }[]
  keys: StackKey[]
  label: string
  xLabel: string
  height?: number
}) {
  const wrap = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(480)
  const [hover, setHover] = useState<number | null>(null)
  const empty = bars.length === 0

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [empty])

  if (empty) {
    return (
      <div ref={wrap} className="grid w-full place-items-center rounded-lg border border-dashed border-line text-[13px] text-ink-faint"
        style={{ height }}>
        No data yet
      </div>
    )
  }

  const w = width - M.left - M.right
  const h = height - M.top - M.bottom
  const slot = w / bars.length
  const barW = Math.min(24, slot * 0.7)
  const sy = (v: number) => M.top + h - (v / 100) * h
  const labelEvery = Math.max(1, Math.ceil(bars.length / Math.max(2, Math.floor(w / 36))))
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const d = e.key === 'ArrowLeft' ? -1 : 1
    setHover((i) => Math.min(bars.length - 1, Math.max(0, (i ?? (d > 0 ? -1 : bars.length)) + d)))
  }
  const hx = hover === null ? 0 : M.left + slot * (hover + 0.5)
  const flip = hx > width * 0.6

  return (
    <div ref={wrap} className="relative w-full select-none">
      <svg width={width} height={height} role="img" aria-label={label} tabIndex={0} onKeyDown={onKey}
        onPointerLeave={() => setHover(null)} onBlur={() => setHover(null)}
        className="block overflow-visible rounded-md focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent">
        {[0, 25, 50, 75, 100].map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + w} y1={sy(t)} y2={sy(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={M.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="fill-ink-faint text-[11px] tabular-nums">{t}%</text>
          </g>
        ))}
        {bars.map((b, i) => {
          const cx = M.left + slot * (i + 0.5)
          let base = 0
          return (
            <g key={b.x} onPointerEnter={() => setHover(i)}>
              <rect x={M.left + slot * i} y={M.top} width={slot} height={h} fill="transparent" />
              {keys.map((k) => {
                const v = b.values[k.id] ?? 0
                if (v <= 0) return null
                const y0 = sy(base)
                base += v
                const y1 = sy(base)
                return (
                  <rect key={k.id} x={cx - barW / 2} y={y1} width={barW} height={Math.max(0, y0 - y1 - 2)} fill={k.color}
                    opacity={hover === null || hover === i ? 1 : 0.55} />
                )
              })}
              {(i % labelEvery === 0 || i === bars.length - 1) && (
                <text x={cx} y={M.top + h + 18} textAnchor="middle" className="fill-ink-faint text-[11px] tabular-nums">{b.x}</text>
              )}
            </g>
          )
        })}
        <text x={M.left + w / 2} y={height - 4} textAnchor="middle" className="fill-ink-faint text-[11px]">{xLabel}</text>
      </svg>

      {hover !== null && (
        <div
          className="pointer-events-none absolute top-1 z-[var(--z-popover)] min-w-[9rem] rounded-lg border border-line-strong bg-surface px-3 py-2 text-[12px] shadow-[0_10px_30px_-12px_rgb(0_0_0/0.5)]"
          style={{ left: flip ? undefined : hx + 16, right: flip ? width - hx + 16 : undefined }}>
          <p className="mb-1 text-ink-faint">{xLabel} {bars[hover].x}</p>
          {[...keys].reverse().map((k) => {
            const v = bars[hover].values[k.id] ?? 0
            return v > 0 && (
              <p key={k.id} className="flex items-center gap-2 whitespace-nowrap">
                <span aria-hidden className="h-2.5 w-2.5 rounded-[2px]" style={{ background: k.color }} />
                <span className="font-semibold text-ink tabular-nums">{Math.round(v)}%</span>
                <span className="text-ink-muted">{k.label}</span>
              </p>
            )
          })}
        </div>
      )}
    </div>
  )
}
