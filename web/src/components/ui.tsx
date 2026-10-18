/** Small shared controls. One button shape, one segmented control, one switch, used everywhere. */

import { useId, type ButtonHTMLAttributes, type ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-ink hover:bg-accent-hover active:translate-y-px shadow-[0_1px_0_0_rgb(0_0_0/0.15)]',
  secondary: 'border border-line-strong text-ink hover:bg-surface-2 hover:border-ink-faint active:translate-y-px',
  ghost: 'text-ink-muted hover:text-ink hover:bg-surface-2',
  // A destructive action, confirmed: the danger colour with the theme's on-colour ink.
  danger: 'bg-danger text-accent-ink hover:brightness-110 active:translate-y-px shadow-[0_1px_0_0_rgb(0_0_0/0.15)]',
}

export function Button({
  variant = 'secondary', size = 'md', icon, children, className = '', ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md'; icon?: ReactNode }) {
  const sizing = size === 'sm' ? 'h-8 px-2.5 text-[13px] gap-1.5' : 'h-10 px-3.5 text-sm gap-2'
  return (
    <button
      type="button"
      className={`inline-flex items-center justify-center rounded-lg font-medium transition-[background-color,border-color,color,transform] duration-150 ease-out-quart disabled:pointer-events-none disabled:opacity-45 ${sizing} ${VARIANTS[variant]} ${className}`}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
}

export function Segmented<T extends string | number>({
  label, value, options, onChange, hideLabel = false, size = 'md',
}: {
  label: string
  value: T
  options: { value: T; label: ReactNode; title?: string }[]
  onChange: (v: T) => void
  hideLabel?: boolean
  size?: 'sm' | 'md'
}) {
  const name = useId()
  const index = Math.max(0, options.findIndex((o) => o.value === value))
  return (
    <fieldset className="min-w-0">
      <legend className={hideLabel ? 'sr-only' : 'mb-1.5 text-[13px] font-medium text-ink-muted'}>{label}</legend>
      <div className="relative flex rounded-lg border border-line bg-bg p-0.5">
        {/* One thumb slides under the options (transform only, so it stays on the compositor). */}
        <span
          aria-hidden
          className="pointer-events-none absolute top-0.5 bottom-0.5 left-0.5 rounded-md bg-surface-2 shadow-[inset_0_0_0_1px_var(--line-strong)] transition-transform duration-[240ms] ease-out-quart"
          style={{ width: `calc((100% - 4px) / ${options.length})`, transform: `translateX(${index * 100}%)` }}
        />
        {options.map((o) => {
          const checked = o.value === value
          return (
            <label
              key={String(o.value)}
              title={o.title}
              className={`relative flex flex-1 cursor-pointer items-center justify-center rounded-md px-2 text-center font-medium transition-colors duration-200 ${
                size === 'sm' ? 'h-7 text-[12.5px]' : 'h-8 text-[13px]'
              } ${checked ? 'text-ink' : 'text-ink-muted hover:text-ink'}`}
            >
              <input
                type="radio"
                name={name}
                className="peer sr-only"
                checked={checked}
                onChange={() => onChange(o.value)}
              />
              <span className="pointer-events-none absolute inset-0 rounded-md peer-focus-visible:outline-2 peer-focus-visible:outline-accent" />
              {o.label}
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

export function Switch({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="group flex cursor-pointer items-center justify-between gap-3 text-[13px] font-medium text-ink-muted">
      {label}
      <span className="relative inline-flex shrink-0">
        <input type="checkbox" role="switch" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-5 w-9 rounded-full border border-line-strong bg-bg transition-[background-color,border-color] duration-200 ease-out-quart group-hover:border-ink-faint peer-checked:border-accent peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent" />
        <span className="pointer-events-none absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-ink-faint shadow-[0_1px_2px_rgb(0_0_0/0.25)] transition-[translate,background-color,scale] duration-[220ms] ease-out-quart group-active:scale-90 peer-checked:translate-x-4 peer-checked:bg-accent-ink" />
      </span>
    </label>
  )
}

/** Animated show/hide for a block: height via grid rows, plus a fade. Hidden content is inert. */
export function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div
      inert={!open}
      className={`grid transition-[grid-template-rows,opacity] duration-[260ms] ease-out-quart ${
        open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
      }`}
    >
      <div className="min-h-0 overflow-hidden">{children}</div>
    </div>
  )
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="font-serif text-[17px] font-semibold text-ink">{children}</h2>
      {aside && <span className="text-[12.5px] text-ink-faint tabular-nums">{aside}</span>}
    </div>
  )
}
