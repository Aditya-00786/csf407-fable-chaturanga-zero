/** Small shared controls. One button shape, one segmented control, one switch, used everywhere. */

import { useId, type ButtonHTMLAttributes, type ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-ink hover:bg-accent-hover active:translate-y-px shadow-[0_1px_0_0_rgb(0_0_0/0.15)]',
  secondary: 'border border-line-strong text-ink hover:bg-surface-2 hover:border-ink-faint active:translate-y-px',
  ghost: 'text-ink-muted hover:text-ink hover:bg-surface-2',
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
  return (
    <fieldset className="min-w-0">
      <legend className={hideLabel ? 'sr-only' : 'mb-1.5 text-[13px] font-medium text-ink-muted'}>{label}</legend>
      <div className="flex rounded-lg border border-line bg-bg p-0.5">
        {options.map((o) => {
          const checked = o.value === value
          return (
            <label
              key={String(o.value)}
              title={o.title}
              className={`relative flex flex-1 cursor-pointer items-center justify-center rounded-md px-2 text-center font-medium transition-colors duration-150 ${
                size === 'sm' ? 'h-7 text-[12.5px]' : 'h-8 text-[13px]'
              } ${checked ? 'bg-surface-2 text-ink shadow-[inset_0_0_0_1px_var(--line-strong)]' : 'text-ink-muted hover:text-ink'}`}
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
    <label className="flex cursor-pointer items-center justify-between gap-3 text-[13px] font-medium text-ink-muted">
      {label}
      <span className="relative inline-flex">
        <input type="checkbox" role="switch" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-5 w-9 rounded-full border border-line-strong bg-bg transition-colors duration-150 peer-checked:border-accent peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent" />
        <span className="pointer-events-none absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-ink-faint transition-transform duration-150 ease-out-quart peer-checked:translate-x-4 peer-checked:bg-accent-ink" />
      </span>
    </label>
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
