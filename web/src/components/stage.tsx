/**
 * Pieces shared by the full-screen board pages (Play, Learn): floating cards and toolbars over
 * the board, their buttons, and the drawers that slide in from either edge.
 */

import { ChevronLeft, X } from 'lucide-react'
import type { ReactNode, Ref } from 'react'

/** A floating surface over the board: status cards and toolbars. */
export function Floating({ className = '', children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`pointer-events-auto rounded-xl border border-line bg-surface/92 shadow-[0_12px_32px_-18px_rgb(0_0_0/0.6)] ${className}`}>
      {children}
    </div>
  )
}

/**
 * Floating controls that sit where a drawer opens. They fade out and become inert while it is
 * open, so nothing peeks out past the drawer's edge.
 */
export function Covered({ by, onHidden, className = '', children }: {
  by: boolean
  /** Called once they have faded out: the moment to tidy them (e.g. fold quick actions) unseen. */
  onHidden?: () => void
  className?: string
  children: ReactNode
}) {
  return (
    <div
      inert={by}
      onTransitionEnd={(e) => {
        if (by && e.target === e.currentTarget && e.propertyName === 'opacity') onHidden?.()
      }}
      className={`transition-opacity duration-200 ease-out-quart ${by ? 'pointer-events-none opacity-0' : 'opacity-100'} ${className}`}
    >
      {children}
    </div>
  )
}

/**
 * Quick actions that slide out of a toolbar beside their chevron. They stay mounted and grow or
 * shrink in width (grid column 0fr to 1fr) with a fade, both ways, so the toolbar never jumps.
 */
export function QuickActions({ id, open, children }: { id: string; open: boolean; children: ReactNode }) {
  return (
    <div
      id={id}
      inert={!open}
      className={`grid transition-[grid-template-columns,opacity] duration-[280ms] ease-out-quart ${
        open ? 'grid-cols-[1fr] opacity-100' : 'grid-cols-[0fr] opacity-0'
      }`}
    >
      <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
        {children}
        <span className="mx-0.5 h-6 w-px shrink-0 bg-line" aria-hidden />
      </div>
    </div>
  )
}

/** The chevron that shows or hides the quick actions; it turns to point the way they will go. */
export function QuickToggle({ open, controls, onClick, describe }: {
  open: boolean; controls: string; onClick: () => void; describe: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      aria-controls={controls}
      aria-label={open ? 'Hide quick actions' : `Show quick actions: ${describe}`}
      title={open ? 'Hide quick actions' : 'Quick actions'}
      className="grid h-10 w-8 shrink-0 place-items-center rounded-lg text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink"
    >
      <ChevronLeft size={18} className={`transition-transform duration-[280ms] ease-out-quart ${open ? 'rotate-180' : ''}`} />
    </button>
  )
}

/** Toolbar button over the board: always an icon, with a text label on wider screens. A toggle passes `pressed`. */
export function ToolButton({ label, icon, onClick, disabled, pressed, title }: {
  label: string; icon: ReactNode; onClick: () => void; disabled?: boolean; pressed?: boolean; title?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
      aria-label={label}
      aria-pressed={pressed}
      className={`inline-flex h-10 shrink-0 items-center gap-2 rounded-lg px-2.5 text-sm font-medium whitespace-nowrap transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:pointer-events-none disabled:opacity-45 sm:px-3 ${
        pressed ? 'text-ink' : 'text-ink-muted'
      }`}
    >
      {icon}
      <span className="hidden md:inline">{label}</span>
    </button>
  )
}

/** A toolbar toggle that opens a drawer. Its drawer covers it (see Covered), so it needs no open state of its own. */
export function DrawerToggle({ ref, id, open, controls, onClick, icon, label, title, badge }: {
  ref?: Ref<HTMLButtonElement>
  id?: string
  open: boolean
  controls: string
  onClick: () => void
  icon: ReactNode
  label: string
  title: string
  badge?: ReactNode
}) {
  return (
    <button
      ref={ref}
      id={id}
      type="button"
      onClick={onClick}
      aria-expanded={open}
      aria-controls={controls}
      title={title}
      className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg px-2.5 text-sm font-medium text-ink transition-colors duration-150 hover:bg-surface-2 sm:px-3"
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
      {badge}
    </button>
  )
}

/** A panel that slides over the board from the left or right edge. Closed panels are inert. */
export function Drawer({ ref, id, label, side, open, className = '', children }: {
  ref?: Ref<HTMLElement>
  id: string
  label: string
  side: 'left' | 'right'
  open: boolean
  className?: string
  children: ReactNode
}) {
  const edge = side === 'right'
    ? 'right-0 border-l shadow-[-24px_0_60px_-30px_rgb(0_0_0/0.6)]'
    : 'left-0 border-r shadow-[24px_0_60px_-30px_rgb(0_0_0/0.6)]'
  const away = side === 'right' ? 'translate-x-full' : '-translate-x-full'
  return (
    <aside
      id={id}
      ref={ref}
      aria-label={label}
      inert={!open}
      // Visible at once on opening (so focus can move in straight away); hidden only once it has
      // slid out on closing. It glides in on a long ease-out and leaves a little quicker; its own
      // compositor layer keeps the slide off the main thread and away from the WebGL canvas.
      className={`absolute inset-y-0 z-[var(--z-sticky)] w-full border-line bg-surface will-change-[translate] ${edge} ${className} ${
        open
          ? 'visible translate-x-0 transition-[translate] duration-[340ms] ease-out-quint'
          : `invisible ${away} transition-[translate,visibility] duration-[260ms] ease-in-out-cubic`
      }`}
    >
      <div className="flex h-full flex-col">{children}</div>
    </aside>
  )
}

export function DrawerHeader({ title, onClose, closeLabel }: { title: ReactNode; onClose: () => void; closeLabel: string }) {
  return (
    <div className="flex h-14 shrink-0 items-center justify-between border-b border-line px-5 sm:px-6">
      <h2 className="font-serif text-[19px] font-semibold text-ink">{title}</h2>
      <button type="button" onClick={onClose} aria-label={closeLabel} title="Close (Esc)"
        className="-mr-2 grid h-9 w-9 place-items-center rounded-lg text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink">
        <X size={18} />
      </button>
    </div>
  )
}
