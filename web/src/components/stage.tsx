/**
 * Pieces shared by the full-screen board pages (Play, Learn): floating cards and toolbars over
 * the board, their buttons, and the drawers that slide in from either edge.
 */

import { X } from 'lucide-react'
import type { ReactNode, Ref } from 'react'

/** A floating surface over the board: status cards and toolbars. */
export function Floating({ className = '', children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`pointer-events-auto rounded-xl border border-line bg-surface/92 shadow-[0_12px_32px_-18px_rgb(0_0_0/0.6)] ${className}`}>
      {children}
    </div>
  )
}

/** Toolbar button over the board: always an icon, with a text label on wider screens. */
export function ToolButton({ label, icon, onClick, disabled }: {
  label: string; icon: ReactNode; onClick: () => void; disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className="inline-flex h-10 items-center gap-2 rounded-lg px-2.5 text-sm font-medium text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:pointer-events-none disabled:opacity-45 sm:px-3"
    >
      {icon}
      <span className="hidden md:inline">{label}</span>
    </button>
  )
}

/** A toolbar toggle that opens a drawer; filled with the accent while its drawer is open. */
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
      className={`inline-flex h-10 items-center gap-2 rounded-lg px-2.5 text-sm font-medium transition-colors duration-150 sm:px-3 ${
        open ? 'bg-accent text-accent-ink hover:bg-accent-hover' : 'text-ink hover:bg-surface-2'
      }`}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
      {!open && badge}
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
      // Visible at once on opening (so focus can move in straight away); hidden only once it
      // has slid out on closing.
      className={`absolute inset-y-0 z-[var(--z-sticky)] w-full border-line bg-surface duration-250 ease-out-quart ${edge} ${className} ${
        open ? 'visible translate-x-0 transition-[translate]' : `invisible ${away} transition-[translate,visibility]`
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
