/**
 * The bar at the foot of the board while reviewing: step through the moves, jump to either
 * end, or replay the game automatically at a chosen speed. ← and → step, Home and End jump.
 */

import { History, Pause, Play, SkipBack, SkipForward, StepBack, StepForward } from 'lucide-react'
import { useEffect, type ReactNode } from 'react'
import { describeMove } from '../lib/board'
import { useGame, type Pace } from '../state/game'
import { Button, Segmented } from './ui'

const KEYS: Record<string, 'start' | 'back' | 'forward' | 'end'> = {
  ArrowLeft: 'back', ArrowRight: 'forward', Home: 'start', End: 'end',
}

const SPEEDS: { value: Pace; label: string }[] = [
  { value: 'slow', label: 'Slow' },
  { value: 'normal', label: 'Normal' },
  { value: 'fast', label: 'Fast' },
]

function StepButton({ label, onClick, disabled, children }: {
  label: string; onClick: () => void; disabled?: boolean; children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="grid h-9 w-9 place-items-center rounded-lg text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:pointer-events-none disabled:opacity-40"
    >
      {children}
    </button>
  )
}

export function ReviewBar() {
  const { game, moves, reviewPly, replaying, replaySpeed, step, startReplay, stopReplay, setReplaySpeed, review } = useGame()
  const total = moves.length
  const at = reviewPly ?? total
  // From the full game record, so the text always matches the move number on show.
  const last = at > 0 ? game?.history[at - 1] : undefined

  // Arrow keys step through the moves, except while the board itself (which uses the arrows to
  // move its cursor) or a form control has the focus.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const to = KEYS[e.key]
      if (!to || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
      const target = e.target instanceof Element ? e.target : null
      if (target?.closest('[role="application"], input, textarea, select, [contenteditable="true"]')) return
      e.preventDefault()
      step(to)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step])

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-[var(--z-stage-overlay)] flex justify-center px-3 sm:bottom-5">
      <div
        role="toolbar"
        aria-label="Review moves"
        className="cz-rise pointer-events-auto flex max-w-full flex-wrap items-center justify-center gap-x-2 gap-y-1.5 rounded-2xl border border-line-strong bg-surface/95 p-1.5 shadow-[0_18px_44px_-20px_rgb(0_0_0/0.6)]"
      >
        {/* Fixed width: the move text varies in length, and a centred bar that resized with it
            would shift every button sideways on each step. */}
        <p className="flex w-[9.5rem] min-w-0 items-center gap-2 pl-2.5 text-[13px] text-ink sm:w-[18.5rem]" aria-live="polite">
          <History size={15} className="shrink-0 text-accent" aria-hidden />
          <span className="shrink-0 tabular-nums whitespace-nowrap">{at === 0 ? 'Start' : `Move ${at} of ${total}`}</span>
          {last && <span className="hidden min-w-0 truncate text-ink-muted sm:inline">· {describeMove(last)}</span>}
        </p>

        <div className="flex items-center">
          <StepButton label="First position (Home)" onClick={() => step('start')} disabled={at === 0}>
            <SkipBack size={17} />
          </StepButton>
          <StepButton label="Previous move (←)" onClick={() => step('back')} disabled={at === 0}>
            <StepBack size={17} />
          </StepButton>
          <button
            type="button"
            onClick={replaying ? stopReplay : startReplay}
            aria-label={replaying ? 'Pause the replay' : at >= total ? 'Replay the game from the start' : 'Replay from here'}
            title={replaying ? 'Pause' : 'Replay'}
            className="mx-0.5 grid h-9 w-9 place-items-center rounded-full bg-accent text-accent-ink transition-colors duration-150 hover:bg-accent-hover"
          >
            {replaying ? <Pause size={16} /> : <Play size={16} className="translate-x-px" />}
          </button>
          <StepButton label="Next move (→)" onClick={() => step('forward')} disabled={at >= total}>
            <StepForward size={17} />
          </StepButton>
          <StepButton label="Last position (End)" onClick={() => step('end')} disabled={at >= total}>
            <SkipForward size={17} />
          </StepButton>
        </div>

        <div className="w-[12.5rem]">
          <Segmented label="Replay speed" hideLabel size="sm" value={replaySpeed} onChange={setReplaySpeed} options={SPEEDS} />
        </div>

        <Button size="sm" onClick={() => review(null)}>Back to game</Button>
      </div>
    </div>
  )
}
