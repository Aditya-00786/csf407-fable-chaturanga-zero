import { AlertTriangle, ChevronLeft, ChevronRight, History, Maximize, Minimize, PanelRight, RotateCcw, Undo2, X } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { PlayPanel, Status } from '../components/PlayPanel'
import { Button } from '../components/ui'
import type { Color } from '../lib/api'
import { explainOutcome, INITIAL_BOARD } from '../lib/board'
import { useGame } from '../state/game'
import { prefersReducedMotion, useSettings } from '../state/settings'
import { BoardScene } from '../three/BoardScene'

// Let the moving piece land before the board turns to the next player.
const TURN_DELAY_MS = 520

function GameEnd() {
  const { game, newGame, review, reviewPly } = useGame()
  const full = useSettings((s) => s.details) === 'full'
  const [hiddenAtPly, setHiddenAtPly] = useState<number | null>(null)
  if (!game?.outcome || reviewPly !== null || hiddenAtPly === game.ply) return null
  const text = explainOutcome(game.outcome, game.side)
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-5 z-[var(--z-stage-overlay)] flex justify-center px-4">
      <div
        role="status"
        className="cz-rise pointer-events-auto w-full max-w-lg rounded-xl border border-line-strong bg-surface/95 p-5 shadow-[0_24px_60px_-20px_rgb(0_0_0/0.55)]"
      >
        <div className="flex items-start justify-between gap-4">
          <h2 className="font-serif text-[22px] leading-snug font-semibold text-ink">{text.title}</h2>
          <button type="button" onClick={() => setHiddenAtPly(game.ply)} aria-label="Hide result"
            className="-m-1 grid h-8 w-8 shrink-0 place-items-center rounded-md text-ink-faint hover:bg-surface-2 hover:text-ink">
            <X size={17} />
          </button>
        </div>
        {full && (
          <>
            <p className="mt-1.5 max-w-[60ch] text-sm leading-relaxed text-ink-muted">{text.detail}</p>
            <p className="mt-2 text-[12.5px] text-ink-faint">Rule: {text.source}</p>
          </>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="primary" onClick={newGame} icon={<RotateCcw size={16} />}>Play again</Button>
          <Button onClick={() => review(Math.max(0, game.ply - 1))} icon={<History size={16} />}>Review moves</Button>
        </div>
      </div>
    </div>
  )
}

/** Toolbar button over the board: always an icon, with a text label on wider screens. */
function ToolButton({ label, icon, onClick, disabled }: {
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

/** The side shown at the bottom. In a two-player game it follows the side to move, after a pause. */
function useBoardOrientation(): Color {
  const { mode, autoRotate, orientation, game, reviewPly } = useGame()
  const following = mode === 'hvh' && autoRotate
  const target: Color = game?.side ?? 1
  const [shown, setShown] = useState<Color>(target)
  useEffect(() => {
    // Hold still while reviewing old positions; catch up afterwards.
    if (!following || reviewPly !== null || shown === target) return
    const t = setTimeout(() => setShown(target), prefersReducedMotion() ? 0 : TURN_DELAY_MS)
    return () => clearTimeout(t)
  }, [following, reviewPly, target, shown])
  return following ? shown : orientation
}

function useFullscreen(): [boolean, () => void, boolean] {
  const [on, setOn] = useState(() => !!document.fullscreenElement)
  useEffect(() => {
    const sync = () => setOn(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])
  const toggle = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void document.documentElement.requestFullscreen?.().catch(() => {})
  }
  return [on, toggle, !!document.fullscreenEnabled]
}

const FLOAT = 'rounded-xl border border-line bg-surface/92 shadow-[0_12px_32px_-18px_rgb(0_0_0/0.6)]'

export function PlayView() {
  const g = useGame()
  useEffect(() => g.init(), []) // eslint-disable-line react-hooks/exhaustive-deps

  const [panelOpen, setPanelOpen] = useState(false)
  // Quick actions (Undo, Full screen, New game) stay tucked away until asked for.
  const [quickOpen, setQuickOpen] = useState(false)
  const [fullscreen, toggleFullscreen, canFullscreen] = useFullscreen()
  const panelRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const orientation = useBoardOrientation()

  const closePanel = () => {
    setPanelOpen(false)
    toggleRef.current?.focus()
  }

  // Esc closes the panel; focus moves into it when it opens.
  useEffect(() => {
    if (!panelOpen) return
    panelRef.current?.querySelector<HTMLElement>('button')?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setPanelOpen(false)
        toggleRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [panelOpen])

  const reviewing = g.reviewPly !== null
  const view = reviewing ? g.reviewGame ?? g.game : g.game
  const last = view?.history[view.history.length - 1]
  const humanTurn =
    !!g.game && !g.game.outcome &&
    (g.mode === 'hvh' || (g.mode === 'hvc' && g.game.side === g.humanColor))
  const interactive = g.status === 'ready' && !reviewing && humanTurn
  const busy = g.status === 'connecting'

  const rotate = () => {
    // Turning the board by hand switches the automatic turning off.
    if (g.mode === 'hvh' && g.autoRotate) {
      g.setAutoRotate(false)
      g.setOrientation((orientation === 1 ? -1 : 1) as Color)
    } else {
      g.flip()
    }
  }

  return (
    <div
      className="relative min-h-[440px] flex-1 overflow-hidden"
      style={{ background: 'radial-gradient(ellipse 70% 60% at 50% 55%, var(--stage-glow), transparent 75%)' }}
    >
      <section aria-label="Board" className="absolute inset-0">
        <BoardScene
          board={view?.board ?? INITIAL_BOARD}
          ply={view?.ply ?? 0}
          legalMoves={interactive ? g.game!.legal_moves : []}
          lastMove={last ? { from: last.from, to: last.to } : null}
          checkSquare={view?.in_check ? view.king_square : null}
          orientation={orientation}
          interactive={interactive}
          onMove={g.play}
        />
      </section>

      {/* Floating status card and toolbar keep the board full-screen. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-[var(--z-stage-overlay)] flex items-start justify-between gap-3 p-3 sm:p-4">
        <div className={`pointer-events-auto max-w-[min(360px,52vw)] px-4 py-3 ${FLOAT}`}>
          <Status view={view} />
        </div>
        <div className={`pointer-events-auto flex shrink-0 items-center gap-0.5 p-1 ${FLOAT}`}>
          {quickOpen && (
            <div id="quick-actions" className="cz-slide-in flex items-center gap-0.5">
              <ToolButton label="Undo" icon={<Undo2 size={17} />} onClick={g.undo}
                disabled={busy || g.moves.length === 0 || (g.status === 'thinking' && g.mode !== 'cvc')} />
              {canFullscreen && (
                <ToolButton label={fullscreen ? 'Exit full screen' : 'Full screen'}
                  icon={fullscreen ? <Minimize size={17} /> : <Maximize size={17} />} onClick={toggleFullscreen} />
              )}
              <ToolButton label="New game" icon={<RotateCcw size={17} />} onClick={g.newGame} disabled={busy} />
              <span className="mx-0.5 h-6 w-px bg-line" aria-hidden />
            </div>
          )}
          <button
            type="button"
            onClick={() => setQuickOpen((open) => !open)}
            aria-expanded={quickOpen}
            aria-controls="quick-actions"
            aria-label={quickOpen ? 'Hide quick actions' : 'Show quick actions: undo, full screen, new game'}
            title={quickOpen ? 'Hide quick actions' : 'Quick actions'}
            className="grid h-10 w-8 place-items-center rounded-lg text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink"
          >
            {quickOpen ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
          <button
            ref={toggleRef}
            type="button"
            onClick={() => {
              setPanelOpen((open) => !open)
              setQuickOpen(false)
            }}
            aria-expanded={panelOpen}
            aria-controls="game-panel"
            title="Mode, moves and display"
            className={`inline-flex h-10 items-center gap-2 rounded-lg px-2.5 text-sm font-medium transition-colors duration-150 sm:px-3 ${
              panelOpen ? 'bg-accent text-accent-ink hover:bg-accent-hover' : 'text-ink hover:bg-surface-2'
            }`}
          >
            <PanelRight size={17} />
            <span className="hidden sm:inline">Game</span>
            {!panelOpen && g.moves.length > 0 && (
              <span className="rounded-md bg-surface-2 px-1.5 text-[11.5px] text-ink-muted tabular-nums" aria-label={`${g.moves.length} moves played`}>
                {g.moves.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {g.status === 'error' && (
        <div role="alert" className="absolute inset-x-0 top-28 z-[var(--z-stage-overlay)] flex justify-center px-4">
          <div className="flex max-w-xl items-center gap-3 rounded-lg border border-danger/40 bg-surface px-4 py-3 text-sm shadow-lg">
            <AlertTriangle size={18} className="shrink-0 text-danger" />
            <span className="text-ink">{g.error}</span>
            <Button size="sm" onClick={() => (g.game ? g.review(null) : g.newGame())}>Retry</Button>
          </div>
        </div>
      )}
      {reviewing && (
        <div className="absolute inset-x-0 bottom-5 z-[var(--z-stage-overlay)] flex justify-center px-4">
          <div className="flex items-center gap-3 rounded-full border border-line-strong bg-surface py-1.5 pr-1.5 pl-4 text-[13px] text-ink shadow-md">
            <History size={15} className="text-accent" />
            Reviewing after ply {g.reviewPly}
            <Button size="sm" variant="primary" onClick={() => g.review(null)}>Back to game</Button>
          </div>
        </div>
      )}
      <GameEnd />

      {/* Game panel: a drawer over the board, closed by default. */}
      <aside
        id="game-panel"
        ref={panelRef}
        aria-label="Game panel"
        inert={!panelOpen}
        className={`absolute inset-y-0 right-0 z-[var(--z-sticky)] w-full border-l border-line bg-surface shadow-[-24px_0_60px_-30px_rgb(0_0_0/0.6)] transition-[translate,visibility] duration-250 ease-out-quart sm:w-[400px] ${
          panelOpen ? 'visible translate-x-0' : 'invisible translate-x-full'
        }`}
      >
        <PlayPanel view={view} onClose={closePanel} onRotate={rotate} />
      </aside>
    </div>
  )
}
