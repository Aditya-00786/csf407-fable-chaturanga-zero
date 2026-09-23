import { AlertTriangle, ChevronLeft, ChevronRight, History, Maximize, Minimize, PanelRight, RotateCcw, Undo2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { PlayPanel, Status } from '../components/PlayPanel'
import { Drawer, DrawerToggle, Floating, ToolButton } from '../components/stage'
import { Button } from '../components/ui'
import type { Color } from '../lib/api'
import { captures, explainOutcome, INITIAL_BOARD } from '../lib/board'
import { useFullscreen } from '../lib/useFullscreen'
import { useGame } from '../state/game'
import { prefersReducedMotion, useDetail } from '../state/settings'
import { BoardScene } from '../three/BoardScene'

// Let the moving piece land before the board turns to the next player.
const TURN_DELAY_MS = 520

function GameEnd() {
  const { game, newGame, review, reviewPly } = useGame()
  const showExplanation = useDetail('explanation')
  const showRule = useDetail('rules')
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
        {showExplanation && <p className="mt-1.5 max-w-[60ch] text-sm leading-relaxed text-ink-muted">{text.detail}</p>}
        {showRule && <p className="mt-2 text-[12.5px] text-ink-faint">Rule: {text.source}</p>}
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="primary" onClick={newGame} icon={<RotateCcw size={16} />}>Play again</Button>
          <Button onClick={() => review(Math.max(0, game.ply - 1))} icon={<History size={16} />}>Review moves</Button>
        </div>
      </div>
    </div>
  )
}

/**
 * The side shown at the bottom, plus a manual "rotate". In a two-player game the board turns to
 * face the side to move shortly after each move (or new game / undo). A manual rotate only
 * overrides that until the turn next changes hands; it never switches auto-turning off.
 */
function useBoardOrientation(): [Color, () => void] {
  const { mode, autoRotate, orientation, game, reviewPly, flip } = useGame()
  const following = mode === 'hvh' && autoRotate
  const target: Color = game?.side ?? 1
  const ply = game?.ply ?? 0
  const [shown, setShown] = useState<Color>(target)
  useEffect(() => {
    // Hold still while reviewing old positions; catch up afterwards. Setting the side that is
    // already shown is a no-op, so a board already facing the player does not move.
    if (!following || reviewPly !== null) return
    const t = setTimeout(() => setShown(target), prefersReducedMotion() ? 0 : TURN_DELAY_MS)
    return () => clearTimeout(t)
  }, [following, reviewPly, target, ply])
  const rotate = () => (following ? setShown((s) => (s === 1 ? -1 : 1)) : flip())
  return [following ? shown : orientation, rotate]
}

export function PlayView() {
  const g = useGame()
  const init = g.init
  useEffect(() => init(), [init]) // init is stable; it changes only when hot reload rebuilds the store

  const [panelOpen, setPanelOpen] = useState(false)
  // Quick actions (Undo, Full screen, New game) stay tucked away until asked for.
  const [quickOpen, setQuickOpen] = useState(false)
  const [fullscreen, toggleFullscreen, canFullscreen] = useFullscreen()
  const panelRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const [orientation, rotate] = useBoardOrientation()

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
          captured={captures(view)}
          checkSquare={view?.in_check ? view.king_square : null}
          orientation={orientation}
          interactive={interactive}
          onMove={g.play}
        />
      </section>

      {/* Floating status card and toolbar keep the board full-screen. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-[var(--z-stage-overlay)] flex items-start justify-between gap-3 p-3 sm:p-4">
        <Floating className="max-w-[min(360px,52vw)] px-4 py-3">
          <Status view={view} />
        </Floating>
        <Floating className="flex shrink-0 items-center gap-0.5 p-1">
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
          <DrawerToggle
            ref={toggleRef}
            open={panelOpen}
            controls="game-panel"
            onClick={() => {
              setPanelOpen((open) => !open)
              setQuickOpen(false)
            }}
            icon={<PanelRight size={17} />}
            label="Game"
            title="Mode, moves and display"
            badge={g.moves.length > 0 && (
              <span className="rounded-md bg-surface-2 px-1.5 text-[11.5px] text-ink-muted tabular-nums" aria-label={`${g.moves.length} moves played`}>
                {g.moves.length}
              </span>
            )}
          />
        </Floating>
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
      <Drawer id="game-panel" ref={panelRef} label="Game panel" side="right" open={panelOpen} className="sm:w-[400px]">
        <PlayPanel view={view} onClose={closePanel} onRotate={rotate} />
      </Drawer>
    </div>
  )
}
