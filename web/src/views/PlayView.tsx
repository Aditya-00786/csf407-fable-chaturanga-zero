import { AlertTriangle, Check, Flag, History, Maximize, Minimize, Pause, PanelRight, Play, RotateCcw, Undo2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { PlayPanel, RecordDot, Status } from '../components/PlayPanel'
import { ReviewBar } from '../components/ReviewBar'
import { Covered, Drawer, DrawerToggle, Floating, QuickActions, QuickToggle, ToolButton } from '../components/stage'
import { Button } from '../components/ui'
import type { Color } from '../lib/api'
import { captures, colorName, explainOutcome, INITIAL_BOARD } from '../lib/board'
import { href } from '../lib/route'
import { useFullscreen } from '../lib/useFullscreen'
import { humanMoved, runLabel, useGame } from '../state/game'
import { prefersReducedMotion, useDetail, useSettings } from '../state/settings'
import { BoardScene } from '../three/BoardScene'

// Let the moving piece land before the board turns to the next player.
const TURN_DELAY_MS = 520

/** Whether the game just over was saved for the results (a game left for a new one says so in the Game panel). */
function Recorded() {
  const { recording, saveRecording } = useGame()
  if (recording.state === 'saving') return <p className="mt-2 text-[13px] text-ink-muted">Saving it to the results…</p>
  if (recording.state === 'failed') {
    return (
      <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-danger">
        Not saved: {recording.message}
        <Button size="sm" onClick={saveRecording}>Try again</Button>
      </p>
    )
  }
  if (recording.state !== 'saved' || recording.ended === 'left') return null
  return (
    <p className="mt-2 flex flex-wrap items-center gap-x-1.5 text-[13px] text-ink-muted">
      <Check size={15} className="shrink-0 text-success" aria-hidden />
      Saved: {recording.games} {recording.games === 1 ? 'game' : 'games'} recorded on this computer.
      <a href={href('results')} className="font-medium text-accent underline-offset-4 hover:underline">See the results</a>
    </p>
  )
}

function GameEnd() {
  const { game, newGame, review, reviewPly, resigned, humanColor } = useGame()
  const showExplanation = useDetail('explanation')
  const showRule = useDetail('rules')
  const [hiddenAtPly, setHiddenAtPly] = useState<number | null>(null)
  if (!game || !(game.outcome || resigned) || reviewPly !== null || hiddenAtPly === game.ply) return null
  const text = game.outcome ? explainOutcome(game.outcome, game.side) : {
    title: `${colorName(-humanColor as Color)} wins: you resigned`,
    detail: 'You gave up the game.',
    source: null,
  }
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
        {showRule && text.source && <p className="mt-2 text-[12.5px] text-ink-faint">Rule: {text.source}</p>}
        <Recorded />
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
  const recordGames = useSettings((s) => s.recordGames)
  const init = g.init
  useEffect(() => init(), [init]) // init is stable; it changes only when hot reload rebuilds the store

  const [panelOpen, setPanelOpen] = useState(false)
  // Quick actions (Undo, or Start / Pause against computers; Full screen; New game) stay tucked away until asked for.
  const [quickOpen, setQuickOpen] = useState(false)
  const [fullscreen, toggleFullscreen, canFullscreen] = useFullscreen()
  const panelRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const [orientation, rotate] = useBoardOrientation()

  // The toggle is inert while the drawer covers it, so let the drawer close before focusing it.
  const closePanel = () => {
    flushSync(() => setPanelOpen(false))
    toggleRef.current?.focus({ preventScroll: true })
  }

  // Esc closes the panel; focus moves into it when it opens.
  useEffect(() => {
    if (!panelOpen) return
    // preventScroll: the drawer is still off-screen as it starts to slide in, and focusing it would
    // otherwise scroll the whole stage sideways to reveal it.
    panelRef.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true })
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        flushSync(() => setPanelOpen(false))
        toggleRef.current?.focus({ preventScroll: true })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [panelOpen])

  const reviewing = g.reviewPly !== null
  const view = reviewing ? g.reviewGame ?? g.game : g.game
  const last = view?.history[view.history.length - 1]
  const humanTurn =
    !!g.game && !g.game.outcome && !g.resigned &&
    (g.mode === 'hvh' || (g.mode === 'hvc' && g.game.side === g.humanColor))
  const interactive = g.status === 'ready' && !reviewing && humanTurn
  const busy = g.status === 'connecting'

  return (
    <div
      className="relative min-h-[440px] flex-1 overflow-clip"
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
        <Covered by={panelOpen} onHidden={() => setQuickOpen(false)} className="shrink-0">
          <Floating className="flex items-center gap-0.5 p-1">
            <QuickActions id="quick-actions" open={quickOpen}>
              {g.mode === 'cvc' ? (
                // Computer vs computer has no move of the user's to take back; it needs Start / Pause instead.
                <ToolButton label={runLabel(g.running, g.moves.length)} icon={g.running ? <Pause size={17} /> : <Play size={17} />}
                  onClick={() => g.setRunning(!g.running)} disabled={busy || !!g.game?.outcome || reviewing} />
              ) : (
                <ToolButton label="Undo" icon={<Undo2 size={17} />} onClick={g.undo}
                  disabled={busy || g.moves.length === 0 || g.status === 'thinking'} />
              )}
              {g.mode === 'hvc' && (
                <>
                  <ToolButton label="Record" pressed={recordGames} onClick={() => g.setRecording(!recordGames)}
                    disabled={!!g.watching}
                    title={recordGames ? 'Recording games for the results: on' : 'Record games for the results'}
                    icon={<span className="grid h-[17px] w-[17px] place-items-center"><RecordDot on={recordGames} /></span>} />
                  <ToolButton label="Resign" icon={<Flag size={17} />} onClick={g.resign}
                    disabled={busy || reviewing || !!g.watching || !g.game || !!g.game.outcome || g.resigned || !humanMoved(g.moves, g.humanColor)} />
                </>
              )}
              {canFullscreen && (
                <ToolButton label={fullscreen ? 'Exit full screen' : 'Full screen'}
                  icon={fullscreen ? <Minimize size={17} /> : <Maximize size={17} />} onClick={toggleFullscreen} />
              )}
              <ToolButton label="New game" icon={<RotateCcw size={17} />} onClick={g.newGame} disabled={busy} />
            </QuickActions>
            <QuickToggle open={quickOpen} controls="quick-actions" onClick={() => setQuickOpen((open) => !open)}
              describe={`${g.mode === 'cvc' ? 'start or pause' : g.mode === 'hvc' ? 'undo, record, resign' : 'undo'}, full screen, new game`} />
            <DrawerToggle
              ref={toggleRef}
              open={panelOpen}
              controls="game-panel"
              onClick={() => setPanelOpen((open) => !open)}
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
        </Covered>
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
      {reviewing && <ReviewBar />}
      <GameEnd />

      {/* Game panel: a drawer over the board, closed by default. */}
      <Drawer id="game-panel" ref={panelRef} label="Game panel" side="right" open={panelOpen} className="sm:w-[400px]">
        <PlayPanel view={view} onClose={closePanel} onRotate={rotate} />
      </Drawer>
    </div>
  )
}
