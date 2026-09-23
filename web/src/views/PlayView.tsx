import { AlertTriangle, History, RotateCcw, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '../components/ui'
import { PlayPanel } from '../components/PlayPanel'
import { explainOutcome, INITIAL_BOARD } from '../lib/board'
import { useGame } from '../state/game'
import { BoardScene } from '../three/BoardScene'

function GameEnd() {
  const { game, newGame, review, reviewPly } = useGame()
  const [hiddenAtPly, setHiddenAtPly] = useState<number | null>(null)
  if (!game?.outcome || reviewPly !== null || hiddenAtPly === game.ply) return null
  const text = explainOutcome(game.outcome, game.side)
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-5 z-[var(--z-stage-overlay)] flex justify-center px-4">
      <div
        role="status"
        className="cz-rise pointer-events-auto w-full max-w-lg rounded-xl border border-line-strong bg-surface/95 p-5 shadow-[0_24px_60px_-20px_rgb(0_0_0/0.55)] backdrop-blur-sm"
      >
        <div className="flex items-start justify-between gap-4">
          <h2 className="font-serif text-[22px] leading-snug font-semibold text-ink">{text.title}</h2>
          <button type="button" onClick={() => setHiddenAtPly(game.ply)} aria-label="Hide result"
            className="-m-1 grid h-8 w-8 shrink-0 place-items-center rounded-md text-ink-faint hover:bg-surface-2 hover:text-ink">
            <X size={17} />
          </button>
        </div>
        <p className="mt-1.5 max-w-[60ch] text-sm leading-relaxed text-ink-muted">{text.detail}</p>
        <p className="mt-2 text-[12.5px] text-ink-faint">Rule: {text.source}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="primary" onClick={newGame} icon={<RotateCcw size={16} />}>Play again</Button>
          <Button onClick={() => review(Math.max(0, game.ply - 1))} icon={<History size={16} />}>Review moves</Button>
        </div>
      </div>
    </div>
  )
}

export function PlayView() {
  const g = useGame()
  useEffect(() => g.init(), []) // eslint-disable-line react-hooks/exhaustive-deps

  const reviewing = g.reviewPly !== null
  const view = reviewing ? g.reviewGame ?? g.game : g.game
  const last = view?.history[view.history.length - 1]
  const humanTurn =
    !!g.game && !g.game.outcome &&
    (g.mode === 'hvh' || (g.mode === 'hvc' && g.game.side === g.humanColor))
  const interactive = g.status === 'ready' && !reviewing && humanTurn

  return (
    <div className="flex flex-col lg:min-h-0 lg:flex-1 lg:flex-row">
      <section
        aria-label="Board"
        className="relative h-[62vh] min-h-[360px] shrink-0 overflow-hidden lg:h-auto lg:flex-1"
        style={{ background: 'radial-gradient(ellipse 70% 60% at 50% 55%, var(--stage-glow), transparent 75%)' }}
      >
        <BoardScene
          board={view?.board ?? INITIAL_BOARD}
          ply={view?.ply ?? 0}
          legalMoves={interactive ? g.game!.legal_moves : []}
          lastMove={last ? { from: last.from, to: last.to } : null}
          checkSquare={view?.in_check ? view.king_square : null}
          orientation={g.orientation}
          interactive={interactive}
          onMove={g.play}
        />

        {g.status === 'error' && (
          <div role="alert" className="absolute inset-x-0 top-4 z-[var(--z-stage-overlay)] flex justify-center px-4">
            <div className="flex max-w-xl items-center gap-3 rounded-lg border border-danger/40 bg-surface px-4 py-3 text-sm shadow-lg">
              <AlertTriangle size={18} className="shrink-0 text-danger" />
              <span className="text-ink">{g.error}</span>
              <Button size="sm" onClick={() => (g.game ? g.review(null) : g.newGame())}>Retry</Button>
            </div>
          </div>
        )}
        {g.status === 'connecting' && (
          <p className="absolute top-4 left-1/2 z-[var(--z-stage-overlay)] -translate-x-1/2 rounded-full border border-line bg-surface px-3 py-1 text-[12.5px] text-ink-muted">
            Connecting to the rules engine…
          </p>
        )}
        {reviewing && (
          <div className="absolute inset-x-0 top-4 z-[var(--z-stage-overlay)] flex justify-center px-4">
            <div className="flex items-center gap-3 rounded-full border border-line-strong bg-surface py-1.5 pr-1.5 pl-4 text-[13px] text-ink shadow-md">
              <History size={15} className="text-accent" />
              Reviewing after ply {g.reviewPly}
              <Button size="sm" variant="primary" onClick={() => g.review(null)}>Back to game</Button>
            </div>
          </div>
        )}
        <GameEnd />
      </section>

      <aside aria-label="Game controls" className="border-line bg-surface lg:w-[400px] lg:shrink-0 lg:border-l">
        <PlayPanel view={view} />
      </aside>
    </div>
  )
}
