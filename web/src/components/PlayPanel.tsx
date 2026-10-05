import { FlipVertical2, Pause, Play } from 'lucide-react'
import { useEffect, useMemo, useRef } from 'react'
import type { Color, GameView } from '../lib/api'
import { colorName, describeMove, explainOutcome, squareName } from '../lib/board'
import { runLabel, useGame, type Mode, type Pace } from '../state/game'
import { useDetail, useSettings, type DetailPart } from '../state/settings'
import { PieceGlyph } from './PieceGlyph'
import { DrawerHeader } from './stage'
import { Button, Collapse, SectionTitle, Segmented, Switch } from './ui'

const AGENTS = [
  { value: 'random', label: 'Random', title: 'Plays a random legal move' },
  { value: 'greedy', label: 'Greedy', title: 'Looks two plies ahead: grabs material, avoids losing pieces' },
  {
    value: 'mcts',
    label: 'MCTS',
    title: 'Monte Carlo Tree Search: 50 simulations per move, guided by a network that is not trained yet',
  },
]

const DETAIL_PARTS: { part: DetailPart; label: string }[] = [
  { part: 'hints', label: 'Move hints' },
  { part: 'counters', label: 'Move and quiet-ply counters' },
  { part: 'explanation', label: 'Why the game ended' },
  { part: 'rules', label: 'Rule sources' },
]

const VALUE: Record<number, number> = { 1: 1, 2: 3, 3: 1.5, 4: 5, 5: 1.5, 6: 0 }

function Swatch({ color }: { color: Color }) {
  return (
    <span
      aria-hidden
      className={`inline-block h-3.5 w-3.5 shrink-0 rounded-full ring-1 ${
        color === 1 ? 'bg-[#f1e6cc] ring-[#8a6a45]' : 'bg-[#2b1e19] ring-[#c9a14c]'
      }`}
    />
  )
}

/**
 * Whose turn it is, shown in the floating card over the board. "Essential" detail keeps it to
 * the headline plus urgent tags (check, thinking); "Full" can add the hint and the counters.
 */
export function Status({ view }: { view: GameView | null }) {
  const { mode, humanColor, status, reviewPly, running, moves } = useGame()
  const showHint = useDetail('hints')
  const showCounters = useDetail('counters')
  if (!view) {
    return (
      <div aria-busy="true">
        <div className="h-6 w-40 animate-pulse rounded-md bg-surface-2" />
        {showHint && <div className="mt-2 h-4 w-56 animate-pulse rounded bg-surface-2" />}
      </div>
    )
  }
  const side = view.side
  let title = `${colorName(side)} to move`
  let hint: string | null = null
  let tag: { text: string; danger?: boolean } | null = null
  if (view.outcome) {
    title = explainOutcome(view.outcome, side).title
  } else if (reviewPly !== null) {
    hint = `Reviewing the position after ${reviewPly} ${reviewPly === 1 ? 'ply' : 'plies'}.`
  } else if (status === 'thinking') {
    tag = { text: 'Thinking…' }
    hint = 'The computer is choosing a move.'
  } else if (view.in_check) {
    tag = { text: 'Check', danger: true }
    hint = `The ${colorName(side)} Raja is attacked and must be saved.`
  } else if (mode === 'hvc') {
    hint = side === humanColor ? 'Your move. Select a piece to see where it can go.' : "The computer's move."
  } else if (mode === 'cvc') {
    hint = running
      ? 'Computer against computer.'
      : moves.length === 0 ? 'Press Start to begin.' : 'Paused. Press Resume to carry on.'
  } else {
    hint = 'Select a piece to see where it can go.'
  }
  const last = view.history[view.history.length - 1]
  return (
    <div>
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 font-serif text-[20px] leading-tight font-semibold text-ink">
        {!view.outcome && <Swatch color={side} />}
        {title}
        {tag && (
          <span className={`rounded-md px-1.5 py-0.5 font-sans text-[12px] font-semibold ${
            tag.danger ? 'bg-danger-soft text-danger' : 'bg-surface-2 text-ink-muted'
          }`}>
            {tag.text}
          </span>
        )}
      </p>
      {showHint && hint && <p className={`mt-1 text-[13.5px] leading-snug ${tag?.danger ? 'text-danger' : 'text-ink-muted'}`}>{hint}</p>}
      {showCounters && (
        <p className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[12px] text-ink-faint tabular-nums">
          <span>Move {Math.floor(view.ply / 2) + 1}</span>
          <span title="Plies since the last capture or Padati move. At 140 the game is drawn.">
            Quiet plies {view.no_progress}/140
          </span>
          {view.repetition > 1 && <span className="text-ink-muted">Position repeated {view.repetition}×</span>}
        </p>
      )}
      <p className="sr-only" aria-live="polite">{last ? `${last.piece > 0 ? 'White' : 'Black'} played ${describeMove(last)}` : ''}</p>
    </div>
  )
}

function MoveList({ view }: { view: GameView | null }) {
  const { reviewPly, review } = useGame()
  const history = view?.history ?? []
  const listRef = useRef<HTMLOListElement>(null)
  useEffect(() => {
    if (reviewPly === null) listRef.current?.scrollTo({ top: listRef.current.scrollHeight })
  }, [history.length, reviewPly])
  const current = reviewPly ?? history.length
  if (history.length === 0) {
    return <p className="text-sm text-ink-muted">No moves yet. White opens the game.</p>
  }
  const rows: number[] = []
  for (let i = 0; i < history.length; i += 2) rows.push(i)
  return (
    <ol ref={listRef} className="max-h-64 overflow-y-auto pr-1 lg:max-h-[min(40vh,22rem)]">
      {rows.map((i) => (
        <li key={i} className="grid grid-cols-[2.2rem_1fr_1fr] items-center gap-1 text-[13px]">
          <span className="text-ink-faint tabular-nums">{i / 2 + 1}.</span>
          {[i, i + 1].map((j) =>
            j < history.length ? (
              <button
                key={j}
                type="button"
                onClick={() => review(j + 1 === history.length ? null : j + 1)}
                className={`flex h-7 items-center gap-1.5 rounded-md px-1.5 text-left tabular-nums transition-colors duration-150 ${
                  current === j + 1 ? 'bg-accent-soft text-ink' : 'text-ink-muted hover:bg-surface-2 hover:text-ink'
                }`}
                aria-label={`Move ${j + 1}: ${describeMove(history[j])}`}
                aria-current={current === j + 1 ? 'step' : undefined}
              >
                <PieceGlyph piece={history[j].piece} size={16} />
                <span>
                  {squareName(history[j].from)}
                  {history[j].captured ? '×' : '–'}
                  {squareName(history[j].to)}
                  {history[j].promotion ? '=M' : ''}
                  {history[j].check ? '+' : ''}
                </span>
              </button>
            ) : (
              <span key={j} />
            ),
          )}
        </li>
      ))}
    </ol>
  )
}

function Captured({ view }: { view: GameView | null }) {
  const taken = useMemo(() => {
    const byWhite: number[] = []
    const byBlack: number[] = []
    for (const h of view?.history ?? []) {
      if (h.captured > 0) byBlack.push(h.captured)
      if (h.captured < 0) byWhite.push(h.captured)
    }
    const sort = (a: number, b: number) => VALUE[Math.abs(b)] - VALUE[Math.abs(a)]
    return { byWhite: byWhite.sort(sort), byBlack: byBlack.sort(sort) }
  }, [view])
  const score = (l: number[]) => l.reduce((s, p) => s + VALUE[Math.abs(p)], 0)
  const diff = score(taken.byWhite) - score(taken.byBlack)
  const row = (label: string, list: number[], lead: number) => (
    <div className="flex min-h-7 items-center gap-3">
      <span className="w-24 shrink-0 text-[13px] text-ink-muted">{label}</span>
      <span className="flex flex-wrap items-center gap-0.5">
        {list.length === 0 ? <span className="text-[13px] text-ink-faint">Nothing yet</span> :
          list.map((p, i) => <PieceGlyph key={i} piece={p} size={20} />)}
        {lead > 0 && <span className="ml-1.5 text-[12.5px] font-medium text-ink-muted tabular-nums">+{lead}</span>}
      </span>
    </div>
  )
  return (
    <div className="space-y-1">
      {row('Taken by White', taken.byWhite, diff)}
      {row('Taken by Black', taken.byBlack, -diff)}
    </div>
  )
}

export function PlayPanel({ view, onClose, onRotate }: { view: GameView | null; onClose: () => void; onRotate: () => void }) {
  const g = useGame()
  const settings = useSettings()
  return (
    <>
      <DrawerHeader title="Game" onClose={onClose} closeLabel="Close the game panel" />
      <div className="flex-1 space-y-7 overflow-y-auto p-5 sm:p-6">

        <section aria-label="Game mode" className="space-y-3.5">
          <Segmented<Mode>
            label="Mode"
            value={g.mode}
            onChange={g.setMode}
            options={[
              { value: 'hvh', label: 'Two players', title: 'Two people take turns on this screen' },
              { value: 'hvc', label: 'Vs computer' },
              { value: 'cvc', label: 'Computers', title: 'Computer against computer' },
            ]}
          />
          {g.mode === 'hvh' && (
            <Switch label="Turn the board after each move" checked={g.autoRotate} onChange={g.setAutoRotate} />
          )}
          {g.mode === 'hvc' && (
            <div className="space-y-3">
              <Segmented<Color> label="You play" value={g.humanColor} onChange={g.setHumanColor} size="sm"
                options={[{ value: 1, label: 'White' }, { value: -1, label: 'Black' }]} />
              <Segmented label="Opponent" value={g.opponent} onChange={g.setOpponent} size="sm" options={AGENTS} />
            </div>
          )}
          {g.mode === 'cvc' && (
            <div className="space-y-3">
              <Segmented label="White" value={g.whiteAgent} onChange={(v) => g.setAgent(1, v)} size="sm" options={AGENTS} />
              <Segmented label="Black" value={g.blackAgent} onChange={(v) => g.setAgent(-1, v)} size="sm" options={AGENTS} />
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <Segmented<Pace> label="Pace" value={g.pace} onChange={g.setPace} size="sm"
                    options={[{ value: 'slow', label: 'Slow' }, { value: 'normal', label: 'Normal' }, { value: 'fast', label: 'Fast' }]} />
                </div>
                <Button size="sm" variant={g.running ? 'secondary' : 'primary'} onClick={() => g.setRunning(!g.running)}
                  disabled={!!view?.outcome} icon={g.running ? <Pause size={15} /> : <Play size={15} />}>
                  {runLabel(g.running, g.moves.length)}
                </Button>
              </div>
            </div>
          )}
        </section>

        <section aria-labelledby="moves-title">
          <SectionTitle aside={view && view.ply > 0 ? `${view.ply} plies` : undefined}>
            <span id="moves-title">Moves</span>
          </SectionTitle>
          <MoveList view={g.game} />
        </section>

        <section aria-labelledby="captured-title">
          <SectionTitle><span id="captured-title">Captured</span></SectionTitle>
          <Captured view={view} />
        </section>

        <section aria-label="Display" className="space-y-3.5">
          <SectionTitle>Display</SectionTitle>
          <div className="grid grid-cols-2 gap-3">
            <Segmented label="View" value={settings.view} onChange={settings.setView} size="sm"
              options={[{ value: '3d', label: '3D' }, { value: 'flat', label: 'Flat' }]} />
            <Segmented label="Piece names" value={settings.labels} onChange={settings.setLabels} size="sm"
              options={[
                { value: 'off', label: 'Off' },
                { value: 'english', label: 'Latin', title: 'Transliterated Sanskrit names' },
                { value: 'sanskrit', label: <span className="font-deva text-[14px]" lang="sa">देव</span>, title: 'Devanagari' },
              ]} />
          </div>
          <Segmented label="Details" value={settings.details} onChange={settings.setDetails} size="sm"
            options={[
              { value: 'essential', label: 'Essential', title: 'Headlines only: whose turn it is, check, and the result' },
              { value: 'full', label: 'Full', title: 'Also hints, move and quiet-ply counters, and the rule behind each result' },
            ]} />
          <div className="-mt-1.5">
            <Collapse open={settings.details === 'essential'}>
              <p className="text-[12.5px] leading-snug text-ink-faint">
                Showing headlines only. Full lets you add hints, move counters and the rule behind each result.
              </p>
            </Collapse>
            <Collapse open={settings.details === 'full'}>
              <fieldset className="mt-0.5 space-y-2.5 rounded-lg bg-bg px-3 py-3">
                <legend className="sr-only">Details to show</legend>
                {DETAIL_PARTS.map((p) => (
                  <Switch key={p.part} label={p.label} checked={settings.detailParts[p.part]}
                    onChange={(on) => settings.setDetailPart(p.part, on)} />
                ))}
              </fieldset>
            </Collapse>
          </div>
          <div>
            <Switch label="Reduced graphics" checked={settings.reducedGraphics} onChange={settings.setReducedGraphics} />
            <p className="mt-1 text-[12.5px] leading-snug text-ink-faint">
              No shadows, glossy finishes or pulsing effects, at a lower resolution. Smoother on slower computers.
            </p>
          </div>
          <Switch label="Sound" checked={settings.sound} onChange={settings.setSound} />
          <Button size="sm" onClick={onRotate} icon={<FlipVertical2 size={15} />}>Rotate the board</Button>
        </section>
      </div>
    </>
  )
}
