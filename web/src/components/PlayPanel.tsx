import { FlipVertical2, Pause, Play } from 'lucide-react'
import { useEffect, useId, useMemo, useRef } from 'react'
import type { Color, GameView } from '../lib/api'
import { colorName, describeMove, explainOutcome, squareName } from '../lib/board'
import { runLabel, useGame, type AgentSlot, type Level, type Mode, type Pace } from '../state/game'
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
  {
    value: 'zero',
    label: 'Trained',
    title: 'ChaturangZero: the same search, guided by the network trained by self-play',
  },
]

const LEVELS: { value: Level; label: string; note: string }[] = [
  { value: 'easy', label: 'Easy', note: 'A 10-simulation search, and it often plays a move other than its favourite.' },
  { value: 'medium', label: 'Medium', note: 'The 50-simulation search it trained with; it usually plays its favourite move.' },
  { value: 'hard', label: 'Hard', note: '200 simulations per move, always its best move.' },
  { value: 'impossible', label: 'Impossible', note: '800 simulations per move, always its best move; about a second to think.' },
]

/** One to four rising bars: how strong a level plays. */
function Strength({ level, on }: { level: number; on: boolean }) {
  return (
    <span aria-hidden className="flex h-3.5 items-end gap-[2px]">
      {[0, 1, 2, 3].map((i) => (
        <span key={i} style={{ height: `${(i + 1) * 25}%` }}
          className={`w-[3px] rounded-[1px] transition-colors duration-200 ${
            i <= level ? (on ? 'bg-accent' : 'bg-ink-muted') : 'bg-line-strong'
          }`} />
      ))}
    </span>
  )
}

/** The trained agent's difficulty, in a panel that hangs from the Trained option above it. */
function LevelPicker({ slot }: { slot: AgentSlot }) {
  const { levels, setLevel, trained } = useGame()
  const name = useId()
  const current = levels[slot]
  const note = LEVELS.find((l) => l.value === current)?.note
  return (
    <div className="relative mt-2.5 rounded-lg border border-line bg-bg px-3 pt-2.5 pb-3">
      {/* The notch points at Trained, the last of the four agents. */}
      <span aria-hidden className="absolute -top-[6px] left-[calc(87.5%-6px)] h-[11px] w-[11px] rotate-45 border-t border-l border-line bg-bg" />
      <fieldset>
        <legend className="mb-2 text-[12.5px] font-medium text-ink-muted">Difficulty</legend>
        <div className="grid grid-cols-4 gap-1">
          {LEVELS.map((l, i) => {
            const on = l.value === current
            return (
              <label key={l.value}
                className={`relative flex cursor-pointer flex-col items-center gap-1.5 rounded-md px-1 py-2 text-[12px] font-medium transition-colors duration-200 ${
                  on ? 'bg-surface-2 text-ink shadow-[inset_0_0_0_1px_var(--line-strong)]' : 'text-ink-muted hover:bg-surface-2/60 hover:text-ink'
                }`}>
                <input type="radio" name={name} className="peer sr-only" checked={on} onChange={() => setLevel(slot, l.value)} />
                <span className="pointer-events-none absolute inset-0 rounded-md peer-focus-visible:outline-2 peer-focus-visible:outline-accent" />
                <Strength level={i} on={on} />
                {l.label}
              </label>
            )
          })}
        </div>
      </fieldset>
      <p key={current} className="cz-swap mt-2 text-[12.5px] leading-snug text-ink-muted">{note}</p>
      <p className="mt-1.5 text-[12px] leading-snug text-ink-faint">
        {trained === undefined ? 'Checking for a trained network'
          : trained === null ? 'No trained network yet: train one first, as the README explains.'
            : `Plays with generation ${trained.generation ?? '?'} of the run “${trained.run ?? 'unknown'}”.`}
      </p>
    </div>
  )
}

/** A computer player's picker; the trained agent adds its difficulty. */
function AgentPicker({ label, slot, value, onChange }: {
  label: string; slot: AgentSlot; value: string; onChange: (id: string) => void
}) {
  const refreshTrained = useGame((s) => s.refreshTrained)
  useEffect(() => {
    if (value === 'zero') refreshTrained()
  }, [value, refreshTrained])
  return (
    <div>
      <Segmented label={label} value={value} onChange={onChange} size="sm" options={AGENTS} />
      <Collapse open={value === 'zero'}>
        <LevelPicker slot={slot} />
      </Collapse>
    </div>
  )
}

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
              <AgentPicker label="Opponent" slot="opponent" value={g.opponent} onChange={g.setOpponent} />
            </div>
          )}
          {g.mode === 'cvc' && (
            <div className="space-y-3">
              <AgentPicker label="White" slot="white" value={g.whiteAgent} onChange={(v) => g.setAgent(1, v)} />
              <AgentPicker label="Black" slot="black" value={g.blackAgent} onChange={(v) => g.setAgent(-1, v)} />
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
