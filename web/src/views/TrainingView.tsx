/**
 * Watch training: a training run's progress, read from the API server (runs/<name>/ on disk).
 * Polls every few seconds while a run is live. A secondary screen for the team: the play and
 * learn screens stay the primary experience.
 */

import { ChevronDown, LoaderCircle, Pause, Play, Plus, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Legend, LineChart, type Series } from '../components/LineChart'
import { Button, Collapse, Segmented, Switch } from '../components/ui'
import { api, ApiError, type GenerationRow, type PauseWhen, type RunDetail, type RunStatus } from '../lib/api'
import { href } from '../lib/route'
import { useGame } from '../state/game'

const LIVE_POLL_MS = 4000
const IDLE_POLL_MS = 30000
// While a pause or a start is under way, so the change shows within a second or so.
const CHANGING_POLL_MS = 1200

const PHASE: Record<string, string> = {
  starting: 'Starting',
  selfplay: 'Playing self-play games',
  training: 'Training the network',
  evaluating: 'Testing the new network',
  generation_done: 'Generation finished',
  finished: 'Finished',
  paused: 'Paused',
  stopped: 'Stopped',
  failed: 'Stopped with an error',
}

const pct = (v: number) => `${Math.round(v * 100)}%`

function ago(seconds: number | null) {
  if (seconds === null) return 'never'
  if (seconds < 90) return 'just now'
  if (seconds < 5400) return `${Math.round(seconds / 60)} minutes ago`
  if (seconds < 129600) return `${Math.round(seconds / 3600)} hours ago`
  return `${Math.round(seconds / 86400)} days ago`
}

function Progress({ value, label }: { value: number; label: string }) {
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}
      className="h-1.5 overflow-hidden rounded-full bg-surface-2">
      <div className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out-quart" style={{ width: `${Math.min(100, value * 100)}%` }} />
    </div>
  )
}

const PHASE_NAME = { selfplay: 'self-play', training: 'training', evaluating: 'testing' } as const

/** Where a paused or stopped run is, in words. */
function whereItStopped(status: RunStatus): string {
  const g = status.generation
  if (status.state === 'paused') {
    if (status.paused_in === 'selfplay') {
      const kept = status.games_done ?? 0
      const of = status.games_total ?? '?'
      if (kept === 0) return `Paused during generation ${g}'s self-play, before any of its ${of} games finished.`
      return `Paused during generation ${g}'s self-play. ${kept === 1 ? 'The 1 finished game' : `The ${kept} finished games`} of ${of} ${kept === 1 ? 'is' : 'are'} kept.`
    }
    if (status.paused_in === 'training') return `Paused while training generation ${g}.`
    if (status.paused_in === 'evaluating') return `Paused while testing generation ${g}.`
    if (status.paused_after === 'selfplay') return `Paused after generation ${g}'s self-play. Training is next.`
    if (status.paused_after === 'training') return `Paused after training generation ${g}. Testing it is next.`
    return `Paused after generation ${g}.`
  }
  if (status.state === 'failed') return 'Stopped with an error; the details are at the end of the run’s train.log.'
  return 'The training process ended without pausing (closed or interrupted). Finished self-play games are kept.'
}

const PAUSE_OPTIONS: { when: PauseWhen; label: string; note: (s: RunStatus) => string }[] = [
  {
    when: 'now', label: 'Pause now',
    note: () => 'Stops within seconds. Finished games are kept; anything half done is redone when you resume.',
  },
  {
    when: 'after_phase', label: 'After this phase',
    note: (s) => s.state === 'selfplay' || s.state === 'training' || s.state === 'evaluating'
      ? `Finishes the current ${PHASE_NAME[s.state]}, then stops.` : 'Finishes the current step of the generation, then stops.',
  },
  {
    when: 'after_generation', label: 'After this generation',
    note: (s) => `Finishes generation ${s.generation}, testing included, then stops.`,
  },
]

const PENDING: Record<PauseWhen, string> = {
  now: 'Pausing now',
  after_phase: 'Will pause after this phase',
  after_generation: 'Will pause after this generation',
}

function Spinner() {
  return <LoaderCircle size={14} className="shrink-0 animate-spin" aria-hidden />
}

/**
 * Eases its height when `swapKey` changes, so the panel grows or shrinks smoothly into what
 * replaces it (the replacement itself fades in with .cz-swap). Size changes within one state,
 * like the pause choices opening, are left to the content's own animation.
 */
function HeightTransition({ swapKey, children }: { swapKey: string; children: ReactNode }) {
  const outer = useRef<HTMLDivElement>(null)
  const inner = useRef<HTMLDivElement>(null)
  const height = useRef<number | null>(null)

  useEffect(() => {
    const el = inner.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      if (!outer.current?.style.height) height.current = el.offsetHeight
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useLayoutEffect(() => {
    const o = outer.current
    const i = inner.current
    if (!o || !i) return
    const from = height.current
    const to = i.offsetHeight
    height.current = to
    if (from === null || from === to) return
    o.style.height = `${from}px`
    void o.offsetHeight // apply the starting height before transitioning from it
    o.style.transition = 'height 300ms var(--ease-out-quart)'
    o.style.height = `${to}px`
    const done = () => {
      o.style.height = ''
      o.style.transition = ''
      height.current = i.offsetHeight
    }
    const onEnd = (e: TransitionEvent) => {
      if (e.target === o) done()
    }
    o.addEventListener('transitionend', onEnd)
    const timer = setTimeout(done, 450) // reduced motion, or a transition that never ran
    return () => {
      o.removeEventListener('transitionend', onEnd)
      clearTimeout(timer)
    }
  }, [swapKey])

  return (
    <div ref={outer} className="overflow-clip">
      <div ref={inner}>{children}</div>
    </div>
  )
}

type ControlMode = 'idle' | 'pausing' | 'starting' | 'stopped' | 'finished'

function RunControls({ status, total, done, onChange }: {
  status: RunStatus; total: number; done: number; onChange: (s?: RunStatus) => void
}) {
  const [choosing, setChoosing] = useState(false)
  const [sending, setSending] = useState<string | null>(null) // which control is waiting for the server
  const [error, setError] = useState<string | null>(null)
  const [more, setMore] = useState(10)
  const act = async (id: string, call: () => Promise<RunStatus>) => {
    if (sending) return
    setSending(id)
    setError(null)
    try {
      onChange(await call())
      setChoosing(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(null)
    }
  }
  const name = status.name
  const finished = status.state === 'finished' && !status.live
  const mode: ControlMode = status.live
    ? status.state === 'starting' ? 'starting' : status.pause_requested ? 'pausing' : 'idle'
    : finished ? 'finished' : 'stopped'
  const target = Math.max(done + 1, total + more)

  let body: ReactNode
  if (mode === 'starting') {
    body = (
      <p className="flex items-center gap-2.5 text-[13.5px] text-ink" role="status">
        <Spinner />
        Starting the run: loading the network and its saved games.
      </p>
    )
  } else if (mode === 'pausing') {
    body = (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="flex items-center gap-2.5 text-[13.5px] text-ink" role="status">
          <Spinner />
          {PENDING[status.pause_requested!]}
        </p>
        <Button size="sm" aria-busy={sending === 'cancel'} onClick={() => act('cancel', () => api.pauseRun(name, null))}
          icon={sending === 'cancel' ? <Spinner /> : undefined}>
          Don’t pause
        </Button>
      </div>
    )
  } else if (mode === 'idle') {
    body = (
      <>
        <Button size="sm" onClick={() => setChoosing((c) => !c)} aria-expanded={choosing} icon={<Pause size={14} />}>
          Pause
        </Button>
        <Collapse open={choosing}>
          <ul className="mt-3 grid gap-2 sm:grid-cols-3">
            {PAUSE_OPTIONS.map((o) => {
              const mine = sending === o.when
              return (
                <li key={o.when}>
                  <button type="button" aria-busy={mine} onClick={() => act(o.when, () => api.pauseRun(name, o.when))}
                    className={`flex h-full w-full flex-col items-start justify-start rounded-xl border px-3.5 py-3 text-left transition-[background-color,border-color,opacity,transform] duration-200 ease-out-quart active:scale-[0.98] ${
                      mine ? 'border-accent bg-accent-soft' : 'border-line bg-bg hover:border-line-strong hover:bg-surface-2'
                    } ${sending && !mine ? 'opacity-40' : ''}`}>
                    <span className="flex w-full items-center justify-between gap-2 text-[13.5px] font-semibold text-ink">
                      {o.label}
                      {mine && <Spinner />}
                    </span>
                    <span className="mt-1 block text-[12.5px] leading-snug text-ink-muted">{o.note(status)}</span>
                  </button>
                </li>
              )
            })}
          </ul>
        </Collapse>
      </>
    )
  } else {
    const starting = sending === 'resume'
    body = (
      <div className="space-y-3">
        {mode === 'stopped' && <p className="max-w-[70ch] text-[13.5px] leading-snug text-ink-muted">{whereItStopped(status)}</p>}
        <div className="flex flex-wrap items-center gap-3">
          {mode === 'finished' && (
            <label className="flex items-center gap-2 text-[13px] text-ink-muted">
              Train further, to
              <input type="number" min={done + 1} max={10000} value={target}
                onChange={(e) => setMore(Math.max(1, Number(e.target.value) - total))}
                className="h-8 w-20 rounded-lg border border-line bg-bg px-2 text-[13px] text-ink tabular-nums" />
              generations
            </label>
          )}
          <Button size="sm" variant="primary" aria-busy={starting} icon={starting ? <Spinner /> : <Play size={14} />}
            onClick={() => act('resume', () => api.resumeRun(name, mode === 'finished' ? target : undefined))}>
            {starting ? 'Starting' : mode === 'finished' ? 'Train further' : 'Resume'}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="mt-5 border-t border-line pt-4">
      <HeightTransition swapKey={mode}>
        <div key={mode} className="cz-swap">{body}</div>
      </HeightTransition>
      {error && <p className="cz-swap mt-2 text-[12.5px] text-danger">{error}</p>}
    </div>
  )
}

function StatusPanel({ status, detail, onChange }: { status: RunStatus; detail: RunDetail; onChange: (s?: RunStatus) => void }) {
  const live = status.live
  // A run whose process ended without saying so (closed, crashed, machine asleep) has stopped.
  const state = !live && !['finished', 'failed', 'paused'].includes(status.state) ? 'stopped' : status.state
  const pausing = live && !!status.pause_requested
  const total = status.generations || detail.config.generations
  const done = detail.generations.filter((g) => g.best !== undefined).length
  let phase = ''
  let phaseValue = 0
  if (live && status.state === 'selfplay' && status.games_total) {
    phase = `${status.games_done ?? 0} of ${status.games_total} games`
    phaseValue = (status.games_done ?? 0) / status.games_total
  } else if (live && status.state === 'training' && status.steps_total) {
    phase = `step ${status.step ?? 0} of ${status.steps_total}`
    phaseValue = (status.step ?? 0) / status.steps_total
  } else if (live && status.state === 'evaluating' && status.games_total) {
    phase = `against ${status.opponent === 'previous' ? 'the previous generation' : status.opponent === 'greedy' ? 'Greedy' : status.opponent}, ${status.games_done ?? 0} of ${status.games_total} games`
    phaseValue = (status.games_done ?? 0) / status.games_total
  }
  // The phase bar folds away when the run stops; keep its last reading on show while it does.
  const [shown, setShown] = useState({ phase, phaseValue })
  if (phase && (phase !== shown.phase || phaseValue !== shown.phaseValue)) setShown({ phase, phaseValue })
  const title = pausing && status.pause_requested === 'now' ? 'Pausing' : PHASE[state] ?? state
  const tone = !live ? (state === 'paused' ? 'bg-accent' : 'bg-ink-faint') : 'bg-success'
  const games = detail.generations.reduce((n, g) => n + (g.games ?? 0), 0)
  const positions = detail.generations.reduce((n, g) => n + (g.positions ?? 0), 0)
  const last = [...detail.generations].reverse().find((g) => g.eval.greedy)
  return (
    <section aria-label="Run status" className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <p className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 font-serif text-[22px] font-semibold text-ink">
          <span aria-hidden className={`relative inline-flex h-2.5 w-2.5 rounded-full transition-colors duration-500 ${tone}`}>
            {live && <span className={`absolute inset-0 animate-ping rounded-full opacity-60 transition-colors duration-500 motion-reduce:hidden ${tone}`} />}
          </span>
          <span key={title} className="cz-swap">{title}</span>
          {live && status.generation > 0 && status.state !== 'generation_done' && status.state !== 'starting' && (
            <span className="font-sans text-[15px] font-normal whitespace-nowrap text-ink-muted">for generation {status.generation}</span>
          )}
        </p>
        <p className="text-[13px] text-ink-faint">{live ? 'Live' : 'Not running'} · last update {ago(status.seconds_since_update)}</p>
      </div>
      <Collapse open={!!phase}>
        <div className="space-y-1.5 pt-3">
          <p className="text-[13px] text-ink-muted tabular-nums">{shown.phase}</p>
          <Progress value={shown.phaseValue} label="This phase" />
        </div>
      </Collapse>
      <div className="mt-5 space-y-1.5">
        <p className="text-[13px] text-ink-muted tabular-nums">{done} of {total} generations</p>
        <Progress value={done / total} label="Generations" />
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-[13px] sm:grid-cols-4">
        {[
          ['Self-play games', games.toLocaleString()],
          ['Positions', positions.toLocaleString()],
          ['Best network', status.best_generation !== undefined ? `Generation ${status.best_generation}` : '–'],
          ['Latest vs Greedy', last ? `${pct(last.eval.greedy.score)} (gen ${last.generation})` : '–'],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-ink-faint">{k}</dt>
            <dd className="mt-0.5 text-[15px] font-semibold text-ink tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <RunControls status={{ ...status, state }} total={total} done={done} onChange={onChange} />
    </section>
  )
}

function ChartCard({ title, note, children }: { title: string; note: string; children: React.ReactNode }) {
  return (
    <section className="min-w-0 rounded-2xl border border-line bg-surface p-5">
      <h3 className="font-serif text-[18px] font-semibold text-ink">{title}</h3>
      <p className="mt-1 mb-3 text-[12.5px] leading-snug text-ink-muted">{note}</p>
      {children}
    </section>
  )
}

function Charts({ detail }: { detail: RunDetail }) {
  const steps = detail.config.steps_per_generation
  const gens = detail.generations
  // A generation's new games are measured before training on them, i.e. at the end of the one before.
  const held = gens.filter((g) => g.held_out)
  const heldX = (g: GenerationRow) => g.held_out?.step ?? (g.generation - 1) * steps
  const lossSeries = (key: 'policy_loss' | 'value_loss'): Series[] => [
    { id: 'train', label: 'training batches', color: 'var(--chart-1)', kind: 'line',
      points: detail.train.map((p) => ({ x: p.step, y: p[key] })) },
    { id: 'new', label: 'new games', color: 'var(--chart-2)', kind: 'dots',
      points: held.map((g) => ({ x: heldX(g), y: g.held_out![key] })) },
  ]
  const lossLegend = [
    { label: 'Training batches', color: 'var(--chart-1)', kind: 'line' as const },
    { label: 'New games, before training on them', color: 'var(--chart-2)', kind: 'dots' as const },
  ]
  const evalSeries: Series[] = [
    { id: 'greedy', label: 'vs Greedy', color: 'var(--chart-1)', kind: 'line',
      points: gens.filter((g) => g.eval.greedy).map((g) => ({ x: g.generation, y: g.eval.greedy.score * 100 })) },
    { id: 'previous', label: 'vs previous generation', color: 'var(--chart-2)', kind: 'line',
      points: gens.filter((g) => g.eval.previous).map((g) => ({ x: g.generation, y: g.eval.previous.score * 100 })) },
  ]
  const decisive: Series[] = [{
    id: 'decisive', label: 'decisive', color: 'var(--chart-1)', kind: 'line',
    points: gens.filter((g) => g.games).map((g) => ({ x: g.generation, y: (1 - (g.draws ?? 0) / g.games!) * 100 })),
  }]
  const fmtLoss = (v: number) => (v === 0 ? '0' : v >= 1 ? v.toFixed(1) : v.toFixed(2))
  const fmtPct = (v: number) => `${Math.round(v)}%`
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ChartCard title="Policy loss" note="How far the network's move choices are from the search's. The dashed line is the lowest it can go.">
        <LineChart label="Policy loss by training step" series={lossSeries('policy_loss')} xLabel="training step" yFormat={fmtLoss}
          reference={{ label: 'lowest possible', points: detail.train.map((p) => ({ x: p.step, y: p.policy_entropy })) }} />
        <div className="mt-2"><Legend items={[...lossLegend,
          { label: 'Lowest possible (the targets’ own entropy)', color: 'var(--ink-faint)', kind: 'ref' }]} /></div>
      </ChartCard>
      <ChartCard title="Value loss" note="How far its prediction of the result is from how games ended. On new games it must beat the dashed line, the score of always predicting a draw.">
        <LineChart label="Value loss by training step" series={lossSeries('value_loss')} xLabel="training step" yFormat={fmtLoss}
          reference={{ label: 'predicting a draw', points: held.filter((g) => g.held_out!.value_baseline !== undefined)
            .map((g) => ({ x: heldX(g), y: g.held_out!.value_baseline! })) }} />
        <div className="mt-2"><Legend items={[...lossLegend,
          { label: 'Predicting a draw, on the new games', color: 'var(--ink-faint)', kind: 'ref' }]} /></div>
      </ChartCard>
      <ChartCard title="Evaluation score" note="Wins plus half the draws, after each generation. Above 50% against the previous generation means it improved.">
        <LineChart label="Evaluation score by generation" series={evalSeries} xLabel="generation" integerX yMin={0} yMax={100} yFormat={fmtPct} />
        <div className="mt-2"><Legend items={[
          { label: 'vs Greedy', color: 'var(--chart-1)', kind: 'line' },
          { label: 'vs previous generation', color: 'var(--chart-2)', kind: 'line' },
        ]} /></div>
      </ChartCard>
      <ChartCard title="Decisive self-play games" note="Games won by checkmate, bare king or stalemate rather than drawn. Decisive games teach the value head the most.">
        <LineChart label="Share of decisive self-play games by generation" series={decisive} xLabel="generation" integerX yMin={0} yMax={100} yFormat={fmtPct} />
      </ChartCard>
    </div>
  )
}

function WatchButton({ run, row }: { run: string; row: GenerationRow }) {
  const watchGame = useGame((s) => s.watchGame)
  const [busy, setBusy] = useState(false)
  const open = async () => {
    setBusy(true)
    try {
      // The shortest decisive game has the most happening per move; fall back to the first game.
      const { games } = await api.selfPlayGames(run, row.generation)
      const decisive = games.filter((g) => g.winner !== null).sort((a, b) => a.plies - b.plies)
      const pick = decisive[0] ?? games[0]
      if (!pick) return
      const game = await api.selfPlayGame(run, row.generation, pick.index)
      watchGame(game.moves)
      location.hash = href('play')
    } finally {
      setBusy(false)
    }
  }
  return (
    <Button size="sm" variant="secondary" onClick={open} disabled={busy || !row.games} icon={<Play size={14} />}
      title="Replay this generation's shortest decisive self-play game on the board"
      aria-label={`Watch a self-play game from generation ${row.generation}`}>
      Watch
    </Button>
  )
}

function GenerationTable({ run, detail }: { run: string; detail: RunDetail }) {
  const rows = [...detail.generations].reverse()
  const best = detail.status.best_generation
  if (!rows.length) return null
  const score = (g: GenerationRow, k: string) => (g.eval[k] ? pct(g.eval[k].score) : '–')
  return (
    <section aria-labelledby="gens-title" className="rounded-2xl border border-line bg-surface p-5">
      <h3 id="gens-title" className="font-serif text-[18px] font-semibold text-ink">Generations</h3>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left text-[13px] tabular-nums">
          <thead>
            <tr className="border-b border-line-strong text-[12px] text-ink-faint">
              {['Gen', 'Self-play W / D / B', 'Mean length', 'Loss', 'Loss on new games', 'vs previous', 'vs Greedy', ''].map((h) => (
                <th key={h} scope="col" className="py-2 pr-4 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => (
              <tr key={g.generation} className="border-b border-line">
                <th scope="row" className="py-2 pr-4 font-semibold text-ink">
                  {g.generation}
                  {best === g.generation && <span className="ml-2 rounded bg-accent-soft px-1.5 py-px text-[11px] font-medium text-accent">best</span>}
                </th>
                <td className="py-2 pr-4 text-ink-muted">{g.games ? `${g.white_wins} / ${g.draws} / ${g.black_wins}` : '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{g.mean_plies ? `${Math.round(g.mean_plies)} plies` : '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{g.train_loss?.loss?.toFixed(2) ?? '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{g.held_out ? g.held_out.loss.toFixed(2) : '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{score(g, 'previous')}</td>
                <td className="py-2 pr-4 text-ink-muted">{score(g, 'greedy')}</td>
                <td className="py-1.5 text-right"><WatchButton run={run} row={g} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function Problem({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8">
      <p className="font-serif text-[20px] font-semibold text-ink">Cannot load training runs</p>
      <p className="mt-2 text-[14px] text-ink-muted">{error}</p>
      <div className="mt-4"><Button size="sm" onClick={onRetry} icon={<RefreshCw size={14} />}>Try again</Button></div>
    </section>
  )
}

/** Seconds per self-play game at 50 simulations, a generation's testing and training included:
 * measured on an M1 with six self-play processes. Used until this machine has a run of its own. */
const FALLBACK_SECONDS_PER_GAME = 5.6

function duration(seconds: number) {
  const minutes = Math.max(1, Math.round(seconds / 60))
  if (minutes < 60) return `${minutes} min`
  const h = Math.floor(minutes / 60)
  const m = Math.round((minutes % 60) / 5) * 5
  return m ? `${h} h ${m} min` : `${h} h`
}

function NumberField({ label, value, onChange, min, max, hint }: {
  label: string; value: number; onChange: (v: number) => void; min: number; max: number; hint?: string
}) {
  const id = useId()
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1.5 block text-[13px] font-medium text-ink-muted">{label}</label>
      <input id={id} type="number" inputMode="numeric" min={min} max={max} value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(e.target.value === '' ? NaN : Number(e.target.value))}
        className="h-9 w-full rounded-lg border border-line bg-bg px-3 text-[14px] text-ink tabular-nums transition-colors duration-150 hover:border-line-strong focus-visible:border-accent" />
      {hint && <p className="mt-1 text-[12px] text-ink-faint">{hint}</p>}
    </div>
  )
}

/** The form for a new training run. */
function NewRun({ runs, first, secondsPerGame, onStarted, onCancel }: {
  runs: RunStatus[]; first: boolean; secondsPerGame: number
  onStarted: (name: string) => void; onCancel?: () => void
}) {
  const names = new Set(runs.map((r) => r.name))
  const suggest = () => {
    if (!names.has('main')) return 'main'
    let n = runs.length + 1
    while (names.has(`run-${n}`)) n++
    return `run-${n}`
  }
  const [name, setName] = useState(suggest)
  const [generations, setGenerations] = useState(20)
  const [games, setGames] = useState(50)
  const [sims, setSims] = useState(50)
  const [evalGames, setEvalGames] = useState(20)
  const [more, setMore] = useState(false)
  const trained = useGame((s) => s.trained)
  const refreshTrained = useGame((s) => s.refreshTrained)
  useEffect(() => refreshTrained(), [refreshTrained])
  // A new run takes over the Trained opponent only if no other run has given it a network yet.
  const [publish, setPublish] = useState<boolean | null>(null)
  const publishing = publish ?? trained === null
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const live = runs.find((r) => r.live)

  const problem =
    !/^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$/.test(name) ? 'Name it with letters, digits, - and _, starting with a letter or digit.'
      : names.has(name) ? `There is already a run named “${name}”.`
        : !(generations >= 1 && generations <= 1000) ? 'Generations: 1 to 1,000.'
          : !(games >= 2 && games <= 1000) ? 'Games per generation: 2 to 1,000.'
            : !(sims >= 2 && sims <= 2000) ? 'Simulations per move: 2 to 2,000.'
              : !(evalGames >= 0 && evalGames <= 200) ? 'Test games: 0 to 200.'
                : null
  const [tried, setTried] = useState(false)
  const shown = tried ? problem ?? error : error

  const start = async () => {
    setTried(true)
    if (problem || sending) return
    setSending(true)
    setError(null)
    try {
      await api.startRun({ name, generations, games_per_generation: games, simulations: sims, eval_games: evalGames,
        publish: publishing })
      onStarted(name)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }

  const total = generations * games
  const estimate = Number.isFinite(total) && Number.isFinite(sims) && total > 0
    ? duration(total * secondsPerGame * (sims / 50) * (0.85 + 0.15 * (evalGames / 20)))
    : null

  return (
    <section aria-label="New training run" className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <h2 className="font-serif text-[22px] font-semibold text-ink">{first ? 'Start the first run' : 'New run'}</h2>
      <p className="mt-1 max-w-[62ch] text-[13.5px] leading-snug text-ink-muted">
        Starts from an untrained network. You can pause it at any time and resume it later.
      </p>
      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <div className="min-w-0">
          <label htmlFor="new-run-name" className="mb-1.5 block text-[13px] font-medium text-ink-muted">Name</label>
          <input id="new-run-name" value={name} onChange={(e) => setName(e.target.value.trim())} spellCheck={false} maxLength={40}
            className="h-9 w-full rounded-lg border border-line bg-bg px-3 text-[14px] text-ink transition-colors duration-150 hover:border-line-strong focus-visible:border-accent" />
        </div>
        <NumberField label="Generations" value={generations} onChange={setGenerations} min={1} max={1000} />
        <NumberField label="Games per generation" value={games} onChange={setGames} min={2} max={1000} />
      </div>
      <button type="button" onClick={() => setMore((m) => !m)} aria-expanded={more}
        className="mt-4 inline-flex items-center gap-1 rounded-md text-[13px] font-medium text-ink-muted transition-colors duration-150 hover:text-ink">
        <ChevronDown size={15} className={`transition-transform duration-200 ease-out-quart ${more ? 'rotate-180' : ''}`} aria-hidden />
        More settings
      </button>
      <Collapse open={more}>
        <div className="grid gap-4 pt-3 sm:grid-cols-3">
          <NumberField label="Simulations per move" value={sims} onChange={setSims} min={2} max={2000}
            hint="More is stronger and slower." />
          <NumberField label="Test games per opponent" value={evalGames} onChange={setEvalGames} min={0} max={200}
            hint="Against the generation before and Greedy." />
        </div>
      </Collapse>
      <div className="mt-5 max-w-md">
        <Switch label="Use its best network as the Trained opponent in Play" checked={publishing} onChange={setPublish} />
        {trained && (
          <p className="mt-1 text-[12.5px] leading-snug text-ink-faint">
            {publishing ? `Replaces the network from the run “${trained.run ?? 'unknown'}” as it improves.`
              : `The Trained opponent keeps the network from the run “${trained.run ?? 'unknown'}”.`}
          </p>
        )}
      </div>
      {live && (
        <p className="mt-5 text-[13.5px] leading-snug text-ink">
          “{live.name}” is training now. Pause it first: one run at a time keeps each one fast.
        </p>
      )}
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-line pt-4">
        <Button variant="primary" size="sm" aria-busy={sending} onClick={start} icon={sending ? <Spinner /> : <Play size={14} />}>
          {sending ? 'Starting' : 'Start training'}
        </Button>
        {onCancel && <Button size="sm" onClick={onCancel}>Cancel</Button>}
        {estimate && (
          <p className="text-[12.5px] text-ink-faint">
            {total.toLocaleString()} self-play games, roughly {estimate} on this machine.
          </p>
        )}
      </div>
      {shown && <p key={shown} className="cz-swap mt-3 text-[13px] text-danger" role="alert">{shown}</p>}
    </section>
  )
}

/** Seconds per self-play game (at 50 simulations, generation included) from a run's own timings.
 * Only a run of real size counts: in a tiny test run the fixed costs swamp the per-game time. */
function measuredSecondsPerGame(detail: RunDetail | null): number {
  const rows = (detail?.generations ?? []).filter((g) => g.seconds && g.games)
  if (!detail || rows.length === 0 || detail.config.games_per_generation < 20 || detail.config.simulations < 20) {
    return FALLBACK_SECONDS_PER_GAME
  }
  const perGame = rows.reduce((t, g) => t + g.seconds! / g.games!, 0) / rows.length
  return perGame * (50 / (detail.config.simulations || 50))
}

/** The run selector: buttons for a few runs, a menu for more. */
function RunPicker({ runs, value, onChange }: { runs: RunStatus[]; value: string; onChange: (v: string) => void }) {
  const id = useId()
  if (runs.length <= 3) {
    return <Segmented label="Run" hideLabel value={value} onChange={onChange} size="sm" options={runs.map((r) => ({ value: r.name, label: r.name }))} />
  }
  return (
    <div>
      <label htmlFor={id} className="sr-only">Run</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}
        className="h-8 w-full rounded-lg border border-line bg-bg px-2.5 text-[13px] text-ink">
        {runs.map((r) => <option key={r.name} value={r.name}>{r.name}{r.live ? ' (training)' : ''}</option>)}
      </select>
    </div>
  )
}

export function TrainingView() {
  const [runs, setRuns] = useState<RunStatus[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const [creating, setCreating] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)

  const refresh = useCallback(async () => {
    try {
      const { runs } = await api.trainingRuns()
      setRuns(runs)
      const name = selected && runs.some((r) => r.name === selected) ? selected : runs[0]?.name ?? null
      if (name !== selected) setSelected(name)
      setDetail(name ? await api.trainingRun(name) : null)
      setError(null)
    } catch (err) {
      setError(err instanceof ApiError && err.status === 0
        ? 'The API server is not running. The README explains how to start it.'
        : err instanceof Error ? err.message : 'Something went wrong.')
    }
  }, [selected])

  useEffect(() => {
    void refresh()
  }, [refresh, tick])

  const live = detail?.status.live ?? false
  useEffect(() => {
    const changing = live && (!!detail?.status.pause_requested || detail?.status.state === 'starting')
    const id = setTimeout(() => setTick((t) => t + 1), changing ? CHANGING_POLL_MS : live ? LIVE_POLL_MS : IDLE_POLL_MS)
    return () => clearTimeout(id)
  }, [live, tick, detail])

  const secondsPerGame = useMemo(() => measuredSecondsPerGame(detail), [detail])
  const started = (name: string) => {
    scroller.current?.scrollTo({ top: 0, behavior: 'smooth' }) // the new run shows at the top
    setCreating(false)
    setSelected(name)
    setDetail(null)
    setTick((t) => t + 1)
  }
  const noRuns = runs !== null && runs.length === 0 && !error

  return (
    <div ref={scroller} className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8 sm:py-12">
        <header>
          <div className="flex items-center justify-between gap-4">
            <h1 className="font-serif text-[34px] leading-[1.1] font-semibold tracking-[-0.015em] text-ink sm:text-[42px]">
              Watch training
            </h1>
            {runs && runs.length > 0 && (
              <Button size="sm" onClick={() => setCreating((c) => !c)} aria-expanded={creating} aria-label="New run"
                className="shrink-0"
                icon={<Plus size={15} className={`transition-transform duration-200 ease-out-quart ${creating ? 'rotate-45' : ''}`} />}>
                <span className="hidden sm:inline">New run</span>
              </Button>
            )}
          </div>
          <p className="mt-3 max-w-[62ch] text-[15px] leading-relaxed text-ink-muted">
            The agent learns by playing itself. Each generation of the newest network plays a batch of self-play
            games with MCTS, trains on the search's choices and the results and is then tested against the
            generation before it and the Greedy baseline.
          </p>
          {runs && runs.length > 1 && selected && (
            <div className="mt-6 flex items-center gap-3">
              <span aria-hidden className="text-[13px] font-medium text-ink-muted">Run</span>
              <div className="w-80 max-w-full min-w-0">
                <RunPicker runs={runs} value={selected} onChange={(v) => { setSelected(v); setDetail(null) }} />
              </div>
            </div>
          )}
        </header>

        <div className="mt-8 space-y-4">
          {runs === null && !error && <div className="h-40 animate-pulse rounded-2xl bg-surface" aria-busy="true" />}
          {error && <Problem error={error} onRetry={() => setTick((t) => t + 1)} />}
          {noRuns && <NewRun runs={[]} first secondsPerGame={secondsPerGame} onStarted={started} />}
          {runs && runs.length > 0 && (
            <Collapse open={creating}>
              {/* Stays mounted so it can fold away; a fresh form after each new run. */}
              <NewRun key={runs.length} runs={runs} first={false} secondsPerGame={secondsPerGame} onStarted={started}
                onCancel={() => setCreating(false)} />
            </Collapse>
          )}
          {detail && selected && (
            <>
              <StatusPanel status={detail.status} detail={detail} onChange={(st) => {
                if (st) setDetail({ ...detail, status: st })
                setTick((t) => t + 1)
              }} />
              <Charts detail={detail} />
              <GenerationTable run={selected} detail={detail} />
              <p className="text-[12.5px] text-ink-faint">
                Network: {detail.config.blocks} residual blocks × {detail.config.channels} filters ·{' '}
                {detail.config.simulations} simulations per move · {detail.config.games_per_generation} games and{' '}
                up to {detail.config.steps_per_generation} training steps per generation.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
