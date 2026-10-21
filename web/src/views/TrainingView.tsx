/**
 * Watch training: a training run's progress, read from the API server (runs/<name>/ on disk).
 * Polls every few seconds while a run is live. A secondary screen for the team: the play and
 * learn screens stay the primary experience.
 */

import { Check, ChevronDown, Clock, ExternalLink, LoaderCircle, Pause, Play, Plus, RefreshCw, Trash2, X } from 'lucide-react'
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Legend, LineChart, type Series } from '../components/LineChart'
import { RunReport } from '../components/RunReport'
import { Button, Collapse, Segmented, Switch } from '../components/ui'
import { api, ApiError, type CloudStatus, type GenerationRow, type PauseWhen, type RunDetail, type RunStatus, type ServerStatus, type Where } from '../lib/api'
import { href } from '../lib/route'
import { useGame } from '../state/game'

const LIVE_POLL_MS = 2000
const IDLE_POLL_MS = 30000
// While a pause or a start is under way, so the change shows within a second or so.
const CHANGING_POLL_MS = 1200

const PHASE: Record<string, string> = {
  starting: 'Starting',
  selfplay: 'Playing self-play games',
  training: 'Training the network',
  evaluating: 'Testing the new network',
  generation_done: 'Generation finished',
  waiting: 'Waiting for Colab',
  uploading: 'Uploading to Kaggle',
  queued: 'Queued on Kaggle',
  cloud_running: 'Training on Kaggle',
  downloading: 'Bringing the run home',
  preparing: 'Setting up the server',
  reporting: 'Making the run report',
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

/** A progress bar. While work is under way a soft sheen runs along the filled part, so it never
 * looks frozen between updates; with no measure yet (`value` null) a short band sweeps across. */
function Progress({ value, label, active = false }: { value: number | null; label: string; active?: boolean }) {
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={value === null ? undefined : Math.round(value * 100)}
      className="relative h-1.5 overflow-hidden rounded-full bg-surface-2">
      {value === null ? (
        <div className="cz-sweep absolute inset-y-0 w-1/4 rounded-full bg-accent" />
      ) : (
        <div className="relative h-full overflow-hidden rounded-full bg-accent transition-[width] duration-700 ease-out-quart"
          style={{ width: `${Math.min(100, Math.max(0, value) * 100)}%` }}>
          {active && <div className="cz-sheen absolute inset-0" />}
        </div>
      )}
    </div>
  )
}

/** Seconds as "45 s" or "3 min 05 s". */
function clock(seconds: number) {
  if (seconds < 60) return `${seconds} s`
  return `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`
}

function remaining(seconds: number) {
  if (seconds < 60) return 'under a minute left'
  const m = Math.round(seconds / 60)
  return m === 1 ? 'about a minute left' : `about ${m} min left`
}

/** The time now, ticking every second while `active`: elapsed times count up between updates. */
function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [active])
  return now
}

const PHASE_NAME = { selfplay: 'self-play', training: 'training', evaluating: 'testing' } as const

/** Where a paused or stopped run is, in words. `lostTrack`: a run on your server that this Mac
 * stopped following while it trained (the Mac restarted, say): it may be training there still. */
function whereItStopped(status: RunStatus, lostTrack = false): string {
  if (lostTrack) {
    return 'This Mac stopped following the run, so it may still be training on the server. Connecting follows it again; '
      + 'if it has stopped there, Resume carries on from where it stopped.'
  }
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
  if (status.state === 'failed') {
    return status.error ? `Stopped with an error. ${status.error}` : 'Stopped with an error; the details are at the end of the run’s train.log.'
  }
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

type ControlMode = 'idle' | 'pausing' | 'starting' | 'reporting' | 'cloud' | 'waiting' | 'stopped' | 'finished'

/** The app's server: what it has and whether this Mac is connected to it (fetched when `wanted`). */
function useServer(wanted: boolean) {
  const [server, setServer] = useState<ServerStatus | null>(null)
  const refresh = useCallback(() => {
    void api.cloudStatus().then((c) => setServer(c.server), () => {})
  }, [])
  useEffect(() => {
    if (wanted) refresh()
  }, [wanted, refresh])
  return { server, setServer, refresh }
}

/** "krish@10.1.19.63": who the app signs in as, from the SSH config when it says. */
function serverLabel(server: Pick<ServerStatus, 'host' | 'hosts'> | null, host?: string | null): string {
  const name = host ?? server?.host ?? ''
  const known = server?.hosts.find((h) => h.name === name)
  if (!known?.user) return name
  return known.hostname === name ? `${known.user}@${name}` : `${name} (${known.user}@${known.hostname})`
}

/** Connect again to a server the app knows: just its password. */
function ServerReconnect({ host, label, note, onConnected }: {
  host: string; label: string; note: ReactNode; onConnected: (s: ServerStatus) => void
}) {
  const id = useId()
  const [password, setPassword] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const connect = async () => {
    if (sending) return
    setSending(true)
    setError(null)
    try {
      const cloud = await api.connectServer({ host, password: password || undefined })
      setPassword('')
      onConnected(cloud.server)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }
  return (
    <div className="rounded-xl border border-line bg-bg px-4 py-3">
      <p className="text-[13.5px] leading-snug text-ink">{note}</p>
      <form className="mt-2.5 flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); void connect() }}>
        <label htmlFor={id} className="sr-only">Password for {label}</label>
        <input id={id} type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="off"
          placeholder="Password"
          className="h-8 w-full max-w-[18rem] min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 text-[13.5px] text-ink placeholder:text-ink-faint focus-visible:border-accent" />
        <Button size="sm" type="submit" aria-busy={sending} icon={sending ? <Spinner /> : undefined}>
          {sending ? 'Connecting' : 'Connect'}
        </Button>
      </form>
      {error && <p className="cz-swap mt-2 text-[12.5px] text-danger">{error}</p>}
    </div>
  )
}

function RunControls({ status, total, done, lostTrack, onChange }: {
  status: RunStatus; total: number; done: number; lostTrack: boolean; onChange: (s?: RunStatus) => void
}) {
  const [choosing, setChoosing] = useState(false)
  const [sending, setSending] = useState<string | null>(null) // which control is waiting for the server
  const [error, setError] = useState<string | null>(null)
  const [more, setMore] = useState(10)
  // A run on your server resumes over the connection: without one, connect first.
  const onServer = status.where === 'server' && !status.live
  const { server, refresh: refreshServer } = useServer(onServer)
  const host = status.server_host ?? server?.host ?? null
  const disconnected = onServer && server !== null && !!host && !(server.connected && server.host === host)
  useEffect(() => {
    if (!disconnected) return
    setError(null) // the form below says it better
  }, [disconnected])
  const act = async (id: string, call: () => Promise<RunStatus>) => {
    if (sending) return
    setSending(id)
    setError(null)
    try {
      onChange(await call())
      setChoosing(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
      if (status.where === 'server') refreshServer() // the connection may have dropped: offer to connect
    } finally {
      setSending(null)
    }
  }
  const name = status.name
  const finished = status.state === 'finished' && !status.live
  const mode: ControlMode = status.live
    ? status.where === 'kaggle' && ['uploading', 'queued', 'cloud_running', 'downloading'].includes(status.state) ? 'cloud'
      : status.state === 'starting' || status.state === 'preparing' ? 'starting' : status.state === 'reporting' ? 'reporting'
        : status.pause_requested ? 'pausing' : 'idle'
    : status.state === 'waiting' ? 'waiting' : finished ? 'finished' : 'stopped'
  const [colabUrl, setColabUrl] = useState<string | null>(null)
  useEffect(() => {
    if (mode === 'waiting') void api.cloudStatus().then((c) => setColabUrl(c.colab.url), () => {})
  }, [mode])
  const target = Math.max(done + 1, total + more)

  let body: ReactNode
  if (mode === 'starting') {
    body = (
      <p className="flex items-center gap-2.5 text-[13.5px] text-ink" role="status">
        <Spinner />
        {status.state === 'preparing' ? 'Getting the server ready; the training starts there next.'
          : 'Starting the run: loading the network and its saved games.'}
      </p>
    )
  } else if (mode === 'cloud') {
    body = (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="text-[13.5px] leading-snug text-ink-muted">
          Kaggle runs the training by itself; to stop it early, stop the session on Kaggle.
        </p>
        {status.kernel_url && (
          <a href={status.kernel_url} target="_blank" rel="noopener noreferrer"
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong px-2.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ink-faint hover:bg-surface-2">
            <ExternalLink size={14} aria-hidden />
            Open on Kaggle
          </a>
        )}
      </div>
    )
  } else if (mode === 'waiting') {
    body = (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="text-[13.5px] leading-snug text-ink-muted">
          Open the notebook in Colab and press Run all; the run shows here as soon as it starts there.
        </p>
        {colabUrl && (
          <a href={colabUrl} target="_blank" rel="noopener noreferrer"
            className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-2.5 text-[13px] font-medium text-accent-ink transition-colors duration-150 hover:bg-accent-hover">
            <ExternalLink size={14} aria-hidden />
            Open in Colab
          </a>
        )}
      </div>
    )
  } else if (mode === 'reporting') {
    body = (
      <p className="flex items-center gap-2.5 text-[13.5px] text-ink" role="status">
        <Spinner />
        Training is done; making the run report and its strength check.
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
        {mode === 'stopped' && <p className="max-w-[70ch] text-[13.5px] leading-snug text-ink-muted">{whereItStopped(status, lostTrack)}</p>}
        {disconnected && host && (
          <ServerReconnect host={host} label={serverLabel(server, host)}
            note={<>This run trains on <span className="font-medium">{serverLabel(server, host)}</span>. Connect to it to{' '}
              {lostTrack ? 'follow the run again' : mode === 'finished' ? 'train further' : 'resume'}.</>}
            onConnected={() => {
              refreshServer()
              onChange()
            }} />
        )}
        <div className={`flex flex-wrap items-center gap-3 transition-opacity duration-200 ${disconnected ? 'pointer-events-none opacity-40' : ''}`}
          aria-disabled={disconnected || undefined}>
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
            onClick={() => act('resume', async () => {
              const resumed = await api.resumeRun(name, mode === 'finished' ? target : undefined)
              if (resumed.colab_url) window.open(resumed.colab_url, '_blank', 'noopener')
              return resumed
            })}>
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
  const state = !live && !['finished', 'failed', 'paused', 'waiting'].includes(status.state) ? 'stopped' : status.state
  // A run on your server that this Mac stopped following mid-run: it may be training there still.
  const lostTrack = status.where === 'server' && !live && !['finished', 'failed', 'paused', 'stopped'].includes(status.state)
  const lostConnection = status.where === 'server' && live && status.server_connected === false
  const pausing = live && !!status.pause_requested
  const total = status.generations || detail.config.generations
  const done = detail.generations.filter((g) => g.best !== undefined).length
  // This phase: what is happening, how far along (smoothly, by moves played against the moves
  // expected, as well as by finished games), how long it has taken and roughly how long is left.
  const now = useNow(live)
  const started = status.phase_started ? Date.parse(status.phase_started) : NaN
  const elapsed = Number.isNaN(started) ? null : Math.max(0, Math.round((now - started) / 1000))
  const gamesDone = status.games_done ?? 0
  const moves = status.moves ?? 0
  const byMoves = status.moves_expected ? Math.min(0.97, moves / status.moves_expected) : 0
  let phase = ''
  let phaseValue: number | null = 0
  if (live && status.state === 'selfplay' && status.games_total) {
    const warming = moves === 0 && gamesDone === 0
    phase = warming ? 'Starting the self-play processes'
      : `${gamesDone} of ${status.games_total} games · ${moves.toLocaleString()} moves played`
    phaseValue = warming ? null : gamesDone >= status.games_total ? 1 : Math.max(gamesDone / status.games_total, byMoves)
  } else if (live && status.state === 'training' && status.steps_total) {
    phase = `step ${status.step ?? 0} of ${status.steps_total}`
    phaseValue = (status.step ?? 0) / status.steps_total
  } else if (live && status.state === 'evaluating' && status.games_total) {
    const against = status.opponent === 'previous' ? 'the previous generation' : status.opponent === 'greedy' ? 'Greedy'
      : status.opponent === 'random' ? 'Random' : status.opponent
    phase = `against ${against}, ${gamesDone} of ${status.games_total} games · ${moves.toLocaleString()} moves played`
    phaseValue = gamesDone >= status.games_total ? 1 : Math.max(gamesDone / status.games_total, byMoves)
  } else if (live && ['uploading', 'queued', 'cloud_running', 'downloading'].includes(status.state)) {
    const checked = status.cloud_checked ? Math.max(0, Math.round((now - Date.parse(status.cloud_checked)) / 1000)) : null
    phase = status.state === 'uploading' ? 'Uploading the code and the run so far'
      : status.state === 'queued' ? 'Waiting for a Kaggle GPU'
        : status.state === 'downloading' ? 'Downloading the finished run'
          : `Checked ${checked === null ? 'just now' : checked < 90 ? `${checked} s ago` : `${Math.round(checked / 60)} min ago`}; its charts arrive with the results`
    phaseValue = null
  } else if (live && status.state === 'preparing') {
    phase = status.setup ?? 'Getting the server ready'
    phaseValue = null
  } else if (live && (status.state === 'reporting' || status.state === 'generation_done')) {
    phase = status.state === 'reporting' ? 'Making the run report' : 'Starting the next generation'
    phaseValue = null
  }
  // Time left: worked out from the last update (how long the phase had taken and how far it had
  // got then), then counted down between updates, so it moves steadily rather than flickering.
  const updated = status.updated ? Date.parse(status.updated) : NaN
  const atUpdate = Number.isNaN(updated) || Number.isNaN(started) ? null : Math.max(0, (updated - started) / 1000)
  let left = ''
  if (phaseValue !== null && phaseValue < 1 && elapsed !== null && elapsed >= 8) {
    if (phaseValue >= 0.97) left = status.state === 'training' ? '' : 'finishing the last games'
    else if (phaseValue >= 0.04 && atUpdate !== null) {
      left = remaining(Math.max(0, (atUpdate * (1 - phaseValue)) / phaseValue - (now - updated) / 1000))
    }
  }
  const timing = elapsed === null ? '' : left ? `${clock(elapsed)} · ${left}` : clock(elapsed)
  // The phase bar folds away when the run stops; keep its last reading on show while it does.
  const [shown, setShown] = useState({ phase, phaseValue, timing })
  if (phase && (phase !== shown.phase || phaseValue !== shown.phaseValue || timing !== shown.timing)) {
    setShown({ phase, phaseValue, timing })
  }
  const title = pausing && status.pause_requested === 'now' ? 'Pausing' : PHASE[state] ?? state
  const tone = !live || lostConnection ? (state === 'paused' ? 'bg-accent' : 'bg-ink-faint') : 'bg-success'
  const games = detail.generations.reduce((n, g) => n + (g.games ?? 0), 0)
  const positions = detail.generations.reduce((n, g) => n + (g.positions ?? 0), 0)
  const last = [...detail.generations].reverse().find((g) => g.eval.greedy)
  return (
    <section aria-label="Run status" className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <p className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 font-serif text-[22px] font-semibold text-ink">
          <span aria-hidden className={`relative inline-flex h-2.5 w-2.5 rounded-full transition-colors duration-500 ${tone}`}>
            {live && !lostConnection && <span className={`absolute inset-0 animate-ping rounded-full opacity-60 transition-colors duration-500 motion-reduce:hidden ${tone}`} />}
          </span>
          <span key={title} className="cz-swap">{title}</span>
          {live && status.generation > 0 && status.state !== 'generation_done' && status.state !== 'starting' && (
            <span className="font-sans text-[15px] font-normal whitespace-nowrap text-ink-muted">for generation {status.generation}</span>
          )}
        </p>
        <p className="text-[13px] text-ink-faint">
          {live ? lostConnection ? 'Not connected' : 'Live' : 'Not running'}
          {status.where === 'server' && status.server_host && ` · on ${status.server_host}`} · last update {ago(status.seconds_since_update)}
        </p>
      </div>
      <Collapse open={lostConnection}>
        <div className="pt-4">
          {status.server_host && (
            <ServerReconnect host={status.server_host} label={status.server_host}
              note="Lost the connection to the server. The run carries on there; connect again to follow it."
              onConnected={() => onChange()} />
          )}
        </div>
      </Collapse>
      <Collapse open={!!phase}>
        <div className="space-y-1.5 pt-3">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 text-[13px] tabular-nums">
            <p className="text-ink-muted">{shown.phase}</p>
            {shown.timing && <p className="text-ink-faint">{shown.timing}</p>}
          </div>
          <Progress value={shown.phaseValue} label="This phase" active={live} />
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
      {status.best_generation !== undefined && <UseInPlay status={status} onChange={onChange} />}
      <RunControls status={{ ...status, state }} total={total} done={done} lostTrack={lostTrack} onChange={onChange} />
    </section>
  )
}

/** Whether the run's best network is Play's Trained opponent, and keeps it as the run improves. */
function UseInPlay({ status, onChange }: { status: RunStatus; onChange: (s?: RunStatus) => void }) {
  const trained = useGame((s) => s.trained)
  const refreshTrained = useGame((s) => s.refreshTrained)
  const [on, setOn] = useState(status.publish)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => setOn(status.publish), [status.publish])
  const inPlay = !!trained && trained.run === status.name
  const set = async (value: boolean) => {
    setOn(value)
    setError(null)
    try {
      onChange(await api.publishRun(status.name, value))
      refreshTrained()
    } catch (err) {
      setOn(!value)
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    }
  }
  const note = on
    ? inPlay && trained?.generation != null
      ? `Play’s Trained opponent uses this run’s best network (generation ${trained.generation}) and follows it as the run improves.`
      : 'Play’s Trained opponent uses this run’s best network and follows it as the run improves.'
    : trained?.run
      ? `Play keeps the network from “${trained.run}”${trained.generation != null ? `, generation ${trained.generation}` : ''}.`
      : 'Play has no trained network yet.'
  return (
    <div className="mt-5 max-w-md border-t border-line pt-4">
      <Switch label="Use in Play" checked={on} onChange={(v) => void set(v)} />
      <p key={note} className="cz-swap mt-1 text-[12.5px] leading-snug text-ink-faint">{note}</p>
      {error && <p className="cz-swap mt-1 text-[12.5px] text-danger">{error}</p>}
    </div>
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
  const allEvalSeries: Series[] = [
    { id: 'greedy', label: 'vs Greedy', color: 'var(--chart-1)', kind: 'line',
      points: gens.filter((g) => g.eval.greedy).map((g) => ({ x: g.generation, y: g.eval.greedy.score * 100 })) },
    { id: 'previous', label: 'vs previous generation', color: 'var(--chart-2)', kind: 'line',
      points: gens.filter((g) => g.eval.previous).map((g) => ({ x: g.generation, y: g.eval.previous.score * 100 })) },
    { id: 'random', label: 'vs Random', color: 'var(--chart-3)', kind: 'line',
      points: gens.filter((g) => g.eval.random).map((g) => ({ x: g.generation, y: g.eval.random.score * 100 })) },
  ]
  const evalSeries = allEvalSeries.filter((series) => series.points.length > 0) // runs before Random was tested
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
      <ChartCard title="Value loss" note={`How far its prediction of the result is from how games ended. On new games it must beat the dashed line, the score of always predicting a draw.${
        detail.config.value_mix ? ' Training batches aim at the result averaged with the search’s own estimate, so their line sits lower.' : ''}`}>
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
          ...(gens.some((g) => g.eval.random) ? [{ label: 'vs Random', color: 'var(--chart-3)', kind: 'line' as const }] : []),
        ]} /></div>
      </ChartCard>
      <ChartCard title="Decisive self-play games" note="Games from the opening won by checkmate, bare king or stalemate rather than drawn (endgame starts are counted in the table). Decisive games teach the value head the most.">
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
      // A game from the opening (the Play board starts there); the shortest decisive one has the
      // most happening per move, else the first.
      const games = (await api.selfPlayGames(run, row.generation)).games.filter((g) => !g.endgame)
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
        <table className="w-full min-w-[960px] border-collapse text-left text-[13px] tabular-nums">
          <thead>
            <tr className="border-b border-line-strong text-[12px] text-ink-faint">
              {['Gen', 'From the opening W / D / B', 'Mean length', 'Endgame starts', 'Loss', 'Loss on new games', 'vs previous', 'vs Greedy', 'vs Random', ''].map((h) => (
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
                <td className="py-2 pr-4 text-ink-muted">
                  {g.endgames ? `${g.endgames.games - g.endgames.draws} of ${g.endgames.games} decisive` : '–'}
                </td>
                <td className="py-2 pr-4 text-ink-muted">{g.train_loss?.loss?.toFixed(2) ?? '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{g.held_out ? g.held_out.loss.toFixed(2) : '–'}</td>
                <td className="py-2 pr-4 text-ink-muted">{score(g, 'previous')}</td>
                <td className="py-2 pr-4 text-ink-muted">{score(g, 'greedy')}</td>
                <td className="py-2 pr-4 text-ink-muted">{score(g, 'random')}</td>
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

function CloudNote({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-line bg-bg px-4 py-3 text-[13px] leading-relaxed text-ink-muted">{children}</div>
}

function ColabNote({ cloud }: { cloud: CloudStatus | null }) {
  if (!cloud) return null
  if (!cloud.colab.drive) {
    return (
      <CloudNote>
        <p className="font-medium text-ink">Colab needs Google Drive for desktop on this Mac</p>
        <p className="mt-1">
          The run lives in your Google Drive, which carries it between Colab and this app. Install Google Drive for
          desktop, sign in with the Google account you use for Colab, then reload this page.
        </p>
      </CloudNote>
    )
  }
  return (
    <CloudNote>
      Starting opens the notebook in Colab: press <span className="font-medium text-ink">Run all</span> there (the first
      time, Colab asks to connect to GitHub and Google Drive). The run trains into your Drive, so its progress shows here,
      and Pause works from here too.
    </CloudNote>
  )
}

function KaggleNote({ cloud, onConnected }: { cloud: CloudStatus | null; onConnected: (c: CloudStatus) => void }) {
  const [username, setUsername] = useState('')
  const [token, setToken] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  if (!cloud) return null
  if (cloud.kaggle.connected) {
    return (
      <CloudNote>
        Runs on a Kaggle GPU from start to finish, as <span className="font-medium text-ink">{cloud.kaggle.username}</span>.
        This app uploads the code privately, checks on the run every minute and brings it home when it ends. Kaggle
        sessions stop at 12 hours, so a long run pauses itself after 11 and resumes in a new session.
      </CloudNote>
    )
  }
  const connect = async () => {
    if (sending) return
    setSending(true)
    setError(null)
    try {
      onConnected(await api.connectKaggle(username, token))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }
  return (
    <CloudNote>
      <p className="font-medium text-ink">Connect Kaggle</p>
      <p className="mt-1">
        On kaggle.com, open Settings, then API, and press Generate New Token. Paste it here with your Kaggle username; it is
        checked with Kaggle and kept on this Mac only.
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1.4fr_auto] sm:items-end">
        <label className="block text-[12.5px] font-medium text-ink-muted">
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value.trim())} autoComplete="off" spellCheck={false}
            className="mt-1 h-9 w-full rounded-lg border border-line bg-surface px-3 text-[14px] text-ink focus-visible:border-accent" />
        </label>
        <label className="block text-[12.5px] font-medium text-ink-muted">
          API token
          <input type="password" value={token} onChange={(e) => setToken(e.target.value.trim())} autoComplete="off"
            className="mt-1 h-9 w-full rounded-lg border border-line bg-surface px-3 text-[14px] text-ink focus-visible:border-accent" />
        </label>
        <Button size="sm" aria-busy={sending} onClick={connect} icon={sending ? <Spinner /> : undefined} className="h-9">
          {sending ? 'Checking' : 'Connect'}
        </Button>
      </div>
      {error && <p className="mt-2 text-[12.5px] text-danger">{error}</p>}
    </CloudNote>
  )
}

const FIELD = 'mt-1 h-9 w-full rounded-lg border border-line bg-surface px-3 text-[14px] text-ink placeholder:text-ink-faint focus-visible:border-accent'

/** Your own server, over SSH: connect with its address, a folder and its password (not kept),
 * then choose how many of its cores self-play uses. */
function ServerNote({ server, onChange, workers, onWorkers, gpuWorkers, onGpuWorkers }: {
  server: ServerStatus | null; onChange: (s: ServerStatus) => void; workers: number; onWorkers: (n: number) => void
  gpuWorkers: number; onGpuWorkers: (n: number) => void
}) {
  const [host, setHost] = useState(server?.host ?? '')
  const [folder, setFolder] = useState(server?.folder ?? '~/sopProject/chaturangzero')
  const [password, setPassword] = useState('')
  const [sending, setSending] = useState<'connect' | 'disconnect' | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (server?.host) setHost((h) => h || server.host!)
  }, [server?.host])
  if (!server) return null
  const act = async (which: 'connect' | 'disconnect', call: () => Promise<CloudStatus>) => {
    if (sending) return
    setSending(which)
    setError(null)
    try {
      onChange((await call()).server)
      setPassword('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(null)
    }
  }
  if (!server.tools) {
    return <CloudNote>Training on a server needs ssh and rsync on this Mac, and one of them is missing.</CloudNote>
  }
  if (server.connected && server.host) {
    const f = server.facts ?? {}
    const cores = f.cores ?? 1
    const has = [f.cores && `${f.cores} cores`, f.gpu ?? 'no NVIDIA GPU', f.memory_gb && `${f.memory_gb} GB of memory`].filter(Boolean)
    return (
      <CloudNote>
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
          <div className="min-w-0">
            <p className="flex items-center gap-2 font-medium text-ink">
              <span aria-hidden className="h-2 w-2 shrink-0 rounded-full bg-success" />
              Connected to {serverLabel(server)}
            </p>
            <p className="mt-0.5">{has.join(' · ')}</p>
          </div>
          <Button size="sm" aria-busy={sending === 'disconnect'} icon={sending === 'disconnect' ? <Spinner /> : undefined}
            onClick={() => act('disconnect', api.disconnectServer)}>
            Disconnect
          </Button>
        </div>
        <p className="mt-2.5">
          Everything goes in <span className="font-medium text-ink">{server.folder}</span> on the server. The first run
          there sets up Python in that folder, PyTorch included (a few GB, a few minutes, once). Training runs at low
          priority, so others using the machine come first, and carries on if this Mac sleeps or the connection drops.
        </p>
        <div className="mt-3 flex flex-wrap items-start gap-x-4 gap-y-3">
          <label className="block w-40 text-[12.5px] font-medium text-ink-muted">
            Self-play processes
            <input type="number" inputMode="numeric" min={1} max={cores} value={Number.isFinite(workers) ? workers : ''}
              onChange={(e) => onWorkers(e.target.value === '' ? NaN : Number(e.target.value))}
              className={`${FIELD} tabular-nums`} />
            <span className="mt-1 block text-[12px] font-normal text-ink-faint">One core each, of its {cores}.</span>
          </label>
          {f.gpu && (
            <label className="block w-40 text-[12.5px] font-medium text-ink-muted">
              Of them on the GPU
              <input type="number" inputMode="numeric" min={0} max={Number.isFinite(workers) ? workers : cores}
                value={Number.isFinite(gpuWorkers) ? gpuWorkers : ''}
                onChange={(e) => onGpuWorkers(e.target.value === '' ? NaN : Number(e.target.value))}
                className={`${FIELD} tabular-nums`} />
              <span className="mt-1 block text-[12px] font-normal text-ink-faint">
                Their network runs on the GPU, 32 games at a time; their search stays on their core.
              </span>
            </label>
          )}
        </div>
        {error && <p className="mt-2 text-[12.5px] text-danger">{error}</p>}
      </CloudNote>
    )
  }
  const connect = () => act('connect', () => api.connectServer({ host: host.trim(), folder: folder.trim(), password: password || undefined }))
  return (
    <CloudNote>
      <p className="font-medium text-ink">Connect to your server</p>
      <p className="mt-1">
        The run trains there over SSH, with Python, its packages and the runs all in one folder of yours. The password
        opens the connection and is not saved anywhere; if the connection drops, connect again. The training carries on
        there either way.
      </p>
      <form className="mt-3 space-y-3" onSubmit={(e) => { e.preventDefault(); void connect() }}>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block text-[12.5px] font-medium text-ink-muted">
            Server
            <input value={host} onChange={(e) => setHost(e.target.value.trim())} autoComplete="off" spellCheck={false}
              placeholder="user@address" className={FIELD} />
          </label>
          <label className="block text-[12.5px] font-medium text-ink-muted">
            Folder on the server
            <input value={folder} onChange={(e) => setFolder(e.target.value.trim())} autoComplete="off" spellCheck={false}
              className={FIELD} />
          </label>
        </div>
        {server.hosts.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-0.5 text-[12px] text-ink-faint">From your SSH config</span>
            {server.hosts.map((h) => {
              const on = host === h.name
              return (
                <button key={h.name} type="button" onClick={() => setHost(h.name)} aria-pressed={on}
                  className={`h-7 rounded-md border px-2 text-[12px] tabular-nums transition-colors duration-150 ${
                    on ? 'border-accent bg-accent-soft text-ink' : 'border-line text-ink-muted hover:border-line-strong hover:text-ink'
                  }`}>
                  {serverLabel(server, h.name)}
                </button>
              )
            })}
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
          <label className="block text-[12.5px] font-medium text-ink-muted">
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="off"
              className={FIELD} />
          </label>
          <Button size="sm" type="submit" aria-busy={sending === 'connect'} disabled={!host.trim()}
            icon={sending === 'connect' ? <Spinner /> : undefined} className="h-9">
            {sending === 'connect' ? 'Connecting' : 'Connect'}
          </Button>
        </div>
      </form>
      {error && <p key={error} className="cz-swap mt-2 text-[12.5px] text-danger">{error}</p>}
    </CloudNote>
  )
}

type NetworkSize = 'standard' | 'large'
const NETWORK_SIZES: Record<NetworkSize, { blocks: number; channels: number }> = {
  standard: { blocks: 6, channels: 64 },
  large: { blocks: 10, channels: 128 },
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
  // The server plays many games at once: 500 a generation gives the value head more games than it
  // can learn by heart. Elsewhere 50 keeps a run to hours. A number typed in stays.
  const [games, setGamesValue] = useState(50)
  const [gamesTyped, setGamesTyped] = useState(false)
  const setGames = (n: number) => {
    setGamesValue(n)
    setGamesTyped(true)
  }
  const [sims, setSims] = useState(50)
  const [evalGames, setEvalGames] = useState(20)
  const [size, setSize] = useState<NetworkSize>('standard')
  const [more, setMore] = useState(false)
  const trained = useGame((s) => s.trained)
  const refreshTrained = useGame((s) => s.refreshTrained)
  useEffect(() => refreshTrained(), [refreshTrained])
  // A new run takes over the Trained opponent only if no other run has given it a network yet.
  const [publish, setPublish] = useState<boolean | null>(null)
  const [autoReport, setAutoReport] = useState(false)
  const publishing = publish ?? trained === null
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [where, setWhere] = useState<Where>('mac')
  const chooseWhere = (w: Where) => {
    setWhere(w)
    if (!gamesTyped) setGamesValue(w === 'server' ? 500 : 50)
  }
  const [cloud, setCloud] = useState<CloudStatus | null>(null)
  const [workers, setWorkers] = useState(NaN)
  const [gpuWorkers, setGpuWorkers] = useState(0)
  useEffect(() => {
    void api.cloudStatus().then((c) => {
      setCloud(c)
      if (c.server.workers) setWorkers(c.server.workers)
      setGpuWorkers(c.server.gpu_workers ?? 0)
    }, () => {})
  }, [])
  const setServer = (server: ServerStatus) => {
    setCloud((c) => (c ? { ...c, server } : c))
    if (server.workers) setWorkers((w) => (Number.isFinite(w) && server.connected ? w : server.workers!))
  }
  const server = cloud?.server ?? null
  // One run trains on this Mac at a time, and one on the server; cloud runs use machines of their own.
  const live = where === 'mac' || where === 'server' ? runs.find((r) => r.live && r.where === where) : undefined
  const cores = server?.facts?.cores ?? 1

  const problem =
    !/^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$/.test(name) ? 'Name it with letters, digits, - and _, starting with a letter or digit.'
      : names.has(name) ? `There is already a run named “${name}”.`
        : !(generations >= 1 && generations <= 1000) ? 'Generations: 1 to 1,000.'
          : !(games >= 2 && games <= 1000) ? 'Games per generation: 2 to 1,000.'
            : !(sims >= 2 && sims <= 2000) ? 'Simulations per move: 2 to 2,000.'
              : !(evalGames >= 0 && evalGames <= 200) ? 'Test games: 0 to 200.'
                : where === 'server' && server?.connected && !(workers >= 1 && workers <= cores)
                  ? `Self-play processes: 1 to ${cores}.`
                  : where === 'server' && server?.connected && !(gpuWorkers >= 0 && gpuWorkers <= workers)
                    ? `On the GPU: 0 to ${workers} of the self-play processes.`
                    : null
  const [tried, setTried] = useState(false)
  const shown = tried ? problem ?? error : error

  const start = async () => {
    setTried(true)
    if (problem || sending) return
    setSending(true)
    setError(null)
    try {
      const started = await api.startRun({ name, generations, games_per_generation: games, simulations: sims,
        eval_games: evalGames, publish: publishing, auto_report: autoReport, where, ...NETWORK_SIZES[size],
        ...(where === 'server' && Number.isFinite(workers) ? { workers, gpu_workers: gpuWorkers } : {}) })
      if (started.colab_url) window.open(started.colab_url, '_blank', 'noopener')
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
      <div className="mt-5 max-w-md">
        <Segmented<Where> label="Where" value={where} onChange={chooseWhere} size="sm" options={[
          { value: 'mac', label: 'This Mac' },
          { value: 'colab', label: 'Colab', title: 'Google Colab, through Google Drive' },
          { value: 'kaggle', label: 'Kaggle', title: 'A Kaggle GPU, started and followed from here' },
          { value: 'server', label: 'Server', title: 'Your own machine, over SSH' },
        ]} />
      </div>
      <Collapse open={where !== 'mac'}>
        <div className="pt-3">
          {where === 'colab' && <ColabNote cloud={cloud} />}
          {where === 'kaggle' && <KaggleNote cloud={cloud} onConnected={setCloud} />}
          {where === 'server' && <ServerNote server={server} onChange={setServer} workers={workers} onWorkers={setWorkers}
            gpuWorkers={gpuWorkers} onGpuWorkers={setGpuWorkers} />}
        </div>
      </Collapse>
      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <div className="min-w-0">
          <label htmlFor="new-run-name" className="mb-1.5 block text-[13px] font-medium text-ink-muted">Name</label>
          <input id="new-run-name" value={name} onChange={(e) => setName(e.target.value.trim())} spellCheck={false} maxLength={40}
            className="h-9 w-full rounded-lg border border-line bg-bg px-3 text-[14px] text-ink transition-colors duration-150 hover:border-line-strong focus-visible:border-accent" />
        </div>
        <NumberField label="Generations" value={generations} onChange={setGenerations} min={1} max={1000} />
        <NumberField label="Games per generation" value={games} onChange={setGames} min={2} max={1000}
          hint={where === 'server' ? 'Hundreds, so the value head learns positions rather than games.' : undefined} />
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
            hint="Against the generation before, Greedy and Random." />
          <div className="min-w-0">
            <Segmented<NetworkSize> label="Network" value={size} onChange={setSize} size="sm" options={[
              { value: 'standard', label: '6 × 64', title: '6 residual blocks of 64 filters (1 M parameters)' },
              { value: 'large', label: '10 × 128', title: '10 residual blocks of 128 filters (3.5 M parameters)' },
            ]} />
            <p className="mt-1 text-[12px] text-ink-faint">
              {size === 'large' ? 'Learns more; each move costs about 3.5 times as much.' : 'Residual blocks × filters. A run keeps its size.'}
            </p>
          </div>
        </div>
      </Collapse>
      <div className="mt-5 max-w-md space-y-4">
        <div>
          <Switch label="Make the run report when it finishes" checked={autoReport} onChange={setAutoReport} />
          <p className="mt-1 text-[12.5px] leading-snug text-ink-faint">
            {autoReport ? 'Includes a strength check against Random and Greedy: a few more minutes at the end.'
              : 'Off: make it from the run’s report section when you want it.'}
          </p>
        </div>
        <div>
          <Switch label="Use its best network as the Trained opponent in Play" checked={publishing} onChange={setPublish} />
          {trained && (
            <p className="mt-1 text-[12.5px] leading-snug text-ink-faint">
              {publishing ? `Replaces the network from the run “${trained.run ?? 'unknown'}” as it improves.`
                : `The Trained opponent keeps the network from the run “${trained.run ?? 'unknown'}”.`}
            </p>
          )}
        </div>
      </div>
      {live && (
        <p className="mt-5 text-[13.5px] leading-snug text-ink">
          “{live.name}” is training {where === 'server' ? 'on the server' : ''} now. Pause it first: one run at a time keeps each one fast.
        </p>
      )}
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-line pt-4">
        <Button variant="primary" size="sm" aria-busy={sending} onClick={start} icon={sending ? <Spinner /> : <Play size={14} />}>
          {sending ? 'Starting' : where === 'colab' ? 'Start on Colab' : where === 'kaggle' ? 'Start on Kaggle'
            : where === 'server' ? 'Start on the server' : 'Start training'}
        </Button>
        {onCancel && <Button size="sm" onClick={onCancel}>Cancel</Button>}
        {estimate && where === 'mac' && (
          <p className="text-[12.5px] text-ink-faint">
            {total.toLocaleString()} self-play games, roughly {estimate} on this machine.
          </p>
        )}
        {where !== 'mac' && Number.isFinite(total) && total > 0 && (
          <p className="text-[12.5px] text-ink-faint">
            {total.toLocaleString()} self-play games on {where === 'colab' ? 'Colab' : where === 'kaggle' ? 'Kaggle'
              : server?.host ? serverLabel(server) : 'your server'}; this Mac stays free.
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

type RunKind = 'training' | 'starting' | 'waiting' | 'paused' | 'finished' | 'stopped' | 'failed'

function runKind(r: RunStatus): RunKind {
  if (r.live) return r.state === 'starting' || r.state === 'uploading' || r.state === 'preparing' ? 'starting' : 'training'
  if (r.state === 'waiting') return 'waiting'
  if (r.state === 'paused' || r.state === 'finished' || r.state === 'failed') return r.state
  return 'stopped' // the process ended without pausing
}

/** A run's state as a small coloured mark, the same in the menu and on its button. */
function RunMark({ kind }: { kind: RunKind }) {
  switch (kind) {
    case 'training':
      return (
        <span aria-hidden className="relative inline-flex h-2 w-2 shrink-0 rounded-full bg-success">
          <span className="absolute inset-0 animate-ping rounded-full bg-success opacity-60 motion-reduce:hidden" />
        </span>
      )
    case 'starting':
      return <LoaderCircle size={13} aria-hidden className="shrink-0 animate-spin text-success" />
    case 'paused':
      return <Pause size={13} aria-hidden className="shrink-0 fill-current text-accent" />
    case 'waiting':
      return <Clock size={13} aria-hidden className="shrink-0 text-accent" />
    case 'finished':
      return <Check size={14} aria-hidden className="shrink-0 text-ink-muted" strokeWidth={2.5} />
    case 'failed':
      return <X size={14} aria-hidden className="shrink-0 text-danger" strokeWidth={2.5} />
    default:
      return <span aria-hidden className="h-2 w-2 shrink-0 rounded-[2px] bg-ink-faint" />
  }
}

/**
 * The run on show: our own menu (a listbox), each run with its state. Opens below its button;
 * arrow keys, Home and End move, Enter or Space picks, Escape or a click elsewhere closes, and
 * typing a letter jumps to the next run whose name starts with it.
 */
function RunMenu({ runs, value, onChange }: { runs: RunStatus[]; value: string; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const id = useId()
  const current = runs.find((r) => r.name === value) ?? runs[0]
  const index = Math.max(0, runs.findIndex((r) => r.name === value))

  const show = () => {
    setActive(index)
    setOpen(true)
  }
  const close = (refocus = true) => {
    setOpen(false)
    if (refocus) button.current?.focus({ preventScroll: true })
  }
  const pick = (i: number) => {
    const run = runs[i]
    if (run && run.name !== value) onChange(run.name)
    close()
  }

  useEffect(() => {
    if (!open) return
    list.current?.focus({ preventScroll: true })
    const outside = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) close(false)
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open])

  useEffect(() => {
    if (open) list.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [open, active])

  const onListKey = (e: React.KeyboardEvent) => {
    const last = runs.length - 1
    const keys: Record<string, () => void> = {
      ArrowDown: () => setActive((a) => Math.min(last, a + 1)),
      ArrowUp: () => setActive((a) => Math.max(0, a - 1)),
      Home: () => setActive(0),
      End: () => setActive(last),
      Enter: () => pick(active),
      ' ': () => pick(active),
      Escape: () => close(),
      Tab: () => close(false),
    }
    if (keys[e.key]) {
      if (e.key !== 'Tab') e.preventDefault()
      keys[e.key]()
    } else if (e.key.length === 1 && /\S/.test(e.key)) {
      const k = e.key.toLowerCase()
      for (let step = 1; step <= runs.length; step++) {
        const i = (active + step) % runs.length
        if (runs[i].name.toLowerCase().startsWith(k)) {
          setActive(i)
          break
        }
      }
    }
  }

  // The state, and where it trains when that is not this Mac.
  const label = (r: RunStatus) =>
    `${runKind(r)}${r.where === 'colab' ? ' · Colab' : r.where === 'kaggle' ? ' · Kaggle' : r.where === 'server' ? ' · Server' : ''}`
  return (
    <div className="flex items-center gap-3">
      <span id={`${id}-label`} className="shrink-0 text-[13px] font-medium text-ink-muted">Run</span>
      <div ref={root} className="relative flex min-w-0 flex-1">
        <button ref={button} type="button" aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-list`}
          aria-labelledby={`${id}-label ${id}-value`}
          onClick={() => (open ? close() : show())}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.preventDefault()
              show()
            }
          }}
          className={`flex h-[34px] min-w-0 flex-1 items-center gap-2.5 rounded-lg border bg-bg pr-2.5 pl-3 text-left text-[13.5px] transition-colors duration-150 ${
            open ? 'border-accent' : 'border-line-strong hover:border-ink-faint'
          }`}>
          <RunMark kind={runKind(current)} />
          <span id={`${id}-value`} className="min-w-0 flex-1 truncate">
            <span className="font-medium text-ink">{current.name}</span>
            <span className="text-ink-faint"> · {label(current)}</span>
          </span>
          <ChevronDown size={15} aria-hidden
            className={`shrink-0 text-ink-muted transition-transform duration-200 ease-out-quart ${open ? 'rotate-180' : ''}`} />
        </button>
        {open && (
          <ul ref={list} id={`${id}-list`} role="listbox" tabIndex={-1} aria-labelledby={`${id}-label`}
            aria-activedescendant={`${id}-opt-${active}`} onKeyDown={onListKey}
            className="cz-menu absolute top-full left-0 z-[var(--z-popover)] mt-1.5 w-max min-w-full max-w-[min(26rem,calc(100vw-2.5rem))] max-h-72 overflow-y-auto rounded-xl border border-line-strong bg-surface p-1 shadow-[0_18px_44px_-18px_rgb(0_0_0/0.6)] outline-none">
            {runs.map((r, i) => {
              const selected = r.name === value
              return (
                <li key={r.name} id={`${id}-opt-${i}`} data-index={i} role="option" aria-selected={selected}
                  onPointerMove={() => setActive(i)} onClick={() => pick(i)}
                  className={`flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] ${
                    i === active ? 'bg-surface-2' : ''
                  }`}>
                  <span className="grid w-3.5 shrink-0 place-items-center"><RunMark kind={runKind(r)} /></span>
                  <span className={`min-w-0 flex-1 truncate ${selected ? 'font-medium text-ink' : 'text-ink'}`}>{r.name}</span>
                  <span className="shrink-0 text-[12.5px] text-ink-faint">{label(r)}</span>
                  <span className="grid w-4 shrink-0 place-items-center">
                    {selected && <Check size={14} aria-hidden className="text-accent" strokeWidth={2.5} />}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}

/** Delete the run on show, after an inline confirmation: everything it saved goes for good. */
function DeleteRun({ run, trainedRun, open, onOpen, onDeleted }: {
  run: RunStatus; trainedRun: string | null | undefined; open: boolean; onOpen: (open: boolean) => void
  onDeleted: () => void
}) {
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const remove = async () => {
    if (sending) return
    setSending(true)
    setError(null)
    try {
      await api.deleteRun(run.name)
      onOpen(false)
      onDeleted()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }
  return (
    <Collapse open={open}>
      <div className="mt-3 rounded-xl border border-danger/40 bg-danger-soft px-4 py-3.5">
        {run.live ? (
          <p className="text-[13.5px] leading-snug text-ink">
            “{run.name}” is training. Pause it before deleting it.
          </p>
        ) : (
          <>
            <p className="text-[13.5px] leading-snug text-ink">
              Delete the run “{run.name}”? Its self-play games, networks and charts are removed for good.
              {trainedRun === run.name && ' The Trained opponent keeps the network it already has from this run.'}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button size="sm" variant="danger" aria-busy={sending} onClick={remove}
                icon={sending ? <Spinner /> : <Trash2 size={14} />}>
                {sending ? 'Deleting' : 'Delete run'}
              </Button>
              <Button size="sm" onClick={() => onOpen(false)}>Keep it</Button>
            </div>
          </>
        )}
        {error && <p className="mt-2 text-[12.5px] text-danger">{error}</p>}
      </div>
    </Collapse>
  )
}

export function TrainingView() {
  const [runs, setRuns] = useState<RunStatus[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const trained = useGame((s) => s.trained)
  const refreshTrained = useGame((s) => s.refreshTrained)
  useEffect(() => refreshTrained(), [refreshTrained])
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
    const changing = live && (!!detail?.status.pause_requested || detail?.status.state === 'starting'
      || detail?.status.state === 'preparing')
    const id = setTimeout(() => setTick((t) => t + 1), changing ? CHANGING_POLL_MS : live ? LIVE_POLL_MS : IDLE_POLL_MS)
    return () => clearTimeout(id)
  }, [live, tick, detail])

  const secondsPerGame = useMemo(() => measuredSecondsPerGame(detail), [detail])
  const started = (name: string) => {
    // The new run shows at the top (on narrow screens the page around this view scrolls instead).
    scroller.current?.scrollTo({ top: 0, behavior: 'smooth' })
    scroller.current?.closest('main')?.scrollTo({ top: 0, behavior: 'smooth' })
    setCreating(false)
    setSelected(name)
    setDetail(null)
    setTick((t) => t + 1)
  }
  const noRuns = runs !== null && runs.length === 0 && !error
  const selectedRun = runs?.find((r) => r.name === selected) ?? null

  return (
    <div ref={scroller} className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8 sm:py-12">
        <header>
          <h1 className="font-serif text-[34px] leading-[1.1] font-semibold tracking-[-0.015em] text-ink sm:text-[42px]">
            Watch training
          </h1>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-muted">
            The agent learns by playing itself. Each generation of the newest network plays a batch of self-play
            games with MCTS, trains on the search's choices and the results and is then tested against the
            generation before it and the Greedy baseline.
          </p>
        </header>

        {/* Which run is on show, starting another and deleting this one: one bar above the run. */}
        {runs && runs.length > 0 && (
          <div className="mt-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="w-full min-w-0 max-w-xs flex-1 sm:flex-none">
                {selected && <RunMenu runs={runs} value={selected} onChange={(v) => {
                  setSelected(v)
                  setDetail(null)
                  setDeleting(false)
                }} />}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {selectedRun && (
                  <Button size="sm" onClick={() => { setDeleting((d) => !d); setCreating(false) }} aria-expanded={deleting}
                    icon={<Trash2 size={14} />} className="hover:text-danger">
                    Delete
                  </Button>
                )}
                <Button size="sm" onClick={() => { setCreating((c) => !c); setDeleting(false) }} aria-expanded={creating}
                  icon={<Plus size={15} className={`transition-transform duration-200 ease-out-quart ${creating ? 'rotate-45' : ''}`} />}>
                  New run
                </Button>
              </div>
            </div>
            {selectedRun && (
              <DeleteRun run={selectedRun} trainedRun={trained?.run} open={deleting} onOpen={setDeleting}
                onDeleted={() => {
                  setSelected(null)
                  setDetail(null)
                  setTick((t) => t + 1)
                }} />
            )}
          </div>
        )}

        <div className={`${runs && runs.length > 0 ? 'mt-4' : 'mt-8'} space-y-4`}>
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
              {!detail.status.live && detail.generations.some((g) => g.best !== undefined) && (
                <RunReport run={selected} finished={detail.status.state === 'finished'} gens={detail.generations}
                  generations={detail.generations.filter((g) => g.best !== undefined).length} />
              )}
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
