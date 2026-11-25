/**
 * Results: games people played against the computer in Play with recording on, and how they are
 * collected from teammates' laptops (a file each, exported and imported, or committed).
 */

import { Check, Download, ExternalLink, FileUp, Info, LoaderCircle, Play, RefreshCw, Upload } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Button } from '../components/ui'
import { api, ApiError, type HumanGame, type HumanGames, type ImportResult, type OpponentTally, type Tally, type TrainedInfo } from '../lib/api'
import { href } from '../lib/route'
import { useGame } from '../state/game'

const RELEASE_URL = 'https://github.com/Aditya-00786/csf407-fable-chaturanga-zero/releases/tag/milestone-4'
const RELEASE_FILE = 'chaturangzero-main-v4-gen100.pt'
const GAMES_SHOWN = 10

const REASONS: Record<string, string> = {
  bare_king: 'bare king', checkmate: 'checkmate', stalemate: 'stalemate', repetition: 'repetition',
  no_progress: '70-move rule', max_plies: '300-ply limit', resigned: 'resigned', left: 'left the game',
}
const RESULT = { win: 'Won', draw: 'Drew', loss: 'Lost' } as const

function Spinner() {
  return <LoaderCircle size={14} className="shrink-0 animate-spin" aria-hidden />
}

function Card({ title, aside, children, label }: { title: string; aside?: ReactNode; children: ReactNode; label: string }) {
  return (
    <section aria-labelledby={label} className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={label} className="font-serif text-[22px] font-semibold text-ink">{title}</h2>
        {aside && <span className="text-[13px] text-ink-faint tabular-nums">{aside}</span>}
      </div>
      {children}
    </section>
  )
}

const percent = (x: number | null) => (x === null ? '–' : `${Math.round(x * 100)}%`)
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

/** Wins, draws and losses as one bar, from the people's side. */
function ScoreBar({ t }: { t: Tally }) {
  if (!t.games) return null
  const part = (n: number) => `${(n / t.games) * 100}%`
  return (
    <span className="hidden h-2 w-20 shrink-0 overflow-hidden rounded-full bg-surface-2 sm:flex md:w-24" aria-hidden>
      <span className="bg-success" style={{ width: part(t.win) }} />
      <span className="bg-ink-faint/60" style={{ width: part(t.draw) }} />
      <span className="bg-danger/80" style={{ width: part(t.loss) }} />
    </span>
  )
}

function TallyCells({ t }: { t: Tally }) {
  return (
    <>
      <td className="py-2.5 pr-2.5 text-right sm:pr-4">{t.games}</td>
      <td className="py-2.5 pr-2.5 text-right sm:pr-4">{t.win}</td>
      <td className="py-2.5 pr-2.5 text-right sm:pr-4">{t.draw}</td>
      <td className="py-2.5 pr-2.5 text-right sm:pr-4">{t.loss}</td>
      <td className="py-2.5">
        <span className="flex items-center gap-2.5">
          <ScoreBar t={t} />
          <span className="w-9 text-right">{percent(t.score)}</span>
        </span>
      </td>
    </>
  )
}

function ResultsTable({ data }: { data: HumanGames }) {
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full min-w-[280px] border-collapse text-left text-[13px] tabular-nums sm:text-[13.5px]">
        <thead>
          <tr className="border-b border-line-strong text-[12px] text-ink-faint">
            <th scope="col" className="py-2 pr-2.5 font-medium sm:pr-4">Opponent</th>
            <th scope="col" className="py-2 pr-2.5 text-right font-medium sm:pr-4">Games</th>
            <th scope="col" className="py-2 pr-2.5 text-right font-medium sm:pr-4">Won</th>
            <th scope="col" className="py-2 pr-2.5 text-right font-medium sm:pr-4">Drawn</th>
            <th scope="col" className="py-2 pr-2.5 text-right font-medium sm:pr-4">Lost</th>
            <th scope="col" className="py-2 font-medium"><span className="sm:hidden">Score</span><span className="hidden sm:inline">People's score</span></th>
          </tr>
        </thead>
        {data.opponents.map((o) => (
          <tbody key={`${o.opponent}/${o.level}/${o.network?.fingerprint ?? o.label}`} className="border-b border-line">
            <tr className="text-ink">
              <th scope="row" className="py-2.5 pr-2.5 font-medium sm:pr-4">
                <OpponentName o={o} />
              </th>
              <TallyCells t={o} />
            </tr>
          </tbody>
        ))}
      </table>
      <p className="mt-3 text-[12.5px] leading-snug text-ink-faint">
        Won, drawn and lost are the people's: a win is a game the person won. The score counts a draw as half
        a win. Resigning, or leaving a recorded game before its end, counts as a loss.
      </p>
    </div>
  )
}

function OpponentName({ o }: { o: OpponentTally }) {
  if (o.opponent !== 'zero') return <>{o.label}</>
  const [name, network] = o.label.split(', ')
  return (
    <span className="flex flex-wrap items-baseline gap-x-2">
      {name}
      <span className={`text-[12px] font-normal ${o.milestone ? 'text-accent' : 'text-ink-faint'}`}>{network}</span>
    </span>
  )
}

function GamesTable({ games }: { games: HumanGame[] }) {
  const [all, setAll] = useState(false)
  const watchGame = useGame((s) => s.watchGame)
  const shown = all ? games : games.slice(0, GAMES_SHOWN)
  const name = (id: string | null) => (id ? id[0].toUpperCase() + id.slice(1) : '')
  const opponent = (g: HumanGame) => (g.opponent === 'zero' ? `Trained · ${name(g.level)}` : g.opponent === 'mcts' ? 'MCTS' : name(g.opponent))
  return (
    <>
      <div className="relative overflow-x-auto">
        <table className="w-full min-w-[440px] border-collapse text-left text-[13px] tabular-nums">
          <thead>
            <tr className="border-b border-line-strong text-[12px] text-ink-faint">
              <th scope="col" className="py-2 pr-4 font-medium">Played as</th>
              <th scope="col" className="py-2 pr-4 font-medium">Opponent</th>
              <th scope="col" className="py-2 pr-4 font-medium">Result</th>
              <th scope="col" className="py-2 pr-4 text-right font-medium">Plies</th>
              <th scope="col" className="py-2"><span className="sr-only">Replay</span></th>
            </tr>
          </thead>
          <tbody>
            {shown.map((g) => (
              <tr key={g.id} className="border-b border-line text-ink-muted">
                <td className="py-2 pr-4 text-ink">
                  <span className="flex items-center gap-2">
                    <span aria-hidden className={`inline-block h-3 w-3 shrink-0 rounded-full ring-1 ${
                      g.human === 'white' ? 'bg-[#f1e6cc] ring-[#8a6a45]' : 'bg-[#2b1e19] ring-[#c9a14c]'
                    }`} />
                    {name(g.human)}
                  </span>
                </td>
                <td className="py-2 pr-4 whitespace-nowrap">{opponent(g)}</td>
                <td className="py-2 pr-4">
                  <span className={g.result === 'win' ? 'font-medium text-success' : g.result === 'loss' ? 'text-danger' : 'text-ink'}>
                    {RESULT[g.result]}
                  </span>
                  <span className="text-ink-faint"> · {REASONS[g.reason] ?? g.reason}</span>
                  {g.takebacks > 0 && <span className="text-ink-faint" title="Moves taken back with Undo"> · {plural(g.takebacks, 'take-back')}</span>}
                </td>
                <td className="py-2 pr-4 text-right">{g.moves.length}</td>
                <td className="py-2 text-right">
                  <Button size="sm" variant="ghost" icon={<Play size={13} />} disabled={g.moves.length === 0}
                    aria-label={`Replay the game played as ${g.human} against ${opponent(g)}, ${RESULT[g.result].toLowerCase()}`}
                    onClick={() => {
                      watchGame(g.moves, 'results')
                      location.hash = href('play')
                    }}>
                    Replay
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {games.length > GAMES_SHOWN && (
        <div className="mt-3">
          <Button size="sm" variant="ghost" onClick={() => setAll((a) => !a)} aria-expanded={all}>
            {all ? `Show the latest ${GAMES_SHOWN}` : `Show all ${games.length}`}
          </Button>
        </div>
      )}
    </>
  )
}

/** The trained opponent this computer plays, and loading the Milestone 4 network where it has none. */
function Opponent({ trained, onLoaded }: { trained: TrainedInfo | null | undefined; onLoaded: () => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null)
  const load = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setMessage(null)
    try {
      const info = await api.loadNetwork(file)
      setMessage({
        ok: true,
        text: info.milestone
          ? `Loaded the ${info.milestone} agent. Play's Trained opponent uses it from its next move.`
          : `Loaded generation ${info.generation ?? '?'} of “${info.run ?? 'unknown'}”. It is not the Milestone 4 agent, so its games are counted apart.`,
      })
      onLoaded()
    } catch (err) {
      setMessage({ ok: false, text: err instanceof Error ? err.message : 'The file could not be loaded.' })
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }
  const milestone = trained?.milestone
  const picker = (
    <>
      <input ref={input} type="file" accept=".pt" className="sr-only" tabIndex={-1} aria-hidden
        onChange={(e) => void load(e.target.files?.[0])} />
      <Button size="sm" variant={milestone ? 'ghost' : 'primary'} onClick={() => input.current?.click()} aria-busy={busy}
        disabled={busy} icon={busy ? <Spinner /> : <FileUp size={14} />}>
        {busy ? 'Loading' : milestone ? 'Load another network file' : 'Load network file'}
      </Button>
    </>
  )
  return (
    <section aria-label="The trained opponent" className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 items-start gap-3">
          <span className={`mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full ${milestone ? 'bg-success-soft text-success' : 'bg-accent-soft text-accent'}`}>
            {milestone ? <Check size={17} /> : <Info size={17} />}
          </span>
          <div className="min-w-0">
            <p className="font-serif text-[19px] leading-snug font-semibold text-ink">
              {trained === undefined ? 'Checking the trained opponent'
                : milestone ? `Play's trained opponent is the ${milestone} agent`
                  : trained ? "Play's trained opponent is not the Milestone 4 agent"
                    : 'This computer has no trained network yet'}
            </p>
            <p className="mt-1 max-w-[68ch] text-[13.5px] leading-relaxed text-ink-muted">
              {milestone ? <>Generation {trained?.generation} of “{trained?.run}”, the network frozen for the milestone, recognised by its file's fingerprint ({trained?.fingerprint?.slice(0, 12)}…).</>
                : trained ? <>It plays generation {trained.generation ?? '?'} of “{trained.run ?? 'unknown'}”. Games against it are counted apart from the Milestone 4 agent's.</>
                  : <>Games against Random, Greedy and MCTS can be recorded already; the Trained opponent needs a network.</>}
            </p>
          </div>
        </div>
        {milestone && <div className="shrink-0">{picker}</div>}
      </div>
      {trained !== undefined && !milestone && (
        <ol className="mt-5 grid gap-3 sm:grid-cols-2">
          <li className="rounded-xl bg-bg p-4">
            <p className="text-[12px] font-medium text-ink-faint">Step 1</p>
            <p className="mt-1 text-[13.5px] leading-snug text-ink">
              Download <span className="font-mono text-[12.5px] break-all">{RELEASE_FILE}</span> (14 MB) from the
              project's GitHub release <span className="font-mono text-[12.5px]">milestone-4</span>.
            </p>
            <a href={RELEASE_URL} target="_blank" rel="noreferrer"
              className="mt-3 inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong px-2.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ink-faint hover:bg-surface-2">
              <ExternalLink size={14} /> Open the release
            </a>
          </li>
          <li className="rounded-xl bg-bg p-4">
            <p className="text-[12px] font-medium text-ink-faint">Step 2</p>
            <p className="mt-1 text-[13.5px] leading-snug text-ink">Choose the downloaded file. It is checked before Play uses it.</p>
            <div className="mt-3">{picker}</div>
          </li>
        </ol>
      )}
      {message && (
        <p role="status" className={`mt-4 text-[13px] leading-snug ${message.ok ? 'text-success' : 'text-danger'}`}>{message.text}</p>
      )}
    </section>
  )
}

/** Sending games from a teammate's laptop, and collecting them on another. */
function Share({ data, onImported }: { data: HumanGames; onImported: () => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ImportResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const importFiles = async (list: FileList | null) => {
    const files = Array.from(list ?? [])
    if (!files.length) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const texts = await Promise.all(files.map(async (f) => ({ name: f.name, text: await f.text() })))
      setResult(await api.importGames(texts))
      onImported()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The files could not be read.')
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }
  return (
    <Card title="Collect everyone's games" label="share-title">
      <div className="grid gap-3 md:grid-cols-3">
        <div className="flex flex-col rounded-xl bg-bg p-4">
          <h3 className="text-[14px] font-semibold text-ink">Send yours</h3>
          <p className="mt-1 flex-1 text-[13px] leading-snug text-ink-muted">
            One file with the {plural(data.total, 'game')} on this computer. Send it by email or chat to whoever
            collects the results.
          </p>
          <a href="/api/human-games/export" download aria-disabled={data.total === 0} tabIndex={data.total === 0 ? -1 : undefined}
            className={`mt-3 inline-flex h-8 items-center justify-center gap-1.5 self-start rounded-lg bg-accent px-2.5 text-[13px] font-medium text-accent-ink shadow-[0_1px_0_0_rgb(0_0_0/0.15)] transition-[background-color,transform] duration-150 hover:bg-accent-hover active:translate-y-px ${
              data.total === 0 ? 'pointer-events-none opacity-45' : ''}`}>
            <Download size={14} /> Export games
          </a>
        </div>
        <div className="flex flex-col rounded-xl bg-bg p-4">
          <h3 className="text-[14px] font-semibold text-ink">Add a teammate's</h3>
          <p className="mt-1 flex-1 text-[13px] leading-snug text-ink-muted">
            Choose the files teammates sent. Every game is replayed and checked; one already here is skipped.
          </p>
          <input ref={input} type="file" accept=".json,.jsonl,application/json" multiple className="sr-only" tabIndex={-1} aria-hidden
            onChange={(e) => void importFiles(e.target.files)} />
          <Button size="sm" onClick={() => input.current?.click()} disabled={busy} aria-busy={busy}
            icon={busy ? <Spinner /> : <Upload size={14} />} className="mt-3 self-start">
            {busy ? 'Importing' : 'Import files'}
          </Button>
        </div>
        <div className="flex flex-col rounded-xl bg-bg p-4">
          <h3 className="text-[14px] font-semibold text-ink">Or commit them</h3>
          <p className="mt-1 text-[13px] leading-snug text-ink-muted">
            Each game is a file of its own in <span className="font-mono text-[12px] break-all text-ink">{data.folder}/</span> in
            the project folder, so nobody's commit touches anyone else's. Commit yours and open a pull request; the
            guide in <span className="font-mono text-[12px] text-ink">docs/HUMAN_GAMES.md</span> has the steps.
          </p>
        </div>
      </div>
      {(result || error) && (
        <div role="status" className="mt-4 text-[13px] leading-snug">
          {error && <p className="text-danger">{error}</p>}
          {result && (
            <>
              <p className={result.added ? 'text-success' : 'text-ink-muted'}>
                {result.added ? `Added ${plural(result.added, 'game')}.` : 'No new games.'}
                {result.already > 0 && ` ${plural(result.already, 'game was', 'games were')} here already.`}
              </p>
              {result.rejected.length > 0 && (
                <ul className="mt-1.5 space-y-0.5 text-danger">
                  {result.rejected.slice(0, 6).map((r, i) => (
                    <li key={i}>Turned away: {r.file}{r.game ? `, game ${r.game}` : ''}: {r.reason}.</li>
                  ))}
                  {result.rejected.length > 6 && <li>And {result.rejected.length - 6} more.</li>}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </Card>
  )
}

export function ResultsView() {
  const [data, setData] = useState<HumanGames | null>(null)
  const [error, setError] = useState<string | null>(null)
  const trained = useGame((s) => s.trained)
  const refreshTrained = useGame((s) => s.refreshTrained)

  const refresh = useCallback(async () => {
    try {
      setData(await api.humanGames())
      setError(null)
    } catch (err) {
      setError(err instanceof ApiError && err.status === 0
        ? 'The API server is not running. The README explains how to start it.'
        : err instanceof Error ? err.message : 'Something went wrong.')
    }
  }, [])
  useEffect(() => {
    refreshTrained()
    void refresh()
    // Games recorded in another tab show up on coming back to this one.
    const onFocus = () => void refresh()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [refresh, refreshTrained])

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 py-10 sm:px-8 sm:py-12">
        <header className="max-w-[68ch]">
          <h1 className="font-serif text-[34px] leading-[1.1] font-semibold tracking-[-0.015em] text-ink sm:text-[42px]">
            People against the agent
          </h1>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-muted">
            Games played in Play against the computer with recording on. The rules engine replays every game
            before it is kept and works out the result itself, so each one here is a legal, finished game.
          </p>
        </header>

        <div className="mt-8 space-y-4">
          <Opponent trained={trained} onLoaded={refreshTrained} />
          {error && (
            <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8">
              <p className="font-serif text-[20px] font-semibold text-ink">Cannot load the recorded games</p>
              <p className="mt-2 text-[14px] text-ink-muted">{error}</p>
              <div className="mt-4"><Button size="sm" onClick={() => void refresh()} icon={<RefreshCw size={14} />}>Try again</Button></div>
            </section>
          )}
          {!data && !error && <div className="h-48 animate-pulse rounded-2xl bg-surface" aria-busy="true" />}
          {data && (
            <>
              <Card title="Results" label="results-title"
                aside={data.total > 0 && plural(data.total, 'game')}>
                {data.total > 0 ? <ResultsTable data={data} /> : (
                  <div className="rounded-xl bg-bg p-5">
                    <p className="text-[14px] font-medium text-ink">No games recorded yet</p>
                    <p className="mt-1 max-w-[62ch] text-[13.5px] leading-relaxed text-ink-muted">
                      In Play, choose <b className="font-medium text-ink">Vs computer</b>, then turn on <b className="font-medium text-ink">Record</b> in
                      the quick actions (the ‹ button) or in the Game panel. Each game you finish then shows up here.
                    </p>
                    <a href={href('play')} className="mt-3 inline-flex text-[13.5px] font-medium text-accent underline-offset-4 hover:underline">Go to Play</a>
                  </div>
                )}
              </Card>
              {data.total > 0 && (
                <Card title="Games" label="games-title" aside="most recently added first">
                  <GamesTable games={data.games} />
                </Card>
              )}
              <Share data={data} onImported={() => void refresh()} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
