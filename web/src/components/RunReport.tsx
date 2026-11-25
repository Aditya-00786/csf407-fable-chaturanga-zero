/**
 * The run report on the Training page, for a run that is not training: what it learned, read from
 * what it saved. Verdicts from its metrics, whether its value head sees material, an optional
 * strength check (the untrained, best and final networks against Random and Greedy, played in the
 * background), the rating ladder and conversion test (played in the background too; a server run's
 * jobs play on its server), and its charts as an image to download.
 */

import { Check, Download, LoaderCircle, Minus, Play, RefreshCw, X } from 'lucide-react'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  api, type GenerationRow, type Ladder, type LadderPlan, type LadderRating, type RunReport as Report, type RunSummary,
  type StrengthCheck, type Verdict,
} from '../lib/api'
import { Legend, LineChart } from './LineChart'
import { StackedBars, type StackKey } from './StackedBars'
import { Button } from './ui'

const pct = (v: number) => `${Math.round(v * 100)}%`
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`
const NETWORK: Record<string, string> = { untrained: 'Untrained', best: 'Best', final: 'Final' }

type Tone = 'good' | 'bad' | 'neutral'

function Mark({ tone }: { tone: Tone }) {
  const cls = 'h-5 w-5 shrink-0 rounded-full p-[3px]'
  if (tone === 'good') return <Check aria-hidden strokeWidth={3} className={`${cls} bg-success-soft text-success`} />
  if (tone === 'bad') return <X aria-hidden strokeWidth={3} className={`${cls} bg-danger-soft text-danger`} />
  return <Minus aria-hidden strokeWidth={3} className={`${cls} bg-surface-2 text-ink-muted`} />
}

function Finding({ title, tone, headline, children }: { title: string; tone: Tone; headline: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-bg p-4">
      <p className="text-[12.5px] font-medium text-ink-faint">{title}</p>
      <p className="mt-1.5 flex items-start gap-2 text-[15px] font-semibold leading-snug text-ink">
        <Mark tone={tone} />
        {headline}
      </p>
      <div className="mt-2 space-y-0.5 text-[13px] leading-snug text-ink-muted tabular-nums">{children}</div>
    </div>
  )
}

const TONE: Record<Verdict, Tone> = { learned: 'good', partly: 'neutral', not_learned: 'bad' }

function change(first: number, last: number): { tone: Tone; word: string } {
  if (last - first >= 0.15) return { tone: 'good', word: 'Stronger' }
  if (first - last >= 0.15) return { tone: 'bad', word: 'Weaker' }
  return { tone: 'neutral', word: 'No clear change' }
}

function Findings({ s }: { s: RunSummary }) {
  const baselines = (['greedy', 'random'] as const).filter((k) => s.strength[k])
  const moves = baselines.map((k) => change(s.strength[k]!.first, s.strength[k]!.last))
  const strength = moves.some((m) => m.tone === 'good') && !moves.some((m) => m.tone === 'bad')
    ? { tone: 'good' as Tone, word: 'Stronger against the baselines' }
    : moves.some((m) => m.tone === 'bad') && !moves.some((m) => m.tone === 'good')
      ? { tone: 'bad' as Tone, word: 'Weaker against the baselines' }
      : { tone: 'neutral' as Tone, word: 'No clear change in strength' }
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {s.policy && (
        <Finding title="Policy: does it predict the search's moves?" tone={TONE[s.policy.verdict]}
          headline={s.policy.verdict === 'learned' ? 'Learned to predict the search' : 'Did not learn to predict the search'}>
          <p>Loss on unseen games {s.policy.first.toFixed(2)} → {s.policy.last.toFixed(2)}</p>
          {s.policy.floor !== null && <p>The lowest it can go is about {s.policy.floor.toFixed(2)}.</p>}
        </Finding>
      )}
      {s.value && (
        <Finding title="Value: can it judge a position?" tone={TONE[s.value.verdict]}
          headline={s.value.verdict === 'learned' ? 'Judges positions better than guessing a draw'
            : s.value.verdict === 'partly' ? 'Sometimes better than guessing a draw' : 'No better than guessing a draw'}>
          <p>Beat predicting a draw on unseen games in {s.value.beat_baseline} of {s.value.generations} generations</p>
          <p>Generation {s.value.last_generation}: {s.value.last.toFixed(3)} against {s.value.last_baseline.toFixed(3)} (lower is better)</p>
        </Finding>
      )}
      {baselines.length > 0 && (
        <Finding title="Strength in the run's tests" tone={strength.tone} headline={strength.word}>
          {baselines.map((k) => {
            const st = s.strength[k]!
            return (
              <p key={k}>
                vs {k === 'greedy' ? 'Greedy' : 'Random'}: {pct(st.first)} → {pct(st.last)}, best {pct(st.best)} (generation {st.best_generation})
              </p>
            )
          })}
        </Finding>
      )}
      {s.selfplay && (
        <Finding title="Self-play" tone="neutral" headline={`${s.selfplay.games.toLocaleString()} games played`}>
          <p>Decisive from the opening {pct(s.selfplay.decisive_first)} → {pct(s.selfplay.decisive_last)}</p>
          <p>Length {Math.round(s.selfplay.plies_first)} → {Math.round(s.selfplay.plies_last)} plies</p>
        </Finding>
      )}
    </div>
  )
}

function Probe({ probe }: { probe: NonNullable<Report['report']>['probe'] }) {
  const final = probe.networks[probe.networks.length - 1]
  const sees = final.material_gap > 0.2
  return (
    <div>
      <h3 className="font-serif text-[18px] font-semibold text-ink">Does the value head see material?</h3>
      <p className="mt-1 text-[13px] leading-snug text-ink-muted">
        The value each network gives these positions, White to move, from −1 (sure to lose) to +1 (sure to win). One that
        understands material rates a side ahead in pieces well above the same side behind.
      </p>
      <p className="mt-2 flex items-start gap-2 text-[14px] font-semibold text-ink">
        <Mark tone={sees ? 'good' : 'bad'} />
        {sees ? 'Yes: the final network rates being ahead in material above being behind'
          : 'Not yet: the final network rates being ahead about the same as being behind'}
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-left text-[13px] tabular-nums">
          <thead>
            <tr className="border-b border-line-strong text-[12px] text-ink-faint">
              <th scope="col" className="py-2 pr-4 font-medium">Position</th>
              {probe.networks.map((n) => (
                <th key={n.label} scope="col" className="py-2 pr-4 text-right font-medium">
                  {NETWORK[n.label]} <span className="text-ink-faint">(gen {n.generation})</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {probe.positions.map((name, i) => (
              <tr key={name} className="border-b border-line">
                <th scope="row" className="py-1.5 pr-4 font-normal text-ink-muted">{name}</th>
                {probe.networks.map((n) => <td key={n.label} className="py-1.5 pr-4 text-right text-ink">{signed(n.values[i])}</td>)}
              </tr>
            ))}
            <tr>
              <th scope="row" className="py-2 pr-4 font-medium text-ink">Ahead minus behind</th>
              {probe.networks.map((n) => <td key={n.label} className="py-2 pr-4 text-right font-semibold text-ink">{signed(n.material_gap)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Where a job plays, while it plays: its progress, time left, and the server's connection. */
function JobProgress({ job, server, reconnect }: {
  job: StrengthCheck | Ladder; server: string | null; reconnect: (note: string) => ReactNode
}) {
  const left = minutesLeft(job)
  const host = job.server_host ?? server
  return (
    <div className="mt-4 space-y-1.5" role="status">
      <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[13px] text-ink">
        <LoaderCircle size={14} aria-hidden className="animate-spin" />
        {job.total ? `Playing ${job.done.toLocaleString()} of ${job.total.toLocaleString()} games` : 'Starting'}
        {host && <span className="text-ink-faint">on {host}</span>}
        {left !== null && <span className="text-ink-faint">· about {left} min left</span>}
      </p>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out-quart"
          style={{ width: `${job.total ? (100 * job.done) / job.total : 2}%` }} />
      </div>
      {job.connected === false && (
        <div className="pt-2">{reconnect(`Not connected to ${host ?? 'the server'}. It plays on there; connect to follow it.`)}</div>
      )}
    </div>
  )
}

/** Minutes left at the pace so far, once enough games have finished to set one (they finish in
 * bursts, and the first wait for the networks to load). */
function minutesLeft(job: { started?: string; done: number; total: number }): number | null {
  if (!job.started || job.done < Math.max(10, job.total * 0.05) || job.total <= job.done) return null
  const spent = (Date.now() - new Date(job.started).getTime()) / 60_000
  return spent > 0 ? Math.max(1, Math.round((spent / job.done) * (job.total - job.done))) : null
}

/** A failed start: the server not connected gets its connect form, anything else its message. */
function StartError({ error, reconnect }: { error: string | null; reconnect: (note: string) => ReactNode }) {
  if (!error) return null
  if (error.startsWith('Not connected')) return <div className="mt-3">{reconnect(error)}</div>
  return <p className="mt-2 text-[12.5px] text-danger">{error}</p>
}

function Strength({ run, check, server, reconnect, onStarted }: {
  run: string; check: StrengthCheck | null; server: string | null; reconnect: (note: string) => ReactNode; onStarted: () => void
}) {
  const [games, setGames] = useState(20)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const start = async () => {
    if (sending) return
    setSending(true)
    setError(null)
    try {
      await api.startStrength(run, games)
      onStarted()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }
  const running = check?.state === 'running'
  const minutes = Math.max(1, Math.round((games * 3 * 2 * 5.5) / 6 / 60))
  return (
    <div>
      <h3 className="font-serif text-[18px] font-semibold text-ink">Strength check</h3>
      <p className="mt-1 text-[13px] leading-snug text-ink-muted">
        The untrained, best and final networks each play Random and Greedy from the opening, colours alternating. It
        shows whether training made the agent stronger than where it started.
      </p>
      {check && check.networks.length > 0 && Object.keys(check.results).length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[480px] border-collapse text-left text-[13px] tabular-nums">
            <thead>
              <tr className="border-b border-line-strong text-[12px] text-ink-faint">
                <th scope="col" className="py-2 pr-4 font-medium">Network</th>
                <th scope="col" className="py-2 pr-4 font-medium">vs Random</th>
                <th scope="col" className="py-2 font-medium">vs Greedy</th>
              </tr>
            </thead>
            <tbody>
              {check.networks.map((n) => (
                <tr key={n.label} className="border-b border-line">
                  <th scope="row" className="py-2 pr-4 font-medium text-ink">
                    {NETWORK[n.label]} <span className="font-normal text-ink-faint">(gen {n.generation})</span>
                  </th>
                  {(['random', 'greedy'] as const).map((o) => {
                    const r = check.results[n.label]?.[o]
                    return (
                      <td key={o} className="py-2 pr-4 text-ink-muted">
                        {r ? <><span className="font-semibold text-ink">{pct(r.score)}</span> · {r.wins} W / {r.draws} D / {r.losses} L</> : '–'}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {check && check.kept && check.kept.length > 0 && (
        <p className="mt-2 text-[12.5px] text-ink-faint">
          Kept from the last check, which played the same {check.kept.length > 1 ? 'networks' : 'network'} the same games:{' '}
          {check.networks.filter((n) => check.kept!.includes(n.label))
            .map((n) => `${NETWORK[n.label] ?? n.label} (gen ${n.generation})`).join(', ')}.
        </p>
      )}
      {running ? (
        <JobProgress job={check} server={server} reconnect={reconnect} />
      ) : (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-[13px] text-ink-muted">
            Games per match
            <input type="number" min={2} max={200} value={games} onChange={(e) => setGames(Math.max(2, Math.min(200, Number(e.target.value) || 2)))}
              className="h-8 w-20 rounded-lg border border-line bg-bg px-2 text-[13px] text-ink tabular-nums" />
          </label>
          <Button size="sm" variant={check?.state === 'done' ? 'secondary' : 'primary'} aria-busy={sending} onClick={start}
            icon={sending ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />}>
            {sending ? 'Starting' : check?.state === 'done' ? 'Measure again' : 'Measure strength'}
          </Button>
          <p className="text-[12.5px] text-ink-faint">{server ? `On ${server}.` : `Roughly ${minutes} min on this machine.`}</p>
        </div>
      )}
      {check?.state === 'failed' && <p className="mt-2 text-[12.5px] text-danger">The check stopped with an error: {check.error}</p>}
      {check?.state === 'stopped' && <p className="mt-2 text-[12.5px] text-ink-muted">The last check ended before it finished.</p>}
      <StartError error={error} reconnect={reconnect} />
    </div>
  )
}

// ---------------------------------------------------------------------- the rating ladder

const entryName = (e: string) => (e === 'random' ? 'Random' : e === 'greedy' ? 'Greedy' : `Generation ${Number(e.slice(4))}`)
const generationOf = (e: string) => (e.startsWith('gen_') ? Number(e.slice(4)) : null)
const elo = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(Math.round(v)).toLocaleString()}`

/** The newest rated generation against the one rated before it: stronger, level or weaker, by the
 * step's own interval (narrower than either rating's, as neighbours' ratings move together). */
function ladderVerdict(ladder: Ladder): { tone: Tone; headline: string; detail: string } | null {
  const steps = ladder.steps ?? []
  if (!steps.length) return null
  const last = steps[steps.length - 1]
  const r = ladder.ratings ?? {}
  const rated = (ladder.entries ?? []).filter((e) => generationOf(e) !== null && r[e])
  const top = rated.reduce((x, y) => (r[y]!.elo > r[x]!.elo ? y : x))
  const signed = (v: number) => `${v >= 0 ? '+' : '−'}${elo(Math.abs(v))}`
  const detail = `Each step: ${steps.map((s) => `${s.from} to ${s.to}, ${signed(s.diff)}`).join('; ')}. `
    + `The highest rated is generation ${generationOf(top)}, at ${elo(r[top]!.elo)}.`
  const range = `${signed(last.low)} to ${signed(last.high)}`
  if (last.low > 0) {
    return { tone: 'good', headline: `Generation ${last.to} is stronger than generation ${last.from}: ${signed(last.diff)} Elo (${range})`, detail }
  }
  if (last.high < 0) {
    return { tone: 'bad', headline: `Generation ${last.to} is weaker than generation ${last.from}: ${signed(last.diff)} Elo (${range})`, detail }
  }
  return { tone: 'neutral', headline: `Generation ${last.to} is level with generation ${last.from}: ${signed(last.diff)} Elo (${range})`, detail }
}

function LadderChart({ ladder }: { ladder: Ladder }) {
  const r = ladder.ratings ?? {}
  const nets = (ladder.entries ?? []).filter((e) => generationOf(e) !== null && r[e])
    .map((e) => ({ x: generationOf(e)!, r: r[e]! })).sort((a, b) => a.x - b.x)
  const greedy = r.greedy
  const xs = nets.map((n) => n.x)
  return (
    <div className="min-w-0 rounded-xl border border-line bg-bg p-4">
      <p className="text-[14px] font-semibold text-ink">Elo by generation</p>
      <p className="mt-0.5 mb-3 text-[12.5px] text-ink-muted">Random is 0. The shaded band is each rating's 95% interval.</p>
      <LineChart label="Elo rating of each generation on the ladder" xLabel="generation" integerX yFormat={elo}
        series={[{ id: 'elo', label: 'Elo', color: 'var(--chart-1)', kind: 'line', markers: true,
          points: nets.map((n) => ({ x: n.x, y: n.r.elo })), band: nets.map((n) => ({ x: n.x, lo: n.r.low, hi: n.r.high })) }]}
        reference={greedy && xs.length > 1 ? { label: 'Greedy', points: [{ x: xs[0], y: greedy.elo }, { x: xs[xs.length - 1], y: greedy.elo }] } : undefined} />
      <div className="mt-2">
        <Legend items={[{ label: 'Generation', color: 'var(--chart-1)', kind: 'line' },
          ...(greedy ? [{ label: 'Greedy', color: 'var(--ink-faint)', kind: 'ref' as const }] : [])]} />
      </div>
    </div>
  )
}

function LadderTable({ ladder }: { ladder: Ladder }) {
  const r = ladder.ratings ?? {}
  const conversion = ladder.conversion ?? {}
  const rows = [...(ladder.entries ?? [])].sort((a, b) => (r[b]?.elo ?? -Infinity) - (r[a]?.elo ?? -Infinity))
  const positions = ladder.suite?.length ?? 0
  // The material balances the test's positions were made from, as "an Ashva against a Padati".
  const balances = [...new Set(ladder.recipes ?? [])].map((r) => r.charAt(0).toLowerCase() + r.slice(1))
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[620px] border-collapse text-left text-[13px] tabular-nums">
        <thead>
          <tr className="border-b border-line-strong text-[12px] text-ink-faint">
            <th scope="col" className="py-2 pr-4 font-medium">Player</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Elo</th>
            <th scope="col" className="py-2 pr-4 font-medium">95% interval</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Score</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Games</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Converts</th>
            <th scope="col" className="py-2 text-right font-medium">Plies to win</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => {
            const x: LadderRating | null | undefined = r[e]
            const c = conversion[e]
            const baseline = generationOf(e) === null
            return (
              <tr key={e} className="border-b border-line">
                <th scope="row" className={`py-1.5 pr-4 font-medium ${baseline ? 'text-ink-muted' : 'text-ink'}`}>{entryName(e)}</th>
                <td className="py-1.5 pr-4 text-right font-semibold text-ink">{x ? elo(x.elo) : '–'}</td>
                <td className="py-1.5 pr-4 text-ink-muted">{x && e !== 'random' ? `${elo(x.low)} to ${elo(x.high)}` : '–'}</td>
                <td className="py-1.5 pr-4 text-right text-ink-muted">{x?.score != null ? pct(x.score) : '–'}</td>
                <td className="py-1.5 pr-4 text-right text-ink-muted">{x ? x.games.toLocaleString() : '–'}</td>
                <td className="py-1.5 pr-4 text-right text-ink-muted">{c ? `${c.won} of ${c.games}` : '–'}</td>
                <td className="py-1.5 text-right text-ink-muted">{c?.median_plies != null ? Math.round(c.median_plies) : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {positions > 0 && (
        <p className="mt-2 text-[12.5px] leading-snug text-ink-faint">
          Converts: of {positions} positions where it is ahead in material
          {balances.length > 0 && ` (${balances.length > 1 ? `${balances.slice(0, -1).join(', ')} or ${balances[balances.length - 1]}` : balances[0]})`},
          how many it won against Greedy, and the median plies those wins took. Greedy playing the side ahead is the reference.
        </p>
      )}
    </div>
  )
}

function LadderSection({ run, ladder, server, reconnect, onStarted }: {
  run: string; ladder: Ladder | null; server: string | null; reconnect: (note: string) => ReactNode; onStarted: () => void
}) {
  const [every, setEvery] = useState(ladder?.every ?? 10)
  const [games, setGames] = useState(ladder?.games ?? 40)
  const [plan, setPlan] = useState<LadderPlan | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const running = ladder?.state === 'running'

  // What these settings would play, asked again as they change (and once a run of it ends).
  useEffect(() => {
    if (running) return
    let stale = false
    const id = setTimeout(async () => {
      try {
        const p = await api.ladderPlan(run, games, every)
        if (!stale) setPlan(p)
      } catch {
        if (!stale) setPlan(null)
      }
    }, 250)
    return () => {
      stale = true
      clearTimeout(id)
    }
  }, [run, games, every, running, ladder?.state])

  const start = async () => {
    if (sending) return
    setSending(true)
    setError(null)
    try {
      await api.startLadder(run, games, every)
      onStarted()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSending(false)
    }
  }
  const rated = Object.entries(ladder?.ratings ?? {}).some(([e, r]) => r && generationOf(e) !== null)
  const verdict = ladder ? ladderVerdict(ladder) : null
  const todo = plan ? plan.ladder_games + plan.conversion_games : null
  const fresh = plan?.new.filter((e) => generationOf(e) !== null).map((e) => generationOf(e)) ?? []
  return (
    <div>
      <h3 className="font-serif text-[18px] font-semibold text-ink">Rating ladder</h3>
      <p className="mt-1 text-[13px] leading-snug text-ink-muted">
        Generations a few apart play each other, Random and Greedy from the opening, and get Elo ratings with Random at 0.
        It shows the progress the run's own tests no longer can: each generation scores about 50% against the one before,
        and Greedy has little left to tell. A conversion test measures how surely and quickly each one turns a material edge into a win.
      </p>
      {verdict && (
        <div className="mt-3">
          <p className="flex items-start gap-2 text-[14px] font-semibold text-ink">
            <Mark tone={verdict.tone} />
            {verdict.headline}
          </p>
          <p className="mt-1 pl-7 text-[13px] text-ink-muted">{verdict.detail}</p>
        </div>
      )}
      {rated && ladder && (
        <div className="mt-4 grid gap-4 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
          <LadderChart ladder={ladder} />
          <LadderTable ladder={ladder} />
        </div>
      )}
      {running ? (
        <JobProgress job={ladder!} server={server} reconnect={reconnect} />
      ) : (
        <div className="mt-4 space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-[13px] text-ink-muted">
              Every
              <input type="number" min={1} max={1000} value={every}
                onChange={(e) => setEvery(Math.max(1, Math.min(1000, Number(e.target.value) || 1)))}
                className="h-8 w-16 rounded-lg border border-line bg-bg px-2 text-[13px] text-ink tabular-nums" />
              generations
            </label>
            <label className="flex items-center gap-2 text-[13px] text-ink-muted">
              Games a pair
              <input type="number" min={2} max={400} value={games}
                onChange={(e) => setGames(Math.max(2, Math.min(400, Number(e.target.value) || 2)))}
                className="h-8 w-20 rounded-lg border border-line bg-bg px-2 text-[13px] text-ink tabular-nums" />
            </label>
            <Button size="sm" variant={rated ? 'secondary' : 'primary'} aria-busy={sending} disabled={todo === 0} onClick={start}
              icon={sending ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />}>
              {sending ? 'Starting' : rated ? 'Rate what is new' : 'Rate generations'}
            </Button>
          </div>
          {plan && (
            <p className="text-[12.5px] text-ink-faint">
              {todo === 0 ? 'Everything at these settings is rated.'
                : `${todo!.toLocaleString()} games to play${fresh.length ? ` (new: generation${fresh.length > 1 ? 's' : ''} ${fresh.join(', ')})` : ''}: `
                  + `${plan.ladder_games.toLocaleString()} on the ladder and ${plan.conversion_games.toLocaleString()} in the conversion test, `
                  + `${plan.where === 'server' ? `on ${server ?? 'the server'}` : 'on this machine'}. The first, latest and best generations are always included.`}
            </p>
          )}
        </div>
      )}
      {ladder?.state === 'failed' && <p className="mt-2 text-[12.5px] text-danger">The ladder stopped with an error: {ladder.error}</p>}
      {ladder?.state === 'stopped' && (
        <p className="mt-2 text-[12.5px] text-ink-muted">The last rating ended before it finished; its finished games are kept.</p>
      )}
      <StartError error={error} reconnect={reconnect} />
    </div>
  )
}

// How self-play games end: decided endings in colour, draws in greys.
const ENDINGS: StackKey[] = [
  { id: 'bare_king', label: 'Bare king', color: 'var(--chart-1)' },
  { id: 'checkmate', label: 'Checkmate', color: 'var(--chart-3)' },
  { id: 'stalemate', label: 'Stalemate', color: 'var(--chart-2)' },
  { id: 'repetition', label: 'Repetition', color: 'var(--ink-muted)' },
  { id: 'no_progress', label: '70 quiet moves', color: 'var(--ink-faint)' },
  { id: 'max_plies', label: '300 plies', color: 'var(--line-strong)' },
]

/** The report's own charts: how self-play games ended, and how long they were. */
function ReportCharts({ gens }: { gens: GenerationRow[] }) {
  const played = gens.filter((g) => g.games && g.reasons)
  return (
    <div>
      <h3 className="font-serif text-[18px] font-semibold text-ink">Self-play over the run</h3>
      <div className="mt-3 grid gap-4 lg:grid-cols-2">
        <div className="min-w-0 rounded-xl border border-line bg-bg p-4">
          <p className="text-[14px] font-semibold text-ink">How games from the opening end</p>
          <p className="mt-0.5 mb-3 text-[12.5px] text-ink-muted">Coloured endings are decided; grey ones are draws.</p>
          <StackedBars label="How self-play games from the opening ended, by generation" xLabel="generation" keys={ENDINGS}
            bars={played.map((g) => ({
              x: g.generation,
              values: Object.fromEntries(Object.entries(g.reasons!).map(([k, n]) => [k, (100 * n) / g.games!])),
            }))} />
          <div className="mt-2"><Legend items={ENDINGS.map((k) => ({ label: k.label, color: k.color, kind: 'box' as const }))} /></div>
        </div>
        <div className="min-w-0 rounded-xl border border-line bg-bg p-4">
          <p className="text-[14px] font-semibold text-ink">Game length</p>
          <p className="mt-0.5 mb-3 text-[12.5px] text-ink-muted">Average plies per game from the opening; games are drawn at 300.</p>
          <LineChart label="Average self-play game length by generation" xLabel="generation" integerX yMin={0} yMax={300}
            yFormat={(v) => String(Math.round(v))}
            series={[{ id: 'plies', label: 'plies', color: 'var(--chart-1)', kind: 'line',
              points: played.filter((g) => g.mean_plies).map((g) => ({ x: g.generation, y: g.mean_plies! })) }]} />
        </div>
      </div>
    </div>
  )
}

function made(when: string) {
  const d = new Date(when)
  return Number.isNaN(d.getTime()) ? when : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function RunReport({ run, generations, finished, gens, server = null, reconnect }: {
  run: string; generations: number; finished: boolean; gens: GenerationRow[]
  /** The run's server, for a server run (its report's jobs play there). */
  server?: string | null
  /** A form to connect to the server again, with a note on why. */
  reconnect: (note: string) => ReactNode
}) {
  const [data, setData] = useState<Report | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [making, setMaking] = useState(false)
  const load = useCallback(async () => {
    try {
      setData(await api.runReport(run))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    }
  }, [run])

  useEffect(() => {
    setData(null)
    void load()
  }, [load])

  // While a strength check or the ladder plays, follow its progress.
  const playing = data?.strength?.state === 'running' || data?.ladder?.state === 'running'
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => void load(), 2000)
    return () => clearInterval(id)
  }, [playing, load])

  const make = async () => {
    if (making) return
    setMaking(true)
    setError(null)
    try {
      setData(await api.makeReport(run))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setMaking(false)
    }
  }

  const report = data?.report ?? null
  const stale = report !== null && report.generations < generations
  return (
    <section aria-labelledby="report-title" className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="report-title" className="font-serif text-[22px] font-semibold text-ink">Run report</h2>
          <p className="mt-1 text-[13.5px] text-ink-muted">
            {report
              ? `Made ${made(report.made)}, after ${report.generations} generation${report.generations === 1 ? '' : 's'}.`
              : finished ? 'What the run learned, worked out from what it saved.'
                : `What the run has learned so far, after ${generations} generations.`}
          </p>
        </div>
        {report && (
          <div className="flex shrink-0 items-center gap-2">
            <Button size="sm" aria-busy={making} onClick={make}
              icon={making ? <LoaderCircle size={14} className="animate-spin" /> : <RefreshCw size={14} />}>
              {making ? 'Updating' : 'Update report'}
            </Button>
            <a href={api.chartsUrl(run)} download={`${run}-training.png`}
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong px-2.5 text-[13px] font-medium text-ink transition-colors duration-150 hover:border-ink-faint hover:bg-surface-2">
              <Download size={14} aria-hidden />
              Download charts
            </a>
          </div>
        )}
      </div>
      {error && <p className="mt-4 text-[13px] text-danger">{error}</p>}
      {!data && !error && <div className="mt-5 h-24 animate-pulse rounded-xl bg-bg" aria-busy="true" />}
      {data && !report && (
        <div className="cz-swap mt-5 rounded-xl border border-dashed border-line-strong px-5 py-6">
          <p className="text-[13.5px] leading-relaxed text-ink-muted">
            The report sums up what the run learned: whether its policy and value heads learned, whether the value sees
            material, how its test scores moved and charts of its self-play. It takes a few seconds; a strength check
            can follow.
          </p>
          <div className="mt-4">
            <Button variant="primary" size="sm" aria-busy={making} onClick={make}
              icon={making ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />}>
              {making ? 'Making the report' : 'Make report'}
            </Button>
          </div>
        </div>
      )}
      {report && (
        <div key={report.made} className="cz-swap mt-5 space-y-7">
          {stale && (
            <p className="text-[13px] text-ink-muted">
              The run has {generations} generations now; update the report to include the latest.
            </p>
          )}
          <Findings s={report.summary} />
          {report.probe.networks.length > 0 && <Probe probe={report.probe} />}
          <ReportCharts gens={gens} />
          <Strength run={run} check={data!.strength} server={server} reconnect={reconnect} onStarted={() => void load()} />
          <LadderSection key={run} run={run} ladder={data!.ladder} server={server} reconnect={reconnect} onStarted={() => void load()} />
        </div>
      )}
    </section>
  )
}
