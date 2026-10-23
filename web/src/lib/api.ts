/** Client for the Python API (chaturanga_server). All rules live on the server. */

export const WHITE = 1
export const BLACK = -1
export type Color = 1 | -1

export interface HistoryEntry {
  move: string
  from: number
  to: number
  piece: number
  captured: number
  promotion: number
  check: boolean
}

export type OutcomeReason = 'bare_king' | 'checkmate' | 'stalemate' | 'repetition' | 'no_progress' | 'max_plies'

export interface Outcome {
  winner: Color | null
  reason: OutcomeReason
}

export interface GameView {
  fen: string
  board: number[]
  side: Color
  ply: number
  in_check: boolean
  king_square: number
  legal_moves: [number, number][]
  outcome: Outcome | null
  no_progress: number
  repetition: number
  history: HistoryEntry[]
}

export interface AgentInfo {
  id: string
  name: string
  description: string
}

export interface TrainedInfo {
  run: string | null
  generation: number | null
}

export interface LevelInfo {
  id: string
  name: string
  simulations: number
  description: string
}

export interface Meta {
  rules: {
    setup: string
    promotion: string
    stalemate: string
    repetition_limit: number
    no_progress_plies: number
    max_plies: number
  }
  agents: AgentInfo[]
  levels: LevelInfo[]
  trained: TrainedInfo | null
}

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, body?: unknown, method?: 'DELETE'): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method: method ?? (body === undefined ? 'GET' : 'POST'),
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError('Cannot reach the rules engine.', 0)
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const data = await res.json()
      if (typeof data?.detail === 'string') detail = data.detail
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  meta: () => request<Meta>('/api/meta'),
  game: (moves: string[], startFen?: string | null) =>
    request<GameView>('/api/game', { moves, start_fen: startFen ?? null }),
  agentMove: (moves: string[], agent: string, startFen?: string | null, level?: string) =>
    request<{ move: string; game: GameView; network?: TrainedInfo }>('/api/agent-move', {
      moves, agent, start_fen: startFen ?? null, level: level ?? null,
    }),
  trainingRuns: () => request<{ runs: RunStatus[] }>('/api/training/runs'),
  trainingRun: (name: string) => request<RunDetail>(`/api/training/runs/${encodeURIComponent(name)}`),
  selfPlayGames: (name: string, generation: number) =>
    request<{ games: SelfPlayGameInfo[] }>(`/api/training/runs/${encodeURIComponent(name)}/games/${generation}`),
  selfPlayGame: (name: string, generation: number, index: number) =>
    request<SelfPlayGame>(`/api/training/runs/${encodeURIComponent(name)}/games/${generation}/${index}`),
  /** Ask a running run to pause; null withdraws the request. */
  pauseRun: (name: string, when: PauseWhen | null) =>
    request<RunStatus>(`/api/training/runs/${encodeURIComponent(name)}/pause`, { when }),
  /** The run's saved report (null until one is made), its strength check and its rating ladder, if played. */
  runReport: (name: string) => request<RunReport>(`/api/training/runs/${encodeURIComponent(name)}/report`),
  /** Make (or remake) the report: findings, material probe and charts. */
  makeReport: (name: string) => request<RunReport>(`/api/training/runs/${encodeURIComponent(name)}/report`, {}),
  /** Play the strength check in the background. */
  startStrength: (name: string, games: number) =>
    request<StrengthCheck>(`/api/training/runs/${encodeURIComponent(name)}/report/strength`, { games }),
  /** What rating with these settings would play: the ladder's entries and the games left. */
  ladderPlan: (name: string, games: number, every: number) =>
    request<LadderPlan>(`/api/training/runs/${encodeURIComponent(name)}/report/ladder/plan?games=${games}&every=${every}`),
  /** Play the rating ladder and conversion test in the background (on the run's server for a server run). */
  startLadder: (name: string, games: number, every: number) =>
    request<Ladder>(`/api/training/runs/${encodeURIComponent(name)}/report/ladder`, { games, every }),
  /** The run's six-panel chart image, for download. */
  chartsUrl: (name: string) => `/api/training/runs/${encodeURIComponent(name)}/charts.png`,
  /** Give Play's Trained opponent one of the run's networks: its best or latest (followed as it trains),
   * a generation (kept), or null (this run stops giving Play its network). */
  choosePlay: (name: string, network: RunStatus['play']) =>
    request<RunStatus>(`/api/training/runs/${encodeURIComponent(name)}/play`, { network }),
  /** Delete a run that is not training, with everything it saved. */
  deleteRun: (name: string) =>
    request<{ deleted: string }>(`/api/training/runs/${encodeURIComponent(name)}`, undefined, 'DELETE'),
  /** Start a new run (on this Mac one at a time; a Colab run comes back with the notebook's link). */
  startRun: (settings: NewRunSettings) => request<RunStatus & { colab_url?: string }>('/api/training/runs', settings),
  cloudStatus: () => request<CloudStatus>('/api/training/cloud'),
  /** Check a Kaggle API token and keep it on this Mac. */
  connectKaggle: (username: string, token: string) =>
    request<CloudStatus>('/api/training/cloud/kaggle', { username, token }),
  /** Open the connection to your server (the password is used for that and not kept). */
  connectServer: (settings: { host: string; folder?: string; password?: string; workers?: number }) =>
    request<CloudStatus>('/api/training/server/connect', settings),
  /** Close it; training there carries on. */
  disconnectServer: () => request<CloudStatus>('/api/training/server/disconnect', {}),
  /** Start a run that is not running, carrying on where it stopped (optionally to more generations). */
  resumeRun: (name: string, generations?: number, settings: RunSettings = {}) =>
    request<RunStatus & { colab_url?: string }>(`/api/training/runs/${encodeURIComponent(name)}/resume`,
      { ...(generations ? { generations } : {}), ...settings }),
}

// ------------------------------------------------------------------ training (Watch training)

export type RunState = 'starting' | 'selfplay' | 'training' | 'evaluating' | 'generation_done' | 'reporting' | 'finished' | 'paused'
  | 'stopped' | 'failed'
  | 'waiting' // a Colab run waiting for its notebook to be run
  | 'uploading' | 'queued' | 'cloud_running' | 'downloading' // a Kaggle run, followed from here
  | 'preparing' // a run on your server, while the server gets ready
export type Where = 'mac' | 'colab' | 'kaggle' | 'server'
export type PauseWhen = 'now' | 'after_phase' | 'after_generation'
type Phase = 'selfplay' | 'training' | 'evaluating'

export interface RunStatus {
  name: string
  state: RunState
  live: boolean
  generation: number
  generations: number
  games_done?: number
  games_total?: number
  step?: number
  steps_total?: number
  opponent?: string | null
  /** While testing: every opponent, their games played together. */
  opponents?: string[]
  /** While the run makes its report at the end: the part under way (its games count in games_done). */
  report_part?: 'report' | 'strength' | 'ladder' | null
  best_generation?: number
  updated?: string
  seconds_since_update: number | null
  /** Where the run trains. */
  where: Where
  /** Which of its networks is the Trained opponent in Play: its best or latest (followed as it
   * trains), one generation (kept), or null (none). One run feeds Play at a time. */
  play: 'best' | 'latest' | number | null
  /** A Kaggle run: its page on Kaggle, when it was started there and last checked. */
  kernel_url?: string
  cloud_started?: string
  cloud_checked?: string
  /** A run on your server: what the server is getting ready, whether this Mac is connected to
   * it (null when not following), the server, and when it was last checked. */
  setup?: string | null
  server_connected?: boolean | null
  server_host?: string | null
  server_checked?: string | null
  /** Copying what the run made back from the server (after a gap, this can take a while). */
  server_syncing?: boolean
  server_workers?: number | null
  server_gpu_workers?: number | null
  /** Why a run failed, when it says. */
  error?: string
  /** Live progress of the current phase: when it began, moves played so far and moves expected. */
  phase_started?: string | null
  moves?: number | null
  moves_expected?: number | null
  /** A pause asked for and not yet reached (live runs only). */
  pause_requested: PauseWhen | null
  /** Where a paused run stopped: in the middle of a phase, or just after one. */
  paused_in?: Phase | null
  paused_after?: Phase | 'generation' | null
}

/** Settings a run can change when it resumes or trains further (from its next generation). */
export interface RunSettings {
  games_per_generation?: number
  simulations?: number
  learning_rate?: number
  late_temperature?: number
  eval_games?: number
  auto_report?: boolean
  /** A run on your server: self-play processes, and how many of them use its GPU. */
  workers?: number
  gpu_workers?: number
}

export interface NewRunSettings {
  name: string
  generations: number
  games_per_generation: number
  simulations: number
  eval_games: number
  /** Make the run's best network the Trained opponent in Play. */
  publish: boolean
  /** Make the run report (strength check included) as soon as the run finishes. */
  auto_report: boolean
  /** Train on this Mac, on Colab (through Google Drive), on Kaggle or on your server. */
  where: Where
  /** Self-play processes, on your server, and how many of them run the network on its GPU. */
  workers?: number
  gpu_workers?: number
  /** The network's size: residual blocks and filters per convolution (a run keeps its size). */
  blocks?: number
  channels?: number
}

export interface CloudStatus {
  /** Colab: the ChaturangZero folder in Google Drive, if Drive is synced here; the notebook's link. */
  colab: { drive: string | null; url: string }
  kaggle: { connected: boolean; username: string | null; tool: boolean }
  server: ServerStatus
}

export interface ServerStatus {
  /** The server new runs go to (null before the first Connect), and the folder everything goes in. */
  host: string | null
  folder: string
  folder_path: string | null
  workers: number | null
  gpu_workers: number
  connected: boolean
  /** What it had when last connected. */
  facts: { hostname?: string; cores?: number; cpu?: string; memory_gb?: number; gpu?: string; python?: string } | null
  /** Hosts named in ~/.ssh/config, as each signs in. */
  hosts: { name: string; user: string | null; hostname: string }[]
  /** ssh and rsync are on this Mac. */
  tools: boolean
}

export interface LossPoint {
  step: number
  generation: number
  loss: number
  value_loss: number
  policy_loss: number
  policy_entropy: number
}

export interface EvalResult {
  wins: number
  draws: number
  losses: number
  score: number
  reasons: Record<string, number>
}

export interface GenerationRow {
  generation: number
  games?: number
  white_wins?: number
  black_wins?: number
  draws?: number
  reasons?: Record<string, number>
  mean_plies?: number
  positions?: number
  seconds?: number
  contributed?: Record<string, number>
  buffer_positions?: number
  held_out?: { loss: number; value_loss: number; policy_loss: number; value_baseline?: number; step?: number }
  /** Self-play games started from a random endgame (the fields above count games from the opening). */
  endgames?: { games: number; white_wins: number; black_wins: number; draws: number; mean_plies: number }
  steps?: number
  eval: Record<string, EvalResult>
  best?: number
  train_loss?: { loss: number; value_loss: number; policy_loss: number }
  finished?: string
}

export interface RunDetail {
  status: RunStatus
  config: { generations: number; games_per_generation: number; simulations: number; steps_per_generation: number;
    blocks: number; channels: number; value_mix?: number; [key: string]: unknown }
  train: LossPoint[]
  generations: GenerationRow[]
}

export interface SelfPlayGameInfo {
  index: number
  winner: Color | null
  reason: OutcomeReason
  plies: number
  /** Started from a random endgame rather than the standard setup. */
  endgame: boolean
}

export interface SelfPlayGame extends SelfPlayGameInfo {
  generation: number
  network: number
  moves: string[]
}

// ------------------------------------------------------------------ the run report

export type Verdict = 'learned' | 'partly' | 'not_learned'

export interface RunSummary {
  generations: number
  policy?: { first: number; last: number; floor: number | null; verdict: Verdict }
  value?: { generations: number; beat_baseline: number; last: number; last_baseline: number; last_generation: number;
    verdict: Verdict }
  strength: Partial<Record<'greedy' | 'random' | 'previous', { first: number; last: number; best: number;
    best_generation: number; first_generation: number; last_generation: number }>>
  selfplay?: { games: number; decisive_first: number; decisive_last: number; plies_first: number; plies_last: number }
}

export interface ValueProbe {
  positions: string[]
  networks: { label: 'untrained' | 'best' | 'final'; generation: number; values: number[]; material_gap: number }[]
}

export interface StrengthResult { wins: number; draws: number; losses: number; score: number; reasons: Record<string, number> }

/** Where a report job plays: a server run's plays on its server, followed from here. */
interface JobPlace {
  where?: 'server'
  server_host?: string
  /** False while the server is not connected (the job plays on there). */
  connected?: boolean
  started?: string
  updated?: string
}

export interface StrengthCheck extends JobPlace {
  state: 'running' | 'done' | 'failed' | 'stopped'
  games: number
  simulations?: number
  networks: { label: 'untrained' | 'best' | 'final'; generation: number }[]
  results: Partial<Record<string, Partial<Record<'random' | 'greedy', StrengthResult>>>>
  /** Networks the last check had measured with the same settings: their results, kept, not replayed. */
  kept?: string[]
  done: number
  total: number
  error?: string
}

/** A ladder entry: "random", "greedy" or a generation's network, "gen_0040". */
export type LadderEntry = string

export interface LadderRating {
  elo: number
  low: number
  high: number
  games: number
  score: number | null
}

export interface Conversion {
  games: number
  won: number
  lost: number
  median_plies: number | null
  mean_plies: number | null
  by_recipe: Record<string, number>
  not_won: Record<string, number>
}

export interface Ladder extends JobPlace {
  state: 'running' | 'done' | 'failed' | 'stopped'
  games: number
  every: number
  simulations?: number
  entries?: LadderEntry[]
  pairs?: Record<string, { a: LadderEntry; b: LadderEntry; games: number; a_wins: number; draws: number; b_wins: number }>
  ratings?: Record<LadderEntry, LadderRating | null>
  suite?: string[]
  recipes?: string[]
  conversion?: Record<LadderEntry, Conversion>
  /** From each rated generation to the next: the change in Elo and its own 95% interval. */
  steps?: { from: number; to: number; diff: number; low: number; high: number }[]
  done: number
  total: number
  error?: string
}

export interface LadderPlan {
  entries: LadderEntry[]
  new: LadderEntry[]
  ladder_games: number
  conversion_games: number
  where: 'server' | 'mac'
}

export interface SavedReport {
  made: string
  generations: number
  summary: RunSummary
  probe: ValueProbe
  charts: boolean
}

export interface RunReport {
  report: SavedReport | null
  strength: StrengthCheck | null
  ladder: Ladder | null
}
