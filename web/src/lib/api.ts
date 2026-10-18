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

async function request<T>(path: string, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
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
  /** Start a run that is not running, carrying on where it stopped (optionally to more generations). */
  resumeRun: (name: string, generations?: number) =>
    request<RunStatus>(`/api/training/runs/${encodeURIComponent(name)}/resume`, generations ? { generations } : {}),
}

// ------------------------------------------------------------------ training (Watch training)

export type RunState = 'starting' | 'selfplay' | 'training' | 'evaluating' | 'generation_done' | 'finished' | 'paused'
  | 'stopped' | 'failed'
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
  opponent?: string
  best_generation?: number
  updated?: string
  seconds_since_update: number | null
  /** A pause asked for and not yet reached (live runs only). */
  pause_requested: PauseWhen | null
  /** Where a paused run stopped: in the middle of a phase, or just after one. */
  paused_in?: Phase | null
  paused_after?: Phase | 'generation' | null
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
  steps?: number
  eval: Record<string, EvalResult>
  best?: number
  train_loss?: { loss: number; value_loss: number; policy_loss: number }
  finished?: string
}

export interface RunDetail {
  status: RunStatus
  config: { generations: number; games_per_generation: number; simulations: number; steps_per_generation: number;
    blocks: number; channels: number; [key: string]: unknown }
  train: LossPoint[]
  generations: GenerationRow[]
}

export interface SelfPlayGameInfo {
  index: number
  winner: Color | null
  reason: OutcomeReason
  plies: number
}

export interface SelfPlayGame extends SelfPlayGameInfo {
  generation: number
  network: number
  moves: string[]
}
