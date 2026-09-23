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
  agentMove: (moves: string[], agent: string, startFen?: string | null) =>
    request<{ move: string; game: GameView }>('/api/agent-move', {
      moves, agent, start_fen: startFen ?? null,
    }),
}
