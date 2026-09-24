/**
 * Play-screen state. The move list is the single source of truth; every change asks the
 * engine for the resulting position. A generation counter drops responses that arrive
 * after the game has moved on (new game, undo, mode change).
 */

import { create } from 'zustand'
import { api, ApiError, type Color, type GameView } from '../lib/api'
import { moveText } from '../lib/board'
import { playMoveSound } from '../lib/sound'

export type Mode = 'hvh' | 'hvc' | 'cvc'
export type Pace = 'slow' | 'normal' | 'fast'

const PACE_MS: Record<Pace, number> = { slow: 1400, normal: 700, fast: 220 }
const HUMAN_OPPONENT_DELAY = 420
/** Time on each position during a replay; long enough for the move's animation to finish. */
const REPLAY_MS: Record<Pace, number> = { slow: 1700, normal: 1000, fast: 520 }

interface GameStore {
  mode: Mode
  humanColor: Color
  opponent: string
  whiteAgent: string
  blackAgent: string
  pace: Pace
  running: boolean // computer vs computer: playing or paused
  orientation: Color // side shown at the bottom of the board
  autoRotate: boolean // two players: turn the board to face the side to move

  moves: string[]
  game: GameView | null
  status: 'connecting' | 'ready' | 'thinking' | 'error'
  error: string | null

  reviewPly: number | null // null: showing the live game
  reviewGame: GameView | null
  replaying: boolean // stepping through the review automatically
  replaySpeed: Pace

  init: () => void
  newGame: () => void
  play: (from: number, to: number) => void
  undo: () => void
  setMode: (mode: Mode) => void
  setHumanColor: (c: Color) => void
  setOpponent: (id: string) => void
  setAgent: (color: Color, id: string) => void
  setPace: (p: Pace) => void
  setRunning: (on: boolean) => void
  /** Show the position after `ply` moves (0 to all of them); null returns to the live game. */
  review: (ply: number | null) => void
  /** Step through the review: -1 / +1 a move, or to the start / end. Pauses a replay. */
  step: (to: 'start' | 'back' | 'forward' | 'end') => void
  /** Replay the game from the start (or carry on from the reviewed position). */
  startReplay: () => void
  stopReplay: () => void
  setReplaySpeed: (p: Pace) => void
  flip: () => void
  setAutoRotate: (on: boolean) => void
}

let generation = 0
let timer: ReturnType<typeof setTimeout> | null = null
let replayTimer: ReturnType<typeof setTimeout> | null = null
// Positions already fetched for review, by move sequence: stepping back and forth is instant.
const reviewCache = new Map<string, GameView>()

function soundFor(view: GameView) {
  const last = view.history[view.history.length - 1]
  if (!last) return
  if (view.outcome) playMoveSound('end')
  else if (last.check) playMoveSound('check')
  else playMoveSound(last.captured ? 'capture' : 'move')
}

export const useGame = create<GameStore>((set, get) => {
  const fail = (err: unknown) => {
    const message = err instanceof ApiError && err.status === 0
      ? 'Cannot reach the rules engine. Start it with: python -m chaturanga_server'
      : err instanceof Error ? err.message : 'Something went wrong.'
    set({ status: 'error', error: message })
  }

  const computerToMove = (s: GameStore) => {
    const g = s.game
    if (!g || g.outcome || s.reviewPly !== null) return null
    if (s.mode === 'hvc' && g.side !== s.humanColor) return s.opponent
    if (s.mode === 'cvc' && s.running) return g.side === 1 ? s.whiteAgent : s.blackAgent
    return null
  }

  const schedule = () => {
    if (timer) clearTimeout(timer)
    timer = null
    const s = get()
    const agent = computerToMove(s)
    if (!agent) {
      if (s.status === 'thinking') set({ status: 'ready' })
      return
    }
    const gen = generation
    set({ status: 'thinking' })
    const delay = s.mode === 'cvc' ? PACE_MS[s.pace] : HUMAN_OPPONENT_DELAY
    timer = setTimeout(async () => {
      try {
        const res = await api.agentMove(get().moves, agent)
        if (gen !== generation) return
        set({ moves: [...get().moves, res.move], game: res.game, status: 'ready', error: null })
        soundFor(res.game)
        schedule()
      } catch (err) {
        if (gen === generation) fail(err)
      }
    }, delay)
  }

  const load = async (moves: string[]) => {
    const gen = ++generation
    if (timer) clearTimeout(timer)
    try {
      const game = await api.game(moves)
      if (gen !== generation) return
      if (replayTimer) clearTimeout(replayTimer)
      set({ moves, game, status: 'ready', error: null, reviewPly: null, reviewGame: null, replaying: false })
      schedule()
    } catch (err) {
      if (gen === generation) fail(err)
    }
  }

  // Advance a running replay once the position on show has loaded; stop at the last move.
  const scheduleReplay = () => {
    if (replayTimer) clearTimeout(replayTimer)
    replayTimer = null
    const s = get()
    if (!s.replaying || s.reviewPly === null) return
    if (s.reviewPly >= s.moves.length) {
      set({ replaying: false })
      return
    }
    replayTimer = setTimeout(() => {
      const t = get()
      if (t.replaying && t.reviewPly !== null) reviewAt(t.reviewPly + 1)
    }, REPLAY_MS[s.replaySpeed])
  }

  const reviewAt = (ply: number | null) => {
    const s = get()
    if (ply === null || ply > s.moves.length || !s.game) {
      if (replayTimer) clearTimeout(replayTimer)
      set({ reviewPly: null, reviewGame: null, replaying: false })
      generation++
      schedule()
      return
    }
    if (timer) clearTimeout(timer)
    const gen = ++generation
    const key = s.moves.slice(0, ply).join(' ')
    const known = ply === s.moves.length ? s.game : reviewCache.get(key)
    if (known) {
      set({ reviewPly: ply, reviewGame: known, status: 'ready' })
      scheduleReplay()
      return
    }
    set({ reviewPly: ply, status: 'ready' })
    void api.game(s.moves.slice(0, ply)).then(
      (reviewGame) => {
        reviewCache.set(key, reviewGame)
        if (gen !== generation) return
        set({ reviewGame })
        scheduleReplay()
      },
      (err) => gen === generation && fail(err),
    )
  }

  return {
    mode: 'hvh',
    humanColor: 1,
    opponent: 'greedy',
    whiteAgent: 'greedy',
    blackAgent: 'random',
    pace: 'normal',
    running: true,
    orientation: 1,
    autoRotate: true,
    moves: [],
    game: null,
    status: 'connecting',
    error: null,
    reviewPly: null,
    reviewGame: null,
    replaying: false,
    replaySpeed: 'normal',

    init: () => {
      if (!get().game) void load([])
    },

    newGame: () => {
      set({ status: get().game ? 'ready' : 'connecting' })
      void load([])
    },

    play: (from, to) => {
      const s = get()
      if (!s.game || s.status === 'thinking' || s.reviewPly !== null) return
      const moves = [...s.moves, moveText(from, to)]
      const gen = ++generation
      void api.game(moves).then(
        (game) => {
          if (gen !== generation) return
          set({ moves, game, status: 'ready', error: null })
          soundFor(game)
          schedule()
        },
        (err) => gen === generation && fail(err),
      )
    },

    undo: () => {
      const s = get()
      let n = s.moves.length
      if (n === 0) return
      if (s.mode === 'hvc') {
        // Take back to the human's turn: one or two plies.
        n -= 1
        const sideAfter = (n % 2 === 0 ? 1 : -1) as Color
        if (sideAfter !== s.humanColor && n > 0) n -= 1
      } else {
        n -= 1
      }
      if (s.mode === 'cvc') set({ running: false })
      void load(s.moves.slice(0, n))
    },

    setMode: (mode) => {
      set(mode === 'hvc' ? { mode, running: true, orientation: get().humanColor } : { mode, running: true })
      generation++
      schedule()
    },
    setHumanColor: (humanColor) => {
      set({ humanColor, orientation: humanColor })
      generation++
      schedule()
    },
    setOpponent: (opponent) => set({ opponent }),
    setAgent: (color, id) => set(color === 1 ? { whiteAgent: id } : { blackAgent: id }),
    setPace: (pace) => set({ pace }),
    setRunning: (running) => {
      set({ running })
      generation++
      schedule()
    },

    flip: () => set({ orientation: (get().orientation === 1 ? -1 : 1) as Color }),
    setAutoRotate: (autoRotate) => set({ autoRotate }),

    review: (ply) => {
      set({ replaying: false })
      reviewAt(ply)
    },

    step: (to) => {
      const s = get()
      const total = s.moves.length
      const at = s.reviewPly ?? total
      const target = to === 'start' ? 0 : to === 'end' ? total : Math.min(total, Math.max(0, at + (to === 'back' ? -1 : 1)))
      set({ replaying: false })
      if (target !== s.reviewPly) reviewAt(target)
    },

    startReplay: () => {
      const s = get()
      if (s.moves.length === 0) return
      set({ replaying: true })
      if (s.reviewPly === null || s.reviewPly >= s.moves.length) reviewAt(0)
      else scheduleReplay()
    },
    stopReplay: () => {
      if (replayTimer) clearTimeout(replayTimer)
      set({ replaying: false })
    },
    setReplaySpeed: (replaySpeed) => {
      set({ replaySpeed })
      scheduleReplay() // restart the wait at the new speed
    },
  }
})

// Development aid: drive the game from the browser console (window.__game.getState()).
if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__game = useGame
