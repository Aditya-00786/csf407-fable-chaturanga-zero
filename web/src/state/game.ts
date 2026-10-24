/**
 * Play-screen state. The move list is the single source of truth; every change asks the
 * engine for the resulting position. A generation counter drops responses that arrive
 * after the game has moved on (new game, undo, mode change).
 */

import { create } from 'zustand'
import { api, ApiError, type Color, type GameView, type HumanGameRecord, type TrainedInfo } from '../lib/api'
import { moveText } from '../lib/board'
import { playMoveSound } from '../lib/sound'
import { useSettings } from './settings'

export type Mode = 'hvh' | 'hvc' | 'cvc'

/** The computer vs computer control's label: Start before the first move, then Pause / Resume. */
export function runLabel(running: boolean, moves: number) {
  return running ? 'Pause' : moves === 0 ? 'Start' : 'Resume'
}
export type Pace = 'slow' | 'normal' | 'fast'
/** The trained agent's strength (its search budget); see chaturanga/agents.py LEVELS. */
export type Level = 'easy' | 'medium' | 'hard' | 'impossible'
export type AgentSlot = 'opponent' | 'white' | 'black'

/**
 * Recording the game against the computer for the results (Results page): this game is being
 * recorded, or the last one recorded is being saved, was saved or could not be.
 */
export type Recording =
  | { state: 'off' }
  | { state: 'on' }
  | { state: 'saving' }
  | { state: 'saved'; ended: HumanGameRecord['ended']; games: number }
  | { state: 'failed'; message: string }

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
  levels: Record<AgentSlot, Level>
  /** The trained network the server plays with, null if none is published (undefined: not asked yet). */
  trained: TrainedInfo | null | undefined
  pace: Pace
  running: boolean // computer vs computer: playing, or waiting for Start / paused
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
  /** Watching a finished game (from training, or a recorded one): the game that was on the board before, to restore. */
  watching: { moves: string[]; mode: Mode; running: boolean; resigned: boolean; from: 'training' | 'results' } | null
  /** Against the computer: the person resigned this game, so it is over. */
  resigned: boolean
  recording: Recording

  init: () => void
  newGame: () => void
  play: (from: number, to: number) => void
  undo: () => void
  /** Against the computer: give up the game (once you have moved). */
  resign: () => void
  /** Try again to save the recorded game that could not be saved. */
  saveRecording: () => void
  /** Record games against the computer, or stop (this game too, if it was being recorded). */
  setRecording: (on: boolean) => void
  setMode: (mode: Mode) => void
  setHumanColor: (c: Color) => void
  setOpponent: (id: string) => void
  setAgent: (color: Color, id: string) => void
  setLevel: (slot: AgentSlot, level: Level) => void
  /** Ask the server which trained network it has (it picks up newly published ones). */
  refreshTrained: () => void
  /** Load a finished game (a self-play game from training, or a recorded one) and replay it from the start. */
  watchGame: (moves: string[], from?: 'training' | 'results') => void
  /** Stop watching: put back the game that was on the board before. */
  leaveWatch: () => void
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
let loading = false // a load is in flight (init must not replace it with an empty game)
let timer: ReturnType<typeof setTimeout> | null = null
let replayTimer: ReturnType<typeof setTimeout> | null = null
// Positions already fetched for review, by move sequence: stepping back and forth is instant.
const reviewCache = new Map<string, GameView>()
// The position being fetched for review, if any. The reviewed ply and its position are only
// shown together, so the board, status and review bar never show a half-updated state.
let pendingPly: number | null = null

const reviewKey = (moves: string[], ply: number) => moves.slice(0, ply).join(' ')

/**
 * The game being recorded: against the computer, from its first move, with one opponent and one
 * colour throughout (the Game panel locks those while it is). Once it is over, the record being
 * saved (kept until it is, so a failed save can be tried again).
 */
interface Draft {
  id: string
  human: Color
  opponent: string
  level: Level
  takebacks: number
  network: TrainedInfo | null // the trained network that played it
}
let draft: Draft | null = null
let unsaved: HumanGameRecord | null = null

function newId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Whether the person has made a move: until then a game can be left without counting. */
export function humanMoved(moves: string[], human: Color) {
  return Math.floor((moves.length + (human === 1 ? 1 : 0)) / 2) > 0
}

/** Fetch a review position into the cache ahead of need (the neighbours of the one on show). */
function prefetch(moves: string[], ply: number) {
  if (ply < 0 || ply >= moves.length) return
  const key = reviewKey(moves, ply)
  if (!reviewCache.has(key)) void api.game(moves.slice(0, ply)).then((v) => reviewCache.set(key, v), () => {})
}

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
    if (!g || g.outcome || s.resigned || s.reviewPly !== null) return null
    if (s.mode === 'hvc' && g.side !== s.humanColor) return { agent: s.opponent, level: s.levels.opponent }
    if (s.mode === 'cvc' && s.running) {
      return g.side === 1 ? { agent: s.whiteAgent, level: s.levels.white } : { agent: s.blackAgent, level: s.levels.black }
    }
    return null
  }

  /** Start recording at the game's first move, if recording is on. */
  const beginRecording = () => {
    const s = get()
    if (draft || s.mode !== 'hvc' || s.moves.length > 0 || s.watching || !useSettings.getState().recordGames) return
    draft = { id: newId(), human: s.humanColor, opponent: s.opponent, level: s.levels.opponent, takebacks: 0, network: null }
    set({ recording: { state: 'on' } })
  }

  /** The recorded game is over: played out, resigned, or left (which counts as resigning). */
  const finishRecording = (ended: HumanGameRecord['ended']) => {
    const d = draft
    if (!d) return
    draft = null
    const s = get()
    const moves = s.watching?.moves ?? s.moves
    if (!useSettings.getState().recordGames || (ended !== 'over' && !humanMoved(moves, d.human))) {
      set({ recording: { state: 'off' } })
      return
    }
    unsaved = {
      id: d.id, human: d.human === 1 ? 'white' : 'black', opponent: d.opponent,
      level: d.opponent === 'zero' ? d.level : null, network: d.network, moves, ended, takebacks: d.takebacks,
    }
    saveRecording()
  }

  /**
   * A setting of the recorded game changed (the Game panel locks them once you have moved): the
   * game counts as left. True if you had not moved yet, so it can start again, recorded.
   */
  const changeRecorded = () => {
    if (!draft) return false
    const s = get()
    const fresh = !humanMoved(s.watching?.moves ?? s.moves, draft.human)
    finishRecording('left')
    return fresh && !s.watching
  }

  const saveRecording = () => {
    const record = unsaved
    if (!record) return
    set({ recording: { state: 'saving' } })
    api.recordGame(record).then(
      (res) => {
        if (unsaved !== record) return
        unsaved = null
        // A new game being recorded by now shows that instead.
        if (!draft) {
          set({ recording: { state: 'saved', ended: record.ended, games: res.total } })
        }
      },
      (err) => {
        if (unsaved === record && !draft) {
          set({ recording: { state: 'failed', message: err instanceof Error ? err.message : 'The game could not be saved.' } })
        }
      },
    )
  }

  const schedule = () => {
    if (timer) clearTimeout(timer)
    timer = null
    const s = get()
    const next = computerToMove(s)
    if (!next) {
      if (s.status === 'thinking') set({ status: 'ready' })
      return
    }
    const gen = generation
    if (s.moves.length === 0) beginRecording()
    set({ status: 'thinking' })
    const delay = s.mode === 'cvc' ? PACE_MS[s.pace] : HUMAN_OPPONENT_DELAY
    timer = setTimeout(async () => {
      try {
        const res = await api.agentMove(get().moves, next.agent, null, next.agent === 'zero' ? next.level : undefined)
        if (gen !== generation) return
        set({ moves: [...get().moves, res.move], game: res.game, status: 'ready', error: null,
          ...(res.network ? { trained: res.network } : {}) })
        if (draft && res.network && !draft.network) draft.network = res.network
        soundFor(res.game)
        if (res.game.outcome) finishRecording('over')
        schedule()
      } catch (err) {
        if (gen === generation) fail(err)
      }
    }, delay)
  }

  const load = async (moves: string[], then?: () => void) => {
    const gen = ++generation
    if (timer) clearTimeout(timer)
    loading = true
    try {
      const game = await api.game(moves)
      if (gen !== generation) return
      loading = false
      if (replayTimer) clearTimeout(replayTimer)
      pendingPly = null
      set({ moves, game, status: 'ready', error: null, reviewPly: null, reviewGame: null, replaying: false, resigned: false })
      if (then) then()
      else schedule()
    } catch (err) {
      if (gen === generation) {
        loading = false
        fail(err)
      }
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
      pendingPly = null
      set({ reviewPly: null, reviewGame: null, replaying: false })
      generation++
      schedule()
      return
    }
    if (timer) clearTimeout(timer)
    // Opening review starts its controls afresh: Normal speed (the caller says whether to replay).
    if (s.reviewPly === null && pendingPly === null) set({ replaySpeed: 'normal' })
    const gen = ++generation
    const moves = s.moves
    const show = (reviewGame: GameView) => {
      pendingPly = null
      set({ reviewPly: ply, reviewGame, status: 'ready' })
      prefetch(moves, ply + 1)
      prefetch(moves, ply - 1)
      scheduleReplay()
    }
    const key = reviewKey(moves, ply)
    const known = ply === moves.length ? s.game : reviewCache.get(key)
    if (known) {
      show(known)
      return
    }
    pendingPly = ply
    void api.game(moves.slice(0, ply)).then(
      (reviewGame) => {
        reviewCache.set(key, reviewGame)
        if (gen === generation) show(reviewGame)
      },
      (err) => {
        if (gen !== generation) return
        pendingPly = null
        fail(err)
      },
    )
  }

  return {
    mode: 'hvh',
    humanColor: 1,
    opponent: 'greedy',
    whiteAgent: 'greedy',
    blackAgent: 'random',
    levels: { opponent: 'medium', white: 'medium', black: 'medium' },
    trained: undefined,
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
    watching: null,
    resigned: false,
    recording: { state: 'off' },

    init: () => {
      if (!get().game && !loading) void load([])
    },

    newGame: () => {
      // A recorded game left before its end counts as resigned (the Game panel then says it was
      // saved); the note that the last finished game was saved is done with.
      const leaving = draft !== null
      finishRecording('left')
      if (!leaving && get().recording.state === 'saved') set({ recording: { state: 'off' } })
      // Computer vs computer waits for Start rather than beginning on its own.
      set({ status: get().game ? 'ready' : 'connecting', running: get().mode !== 'cvc', watching: null })
      void load([])
    },

    play: (from, to) => {
      const s = get()
      if (!s.game || s.status === 'thinking' || s.reviewPly !== null || s.resigned) return
      if (s.moves.length === 0) beginRecording()
      const moves = [...s.moves, moveText(from, to)]
      const gen = ++generation
      void api.game(moves).then(
        (game) => {
          if (gen !== generation) return
          set({ moves, game, status: 'ready', error: null })
          soundFor(game)
          if (game.outcome) finishRecording('over')
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
      if (draft && !s.watching) draft.takebacks += 1
      void load(s.moves.slice(0, n))
    },

    resign: () => {
      const s = get()
      if (s.mode !== 'hvc' || s.watching || !s.game || s.game.outcome || s.resigned || !humanMoved(s.moves, s.humanColor)) return
      generation++ // a computer move on its way is dropped
      if (timer) clearTimeout(timer)
      timer = null
      set({ resigned: true, status: 'ready' })
      playMoveSound('end')
      finishRecording('resigned')
    },
    saveRecording,
    setRecording: (on) => {
      useSettings.getState().setRecordGames(on)
      const s = get()
      if (!on) {
        draft = null
        if (s.recording.state === 'on') set({ recording: { state: 'off' } })
        return
      }
      if (s.mode !== 'hvc' || s.watching || draft) return
      // A game is recorded from its first move. Before yours, it starts now (or again, the
      // computer's opening move played afresh); once you have moved, with the next game.
      if (s.moves.length === 0) beginRecording()
      else if (!s.game?.outcome && !s.resigned && !humanMoved(s.moves, s.humanColor)) get().newGame()
    },

    setMode: (mode) => {
      if (mode !== get().mode) finishRecording('left')
      // Switching to computer vs computer does not start it: the user presses Start.
      const running = mode !== 'cvc'
      set(mode === 'hvc' ? { mode, running, orientation: get().humanColor, watching: null } : { mode, running, watching: null })
      generation++
      schedule()
    },
    setHumanColor: (humanColor) => {
      if (humanColor === get().humanColor) return
      const restart = changeRecorded()
      set({ humanColor, orientation: humanColor })
      if (restart) {
        get().newGame()
        return
      }
      generation++
      schedule()
    },
    setOpponent: (opponent) => {
      if (opponent === get().opponent) return
      const restart = changeRecorded()
      set({ opponent })
      if (restart) get().newGame()
    },
    setAgent: (color, id) => set(color === 1 ? { whiteAgent: id } : { blackAgent: id }),
    setLevel: (slot, level) => {
      if (level === get().levels[slot]) return
      const restart = slot === 'opponent' && changeRecorded()
      set({ levels: { ...get().levels, [slot]: level } })
      if (restart) get().newGame()
    },
    refreshTrained: () => {
      void api.meta().then((m) => set({ trained: m.trained }), () => {})
    },
    watchGame: (moves, from = 'training') => {
      const s = get()
      // Keep the game that was on the board (unless already watching one), to put it back after. A
      // game being recorded waits, untouched, until it is back.
      const watching = { ...(s.watching ?? { moves: s.moves, mode: s.mode, running: s.running, resigned: s.resigned }), from }
      // Two players: nobody moves on their own, so the replay has the board to itself.
      set({ mode: 'hvh', running: true, status: s.game ? 'ready' : 'connecting', watching })
      void load(moves, () => {
        reviewAt(0) // opening review resets its controls...
        set({ replaying: true }) // ...and a watched game plays straight away
        scheduleReplay() // (if the start was cached and is already on show; otherwise it starts once it is)
      })
    },
    leaveWatch: () => {
      const before = get().watching
      if (!before) return
      // A computer-vs-computer game comes back paused, waiting for Start.
      set({ watching: null, mode: before.mode, running: before.mode === 'cvc' ? false : before.running, replaying: false })
      void load(before.moves, () => {
        set({ resigned: before.resigned })
        schedule()
      })
    },
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
      const at = pendingPly ?? s.reviewPly ?? total
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
