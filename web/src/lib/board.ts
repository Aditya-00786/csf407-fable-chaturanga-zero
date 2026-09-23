/** Square helpers, the starting position and human-readable descriptions of results. */

import type { Color, HistoryEntry, Outcome } from './api'
import { PIECES, type PieceKind } from './pieces'

export const FILES = 'abcdefgh'

export const fileOf = (sq: number) => sq & 7
export const rankOf = (sq: number) => sq >> 3
export const squareName = (sq: number) => FILES[fileOf(sq)] + String(rankOf(sq) + 1)
export const squareIndex = (name: string) => FILES.indexOf(name[0]) + (Number(name[1]) - 1) * 8
export const moveText = (from: number, to: number) => squareName(from) + squareName(to)

export const colorName = (c: Color) => (c === 1 ? 'White' : 'Black')

/** The traditional Ashtapada markings: corners, the middle of each edge, and the centre (Murray p. 64). */
export const MARKED_SQUARES = ['a1', 'h1', 'a8', 'h8', 'd1', 'e1', 'd8', 'e8', 'a4', 'a5', 'h4', 'h5', 'd4', 'e4', 'd5', 'e5']
  .map(squareIndex)

/** Standard crosswise setup (docs/RULES.md section 2), used before the engine responds. */
export const INITIAL_BOARD: number[] = (() => {
  const white = [4, 2, 3, 5, 6, 3, 2, 4]
  const black = [4, 2, 3, 6, 5, 3, 2, 4]
  const board = new Array(64).fill(0)
  for (let f = 0; f < 8; f++) {
    board[f] = white[f]
    board[8 + f] = 1
    board[48 + f] = -1
    board[56 + f] = -black[f]
  }
  return board
})()

export function pieceLabel(piece: number) {
  return PIECES[Math.abs(piece) as PieceKind]
}

/** Short notation like "Gaja c1–e3" or "Padati d7×e6". */
export function describeMove(h: HistoryEntry) {
  const info = pieceLabel(h.piece)
  const sep = h.captured ? '×' : '–'
  let text = `${info.name} ${squareName(h.from)}${sep}${squareName(h.to)}`
  if (h.promotion) text += ` = ${pieceLabel(h.promotion).name}`
  if (h.check) text += ' +'
  return text
}

export interface OutcomeText {
  title: string
  detail: string
  source: string
}

/** Plain-language explanation of why the game ended (PRODUCT.md principle 4). */
export function explainOutcome(outcome: Outcome, sideToMove: Color): OutcomeText {
  const w = outcome.winner === null ? null : colorName(outcome.winner)
  const loser = outcome.winner === null ? null : colorName(-outcome.winner as Color)
  switch (outcome.reason) {
    case 'bare_king':
      if (w === null) {
        return { title: 'Draw: both kings bare', detail: 'Neither side has anything left but its Raja.', source: 'RULES.md §5.1' }
      }
      return {
        title: `${w} wins: bare king`,
        detail: `${loser} has only the Raja left. Stripping the enemy army down to its king wins at once, even if that Raja could capture back.`,
        source: 'Bare-king rule, Murray (1913) p. 61',
      }
    case 'checkmate':
      return {
        title: `${w} wins by checkmate`,
        detail: `${loser}'s Raja is attacked and has no legal way out.`,
        source: 'RULES.md §5.2',
      }
    case 'stalemate':
      return {
        title: `${w} wins by stalemate`,
        detail: `${colorName(sideToMove)} has no legal move but is not in check. In Chaturanga the stalemated player wins, the opposite of modern chess.`,
        source: 'Old Indian rule reported by al-Adli, Murray (1913) pp. 60–61',
      }
    case 'repetition':
      return {
        title: 'Draw by repetition',
        detail: 'The same position appeared for the third time with the same side to move.',
        source: 'Shatranj practice, Murray (1913) p. 229',
      }
    case 'no_progress':
      return {
        title: 'Draw: no progress',
        detail: '70 moves each passed without a capture or a Padati move.',
        source: "Shatranj's 70-move rule",
      }
    case 'max_plies':
      return {
        title: 'Draw: move limit reached',
        detail: 'The game reached 300 plies, the engine\'s cap so that self-play games always finish.',
        source: 'Engineering rule, RULES.md §5.6',
      }
  }
}
