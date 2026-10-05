/**
 * Gives every piece a stable identity across positions, so a move animates the moving piece
 * instead of re-creating the board, and a captured piece leaves its square for its place beside
 * the board. Going one move back (stepping back through a review) plays the move in reverse,
 * bringing a captured piece back from its tray. Any other change (new game, a jump in review)
 * rebuilds the identities from the board and the list of captures.
 */

import { useState } from 'react'

export interface TrackedPiece {
  id: number
  piece: number // signed: > 0 White, < 0 Black
  /** The square it stands on, or for a captured piece, the square it was taken on. */
  sq: number
  /** Place in its colour's line of captured pieces, or null while in play. */
  taken: number | null
}

type Move = { from: number; to: number }

interface Snapshot {
  board: number[]
  ply: number
  lastMove: Move | null
  pieces: TrackedPiece[]
}

const takenCount = (pieces: TrackedPiece[], white: boolean) =>
  pieces.filter((p) => p.taken !== null && p.piece > 0 === white).length

let nextId = 1

function build(board: number[], captured: number[]): TrackedPiece[] {
  const pieces: TrackedPiece[] = []
  board.forEach((p, sq) => {
    if (p) pieces.push({ id: nextId++, piece: p, sq, taken: null })
  })
  for (const p of captured) pieces.push({ id: nextId++, piece: p, sq: -1, taken: takenCount(pieces, p > 0) })
  return pieces
}

function sameExcept(a: number[], b: number[], skip: number[]) {
  for (let i = 0; i < 64; i++) if (!skip.includes(i) && a[i] !== b[i]) return false
  return true
}

function advance(prev: Snapshot, board: number[], ply: number, lastMove: Move | null, captured: number[]): Snapshot {
  const next = (pieces: TrackedPiece[]): Snapshot => ({ board, ply, lastMove, pieces })
  const oneMoveOn =
    lastMove && ply === prev.ply + 1 &&
    prev.board[lastMove.from] !== 0 && board[lastMove.from] === 0 && board[lastMove.to] !== 0 &&
    sameExcept(prev.board, board, [lastMove.from, lastMove.to])
  if (oneMoveOn && lastMove) {
    return next(prev.pieces.map((p) => {
      if (p.taken !== null) return p
      if (p.sq === lastMove.to) return { ...p, taken: takenCount(prev.pieces, p.piece > 0) }
      if (p.sq === lastMove.from) return { ...p, sq: lastMove.to, piece: board[lastMove.to] }
      return p
    }))
  }
  const undone = prev.lastMove
  const oneMoveBack =
    undone && ply === prev.ply - 1 &&
    prev.board[undone.from] === 0 && board[undone.from] !== 0 &&
    sameExcept(prev.board, board, [undone.from, undone.to])
  if (oneMoveBack && undone) {
    // The piece that moved goes back (as it was before promoting); a piece it captured, the
    // latest one taken of its colour, returns from the tray.
    const back = board[undone.to]
    let returning: TrackedPiece | null = null
    if (back !== 0) {
      for (const p of prev.pieces) {
        if (p.taken !== null && p.piece === back && (returning === null || p.taken > (returning.taken ?? -1))) returning = p
      }
    }
    if (back === 0 || returning) {
      return next(prev.pieces.map((p) => {
        if (p === returning) return { ...p, sq: undone.to, taken: null }
        if (p.taken === null && p.sq === undone.to) return { ...p, sq: undone.from, piece: board[undone.from] }
        return p
      }))
    }
  }
  return next(build(board, captured))
}

/** `captured` lists the pieces taken so far (signed codes), in the order they were taken. */
export function usePieces(board: number[], ply: number, lastMove: Move | null, captured: number[]) {
  const [snap, setSnap] = useState<Snapshot>(() => ({ board, ply, lastMove, pieces: build(board, captured) }))
  if (snap.board !== board) {
    // Derive from the previous render's snapshot (React's "adjust state on prop change" pattern).
    const next = advance(snap, board, ply, lastMove, captured)
    setSnap(next)
    return next.pieces
  }
  return snap.pieces
}
