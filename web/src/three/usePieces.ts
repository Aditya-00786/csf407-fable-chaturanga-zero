/**
 * Gives every piece a stable identity across positions, so a move animates the moving piece
 * instead of re-creating the board. When the new position is not exactly one move on from
 * the previous one (new game, undo, review jump), identities are rebuilt.
 */

import { useState } from 'react'

export interface TrackedPiece {
  id: number
  piece: number // signed: > 0 White, < 0 Black
  sq: number
}

interface Snapshot {
  board: number[]
  ply: number
  pieces: TrackedPiece[]
}

let nextId = 1

function build(board: number[]): TrackedPiece[] {
  const pieces: TrackedPiece[] = []
  board.forEach((p, sq) => {
    if (p) pieces.push({ id: nextId++, piece: p, sq })
  })
  return pieces
}

function sameExcept(a: number[], b: number[], skip: number[]) {
  for (let i = 0; i < 64; i++) if (!skip.includes(i) && a[i] !== b[i]) return false
  return true
}

function advance(prev: Snapshot, board: number[], ply: number, lastMove: { from: number; to: number } | null): Snapshot {
  const oneMoveOn =
    lastMove && ply === prev.ply + 1 &&
    prev.board[lastMove.from] !== 0 && board[lastMove.from] === 0 && board[lastMove.to] !== 0 &&
    sameExcept(prev.board, board, [lastMove.from, lastMove.to])
  if (!oneMoveOn || !lastMove) return { board, ply, pieces: build(board) }
  const pieces = prev.pieces
    .filter((p) => p.sq !== lastMove.to)
    .map((p) => (p.sq === lastMove.from ? { ...p, sq: lastMove.to, piece: board[lastMove.to] } : p))
  return { board, ply, pieces }
}

export function usePieces(board: number[], ply: number, lastMove: { from: number; to: number } | null) {
  const [snap, setSnap] = useState<Snapshot>(() => ({ board, ply, pieces: build(board) }))
  if (snap.board !== board) {
    // Derive from the previous render's snapshot (React's "adjust state on prop change" pattern).
    const next = advance(snap, board, ply, lastMove)
    setSnap(next)
    return next.pieces
  }
  return snap.pieces
}
