/** World coordinates: one square = 1 unit, a1 at (-3.5, 0, 3.5), the playing surface at y = 0. */

import type { PieceKind } from '../lib/pieces'

export function squareToWorld(sq: number): [number, number] {
  return [(sq & 7) - 3.5, 3.5 - (sq >> 3)]
}

export function worldToSquare(x: number, z: number): number | null {
  const file = Math.floor(x + 4)
  const rank = Math.floor(4 - z)
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null
  return rank * 8 + file
}

/** Height of each piece model's top (world units), for placing labels above it. */
export const PIECE_TOP: Record<PieceKind, number> = { 1: 0.97, 2: 0.86, 3: 0.88, 4: 0.97, 5: 1.05, 6: 1.38 }
