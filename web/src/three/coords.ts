/** World coordinates: one square = 1 unit, a1 at (-3.5, 0, 3.5), the playing surface at y = 0. */

export function squareToWorld(sq: number): [number, number] {
  return [(sq & 7) - 3.5, 3.5 - (sq >> 3)]
}

export function worldToSquare(x: number, z: number): number | null {
  const file = Math.floor(x + 4)
  const rank = Math.floor(4 - z)
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null
  return rank * 8 + file
}
