/** Piece metadata and 2D glyph silhouettes (100x100 SVG space, baseline at y = 92). */

export const PADATI = 1
export const ASHVA = 2
export const GAJA = 3
export const RATHA = 4
export const MANTRI = 5
export const RAJA = 6

export type PieceKind = 1 | 2 | 3 | 4 | 5 | 6

export interface PieceInfo {
  kind: PieceKind
  letter: string
  name: string // transliterated Sanskrit
  devanagari: string
  english: string
  chess: string
  move: string
}

export const PIECES: Record<PieceKind, PieceInfo> = {
  6: {
    kind: 6, letter: 'K', name: 'Raja', devanagari: 'राजा', english: 'King', chess: 'King',
    move: 'One step in any direction. No castling.',
  },
  5: {
    kind: 5, letter: 'M', name: 'Mantri', devanagari: 'मन्त्री', english: 'Minister', chess: 'Ferz (early queen)',
    move: 'One step diagonally.',
  },
  4: {
    kind: 4, letter: 'R', name: 'Ratha', devanagari: 'रथ', english: 'Chariot', chess: 'Rook',
    move: 'Any distance along a rank or file. Cannot jump.',
  },
  3: {
    kind: 3, letter: 'G', name: 'Gaja', devanagari: 'गज', english: 'Elephant', chess: 'Alfil (early bishop)',
    move: 'Exactly two squares diagonally, jumping over the square between.',
  },
  2: {
    kind: 2, letter: 'N', name: 'Ashva', devanagari: 'अश्व', english: 'Horse', chess: 'Knight',
    move: 'An L-shaped leap, like a knight. Jumps over pieces.',
  },
  1: {
    kind: 1, letter: 'P', name: 'Padati', devanagari: 'पदाति', english: 'Foot soldier', chess: 'Pawn',
    move: 'One step forward, captures one step diagonally forward. No double step. Becomes a Mantri on the last rank.',
  },
}

export const PIECE_ORDER: PieceKind[] = [6, 5, 4, 3, 2, 1]

const PLINTH = 'M24 80 H76 Q81 80 81 85 V92 H19 V85 Q19 80 24 80 Z'
/** A full circle as a path, for eyes, heads, wheels and shields (nested circles cut rings with evenodd). */
const ring = (cx: number, cy: number, r: number) =>
  `M${cx - r} ${cy} A${r} ${r} 0 1 0 ${cx + r} ${cy} A${r} ${r} 0 1 0 ${cx - r} ${cy} Z`

/**
 * Silhouettes shared by the 2D glyphs and the flat-view tokens, drawn after the 3D set: robed
 * court figures, a temple chariot, a caparisoned elephant and a horse. Fill with evenodd; parts
 * are kept apart by thin gaps, which read as carved lines.
 */
export const GLYPHS: Record<PieceKind, string> = {
  // Raja: robed figure under a tiered mukut.
  6: `M50 4 L53.5 12 H56 L58 21 H42 L44 12 H46.5 Z M40 23 H60 L61 29 H39 Z ${ring(50, 36.5, 5.5)}
      M41 44.5 Q50 42.5 59 44.5 Q62 45.5 62 49 L60 57 H40 L38 49 Q38 45.5 41 44.5 Z
      M40.5 59.5 H59.5 Q63 70 71 78 H29 Q37 70 40.5 59.5 Z ${PLINTH}`,
  // Mantri: robed courtier in a pagdi with a sarpech (cut out) and a plume.
  5: `M54 5 Q63 8 61 18 L58 18 Q59 11 53 8 Z M38.5 29 Q38 19.5 50 19 Q62 19.5 61.5 29 Z ${ring(50, 24, 2)} ${ring(50, 36, 5.5)}
      M42 43.5 Q50 42 58 43.5 Q61 44.5 61 48 L59 55 H41 L39 48 Q39 44.5 42 43.5 Z
      M41.5 57.5 H58.5 Q62 68 69 78 H31 Q38 68 41.5 57.5 Z ${PLINTH}`,
  // Padati: foot soldier with a spear and a round shield.
  1: `M73 5 L76.5 17 H74.6 V78 H72.4 V17 H70.5 Z
      M43 31 Q43 21 50 17 Q57 21 57 31 Z ${ring(50, 37, 5)}
      M44 44.5 Q50 43 56 44.5 Q59 45.5 59 48.5 L58 55 H42 L41 48.5 Q41 45.5 44 44.5 Z
      M42 57 H58 L61.5 66 H38.5 Z M42.5 68 H48 V78 H42.5 Z M52 68 H57.5 V78 H52 Z
      ${ring(29, 52, 9)} ${ring(29, 52, 6.5)} ${ring(29, 52, 2.5)} ${PLINTH}`,
  // Ratha: temple chariot, a stepped shikhara over the car, one great wheel.
  4: `M50 3 L53 11 H47 Z M50 12 L63 24 H37 Z M34 25.5 H66 V29.5 H34 Z M30 31 H70 V35 H30 Z
      M33 36.5 H36.5 V46 H33 Z M63.5 36.5 H67 V46 H63.5 Z M48.2 36.5 H51.8 V46 H48.2 Z
      M27 47.5 H73 V58 H27 Z M27 52 L13 57 V59.5 L27 55 Z
      ${ring(50, 69.5, 10)} ${ring(50, 69.5, 7)} ${ring(50, 69.5, 2.6)} ${PLINTH}`,
  // Gaja: caparisoned elephant facing left, with a howdah under a canopy.
  3: `${ring(53, 6.5, 2.5)} M53 10 Q63 10 65 20 H41 Q43 10 53 10 Z
      M43 21.5 H45.5 V29 H43 Z M60.5 21.5 H63 V29 H60.5 Z M41 30.5 H65 V37 H41 Z
      M36 39.5 Q58 36.5 75 40.5 Q85 44 85 56 Q85 63 81 65 V78 H72.5 V67 H50.5 V78 H42 V66 Q37 64 34 61
      Q31 66 28 71 Q25 77 20 78 Q15 78 16 74 Q19 73 21 70 Q24 64 24 56 Q22 46 28 42 Q31 40 36 39.5 Z
      ${ring(27.5, 49, 1.7)} M34 45 Q43 44 44 52 Q43 60 36 58 Q32 52 34 45 Z ${PLINTH}`,
  // Ashva: horse facing left, ears curving in, with a plume and a saddle cloth line.
  2: `M27 3 Q33 4 32 11 L30 11 Q30 6 26 5 Z
      M13 37 Q10 34 13 31 L22 19 L23.5 11 L27 16.5 L29.5 11.5 L31.5 18 Q41 22 46 37 Q56 40 70 39 Q79 40 81 46
      Q88 50 87 64 L84.5 64 Q84.5 54 81 51 Q81 57 79 62 L80 78 H75 L73.5 64 Q68 60 62 60 H48 Q44 61 43 64 L42 78 H37 L36.5 64
      Q33 58 32 49 Q30 41 26 35.5 Q22 38.5 18 39 Q15 39 13 37 Z ${ring(20.5, 26.5, 1.5)} M50 41 H66 V43.5 H50 Z ${PLINTH}`,
}
