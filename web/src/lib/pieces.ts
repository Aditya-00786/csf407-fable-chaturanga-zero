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

/** Silhouettes shared by the 2D glyphs and the flat-view tokens. Fill with evenodd. */
export const GLYPHS: Record<PieceKind, string> = {
  // Foot soldier: helmeted head, tapering body, round shield.
  1: `M50 11 L55 20 A11 11 0 1 1 45 20 Z M43 42 H57 L64 78 H36 Z ${PLINTH}`,
  // Minister beneath a chhatri (royal parasol).
  5: `M50 5 A4 4 0 1 1 49.9 5 Z M20 36 Q50 4 80 36 Z M47.5 36 H52.5 V44 H47.5 Z M41 44 H59 L66 78 H34 Z ${PLINTH}`,
  // King: three-pointed crown over a banded body.
  6: `M28 34 L31 12 L41 24 L50 6 L59 24 L69 12 L72 34 Z M29 36 H71 V43 H29 Z M39 43 H61 L69 78 H31 Z ${PLINTH}`,
  // Chariot: canopy, body and two spoked wheels.
  4: `M28 38 L50 20 L72 38 Z M25 40 H75 V62 H25 Z
      M36 58 A13 13 0 1 1 35.9 58 Z M36 66 A5 5 0 1 0 36.1 66 Z
      M64 58 A13 13 0 1 1 63.9 58 Z M64 66 A5 5 0 1 0 64.1 66 Z ${PLINTH}`,
  // Elephant facing left, trunk lowered, with an eye.
  3: `M83 80 L83 56 Q83 33 61 31 Q47 29 39 33 Q27 29 20 40 Q14 49 16 60 Q17 70 12 75 Q9 80 15 80
      Q22 79 24 69 Q25 61 30 58 L34 80 Z M29 43 A2.6 2.6 0 1 0 29.1 43 Z ${PLINTH}`,
  // Horse head facing left, with an eye.
  2: `M71 80 L69 60 Q72 40 63 28 L66 14 L56 21 Q50 17 44 21 Q33 28 25 42 Q21 50 26 54 Q33 58 40 54
      Q46 51 51 55 Q45 66 40 80 Z M45 32 A2.6 2.6 0 1 0 45.1 32 Z ${PLINTH}`,
}
