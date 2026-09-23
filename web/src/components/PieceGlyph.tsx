import { GLYPHS, PIECES, type PieceKind } from '../lib/pieces'

/** 2D piece icon: the same silhouettes as the flat-view tokens, as ivory or ebony. */
export function PieceGlyph({ piece, size = 20, title }: { piece: number; size?: number; title?: string }) {
  const kind = Math.abs(piece) as PieceKind
  const white = piece > 0
  const label = title ?? `${white ? 'White' : 'Black'} ${PIECES[kind].name}`
  return (
    <svg viewBox="0 0 100 100" width={size} height={size} role="img" aria-label={label} className="shrink-0">
      <path
        d={GLYPHS[kind]}
        fillRule="evenodd"
        fill={white ? '#f1e6cc' : '#2b1e19'}
        stroke={white ? '#6b4a2e' : '#c9a14c'}
        strokeWidth={4}
        strokeLinejoin="round"
      />
    </svg>
  )
}
