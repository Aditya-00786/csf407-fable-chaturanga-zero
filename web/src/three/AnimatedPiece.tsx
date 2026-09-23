import { animated, easings, to, useSpring } from '@react-spring/three'
import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { memo, useEffect, useRef } from 'react'
import { ASHVA, GAJA, PIECES, type PieceKind } from '../lib/pieces'
import type { LabelMode } from '../state/settings'
import { squareToWorld } from './coords'
import { PieceModel, PieceToken } from './PieceModel'

interface Props {
  piece: number
  sq: number
  lifted: boolean
  flat: boolean
  flipped: boolean
  labels: LabelMode
  reducedMotion: boolean
  onPick: (e: ThreeEvent<PointerEvent>, sq: number) => void
}

export const AnimatedPiece = memo(function AnimatedPiece({ piece, sq, lifted, flat, flipped, labels, reducedMotion, onPick }: Props) {
  const kind = Math.abs(piece) as PieceKind
  const white = piece > 0
  const from = useRef(sq)
  const [{ t, lift }, api] = useSpring(() => ({ t: 1, lift: 0 }))

  useEffect(() => {
    if (from.current === sq) return
    api.start({
      from: { t: 0 },
      to: { t: 1 },
      immediate: reducedMotion,
      config: { duration: 360, easing: easings.easeOutQuart },
      onRest: () => { from.current = sq },
    })
  }, [sq, api, reducedMotion])

  useEffect(() => {
    api.start({ lift: lifted ? 0.14 : 0, immediate: reducedMotion, config: { duration: 180, easing: easings.easeOutQuart } })
  }, [lifted, api, reducedMotion])

  // Leapers (Gaja, Ashva) hop in an arc; everything else glides.
  const arc = kind === GAJA || kind === ASHVA ? 0.55 : 0.12
  // The callback runs per animation frame, not during render, so reading the ref is safe
  // (and needed: it must see the latest start square).
  // oxlint-disable-next-line react/refs
  const position = to([t, lift], (v: number, l: number) => {
    const [ax, az] = squareToWorld(from.current)
    const [bx, bz] = squareToWorld(sq)
    return [ax + (bx - ax) * v, Math.sin(Math.PI * v) * arc + l, az + (bz - az) * v] as [number, number, number]
  })

  const info = PIECES[kind]
  // Profile pieces face left for White and right for Black, like knights in a chess set.
  const facing = (white ? 1 : -1) * (flipped ? -1 : 1)
  return (
    <animated.group position={position} onPointerDown={(e: ThreeEvent<PointerEvent>) => onPick(e, sq)}>
      {flat ? (
        <group rotation={[0, flipped ? Math.PI : 0, 0]}>
          <PieceToken kind={kind} white={white} />
        </group>
      ) : (
        <group scale={[facing, 1, 1]} rotation={[0, flipped ? Math.PI : 0, 0]}>
          <PieceModel kind={kind} white={white} />
        </group>
      )}
      {labels !== 'off' && (
        <Html position={flat ? [0, 0.1, flipped ? -0.52 : 0.52] : [0, 1.05, 0]} center zIndexRange={[5, 0]} style={{ pointerEvents: 'none' }}>
          <span
            className={
              'whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium shadow-sm ' +
              (white ? 'bg-[#f3e9d2] text-[#3a2118]' : 'bg-[#2a1d18] text-[#f3e9d2]') +
              (labels === 'sanskrit' ? ' font-deva text-[13px] leading-5' : '')
            }
          >
            {labels === 'sanskrit' ? info.devanagari : info.name}
          </span>
        </Html>
      )}
    </animated.group>
  )
})
