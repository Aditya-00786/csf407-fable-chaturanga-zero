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

const smoothstep = (a: number, b: number, v: number) => {
  const x = Math.min(1, Math.max(0, (v - a) / (b - a)))
  return x * x * (3 - 2 * x)
}

export const AnimatedPiece = memo(function AnimatedPiece({ piece, sq, lifted, flat, flipped, labels, reducedMotion, onPick }: Props) {
  const kind = Math.abs(piece) as PieceKind
  const white = piece > 0
  const from = useRef(sq)
  // t: travel between squares. lift: raised when selected (3D). pop: enlarged when selected
  // (flat, where a lift can't be seen from above). f: 0 = 3D model, 1 = flat token.
  const [{ t, lift, pop, f }, api] = useSpring(() => ({ t: 1, lift: 0, pop: 1, f: flat ? 1 : 0 }))

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
    api.start({
      lift: lifted && !flat ? 0.14 : 0,
      pop: lifted && flat ? 1.1 : 1,
      immediate: reducedMotion,
      config: { duration: 180, easing: easings.easeOutQuart },
    })
  }, [lifted, flat, api, reducedMotion])

  // Morph between the carved piece and the flat token, in step with the camera's glide.
  useEffect(() => {
    api.start({ f: flat ? 1 : 0, immediate: reducedMotion, config: { duration: 420, easing: easings.easeInOutCubic } })
  }, [flat, api, reducedMotion])

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
  // The model settles into its base over the first part of the morph; the token rises over the rest.
  const modelScale = to([f], (v: number) => [facing, Math.max(0.001, 1 - smoothstep(0, 0.6, v)), 1] as [number, number, number])
  const tokenScale = to([f, pop], (v: number, p: number) => {
    const s = Math.max(0.001, smoothstep(0.35, 1, v)) * p
    return [s, s, s] as [number, number, number]
  })
  return (
    <animated.group position={position} onPointerDown={(e: ThreeEvent<PointerEvent>) => onPick(e, sq)}>
      <animated.group scale={modelScale} rotation={[0, flipped ? Math.PI : 0, 0]} visible={f.to((v) => v < 0.999)}>
        <PieceModel kind={kind} white={white} />
      </animated.group>
      {/* Tokens keep a fixed orientation: each side's glyphs face their own player. */}
      <animated.group scale={tokenScale} visible={f.to((v) => v > 0.001)}>
        <PieceToken kind={kind} white={white} />
      </animated.group>
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
