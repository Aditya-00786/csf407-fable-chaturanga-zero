import { animated, easings, to, useSpring } from '@react-spring/three'
import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { memo, useEffect, useMemo, useRef } from 'react'
import { ASHVA, GAJA, PIECES, type PieceKind } from '../lib/pieces'
import type { LabelMode } from '../state/settings'
import { PIECE_TOP, TAKEN_SCALE, squareToWorld, takenToWorld } from './coords'
import { PieceModel, PieceToken } from './PieceModel'
import { markShadowsDirty } from './shadows'

interface Props {
  piece: number
  sq: number
  /** Place in the line of captured pieces beside the board, or null while in play. */
  taken: number | null
  lifted: boolean
  flat: boolean
  /** Only used to place flat-view labels on the viewer's side; false when labels are off. */
  labelFlipped: boolean
  labels: LabelMode
  reducedMotion: boolean
  lite: boolean
  onPick: (e: ThreeEvent<PointerEvent>, sq: number) => void
}

/** How high a piece is lifted on its way to or from a capture tray. */
const CAPTURE_ARC = 1.1

/**
 * Extra turn for a piece in a capture tray, so it faces the board. On the board a model faces the
 * opponent; a quarter-turn more points White's (on the left tray) and Black's (on the right tray)
 * inwards alike.
 */
const TRAY_TURN = -Math.PI / 2

/** Where a piece stands, at what size and turned how far from its usual facing: x, y, z, scale, turn. */
type Place = [number, number, number, number, number]

function placeOf(piece: number, sq: number, taken: number | null): Place {
  if (taken !== null) return [...takenToWorld(piece, taken), TAKEN_SCALE, TRAY_TURN]
  const [x, z] = squareToWorld(sq)
  return [x, 0, z, 1, 0]
}

const smoothstep = (a: number, b: number, v: number) => {
  const x = Math.min(1, Math.max(0, (v - a) / (b - a)))
  return x * x * (3 - 2 * x)
}

export const AnimatedPiece = memo(function AnimatedPiece({ piece, sq, taken, lifted, flat, labelFlipped, labels, reducedMotion, lite, onPick }: Props) {
  const kind = Math.abs(piece) as PieceKind
  const white = piece > 0
  const captured = taken !== null
  const target = useMemo(() => placeOf(piece, sq, taken), [piece, sq, taken])
  // The current journey's end points. They change only when its animation starts, together with
  // resetting t, so a new place is never drawn before the piece has travelled there (a delayed
  // start would otherwise show the piece at its destination, then snap back and fly again).
  const journey = useRef({ from: target, to: target, arc: 0 })
  // t: travel between squares. lift: raised when selected (3D). pop: enlarged when selected
  // (flat, where a lift can't be seen from above). f: 0 = 3D model, 1 = flat token.
  const [{ t, lift, pop, f }, api] = useSpring(() => ({ t: 1, lift: 0, pop: 1, f: flat ? 1 : 0 }))

  // Leapers (Gaja, Ashva) hop in an arc, captured pieces are lifted clear of the board, and
  // everything else glides.
  const arc = captured ? CAPTURE_ARC : kind === GAJA || kind === ASHVA ? 0.55 : 0.12

  useEffect(() => {
    const j = journey.current
    if (j.to === target) return
    // Set off from wherever the piece is now, even if it is still on its way (or mid-hop) elsewhere.
    const v = t.get()
    const here = j.from.map((a, i) => a + (j.to[i] - a) * v) as Place
    here[1] += Math.sin(Math.PI * v) * j.arc
    // Anything going to or from a tray (a capture, or a capture undone in review) is lifted clear
    // of the frame.
    const viaTray = here[3] !== 1 || target[3] !== 1
    journey.current = { from: here, to: target, arc: viaTray ? CAPTURE_ARC : arc }
    api.set({ t: 0 })
    api.start({
      t: 1,
      immediate: reducedMotion,
      // A captured piece waits a beat for its captor, then is carried off to the side.
      delay: captured ? 140 : 0,
      config: { duration: captured ? 620 : 360, easing: captured ? easings.easeInOutCubic : easings.easeOutQuart },
      onChange: markShadowsDirty,
    })
  }, [target, captured, arc, t, api, reducedMotion])

  useEffect(() => {
    api.start({
      lift: lifted && !flat ? 0.14 : 0,
      pop: lifted && flat ? 1.1 : 1,
      immediate: reducedMotion,
      config: { duration: 180, easing: easings.easeOutQuart },
      onChange: markShadowsDirty,
    })
  }, [lifted, flat, api, reducedMotion])

  // Morph between the carved piece and the flat token, in step with the camera's glide.
  useEffect(() => {
    api.start({
      f: flat ? 1 : 0,
      immediate: reducedMotion,
      config: { duration: 420, easing: easings.easeInOutCubic },
      onChange: markShadowsDirty,
    })
  }, [flat, api, reducedMotion])

  // Appearing, disappearing (capture) or changing kind (promotion) changes the shadows too.
  useEffect(() => {
    markShadowsDirty()
    return markShadowsDirty
  }, [piece, sq, flat, lifted])

  // The callbacks run per animation frame, not during render, so reading the ref is safe
  // (and needed: they must see the current journey).
  // oxlint-disable-next-line react/refs
  const position = to([t, lift], (v: number, l: number) => {
    const { from: a, to: b, arc: h } = journey.current
    return [a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v + Math.sin(Math.PI * v) * h + l, a[2] + (b[2] - a[2]) * v] as [number, number, number]
  })
  // oxlint-disable-next-line react/refs
  const size = to([t], (v: number) => {
    const { from: a, to: b } = journey.current
    const s = a[3] + (b[3] - a[3]) * v
    return [s, s, s] as [number, number, number]
  })
  // The model turns to face the board on its way into a tray, and back on its way out.
  // oxlint-disable-next-line react/refs
  const turn = to([t], (v: number) => {
    const { from: a, to: b } = journey.current
    return [0, a[4] + (b[4] - a[4]) * v, 0] as [number, number, number]
  })

  const info = PIECES[kind]
  // Each model faces its own way (see PieceModel), independent of how the board is turned.
  // The model settles into its base over the first part of the morph; the token rises over the rest.
  const modelScale = to([f], (v: number) => [1, Math.max(0.001, 1 - smoothstep(0, 0.6, v)), 1] as [number, number, number])
  const tokenScale = to([f, pop], (v: number, p: number) => {
    const s = Math.max(0.001, smoothstep(0.35, 1, v)) * p
    return [s, s, s] as [number, number, number]
  })
  return (
    <animated.group
      position={position}
      scale={size}
      // Captured pieces are out of play: clicks on them do nothing.
      onPointerDown={captured ? undefined : (e: ThreeEvent<PointerEvent>) => onPick(e, sq)}
    >
      <animated.group scale={modelScale} rotation={turn} visible={f.to((v) => v < 0.999)}>
        <PieceModel kind={kind} white={white} lite={lite} />
      </animated.group>
      {/* Tokens keep a fixed orientation: each side's glyphs face their own player. */}
      <animated.group scale={tokenScale} visible={f.to((v) => v > 0.001)}>
        <PieceToken kind={kind} white={white} lite={lite} />
      </animated.group>
      {labels !== 'off' && !captured && (
        <Html position={flat ? [0, 0.1, labelFlipped ? -0.52 : 0.52] : [0, PIECE_TOP[kind] + 0.14, 0]} center zIndexRange={[5, 0]} style={{ pointerEvents: 'none' }}>
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
