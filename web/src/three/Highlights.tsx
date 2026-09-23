/**
 * Square overlays. Each kind of highlight has a distinct shape as well as colour, so none
 * relies on colour alone: plain ring = move, corner brackets = capture, outline = last move,
 * thick ring = check, dashed square = keyboard cursor.
 */

import { Line } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { squareToWorld } from './coords'

export interface HighlightColors {
  accent: string
  danger: string
  ink: string
}

const Y = 0.012

function squareOutline(inset: number): [number, number, number][] {
  const h = 0.5 - inset
  return [[-h, Y, -h], [h, Y, -h], [h, Y, h], [-h, Y, h], [-h, Y, -h]]
}

function MoveTarget({ sq, capture, hovered, color }: { sq: number; capture: boolean; hovered: boolean; color: string }) {
  const [x, z] = squareToWorld(sq)
  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ color, transparent: true, opacity: hovered ? 1 : 0.85, depthWrite: false }),
    [color, hovered],
  )
  if (capture) {
    return (
      <group position={[x, Y, z]} rotation={[-Math.PI / 2, 0, 0]}>
        {[0, 1, 2, 3].map((i) => (
          <mesh key={i} material={material}>
            <ringGeometry args={[0.36, hovered ? 0.46 : 0.43, 10, 1, Math.PI / 4 + (i * Math.PI) / 2 - 0.42, 0.84]} />
          </mesh>
        ))}
      </group>
    )
  }
  return (
    <mesh position={[x, Y, z]} rotation={[-Math.PI / 2, 0, 0]} material={material}>
      <ringGeometry args={[hovered ? 0.15 : 0.13, hovered ? 0.24 : 0.19, 40]} />
    </mesh>
  )
}

function CheckRing({ sq, color, animate }: { sq: number; color: string; animate: boolean }) {
  const [x, z] = squareToWorld(sq)
  const ref = useRef<THREE.MeshBasicMaterial>(null)
  useFrame(({ clock, invalidate }) => {
    if (!ref.current || !animate) return
    ref.current.opacity = 0.65 + Math.sin(clock.elapsedTime * 4) * 0.3
    invalidate() // the scene renders on demand; keep frames coming while the pulse runs
  })
  // A thick ring plus a soft wash under the piece, so check reads even around a wide plinth.
  return (
    <group position={[x, Y + 0.002, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <mesh>
        <circleGeometry args={[0.48, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.28} depthWrite={false} />
      </mesh>
      <mesh>
        <ringGeometry args={[0.39, 0.49, 48]} />
        <meshBasicMaterial ref={ref} color={color} transparent opacity={0.9} depthWrite={false} />
      </mesh>
    </group>
  )
}

export function Highlights(props: {
  colors: HighlightColors
  selected: number | null
  targets: { sq: number; capture: boolean }[]
  hovered: number | null
  lastMove: { from: number; to: number } | null
  checkSquare: number | null
  cursor: number | null
  emphasis: number[]
  animate: boolean
}) {
  const { colors, selected, targets, hovered, lastMove, checkSquare, cursor, emphasis, animate } = props
  const outline = useMemo(() => squareOutline(0.04), [])
  const cursorOutline = useMemo(() => squareOutline(0.07), [])
  return (
    <group>
      {lastMove &&
        [lastMove.from, lastMove.to].map((sq) => {
          const [x, z] = squareToWorld(sq)
          return (
            <group key={`lm${sq}`} position={[x, 0, z]}>
              <mesh position={[0, Y - 0.004, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[0.96, 0.96]} />
                <meshBasicMaterial color={colors.accent} transparent opacity={0.13} depthWrite={false} />
              </mesh>
              <Line points={outline} color={colors.accent} lineWidth={1.6} transparent opacity={0.7} />
            </group>
          )
        })}
      {emphasis.map((sq) => {
        const [x, z] = squareToWorld(sq)
        return (
          <mesh key={`em${sq}`} position={[x, Y - 0.003, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[0.96, 0.96]} />
            <meshBasicMaterial color={colors.accent} transparent opacity={0.22} depthWrite={false} />
          </mesh>
        )
      })}
      {selected !== null && (() => {
        const [x, z] = squareToWorld(selected)
        return (
          <mesh position={[x, Y, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.47, 48]} />
            <meshBasicMaterial color={colors.accent} transparent opacity={0.3} depthWrite={false} />
          </mesh>
        )
      })()}
      {targets.map((t) => (
        <MoveTarget key={`t${t.sq}`} sq={t.sq} capture={t.capture} hovered={hovered === t.sq} color={colors.accent} />
      ))}
      {checkSquare !== null && <CheckRing sq={checkSquare} color={colors.danger} animate={animate} />}
      {cursor !== null && (() => {
        const [x, z] = squareToWorld(cursor)
        return (
          <group position={[x, 0.004, z]}>
            <Line points={cursorOutline} color={colors.ink} lineWidth={2.2} dashed dashSize={0.12} gapSize={0.08} />
          </group>
        )
      })()}
    </group>
  )
}
