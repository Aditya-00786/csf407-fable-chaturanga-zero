/** The Ashtapada: a carved rosewood frame around an uncheckered field with inlaid lines and brass crosses. */

import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { MARKED_SQUARES } from '../lib/board'
import { prefersReducedMotion } from '../state/settings'
import { TRAY, squareToWorld } from './coords'
import { brass, darkBrass, getFieldTexture, getFrameTexture, makeCoordinateTexture } from './materials'

const FIELD = 8.24
const OUTER = 10
const FRAME_DEPTH = 0.44
const FRAME_RAISE = 0.09

function roundedRect(half: number, r: number) {
  const s = new THREE.Shape()
  s.moveTo(-half + r, -half)
  s.lineTo(half - r, -half)
  s.quadraticCurveTo(half, -half, half, -half + r)
  s.lineTo(half, half - r)
  s.quadraticCurveTo(half, half, half - r, half)
  s.lineTo(-half + r, half)
  s.quadraticCurveTo(-half, half, -half, half - r)
  s.lineTo(-half, -half + r)
  s.quadraticCurveTo(-half, -half, -half + r, -half)
  return s
}

/** A rounded rectangle centred at (cx, cy). */
function roundedRectAt(cx: number, cy: number, w: number, h: number, r: number) {
  const s = new THREE.Shape()
  const [x0, x1, y0, y1] = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
  s.moveTo(x0 + r, y0)
  s.lineTo(x1 - r, y0)
  s.quadraticCurveTo(x1, y0, x1, y0 + r)
  s.lineTo(x1, y1 - r)
  s.quadraticCurveTo(x1, y1, x1 - r, y1)
  s.lineTo(x0 + r, y1)
  s.quadraticCurveTo(x0, y1, x0, y1 - r)
  s.lineTo(x0, y0 + r)
  s.quadraticCurveTo(x0, y0, x0 + r, y0)
  return s
}

/**
 * The two trays beside the board where captured pieces are set down. Extruded in the frame's
 * own coordinates, so the rosewood grain continues from the frame.
 */
function useTrayGeometry() {
  return useMemo(() => {
    const shapes = [-1, 1].map((side) => roundedRectAt(side * TRAY.x, 0, TRAY.width, TRAY.length, 0.12))
    const g = new THREE.ExtrudeGeometry(shapes, {
      depth: TRAY.depth, bevelEnabled: true, bevelSize: 0.04, bevelThickness: 0.04, bevelSegments: 3, curveSegments: 8,
    })
    g.rotateX(-Math.PI / 2)
    g.translate(0, TRAY.top - TRAY.depth - 0.04, 0)
    return g
  }, [])
}

function useFrameGeometry() {
  return useMemo(() => {
    const shape = roundedRect(OUTER / 2 - 0.06, 0.28)
    const hole = new THREE.Path()
    const h = FIELD / 2
    hole.moveTo(-h, -h)
    hole.lineTo(-h, h)
    hole.lineTo(h, h)
    hole.lineTo(h, -h)
    hole.lineTo(-h, -h)
    shape.holes.push(hole)
    const g = new THREE.ExtrudeGeometry(shape, {
      depth: FRAME_DEPTH, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.06, bevelSegments: 4, curveSegments: 12,
    })
    g.rotateX(-Math.PI / 2)
    g.translate(0, FRAME_RAISE - FRAME_DEPTH - 0.06, 0)
    return g
  }, [])
}

function Cross({ sq }: { sq: number }) {
  const [x, z] = squareToWorld(sq)
  return (
    <group position={[x, 0.004, z]}>
      {[Math.PI / 4, -Math.PI / 4].map((r) => (
        <mesh key={r} rotation={[0, r, 0]} material={brass} receiveShadow>
          <boxGeometry args={[0.7, 0.008, 0.03]} />
        </mesh>
      ))}
    </group>
  )
}

/** Letters readable from White's side and from Black's, crossfaded as the board turns. */
function CoordinateOverlays({ white, black, flipped }: { white: THREE.Texture; black: THREE.Texture; flipped: boolean }) {
  const whiteMat = useRef<THREE.MeshStandardMaterial>(null)
  const blackMat = useRef<THREE.MeshStandardMaterial>(null)
  const [initial] = useState(flipped ? 0 : 1)
  useFrame(({ invalidate }, dt) => {
    if (!whiteMat.current || !blackMat.current) return
    const target = flipped ? 0 : 1
    const current = whiteMat.current.opacity
    if (current === target) return
    const next = prefersReducedMotion() || Math.abs(target - current) < 0.003
      ? target
      : current + (target - current) * (1 - Math.exp(-7 * dt))
    whiteMat.current.opacity = next
    blackMat.current.opacity = 1 - next
    invalidate()
  })
  const plane = [OUTER - 0.12, OUTER - 0.12] as [number, number]
  return (
    <>
      <mesh position={[0, FRAME_RAISE + 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={plane} />
        <meshStandardMaterial ref={whiteMat} map={white} transparent opacity={initial} depthWrite={false} metalness={0.6} roughness={0.35} />
      </mesh>
      <mesh position={[0, FRAME_RAISE + 0.003, 0]} rotation={[-Math.PI / 2, 0, Math.PI]}>
        <planeGeometry args={plane} />
        <meshStandardMaterial ref={blackMat} map={black} transparent opacity={1 - initial} depthWrite={false} metalness={0.6} roughness={0.35} />
      </mesh>
    </>
  )
}

export function Board({ flipped, lite }: { flipped: boolean; lite: boolean }) {
  const frameGeometry = useFrameGeometry()
  const trayGeometry = useTrayGeometry()
  // Reduced graphics drops the lacquer (clear-coat) layers for a much cheaper shader.
  const fieldMaterials = useMemo(() => {
    const side = new THREE.MeshStandardMaterial({ color: '#7a4a2a', roughness: 0.7 })
    const top = lite
      ? new THREE.MeshStandardMaterial({ map: getFieldTexture(), roughness: 0.6 })
      : new THREE.MeshPhysicalMaterial({ map: getFieldTexture(), roughness: 0.55, clearcoat: 0.25, clearcoatRoughness: 0.5 })
    return [side, side, top, side, side, side]
  }, [lite])
  const frameMaterial = useMemo(
    () => lite
      ? new THREE.MeshStandardMaterial({ map: getFrameTexture(), roughness: 0.5 })
      : new THREE.MeshPhysicalMaterial({ map: getFrameTexture(), roughness: 0.45, clearcoat: 0.55, clearcoatRoughness: 0.3 }),
    [lite],
  )

  // Draw both sets of coordinates once, when the serif font is available (so they don't fall
  // back to Georgia). Flipping the board then crossfades between them instead of redrawing a
  // large texture mid-turn.
  const [coords, setCoords] = useState<{ white: THREE.CanvasTexture; black: THREE.CanvasTexture } | null>(null)
  useEffect(() => {
    let alive = true
    let made: { white: THREE.CanvasTexture; black: THREE.CanvasTexture } | null = null
    document.fonts.load('600 40px "Newsreader Variable"').finally(() => {
      if (!alive) return
      made = { white: makeCoordinateTexture(false), black: makeCoordinateTexture(true) }
      setCoords(made)
    })
    return () => {
      alive = false
      made?.white.dispose()
      made?.black.dispose()
    }
  }, [])

  return (
    <group>
      {/* Playing field; its top surface is y = 0. */}
      <mesh position={[0, -0.15, 0]} material={fieldMaterials} receiveShadow>
        <boxGeometry args={[FIELD, 0.3, FIELD]} />
      </mesh>
      <mesh geometry={frameGeometry} material={frameMaterial} castShadow receiveShadow />
      <mesh geometry={trayGeometry} material={frameMaterial} castShadow receiveShadow />
      {coords && <CoordinateOverlays white={coords.white} black={coords.black} flipped={flipped} />}
      {/* Thin brass lip where the frame meets the field. */}
      {[0, 1, 2, 3].map((i) => (
        <group key={i} rotation={[0, (i * Math.PI) / 2, 0]}>
          <mesh position={[0, 0.012, FIELD / 2 + 0.005]} material={darkBrass}>
            <boxGeometry args={[FIELD + 0.03, 0.02, 0.025]} />
          </mesh>
        </group>
      ))}
      {/* Brass diamonds at the frame corners. */}
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sz], i) => (
        <mesh key={i} position={[sx * 4.56, FRAME_RAISE + 0.01, sz * 4.56]} rotation={[0, Math.PI / 4, 0]} material={brass}>
          <boxGeometry args={[0.2, 0.02, 0.2]} />
        </mesh>
      ))}
      {MARKED_SQUARES.map((sq) => <Cross key={sq} sq={sq} />)}
      {/* Invisible table that catches the board's shadow (not needed without shadows). */}
      {!lite && (
        <mesh position={[0, -0.62, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[40, 40]} />
          <shadowMaterial opacity={0.35} />
        </mesh>
      )}
    </group>
  )
}
