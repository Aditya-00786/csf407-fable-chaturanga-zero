/** The Ashtapada: a carved rosewood frame around an uncheckered field with inlaid lines and brass crosses. */

import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { MARKED_SQUARES } from '../lib/board'
import { squareToWorld } from './coords'
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

export function Board({ flipped }: { flipped: boolean }) {
  const frameGeometry = useFrameGeometry()
  const fieldMaterials = useMemo(() => {
    const side = new THREE.MeshStandardMaterial({ color: '#7a4a2a', roughness: 0.7 })
    const top = new THREE.MeshPhysicalMaterial({
      map: getFieldTexture(), roughness: 0.55, clearcoat: 0.25, clearcoatRoughness: 0.5,
    })
    return [side, side, top, side, side, side]
  }, [])
  const frameMaterial = useMemo(
    () => new THREE.MeshPhysicalMaterial({ map: getFrameTexture(), roughness: 0.45, clearcoat: 0.55, clearcoatRoughness: 0.3 }),
    [],
  )

  // Draw the coordinates once the serif font is available, so they don't fall back to Georgia.
  const [coords, setCoords] = useState<THREE.CanvasTexture | null>(null)
  useEffect(() => {
    let alive = true
    let tex: THREE.CanvasTexture | null = null
    document.fonts.load('600 40px "Newsreader Variable"').finally(() => {
      if (!alive) return
      tex = makeCoordinateTexture(flipped)
      setCoords(tex)
    })
    return () => {
      alive = false
      tex?.dispose()
    }
  }, [flipped])

  return (
    <group>
      {/* Playing field; its top surface is y = 0. */}
      <mesh position={[0, -0.15, 0]} material={fieldMaterials} receiveShadow>
        <boxGeometry args={[FIELD, 0.3, FIELD]} />
      </mesh>
      <mesh geometry={frameGeometry} material={frameMaterial} castShadow receiveShadow />
      {coords && (
        <mesh position={[0, FRAME_RAISE + 0.002, 0]} rotation={[-Math.PI / 2, 0, flipped ? Math.PI : 0]}>
          <planeGeometry args={[OUTER - 0.12, OUTER - 0.12]} />
          <meshStandardMaterial map={coords} transparent depthWrite={false} metalness={0.6} roughness={0.35} />
        </mesh>
      )}
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
      {/* Invisible table that catches the board's shadow. */}
      <mesh position={[0, -0.62, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[40, 40]} />
        <shadowMaterial opacity={0.35} />
      </mesh>
    </group>
  )
}
