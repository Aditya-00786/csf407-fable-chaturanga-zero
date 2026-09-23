/**
 * Procedural piece models. Turned (lathe) bodies for the human figures, extruded carved
 * silhouettes for the elephant and horse, and a built-up chariot. All share a plinth with a
 * brass ring. Geometries are created once and reused by every piece.
 */

import { memo, useEffect, useState } from 'react'
import * as THREE from 'three'
import { SVGLoader } from 'three/examples/jsm/loaders/SVGLoader.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { PieceKind } from '../lib/pieces'
import { brass, ebony, getTokenMaterials, ivory } from './materials'
import { markShadowsDirty } from './shadows'

const PLINTH_TOP = 0.1

const lathe = (pts: [number, number][], segments = 48) =>
  new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), segments)

function extrudeSvg(d: string, scale: number, depth: number, lift: number, cx: number) {
  const data = new SVGLoader().parse(`<svg xmlns="http://www.w3.org/2000/svg"><path d="${d}"/></svg>`)
  const shapes = data.paths.flatMap((p) => p.toShapes())
  const geom = new THREE.ExtrudeGeometry(shapes, {
    depth,
    bevelEnabled: true,
    bevelSize: 2.6,
    bevelThickness: 0.03,
    bevelSegments: 4,
    curveSegments: 18,
  })
  // Flip y (SVG is y-down) and z together so the triangle winding stays correct.
  geom.scale(scale, -scale, -1)
  geom.translate(-cx * scale, 80 * scale + lift, depth / 2)
  geom.computeVertexNormals()
  return geom
}

const G = {
  plinth: lathe([[0, 0], [0.36, 0], [0.378, 0.022], [0.372, 0.06], [0.335, 0.083], [0.3, 0.1], [0, 0.1]]),
  plinthRing: new THREE.TorusGeometry(0.3, 0.017, 12, 48),
  padati: lathe([[0, 0.1], [0.26, 0.1], [0.2, 0.15], [0.155, 0.22], [0.13, 0.33], [0.125, 0.37], [0.165, 0.39],
    [0.165, 0.415], [0.1, 0.43], [0, 0.43]]),
  padatiHead: new THREE.SphereGeometry(0.105, 32, 20),
  padatiHelmet: new THREE.ConeGeometry(0.06, 0.1, 24),
  mantri: lathe([[0, 0.1], [0.25, 0.1], [0.2, 0.14], [0.15, 0.21], [0.12, 0.31], [0.115, 0.42], [0.165, 0.45],
    [0.165, 0.48], [0.1, 0.5], [0, 0.5]]),
  pole: new THREE.CylinderGeometry(0.02, 0.02, 0.22, 12),
  chhatri: new THREE.SphereGeometry(0.31, 48, 12, 0, Math.PI * 2, 0, Math.PI / 3),
  chhatriRim: new THREE.TorusGeometry(0.268, 0.014, 10, 56),
  bead: new THREE.SphereGeometry(0.02, 10, 8),
  finial: new THREE.SphereGeometry(0.038, 16, 12),
  raja: lathe([[0, 0.1], [0.3, 0.1], [0.23, 0.17], [0.185, 0.33], [0.16, 0.58], [0.215, 0.61], [0.215, 0.655],
    [0.14, 0.685], [0, 0.685]]),
  crownBand: new THREE.TorusGeometry(0.155, 0.032, 12, 40),
  crownPoint: new THREE.ConeGeometry(0.036, 0.13, 16),
  crownTop: new THREE.ConeGeometry(0.05, 0.16, 20),
  rathaBase: lathe([[0, 0], [0.36, 0], [0.378, 0.022], [0.372, 0.06], [0.335, 0.083], [0.3, 0.1], [0, 0.1]]),
  rathaBody: new THREE.BoxGeometry(0.46, 0.24, 0.34),
  wheel: new THREE.TorusGeometry(0.15, 0.03, 12, 36),
  spoke: new THREE.BoxGeometry(0.27, 0.022, 0.022),
  hub: new THREE.CylinderGeometry(0.035, 0.035, 0.06, 16),
  post: new THREE.CylinderGeometry(0.018, 0.018, 0.2, 10),
  roof: new THREE.ConeGeometry(0.34, 0.18, 4),
  // Carved elephant, facing left, with ears and a small howdah on its back.
  gaja: extrudeSvg(
    'M83 80 L83 56 Q83 33 61 31 Q47 29 39 33 Q27 29 20 40 Q14 49 16 60 Q17 70 12 75 Q9 80 15 80 Q22 79 24 69 Q25 61 30 58 L34 80 Z',
    0.009, 0.24, PLINTH_TOP, 47,
  ),
  gajaEar: extrudeSvg('M40 36 Q56 33 57 48 Q56 62 42 58 Q37 47 40 36 Z', 0.009, 0.04, PLINTH_TOP, 47),
  tusk: new THREE.ConeGeometry(0.02, 0.13, 12),
  howdah: new THREE.SphereGeometry(0.075, 24, 10, 0, Math.PI * 2, 0, Math.PI / 2),
  howdahBase: new THREE.CylinderGeometry(0.09, 0.09, 0.03, 24),
  // Carved horse head, facing left.
  ashva: extrudeSvg(
    'M71 80 L69 60 Q72 40 63 28 L66 14 L56 21 Q50 17 44 21 Q33 28 25 42 Q21 50 26 54 Q33 58 40 54 Q46 51 51 55 Q45 66 40 80 Z',
    0.0095, 0.22, PLINTH_TOP, 47,
  ),
  eye: new THREE.SphereGeometry(0.022, 12, 10),
  token: new THREE.CylinderGeometry(0.4, 0.4, 0.08, 48),
  tokenRim: new THREE.TorusGeometry(0.4, 0.02, 10, 48),
}

// The same SVG → world mapping the extrusions use, for placing accents.
const svgToWorld = (x: number, y: number, scale: number, cx = 47): [number, number] =>
  [(x - cx) * scale, (80 - y) * scale + PLINTH_TOP]

type Mat = 'body' | 'brass' | 'ivory'
interface Part {
  g: THREE.BufferGeometry
  m: Mat
  p?: [number, number, number]
  r?: [number, number, number]
  s?: [number, number, number] | number
}

const X90: [number, number, number] = [Math.PI / 2, 0, 0]
const plinth = (): Part[] => [{ g: G.plinth, m: 'body' }, { g: G.plinthRing, m: 'brass', p: [0, PLINTH_TOP, 0], r: X90 }]

/** Every piece as a list of parts: geometry, material and placement. */
function partsFor(kind: PieceKind): Part[] {
  switch (kind) {
    case 1:
      return [
        ...plinth(),
        { g: G.padati, m: 'body' },
        { g: G.padatiHead, m: 'body', p: [0, 0.52, 0] },
        { g: G.padatiHelmet, m: 'brass', p: [0, 0.645, 0] },
      ]
    case 5:
      return [
        ...plinth(),
        { g: G.mantri, m: 'body' },
        { g: G.pole, m: 'brass', p: [0, 0.6, 0] },
        // Chhatri: a shallow royal parasol with a beaded brass fringe.
        { g: G.chhatri, m: 'body', p: [0, 0.58, 0], s: [1, 0.5, 1] },
        { g: G.chhatriRim, m: 'brass', p: [0, 0.657, 0], r: X90 },
        ...Array.from({ length: 14 }, (_, i): Part => {
          const a = (i / 14) * Math.PI * 2
          return { g: G.bead, m: 'brass', p: [Math.cos(a) * 0.268, 0.63, Math.sin(a) * 0.268] }
        }),
        { g: G.finial, m: 'brass', p: [0, 0.77, 0] },
      ]
    case 6:
      return [
        ...plinth(),
        { g: G.raja, m: 'body' },
        { g: G.crownBand, m: 'brass', p: [0, 0.71, 0], r: X90 },
        ...[0, 1, 2, 3, 4, 5].map((i): Part => {
          const a = (i / 6) * Math.PI * 2
          return {
            g: G.crownPoint, m: 'brass',
            p: [Math.cos(a) * 0.14, 0.79, Math.sin(a) * 0.14],
            r: [Math.sin(a) * 0.35, 0, -Math.cos(a) * 0.35],
          }
        }),
        { g: G.crownTop, m: 'body', p: [0, 0.8, 0] },
        { g: G.finial, m: 'brass', p: [0, 0.9, 0] },
      ]
    case 4:
      return [
        { g: G.rathaBase, m: 'body' },
        { g: G.plinthRing, m: 'brass', p: [0, PLINTH_TOP, 0], r: X90 },
        { g: G.rathaBody, m: 'body', p: [0, 0.3, 0] },
        ...[-1, 1].flatMap((side): Part[] => {
          const p: [number, number, number] = [0, 0.27, side * 0.2]
          return [
            { g: G.wheel, m: 'brass', p },
            { g: G.spoke, m: 'brass', p },
            { g: G.spoke, m: 'brass', p, r: [0, 0, Math.PI / 3] },
            { g: G.spoke, m: 'brass', p, r: [0, 0, -Math.PI / 3] },
            { g: G.hub, m: 'brass', p, r: X90 },
          ]
        }),
        ...[[-0.19, -0.13], [0.19, -0.13], [-0.19, 0.13], [0.19, 0.13]].map(([x, z]): Part => ({ g: G.post, m: 'brass', p: [x, 0.52, z] })),
        { g: G.roof, m: 'body', p: [0, 0.7, 0], r: [0, Math.PI / 4, 0] },
        { g: G.finial, m: 'brass', p: [0, 0.81, 0] },
      ]
    case 3: {
      const [tx, ty] = svgToWorld(24, 61, 0.009)
      const [hx, hy] = svgToWorld(62, 31, 0.009)
      return [
        ...plinth(),
        { g: G.gaja, m: 'body' },
        { g: G.gajaEar, m: 'body', p: [0, 0, 0.14] },
        { g: G.gajaEar, m: 'body', p: [0, 0, -0.14] },
        ...[-0.06, 0.06].map((z): Part => ({ g: G.tusk, m: 'ivory', p: [tx - 0.05, ty - 0.03, z], r: [0, 0, 2.3] })),
        { g: G.howdahBase, m: 'brass', p: [hx, hy - 0.005, 0] },
        { g: G.howdah, m: 'brass', p: [hx, hy + 0.01, 0], s: [1, 0.8, 1] },
        { g: G.finial, m: 'brass', p: [hx, hy + 0.08, 0], s: 0.6 },
      ]
    }
    case 2: {
      const [ex, ey] = svgToWorld(45, 32, 0.0095)
      return [
        ...plinth(),
        { g: G.ashva, m: 'body' },
        ...[-0.14, 0.14].map((z): Part => ({ g: G.eye, m: 'brass', p: [ex, ey, z] })),
      ]
    }
  }
}

function merge(parts: Part[]): THREE.BufferGeometry | null {
  if (parts.length === 0) return null
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const geoms = parts.map((part) => {
    const s = part.s ?? 1
    q.setFromEuler(new THREE.Euler(...(part.r ?? [0, 0, 0])))
    m.compose(
      new THREE.Vector3(...(part.p ?? [0, 0, 0])),
      q,
      typeof s === 'number' ? new THREE.Vector3(s, s, s) : new THREE.Vector3(...s),
    )
    // Mixed indexed/non-indexed sources (extrusions are non-indexed) must match to merge.
    const g = part.g.index ? part.g.toNonIndexed() : part.g.clone()
    g.applyMatrix4(m)
    for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(name)) g.deleteAttribute(name)
    return g
  })
  const merged = mergeGeometries(geoms, false)
  geoms.forEach((g) => g.dispose())
  return merged
}

interface Merged { body: THREE.BufferGeometry; brass: THREE.BufferGeometry; ivory: THREE.BufferGeometry | null; bodyWithIvory: THREE.BufferGeometry }
const mergedCache = new Map<PieceKind, Merged>()

/**
 * Each piece merged into one geometry per material (body, brass, and ivory tusks), so a piece
 * costs 2-3 draw calls instead of 10-20. Built once per kind and shared by every piece.
 */
function mergedFor(kind: PieceKind): Merged {
  let cached = mergedCache.get(kind)
  if (!cached) {
    const parts = partsFor(kind)
    const pick = (mat: Mat) => parts.filter((p) => p.m === mat)
    cached = {
      body: merge(pick('body'))!,
      brass: merge(pick('brass'))!,
      ivory: merge(pick('ivory')),
      bodyWithIvory: merge([...pick('body'), ...pick('ivory')])!,
    }
    mergedCache.set(kind, cached)
  }
  return cached
}

export const PieceModel = memo(function PieceModel({ kind, white }: { kind: PieceKind; white: boolean }) {
  const g = mergedFor(kind)
  return (
    <group>
      {/* White's tusks are ivory like its body, so they share the body mesh. */}
      <mesh geometry={white ? g.bodyWithIvory : g.body} material={white ? ivory : ebony} castShadow receiveShadow />
      <mesh geometry={g.brass} material={brass} castShadow />
      {!white && g.ivory && <mesh geometry={g.ivory} material={ivory} castShadow />}
    </group>
  )
})

/** Flat-view token: a disc with the piece glyph, for the top-down view and slow machines. */
export const PieceToken = memo(function PieceToken({ kind, white }: { kind: PieceKind; white: boolean }) {
  const materials = getTokenMaterials(kind, white)
  return (
    <group>
      <mesh geometry={G.token} material={materials} position={[0, 0.04, 0]} castShadow />
      <mesh geometry={G.tokenRim} material={brass} position={[0, 0.08, 0]} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  )
})

/**
 * Shader warm-up: renders one of every model and token for a moment, hidden under the board,
 * so the first switch between 3D and flat view never stalls to compile shaders or upload textures.
 */
export function PieceWarmup() {
  const [done, setDone] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setDone(true), 1500)
    return () => {
      clearTimeout(t)
      markShadowsDirty()
    }
  }, [])
  if (done) return null
  const kinds: PieceKind[] = [1, 2, 3, 4, 5, 6]
  return (
    <group position={[0, -3, 0]} scale={0.2}>
      {kinds.flatMap((k) => [true, false].map((white) => (
        <group key={`${k}${white}`}>
          <PieceModel kind={k} white={white} />
          <PieceToken kind={k} white={white} />
        </group>
      )))}
    </group>
  )
}
