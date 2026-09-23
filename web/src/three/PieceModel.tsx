/**
 * Procedural piece models in classical turned proportions: a graduated base, a tapering stem
 * and a collar, crowned by each piece's own emblem (an onion dome and kalash for the Raja, a
 * lotus coronet for the Mantri, a pavilion and wheels for the Ratha). The Gaja and Ashva are
 * carved busts on a turned stand, like the knight of a Staunton set. Geometries are built once
 * and shared by every piece.
 */

import { memo, useEffect, useState } from 'react'
import * as THREE from 'three'
import { SVGLoader } from 'three/examples/jsm/loaders/SVGLoader.js'
import { mergeGeometries, toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { PieceKind } from '../lib/pieces'
import { brass, getTokenMaterials, ivory, ivoryLite, pieceMaterials } from './materials'
import { markShadowsDirty } from './shadows'

/** A turned profile as (radius, height) points, drawn with lines and quadratic curves. */
class Profile {
  pts = [new THREE.Vector2(0, 0)]
  L(r: number, y: number) {
    this.pts.push(new THREE.Vector2(r, y))
    return this
  }
  Q(cr: number, cy: number, r: number, y: number, n = 6) {
    const a = this.pts[this.pts.length - 1].clone()
    for (let i = 1; i <= n; i++) {
      const t = i / n
      const u = 1 - t
      this.pts.push(new THREE.Vector2(u * u * a.x + 2 * u * t * cr + t * t * r, u * u * a.y + 2 * u * t * cy + t * t * y))
    }
    return this
  }
  lathe(segments = 44) {
    return new THREE.LatheGeometry(this.pts, segments)
  }
}

/** The shared graduated base: felt edge, rounded rim, a notch for the brass ring, then a bead. */
const base = (R: number) => new Profile()
  .L(R, 0).L(R, 0.022)
  .Q(R * 1.012, 0.052, R * 0.93, 0.06)
  .L(R * 0.86, 0.066)
  .Q(R * 0.905, 0.084, R * 0.84, 0.1)
  .L(R * 0.72, 0.108)

const R = { padati: 0.26, ashva: 0.3, gaja: 0.31, ratha: 0.31, mantri: 0.3, raja: 0.33 }

/** The short column the Gaja and Ashva busts stand on; its top is STAND_TOP. */
const STAND_TOP = 0.2
const stand = (r: number) => base(r)
  .Q(0.17, 0.114, 0.158, 0.16)
  .L(0.15, 0.172)
  .Q(0.19, 0.172, 0.19, 0.187)
  .Q(0.19, 0.202, 0.16, 0.206)
  .L(0, 0.206)
  .lathe()

interface Bust {
  scale: number
  depth: number
  round: number
  cx: number
  /**
   * Thickness factor by SVG x, so the bust narrows toward the muzzle or trunk. Linear, which
   * keeps the sides flat (a curved side would shade in streaks across its long triangles).
   */
  taper?: (x: number) => number
}

/** Extrude an SVG path (100-unit box, facing left, baseline y = 80) as a rounded, tapered slab. */
function carve(d: string, { scale, depth, round, cx, taper }: Bust) {
  const data = new SVGLoader().parse(`<svg xmlns="http://www.w3.org/2000/svg"><path d="${d}"/></svg>`)
  const shapes = data.paths.flatMap((p) => p.toShapes())
  const geom = new THREE.ExtrudeGeometry(shapes, {
    depth: depth / scale,
    bevelEnabled: true,
    // The rounding eats inward, so the silhouette stays exactly the drawn path.
    bevelSize: round / scale,
    bevelOffset: -round / scale,
    bevelThickness: round / scale,
    bevelSegments: 4,
    curveSegments: 10,
  })
  if (taper) {
    const pos = geom.attributes.position
    const mid = depth / scale / 2
    for (let i = 0; i < pos.count; i++) pos.setZ(i, mid + (pos.getZ(i) - mid) * taper(pos.getX(i)))
  }
  // Flip y (SVG is y-down) and z together so the triangle winding stays correct.
  geom.scale(scale, -scale, -scale)
  geom.translate(-cx * scale, 80 * scale + STAND_TOP, depth / 2)
  geom.deleteAttribute('normal')
  // Smooth shading around the rounded bevel...
  const out = toCreasedNormals(geom, Math.PI / 3)
  // ...but the flat sides (group 0) keep their true face normal: every side vertex lies on the
  // outline, so smoothed normals there would streak across the face.
  const caps = out.groups[0]
  const pos = out.attributes.position
  const normal = out.attributes.normal
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3(), was = new THREE.Vector3()
  for (let i = caps.start; i < caps.start + caps.count; i += 3) {
    a.fromBufferAttribute(pos, i)
    b.fromBufferAttribute(pos, i + 1)
    c.fromBufferAttribute(pos, i + 2)
    n.subVectors(c, b).cross(a.sub(b)).normalize()
    if (n.dot(was.fromBufferAttribute(normal, i)) < 0) n.negate()
    for (let k = 0; k < 3; k++) normal.setXYZ(i + k, n.x, n.y, n.z)
  }
  return out
}

// Bust dimensions, shared with the accents placed on them. The trunk and the muzzle are
// narrower than the head and neck behind them.
const GAJA_BUST: Bust = { scale: 0.0102, depth: 0.16, round: 0.034, cx: 50, taper: (x) => 0.55 + 0.45 * (x - 18) / 62 }
const ASHVA_BUST: Bust = {
  scale: 0.0104, depth: 0.13, round: 0.032, cx: 48,
  taper: (x) => 0.6 + 0.4 * (x - 14) / 62,
}

/** Half the bust's thickness at an SVG point, for setting accents on its surface. */
const bustSide = (b: Bust, x: number) => (b.depth / 2 + b.round) * (b.taper?.(x) ?? 1)

/** The same SVG → world mapping the busts use, for placing accents. */
const onBust = (x: number, y: number, bust: { scale: number; cx: number }): [number, number] =>
  [(x - bust.cx) * bust.scale, (80 - y) * bust.scale + STAND_TOP]

const petal = new THREE.SphereGeometry(1, 12, 8)
const ring = (r: number, tube = 0.012) => new THREE.TorusGeometry(r, tube, 8, 44)

const G = {
  padati: base(R.padati)
    .Q(0.12, 0.118, 0.112, 0.2)
    .Q(0.086, 0.31, 0.08, 0.42)
    .L(0.08, 0.438)
    .Q(0.15, 0.438, 0.153, 0.456)
    .Q(0.156, 0.476, 0.108, 0.486)
    .L(0.074, 0.498)
    .L(0.07, 0.53)
    .L(0, 0.53)
    .lathe(),
  mantri: base(R.mantri)
    .Q(0.13, 0.116, 0.124, 0.22)
    .Q(0.094, 0.44, 0.084, 0.64)
    .L(0.084, 0.658)
    .Q(0.168, 0.658, 0.172, 0.678)
    .Q(0.176, 0.698, 0.118, 0.708)
    .L(0.128, 0.718)
    .Q(0.138, 0.728, 0.098, 0.738)
    .Q(0.086, 0.84, 0.185, 0.925)
    .L(0.196, 0.945)
    .Q(0.188, 0.962, 0.14, 0.962)
    .L(0, 0.962)
    .lathe(),
  raja: base(R.raja)
    .Q(0.145, 0.116, 0.138, 0.24)
    .Q(0.104, 0.5, 0.094, 0.74)
    .L(0.094, 0.758)
    .Q(0.188, 0.758, 0.192, 0.782)
    .Q(0.196, 0.804, 0.128, 0.814)
    .L(0.142, 0.825)
    .Q(0.152, 0.836, 0.108, 0.846)
    .Q(0.096, 0.96, 0.198, 1.045)
    .L(0.208, 1.066)
    .Q(0.2, 1.082, 0.15, 1.082)
    .L(0.132, 1.09)
    // Onion dome, swelling out and drawing up to a spire.
    .Q(0.212, 1.11, 0.198, 1.19, 10)
    .Q(0.184, 1.27, 0.07, 1.32, 10)
    .Q(0.028, 1.345, 0.022, 1.38)
    .L(0, 1.38)
    .lathe(),
  rathaBody: base(R.ratha)
    .Q(0.214, 0.114, 0.212, 0.16)
    .Q(0.218, 0.4, 0.2, 0.54)
    .Q(0.2, 0.572, 0.25, 0.588)
    .L(0.255, 0.642)
    .Q(0.257, 0.662, 0.234, 0.664)
    .L(0.206, 0.664)
    .L(0.194, 0.65)
    .L(0, 0.65)
    .lathe(),
  // Pavilion canopy over the chariot: a shallow dome with an upturned rim.
  rathaCanopy: new Profile()
    .L(0.215, 0)
    .Q(0.232, 0.004, 0.228, 0.022)
    .Q(0.2, 0.024, 0.18, 0.04)
    .Q(0.12, 0.11, 0.03, 0.125)
    .L(0, 0.128)
    .lathe(),
  gajaStand: stand(R.gaja),
  ashvaStand: stand(R.ashva),
  // Elephant bust facing left: domed brow, the trunk falling clear of the chest and curling in.
  gaja: carve(
    'M46 80 L80 80 Q85 60 79 42 Q74 22 62 13 Q52 7 42 10 Q33 13 30 24 Q28 34 25 44 Q21 56 20 66 Q19 76 26 77 Q31 77 31 72 Q30 66 32 58 Q34 50 38 48 Q42 50 44 56 Q46 64 47 70 Q48 76 46 80 Z',
    GAJA_BUST,
  ),
  gajaEar: carve('M53 20 Q67 18 72 31 Q75 44 67 54 Q62 60 57 64 Q55 52 54 42 Q51 30 53 20 Z', { ...GAJA_BUST, depth: 0.004, round: 0.01, taper: undefined }),
  // Horse head facing left: arched neck, pricked ears.
  ashva: carve(
    'M36 80 L76 80 Q79 62 74 46 Q70 32 63 22 L66 9 L58 15 Q54 11 50 12 L47 5 L43 15 Q32 21 26 32 Q20 42 16 49 Q13 55 17 58 Q23 61 30 57 Q36 53 42 51 Q45 57 41 65 Q37 72 36 80 Z',
    ASHVA_BUST,
  ),
  bead: new THREE.SphereGeometry(1, 16, 12),
  tusk: new THREE.ConeGeometry(0.017, 0.13, 14),
  post: new THREE.CylinderGeometry(0.011, 0.011, 0.14, 10),
  wheel: new THREE.TorusGeometry(0.125, 0.016, 10, 40),
  spoke: new THREE.BoxGeometry(0.23, 0.012, 0.012),
  hub: new THREE.CylinderGeometry(0.028, 0.028, 0.05, 16),
  spire: new THREE.ConeGeometry(0.016, 0.1, 16),
  token: new THREE.CylinderGeometry(0.4, 0.4, 0.08, 48),
  tokenRim: new THREE.TorusGeometry(0.4, 0.02, 10, 48),
}

type Mat = 'body' | 'brass' | 'ivory'
interface Part {
  g: THREE.BufferGeometry
  m: Mat
  p?: [number, number, number]
  r?: [number, number, number]
  /** Euler order for r; 'YXZ' reads as "turn to face, then tilt". */
  o?: THREE.EulerOrder
  s?: [number, number, number] | number
}

const X90: [number, number, number] = [Math.PI / 2, 0, 0]
/** The brass ring that sits in every base's notch. */
const baseRing = (r: number): Part => ({ g: ring(r * 0.875, 0.013), m: 'brass', p: [0, 0.064, 0], r: X90 })
const bead = (p: [number, number, number], r: number): Part => ({ g: G.bead, m: 'brass', p, s: r })

/** Every piece as a list of parts: geometry, material and placement. */
function partsFor(kind: PieceKind): Part[] {
  switch (kind) {
    case 1:
      return [
        { g: G.padati, m: 'body' },
        baseRing(R.padati),
        { g: G.bead, m: 'body', p: [0, 0.636, 0], s: 0.118 },
        bead([0, 0.762, 0], 0.022),
      ]
    case 5:
      return [
        { g: G.mantri, m: 'body' },
        baseRing(R.mantri),
        { g: ring(0.17), m: 'brass', p: [0, 0.679, 0], r: X90 },
        // Lotus coronet: petals leaning out around a domed centre.
        ...Array.from({ length: 10 }, (_, i): Part => {
          const a = (i / 10) * Math.PI * 2
          return {
            g: petal, m: 'body',
            p: [Math.cos(a) * 0.132, 1.01, Math.sin(a) * 0.132],
            r: [0.42, Math.PI / 2 - a, 0], o: 'YXZ',
            s: [0.048, 0.078, 0.02],
          }
        }),
        { g: G.bead, m: 'body', p: [0, 0.975, 0], s: [0.1, 0.085, 0.1] },
        { g: ring(0.1, 0.01), m: 'brass', p: [0, 0.972, 0], r: X90 },
        bead([0, 1.085, 0], 0.034),
      ]
    case 6:
      return [
        { g: G.raja, m: 'body' },
        baseRing(R.raja),
        { g: ring(0.19), m: 'brass', p: [0, 0.783, 0], r: X90 },
        { g: ring(0.14, 0.014), m: 'brass', p: [0, 1.092, 0], r: X90 },
        // Kalash finial: a brass pot and spire.
        bead([0, 1.41, 0], 0.038),
        { g: G.bead, m: 'brass', p: [0, 1.452, 0], s: [0.022, 0.012, 0.022] },
        { g: G.spire, m: 'brass', p: [0, 1.5, 0] },
      ]
    case 4:
      return [
        { g: G.rathaBody, m: 'body' },
        baseRing(R.ratha),
        { g: ring(0.252), m: 'brass', p: [0, 0.615, 0], r: X90 },
        // Wheels on the axle, one each side.
        ...[-1, 1].flatMap((side): Part[] => {
          const p: [number, number, number] = [0, 0.31, side * 0.228]
          return [
            { g: G.wheel, m: 'brass', p },
            ...[0, 1, 2, 3].map((i): Part => ({ g: G.spoke, m: 'brass', p, r: [0, 0, (i * Math.PI) / 4] })),
            { g: G.hub, m: 'brass', p, r: X90 },
          ]
        }),
        // Pavilion: slender posts carrying the canopy.
        ...[0, 1, 2, 3, 4, 5].map((i): Part => {
          const a = (i / 6) * Math.PI * 2 + Math.PI / 6
          return { g: G.post, m: 'brass', p: [Math.cos(a) * 0.15, 0.715, Math.sin(a) * 0.15] }
        }),
        { g: G.rathaCanopy, m: 'body', p: [0, 0.778, 0] },
        bead([0, 0.906, 0], 0.028),
      ]
    case 3: {
      const [tx, ty] = onBust(39, 53, GAJA_BUST)
      const [ex, ey] = onBust(42, 24, GAJA_BUST)
      const side = bustSide(GAJA_BUST, 62)
      const eye = bustSide(GAJA_BUST, 42) - 0.008
      // Tusks run forward and down beside the trunk.
      const dir = new THREE.Vector2(-0.75, -0.66)
      const tusk = (z: number): Part => ({
        g: G.tusk, m: 'ivory', p: [tx + dir.x * 0.065, ty + dir.y * 0.065, z], r: [0, 0, Math.atan2(-dir.x, dir.y)],
      })
      return [
        { g: G.gajaStand, m: 'body' },
        baseRing(R.gaja),
        { g: G.gaja, m: 'body' },
        // Ears lie against the head and flare out behind.
        { g: G.gajaEar, m: 'body', p: [0, 0, side], r: [0, -0.28, 0] },
        { g: G.gajaEar, m: 'body', p: [0, 0, -side], r: [0, 0.28, 0] },
        tusk(0.072),
        tusk(-0.072),
        bead([ex, ey, eye], 0.017),
        bead([ex, ey, -eye], 0.017),
      ]
    }
    case 2: {
      const [ex, ey] = onBust(44, 25, ASHVA_BUST)
      const eye = bustSide(ASHVA_BUST, 44) - 0.01
      return [
        { g: G.ashvaStand, m: 'body' },
        baseRing(R.ashva),
        { g: G.ashva, m: 'body' },
        bead([ex, ey, eye], 0.016),
        bead([ex, ey, -eye], 0.016),
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
    q.setFromEuler(new THREE.Euler(...(part.r ?? [0, 0, 0]), part.o ?? 'XYZ'))
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

export const PieceModel = memo(function PieceModel({ kind, white, lite = false }: { kind: PieceKind; white: boolean; lite?: boolean }) {
  const g = mergedFor(kind)
  return (
    <group>
      {/* White's tusks are ivory like its body, so they share the body mesh. */}
      <mesh geometry={white ? g.bodyWithIvory : g.body} material={pieceMaterials(white, lite)} castShadow receiveShadow />
      <mesh geometry={g.brass} material={brass} castShadow />
      {!white && g.ivory && <mesh geometry={g.ivory} material={lite ? ivoryLite : ivory} castShadow />}
    </group>
  )
})

/** Flat-view token: a disc with the piece glyph, for the top-down view and slow machines. */
export const PieceToken = memo(function PieceToken({ kind, white, lite = false }: { kind: PieceKind; white: boolean; lite?: boolean }) {
  const materials = getTokenMaterials(kind, white, lite)
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
export function PieceWarmup({ lite }: { lite: boolean }) {
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
          <PieceModel kind={k} white={white} lite={lite} />
          <PieceToken kind={k} white={white} lite={lite} />
        </group>
      )))}
    </group>
  )
}
