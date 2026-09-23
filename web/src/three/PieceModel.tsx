/**
 * Procedural piece models after the carved figural sets of Indian courts: robed figures for the
 * Raja, Mantri and Padati, a temple chariot for the Ratha, and a caparisoned elephant and a
 * Marwari horse for the Gaja and Ashva, each on a lotus pedestal. Figures face the opponent;
 * the chariot and the animals stand in profile, like knights in a chess set. Geometries are
 * built once and shared by every piece.
 */

import { memo, useEffect, useState } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { PieceKind } from '../lib/pieces'
import { brass, getTokenMaterials, ivory, ivoryLite, pieceMaterials } from './materials'
import { markShadowsDirty } from './shadows'

type V3 = [number, number, number]

/** A turned profile as (radius, height) points, drawn with lines and quadratic curves. */
class Profile {
  pts: THREE.Vector2[]
  constructor(y0 = 0) {
    this.pts = [new THREE.Vector2(0, y0)]
  }
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
  lathe(segments = 32) {
    return new THREE.LatheGeometry(this.pts, segments)
  }
}

/** A smooth tube through the points whose radius eases from r0 to r1: limbs, trunks, tails. */
function tube(points: V3[], r0: number, r1: number, tubular = 16, radial = 10) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)))
  const g = new THREE.TubeGeometry(curve, tubular, 1, radial, false)
  const pos = g.attributes.position
  const c = new THREE.Vector3()
  const v = new THREE.Vector3()
  for (let i = 0; i <= tubular; i++) {
    curve.getPointAt(i / tubular, c)
    const r = r0 + (r1 - r0) * (i / tubular)
    for (let j = 0; j <= radial; j++) {
      const k = i * (radial + 1) + j
      v.fromBufferAttribute(pos, k).sub(c).multiplyScalar(r).add(c)
      pos.setXYZ(k, v.x, v.y, v.z)
    }
  }
  return g
}

/** Top of every pedestal. */
const PED = 0.142

/** Padmapitha: a plinth band, then a lotus cushion under a lip, as under a temple image. */
const pedestal = (R: number) => new Profile()
  .L(R, 0).L(R, 0.024)
  .Q(R * 1.01, 0.05, R * 0.9, 0.056)
  .L(R * 0.84, 0.062)
  .Q(R * 0.98, 0.08, R * 0.86, 0.104)
  .L(R * 0.8, 0.11)
  .Q(R * 0.86, 0.118, R * 0.84, 0.13)
  .Q(R * 0.83, 0.14, R * 0.76, PED)
  .L(0, PED)
  .lathe()

const R = { padati: 0.25, ashva: 0.3, gaja: 0.32, ratha: 0.32, mantri: 0.3, raja: 0.33 }

const ball = new THREE.SphereGeometry(1, 20, 14)
const bead = new THREE.SphereGeometry(1, 10, 7)
const petal = new THREE.SphereGeometry(1, 8, 5)
const ring = (r: number, tube = 0.01) => new THREE.TorusGeometry(r, tube, 6, 36)
const X90: V3 = [Math.PI / 2, 0, 0]

type Mat = 'body' | 'brass' | 'ivory'
interface Part {
  g: THREE.BufferGeometry
  m: Mat
  p?: V3
  r?: V3
  s?: V3 | number
}

const brassBead = (p: V3, r: number): Part => ({ g: bead, m: 'brass', p, s: r })

/** Lotus petals in relief around a pedestal's cushion, and the brass ring above them. */
function pedestalParts(R: number): Part[] {
  const n = 16
  return [
    { g: pedestal(R), m: 'body' },
    { g: ring(R * 0.81), m: 'brass', p: [0, 0.109, 0], r: X90 },
    ...Array.from({ length: n }, (_, i): Part => {
      const a = ((i + 0.5) / n) * Math.PI * 2
      return {
        g: petal, m: 'body',
        p: [Math.cos(a) * R * 0.9, 0.084, Math.sin(a) * R * 0.9],
        r: [0, -a, -0.5],
        s: [0.022, 0.03, (Math.PI * R * 0.9) / n],
      }
    }),
  ]
}

interface FigureSpec {
  hemY: number
  hem: number
  waistY: number
  waist: number
  shoulderY: number
  chest: number
  neck: number
  head: number
  arm: number
  /** A brass border along the hem, for court robes. */
  border?: boolean
  /** Where the hands rest; by default together in front of the waist. */
  hands?: [V3, V3]
}

/** Figures are deeper side to side than front to back; their front is -x. */
const DEPTH = 0.78

/** A robed figure: flared jama, sash, shoulders, arms, head. Returns the parts and the head's top. */
function figure(f: FigureSpec): { parts: Part[]; headTop: number; headY: number } {
  const robe = new Profile(f.hemY)
    .L(f.hem, f.hemY)
    .Q(f.hem * 1.04, f.hemY + 0.024, f.hem * 0.92, f.hemY + 0.04)
    .Q(f.waist * 1.05, f.hemY + (f.waistY - f.hemY) * 0.6, f.waist, f.waistY)
    .Q(f.chest * 1.06, f.waistY + (f.shoulderY - f.waistY) * 0.45, f.chest, f.shoulderY - 0.03)
    .Q(f.chest * 0.98, f.shoulderY + 0.01, f.neck, f.shoulderY + 0.02)
    .L(f.neck, f.shoulderY + 0.05)
    .L(0, f.shoulderY + 0.05)
    .lathe()
  const headY = f.shoulderY + 0.05 + f.head * 0.8
  const front = -f.chest * DEPTH
  const hands = f.hands ?? [
    [front - 0.012, f.waistY + 0.05, 0.026],
    [front - 0.012, f.waistY + 0.05, -0.026],
  ]
  const elbowY = (f.waistY + f.shoulderY) / 2 - 0.01
  const arms = hands.map((h): Part[] => {
    const side = Math.sign(h[2]) || 1
    return [
      { g: tube([[0, f.shoulderY - 0.025, side * f.chest * 0.92], [0.012, elbowY, side * (f.chest + f.arm * 0.6)], h], f.arm, f.arm * 0.8), m: 'body' },
      { g: bead, m: 'body', p: h, s: f.arm * 0.95 },
    ]
  }).flat()
  return {
    parts: [
      { g: robe, m: 'body', s: [DEPTH, 1, 1] },
      // Kamarband (waist sash) and necklace in brass.
      { g: ring(f.waist * 1.02, 0.011), m: 'brass', p: [0, f.waistY, 0], r: X90, s: [DEPTH, 1, 1] },
      { g: ring(f.neck + 0.012, 0.008), m: 'brass', p: [0, f.shoulderY + 0.02, 0], r: X90 },
      ...(f.border ? [{ g: ring(f.hem * 0.985, 0.012), m: 'brass', p: [0, f.hemY + 0.03, 0], r: X90, s: [DEPTH, 1, 1] } satisfies Part] : []),
      ...arms,
      { g: ball, m: 'body', p: [0, headY, 0], s: [f.head * 0.95, f.head, f.head * 0.92] },
    ],
    headY,
    headTop: headY + f.head,
  }
}

/** A draped cloth over an animal's back: a shell just proud of the body, edged in brass. */
function caparison(center: V3, size: V3, reach = 1.0): Part[] {
  const grow = 1.05
  const shell = new THREE.SphereGeometry(1, 24, 8, 0, Math.PI * 2, 0, reach)
  const edgeY = center[1] + size[1] * grow * Math.cos(reach)
  const k = Math.sin(reach) * grow
  return [
    { g: shell, m: 'body', p: center, s: [size[0] * grow, size[1] * grow, size[2] * grow] },
    { g: new THREE.TorusGeometry(1, 0.05, 6, 40), m: 'brass', p: [center[0], edgeY, center[2]], r: X90, s: [size[0] * k, size[2] * k, 0.18] },
  ]
}

// Shared small parts.
const post = new THREE.CylinderGeometry(0.01, 0.01, 1, 8)
const cone = new THREE.ConeGeometry(1, 1, 16)
const box = new THREE.BoxGeometry(1, 1, 1)
const cylinder = new THREE.CylinderGeometry(1, 1, 1, 20)
const dome = new THREE.SphereGeometry(1, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2)

/** A vertical post from y0 to y1. */
const postAt = (x: number, z: number, y0: number, y1: number, m: Mat = 'brass'): Part =>
  ({ g: post, m, p: [x, (y0 + y1) / 2, z], s: [1, y1 - y0, 1] })

/** Every piece as a list of parts: geometry, material and placement. Front is -x. */
function partsFor(kind: PieceKind): Part[] {
  switch (kind) {
    case 6: {
      // Raja: the tallest figure, crowned with a tiered brass mukut.
      const f = figure({ hemY: PED, hem: 0.22, waistY: 0.66, waist: 0.085, shoulderY: 0.93, chest: 0.12, neck: 0.036, head: 0.068, arm: 0.03, border: true })
      const y = f.headY + 0.03
      const mukut = new Profile(y)
        .L(0.066, y)
        .Q(0.088, y + 0.03, 0.08, y + 0.062)
        .Q(0.084, y + 0.075, 0.066, y + 0.085)
        .Q(0.062, y + 0.12, 0.05, y + 0.14)
        .L(0.056, y + 0.15)
        .Q(0.04, y + 0.19, 0.026, y + 0.21)
        .L(0.03, y + 0.22)
        .Q(0.016, y + 0.26, 0.004, y + 0.29)
        .L(0, y + 0.29)
        .lathe(24)
      return [
        ...pedestalParts(R.raja),
        ...f.parts,
        { g: mukut, m: 'brass' },
        brassBead([0, y + 0.3, 0], 0.016),
      ]
    }
    case 5: {
      // Mantri: a robed courtier in a pagdi with a jewelled sarpech and plume.
      const f = figure({ hemY: PED, hem: 0.2, waistY: 0.57, waist: 0.078, shoulderY: 0.8, chest: 0.108, neck: 0.033, head: 0.062, arm: 0.027, border: true })
      const y = f.headY
      const pagdi = new Profile(y - 0.01)
        .L(0.062, y - 0.01)
        .Q(0.086, y + 0.01, 0.085, y + 0.045)
        .Q(0.084, y + 0.085, 0, y + 0.09)
        .lathe(28)
      return [
        ...pedestalParts(R.mantri),
        ...f.parts,
        { g: pagdi, m: 'body', s: [0.95, 1, 1] },
        { g: ring(0.078, 0.009), m: 'brass', p: [0, y + 0.012, 0], r: X90, s: [0.95, 1, 1] },
        // Sarpech at the front, its plume sweeping back.
        { g: bead, m: 'brass', p: [-0.078, y + 0.05, 0], s: [0.012, 0.024, 0.014] },
        { g: cone, m: 'brass', p: [-0.052, y + 0.108, 0], r: [0, 0, -0.55], s: [0.012, 0.09, 0.012] },
      ]
    }
    case 1: {
      // Padati: a foot soldier in a knee-length tunic, with a dhal shield and a spear.
      const f = figure({
        hemY: 0.33, hem: 0.1, waistY: 0.44, waist: 0.06, shoulderY: 0.58, chest: 0.083, neck: 0.027, head: 0.052, arm: 0.022,
        hands: [[-0.02, 0.47, 0.11], [-0.045, 0.5, -0.1]],
      })
      const y = f.headY
      const turban = new Profile(y - 0.01)
        .L(0.052, y - 0.01)
        .Q(0.068, y + 0.01, 0.064, y + 0.04)
        .Q(0.052, y + 0.08, 0.008, y + 0.1)
        .L(0, y + 0.1)
        .lathe(24)
      return [
        ...pedestalParts(R.padati),
        // Legs below the tunic.
        ...[-0.036, 0.036].map((z): Part => ({ g: tube([[0, PED, z], [0, 0.35, z]], 0.028, 0.026, 2, 10), m: 'body' })),
        ...f.parts,
        { g: turban, m: 'body' },
        brassBead([0, y + 0.1, 0], 0.012),
        // Dhal on the left arm: a domed shield with a brass rim and bosses.
        { g: dome, m: 'body', p: [-0.01, 0.48, 0.13], r: X90, s: [0.085, 0.028, 0.085] },
        { g: ring(0.083, 0.008), m: 'brass', p: [-0.01, 0.48, 0.13] },
        ...[0, 1, 2, 3].map((i): Part => {
          const a = (i / 4) * Math.PI * 2 + Math.PI / 4
          return brassBead([-0.01 + Math.cos(a) * 0.035, 0.48 + Math.sin(a) * 0.035, 0.152], 0.009)
        }),
        // Bhala (spear) in the right hand, standing on the pedestal.
        postAt(-0.045, -0.1, PED, 0.9, 'brass'),
        { g: cone, m: 'brass', p: [-0.045, 0.93, -0.1], s: [0.016, 0.07, 0.016] },
        brassBead([-0.045, 0.89, -0.1], 0.012),
      ]
    }
    case 4: {
      // Ratha: a temple chariot with a stepped shikhara roof and a kalash finial.
      const axle: V3 = [0.03, 0.28, 0]
      return [
        ...pedestalParts(R.ratha),
        { g: box, m: 'body', p: [0, 0.33, 0], s: [0.46, 0.045, 0.26] },
        { g: cylinder, m: 'brass', p: axle, r: X90, s: [0.016, 0.44, 0.016] },
        ...[-1, 1].flatMap((side): Part[] => {
          const p: V3 = [axle[0], axle[1], side * 0.2]
          return [
            { g: new THREE.TorusGeometry(0.135, 0.017, 8, 40), m: 'brass', p },
            ...[0, 1, 2, 3].map((i): Part => ({ g: box, m: 'brass', p, r: [0, 0, (i * Math.PI) / 4], s: [0.25, 0.011, 0.011] })),
            { g: cylinder, m: 'brass', p, r: X90, s: [0.028, 0.05, 0.028] },
          ]
        }),
        // Car body with a moulded rail.
        { g: box, m: 'body', p: [0.02, 0.43, 0], s: [0.36, 0.15, 0.24] },
        { g: box, m: 'body', p: [0.02, 0.51, 0], s: [0.39, 0.022, 0.27] },
        ...[[-0.14, -0.09], [0.18, -0.09], [-0.14, 0.09], [0.18, 0.09]].map(([x, z]) => postAt(x, z, 0.52, 0.63)),
        // Shikhara: three receding tiers, a pyramid, an amalaka and the kalash.
        { g: box, m: 'body', p: [0.02, 0.645, 0], s: [0.42, 0.032, 0.3] },
        { g: box, m: 'body', p: [0.02, 0.677, 0], s: [0.34, 0.032, 0.24] },
        { g: box, m: 'body', p: [0.02, 0.709, 0], s: [0.26, 0.032, 0.18] },
        { g: new THREE.ConeGeometry(0.15, 0.13, 4), m: 'body', p: [0.02, 0.79, 0], r: [0, Math.PI / 4, 0], s: [1, 1, 0.75] },
        { g: ball, m: 'brass', p: [0.02, 0.86, 0], s: [0.04, 0.018, 0.04] },
        brassBead([0.02, 0.885, 0], 0.022),
        { g: cone, m: 'brass', p: [0.02, 0.93, 0], s: [0.012, 0.07, 0.012] },
        // Draught pole running forward.
        { g: tube([[-0.2, 0.33, 0], [-0.3, 0.3, 0], [-0.38, 0.3, 0]], 0.014, 0.012, 6, 8), m: 'body' },
        brassBead([-0.385, 0.3, 0], 0.02),
      ]
    }
    case 3: {
      // Gaja: an elephant in a caparison, with a howdah under a brass canopy.
      const body: V3 = [0.03, 0.47, 0]
      const size: V3 = [0.24, 0.16, 0.15]
      return [
        ...pedestalParts(R.gaja),
        ...[[-0.12, -0.085], [-0.12, 0.085], [0.17, -0.085], [0.17, 0.085]].flatMap(([x, z]): Part[] => [
          { g: tube([[x, 0.46, z], [x, PED, z]], 0.055, 0.058, 2, 16), m: 'body' },
          { g: ring(0.059, 0.009), m: 'brass', p: [x, 0.19, z], r: X90 },
        ]),
        { g: ball, m: 'body', p: body, s: size },
        ...caparison(body, size, 1.05),
        // Head, ears, trunk curling in towards the chest, tusks and eyes.
        { g: ball, m: 'body', p: [-0.2, 0.57, 0], s: [0.11, 0.13, 0.115] },
        ...[-1, 1].map((side): Part => ({ g: ball, m: 'body', p: [-0.14, 0.56, side * 0.112], r: [0, -side * 0.35, 0], s: [0.085, 0.115, 0.018] })),
        { g: tube([[-0.28, 0.58, 0], [-0.33, 0.48, 0], [-0.345, 0.34, 0], [-0.33, 0.23, 0], [-0.29, 0.2, 0]], 0.048, 0.02, 20, 12), m: 'body' },
        ...[-1, 1].map((side): Part => ({ g: tube([[-0.27, 0.47, side * 0.05], [-0.33, 0.41, side * 0.065], [-0.39, 0.43, side * 0.07]], 0.017, 0.005, 8, 8), m: 'ivory' })),
        ...[-1, 1].map((side) => brassBead([-0.265, 0.61, side * 0.078], 0.011)),
        // Brass forehead ornament.
        { g: bead, m: 'brass', p: [-0.29, 0.62, 0], s: [0.012, 0.03, 0.03] },
        { g: tube([[0.26, 0.52, 0], [0.285, 0.42, 0], [0.28, 0.32, 0]], 0.012, 0.007, 8, 6), m: 'body' },
        // Howdah.
        { g: box, m: 'body', p: [0.05, 0.66, 0], s: [0.17, 0.06, 0.15] },
        { g: box, m: 'brass', p: [0.05, 0.692, 0], s: [0.18, 0.01, 0.16] },
        ...[[-0.025, -0.06], [0.125, -0.06], [-0.025, 0.06], [0.125, 0.06]].map(([x, z]) => postAt(x, z, 0.69, 0.79)),
        { g: dome, m: 'brass', p: [0.05, 0.788, 0], s: [0.1, 0.065, 0.09] },
        brassBead([0.05, 0.862, 0], 0.014),
      ]
    }
    case 2: {
      // Ashva: a Marwari horse, ears curving in to meet, with a saddle cloth and a kalgi plume.
      const body: V3 = [0.02, 0.48, 0]
      const size: V3 = [0.2, 0.1, 0.088]
      return [
        ...pedestalParts(R.ashva),
        ...[[-0.11, -0.05], [-0.11, 0.05], [0.15, -0.05], [0.15, 0.05]].flatMap(([x, z]): Part[] => [
          { g: tube([[x, 0.46, z], [x + (x < 0 ? -0.008 : 0.008), 0.3, z], [x, PED, z]], 0.036, 0.024, 8, 12), m: 'body' },
          { g: ring(0.026, 0.007), m: 'brass', p: [x, PED + 0.016, z], r: X90 },
        ]),
        { g: ball, m: 'body', p: body, s: size },
        // Deep chest and rounded quarters.
        { g: ball, m: 'body', p: [-0.1, 0.5, 0], s: [0.1, 0.108, 0.09] },
        { g: ball, m: 'body', p: [0.14, 0.495, 0], s: [0.1, 0.11, 0.094] },
        ...caparison([0.03, 0.48, 0], [0.12, 0.11, 0.095], 1.1),
        // Neck arching up to the head, which drops towards the muzzle.
        { g: tube([[-0.12, 0.52, 0], [-0.17, 0.61, 0], [-0.2, 0.7, 0]], 0.076, 0.05, 12, 14), m: 'body' },
        { g: tube([[-0.2, 0.735, 0], [-0.25, 0.7, 0], [-0.3, 0.645, 0]], 0.055, 0.038, 12, 14), m: 'body' },
        { g: bead, m: 'body', p: [-0.2, 0.735, 0], s: 0.055 },
        { g: bead, m: 'body', p: [-0.303, 0.64, 0], s: [0.04, 0.036, 0.038] },
        // Mane along the crest of the neck.
        { g: tube([[-0.06, 0.55, 0], [-0.12, 0.65, 0], [-0.17, 0.76, 0]], 0.026, 0.018, 10, 8), m: 'body' },
        ...[-1, 1].map((side): Part => ({ g: cone, m: 'body', p: [-0.185, 0.8, side * 0.024], r: [-side * 0.45, 0, 0.15], s: [0.015, 0.065, 0.012] })),
        ...[-1, 1].map((side) => brassBead([-0.235, 0.745, side * 0.042], 0.01)),
        { g: cone, m: 'brass', p: [-0.19, 0.81, 0], r: [0, 0, -0.5], s: [0.011, 0.075, 0.011] },
        // Bridle band.
        { g: ring(0.045, 0.007), m: 'brass', p: [-0.262, 0.685, 0], r: [0, Math.PI / 2, 0.8] },
        { g: tube([[0.225, 0.51, 0], [0.28, 0.43, 0], [0.275, 0.3, 0]], 0.024, 0.014, 8, 8), m: 'body' },
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

/** Figures turn to face the opponent; the chariot and animals stand in profile, facing left for White. */
const FIGURES = new Set<PieceKind>([1, 5, 6])

export const PieceModel = memo(function PieceModel({ kind, white, lite = false }: { kind: PieceKind; white: boolean; lite?: boolean }) {
  const g = mergedFor(kind)
  const figure = FIGURES.has(kind)
  return (
    <group rotation={[0, figure ? (white ? -Math.PI / 2 : Math.PI / 2) : 0, 0]} scale={[figure || white ? 1 : -1, 1, 1]}>
      {/* White's tusks are ivory like its body, so they share the body mesh. */}
      <mesh geometry={white ? g.bodyWithIvory : g.body} material={pieceMaterials(white, lite)} castShadow receiveShadow />
      <mesh geometry={g.brass} material={brass} castShadow />
      {!white && g.ivory && <mesh geometry={g.ivory} material={lite ? ivoryLite : ivory} castShadow />}
    </group>
  )
})

const TOKEN = { disc: new THREE.CylinderGeometry(0.4, 0.4, 0.08, 48), rim: new THREE.TorusGeometry(0.4, 0.02, 10, 48) }

/** Flat-view token: a disc with the piece glyph, for the top-down view and slow machines. */
export const PieceToken = memo(function PieceToken({ kind, white, lite = false }: { kind: PieceKind; white: boolean; lite?: boolean }) {
  const materials = getTokenMaterials(kind, white, lite)
  return (
    <group>
      <mesh geometry={TOKEN.disc} material={materials} position={[0, 0.04, 0]} castShadow />
      <mesh geometry={TOKEN.rim} material={brass} position={[0, 0.08, 0]} rotation={[Math.PI / 2, 0, 0]} />
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
