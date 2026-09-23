/** Procedural materials: rosewood frame, honey-toned playing field, ivory, ebony and brass. */

import * as THREE from 'three'
import { GLYPHS, type PieceKind } from '../lib/pieces'

// Small deterministic value noise so the wood grain looks the same on every load.
function makeNoise(seed: number) {
  const perm = new Uint8Array(512)
  let s = seed
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  const p = Array.from({ length: 256 }, (_, i) => i)
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[p[i], p[j]] = [p[j], p[i]]
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255]
  const grad = (h: number) => (h / 255) * 2 - 1
  const fade = (t: number) => t * t * (3 - 2 * t)
  return (x: number, y: number) => {
    const xi = Math.floor(x) & 255
    const yi = Math.floor(y) & 255
    const xf = x - Math.floor(x)
    const yf = y - Math.floor(y)
    const a = grad(perm[perm[xi] + yi])
    const b = grad(perm[perm[xi + 1] + yi])
    const c = grad(perm[perm[xi] + yi + 1])
    const d = grad(perm[perm[xi + 1] + yi + 1])
    const u = fade(xf)
    const v = fade(yf)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
  }
}

interface WoodOptions {
  size: number
  light: [number, number, number]
  dark: [number, number, number]
  rings: number
  seed: number
}

function paintWood(ctx: CanvasRenderingContext2D, { size, light, dark, rings, seed }: WoodOptions) {
  const noise = makeNoise(seed)
  const img = ctx.createImageData(size, size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = x / size
      const ny = y / size
      // Long grain along x: stretched noise warps the ring pattern.
      const warp = noise(nx * 2.2, ny * 9) * 0.55 + noise(nx * 7, ny * 30) * 0.12
      const ring = Math.sin((ny * rings + warp) * Math.PI * 2) * 0.5 + 0.5
      const fine = noise(nx * 18, ny * 180) * 0.5 + 0.5
      const t = Math.min(1, Math.max(0, ring * ring * 0.5 + fine * 0.34))
      const i = (y * size + x) * 4
      img.data[i] = light[0] + (dark[0] - light[0]) * t
      img.data[i + 1] = light[1] + (dark[1] - light[1]) * t
      img.data[i + 2] = light[2] + (dark[2] - light[2]) * t
      img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

function canvas(size: number) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  return c
}

let fieldTexture: THREE.CanvasTexture | null = null
let frameTexture: THREE.CanvasTexture | null = null

/** Honey sheesham playing field with inlaid, burnt grid lines. */
export function getFieldTexture() {
  if (fieldTexture) return fieldTexture
  const size = 1024
  const c = canvas(size)
  const ctx = c.getContext('2d')!
  paintWood(ctx, { size, light: [214, 160, 102], dark: [168, 110, 63], rings: 34, seed: 7 })
  // Grid: the field is 8.24 units wide; lines sit on square edges.
  const unit = size / 8.24
  const offset = unit * 0.12
  ctx.strokeStyle = 'rgba(52, 24, 12, 0.88)'
  ctx.lineWidth = unit * 0.035
  ctx.lineCap = 'square'
  for (let i = 0; i <= 8; i++) {
    const p = offset + i * unit
    ctx.beginPath()
    ctx.moveTo(offset, p)
    ctx.lineTo(offset + 8 * unit, p)
    ctx.moveTo(p, offset)
    ctx.lineTo(p, offset + 8 * unit)
    ctx.stroke()
  }
  fieldTexture = new THREE.CanvasTexture(c)
  fieldTexture.colorSpace = THREE.SRGBColorSpace
  fieldTexture.anisotropy = 8
  return fieldTexture
}

/** Dark rosewood for the carved frame. */
export function getFrameTexture() {
  if (frameTexture) return frameTexture
  const size = 512
  const c = canvas(size)
  paintWood(c.getContext('2d')!, { size, light: [96, 44, 28], dark: [52, 22, 14], rings: 40, seed: 23 })
  frameTexture = new THREE.CanvasTexture(c)
  frameTexture.colorSpace = THREE.SRGBColorSpace
  // Extrude UVs are world x/y (about -5..5); map them into 0..1 so the non-tiling grain has no seam.
  frameTexture.repeat.set(0.095, 0.095)
  frameTexture.offset.set(0.5, 0.5)
  frameTexture.anisotropy = 8
  return frameTexture
}

/** Transparent overlay with file letters and rank numbers engraved on the frame. */
export function makeCoordinateTexture(flipped: boolean) {
  const size = 1536
  const c = canvas(size)
  const ctx = c.getContext('2d')!
  const unit = size / 10 // the overlay spans the full 10-unit frame
  ctx.fillStyle = 'rgba(228, 194, 120, 0.82)'
  ctx.font = `600 ${unit * 0.34}px "Newsreader Variable", Georgia, serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const files = 'abcdefgh'
  for (let i = 0; i < 8; i++) {
    const pos = unit * (1 + i) + unit / 2
    const file = files[flipped ? 7 - i : i]
    const rank = String(flipped ? i + 1 : 8 - i)
    ctx.fillText(file, pos, size - unit * 0.46)
    ctx.fillText(file, pos, unit * 0.46)
    ctx.fillText(rank, unit * 0.46, pos)
    ctx.fillText(rank, size - unit * 0.46, pos)
  }
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  return tex
}

export const ivory = new THREE.MeshPhysicalMaterial({
  color: '#efe3c8', roughness: 0.42, clearcoat: 0.35, clearcoatRoughness: 0.35, sheen: 0.4, sheenColor: '#fff4dd',
})
export const ebony = new THREE.MeshPhysicalMaterial({
  color: '#2a1d18', roughness: 0.36, clearcoat: 0.75, clearcoatRoughness: 0.22,
})
export const brass = new THREE.MeshStandardMaterial({ color: '#c9a14c', metalness: 1, roughness: 0.3 })
export const darkBrass = new THREE.MeshStandardMaterial({ color: '#8f6c2c', metalness: 1, roughness: 0.4 })

/** Cached token materials (side, top, side), one set per piece type and colour. */
const tokenMaterialCache = new Map<string, THREE.Material[]>()
export function getTokenMaterials(kind: PieceKind, white: boolean) {
  const key = `${kind}${white ? 'w' : 'b'}`
  let materials = tokenMaterialCache.get(key)
  if (!materials) {
    const side = white ? ivory : ebony
    const top = new THREE.MeshStandardMaterial({ map: getTokenTexture(kind, white), roughness: 0.55 })
    materials = [side, top, side]
    tokenMaterialCache.set(key, materials)
  }
  return materials
}

/** Top-face textures for flat-view tokens: glyph on an ivory or ebony disc. */
const tokenCache = new Map<string, THREE.CanvasTexture>()
export function getTokenTexture(kind: PieceKind, white: boolean) {
  const key = `${kind}${white ? 'w' : 'b'}`
  const cached = tokenCache.get(key)
  if (cached) return cached
  const size = 256
  const c = canvas(size)
  const ctx = c.getContext('2d')!
  ctx.fillStyle = white ? '#efe3c8' : '#2a1d18'
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2)
  ctx.fill()
  ctx.save()
  ctx.translate(size * 0.14, size * 0.12)
  ctx.scale((size * 0.72) / 100, (size * 0.72) / 100)
  ctx.fillStyle = white ? '#3a2118' : '#e9d9b6'
  ctx.fill(new Path2D(GLYPHS[kind]), 'evenodd')
  ctx.restore()
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 4
  // Cylinder cap UVs sit a quarter turn off the board. Turn each side's glyph to face its own
  // player: White's reads upright from rank 1, Black's from rank 8.
  tex.center.set(0.5, 0.5)
  tex.rotation = white ? Math.PI / 2 : -Math.PI / 2
  tokenCache.set(key, tex)
  return tex
}
