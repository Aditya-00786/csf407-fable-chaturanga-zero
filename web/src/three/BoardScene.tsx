/**
 * The interactive 3D board. Presentational: it is given the position and the legal moves
 * (from the engine) and reports the move the user picks through onMove.
 */

import { CameraControls, Environment, Lightformer } from '@react-three/drei'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type RefObject } from 'react'
import * as THREE from 'three'
import type { Color } from '../lib/api'
import { squareName } from '../lib/board'
import { PIECES, type PieceKind } from '../lib/pieces'
import { prefersReducedMotion, useSettings, type Theme } from '../state/settings'
import { AnimatedPiece } from './AnimatedPiece'
import { PieceWarmup } from './PieceModel'
import { Board } from './Board'
import { worldToSquare } from './coords'
import { Highlights, type HighlightColors } from './Highlights'
import { usePieces } from './usePieces'

export interface BoardSceneProps {
  board: number[]
  ply: number
  legalMoves: [number, number][]
  lastMove: { from: number; to: number } | null
  checkSquare: number | null
  orientation: Color
  interactive: boolean
  onMove: (from: number, to: number) => void
  emphasis?: number[]
  label?: string
}

const SCENE: Record<Theme, { colors: HighlightColors }> = {
  dark: { colors: { accent: '#dcb562', danger: '#e36a4c', ink: '#f1e8d8' } },
  light: { colors: { accent: '#8d6420', danger: '#b8432e', ink: '#3a2118' } },
}

// One fixed set of lights for both themes: only colours and intensities change, so switching
// theme never changes the light count (which would force every shader to recompile).
const LIGHTING: Record<Theme, { sky: string; ground: string; hemi: number; key: string; keyI: number; rim: number; env: number }> = {
  dark: { sky: '#ffe0b8', ground: '#2b140c', hemi: 0.45, key: '#ffd29a', keyI: 2.5, rim: 0.35, env: 0.55 },
  light: { sky: '#ffffff', ground: '#d8c4b3', hemi: 1.0, key: '#fff5e8', keyI: 2.6, rim: 0, env: 0.9 },
}

// Longest time step (seconds) any animation may advance in a single frame.
const MAX_FRAME_DELTA = 1 / 30

/** Low-end hint: few cores or little memory. Such machines get a lower pixel ratio and smaller shadows. */
const LOW_POWER =
  (navigator.hardwareConcurrency ?? 8) <= 4 ||
  ((navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8) <= 4

function Lights({ theme, instant }: { theme: Theme; instant: boolean }) {
  const hemi = useRef<THREE.HemisphereLight>(null)
  const key = useRef<THREE.DirectionalLight>(null)
  const rim = useRef<THREE.DirectionalLight>(null)
  const env = useRef(LIGHTING[theme].env)
  const targets = useMemo(() => {
    const l = LIGHTING[theme]
    return { ...l, sky: new THREE.Color(l.sky), ground: new THREE.Color(l.ground), key: new THREE.Color(l.key) }
  }, [theme])

  useFrame(({ scene, invalidate }, dt) => {
    if (!hemi.current || !key.current || !rim.current) return
    // Ease towards the theme's lighting; keep requesting frames until it has settled.
    const k = instant ? 1 : 1 - Math.exp(-9 * Math.min(dt, 0.05))
    hemi.current.color.lerp(targets.sky, k)
    hemi.current.groundColor.lerp(targets.ground, k)
    key.current.color.lerp(targets.key, k)
    hemi.current.intensity += (targets.hemi - hemi.current.intensity) * k
    key.current.intensity += (targets.keyI - key.current.intensity) * k
    rim.current.intensity += (targets.rim - rim.current.intensity) * k
    env.current += (targets.env - env.current) * k
    scene.environmentIntensity = env.current
    const near = (a: number, b: number) => Math.abs(a - b) < 0.002
    const nearColor = (a: THREE.Color, b: THREE.Color) => near(a.r, b.r) && near(a.g, b.g) && near(a.b, b.b)
    const settled =
      near(targets.hemi, hemi.current.intensity) && near(targets.keyI, key.current.intensity) &&
      near(targets.rim, rim.current.intensity) && near(targets.env, env.current) &&
      nearColor(hemi.current.color, targets.sky) && nearColor(hemi.current.groundColor, targets.ground) &&
      nearColor(key.current.color, targets.key)
    if (!settled) {
      invalidate()
    } else if (hemi.current.intensity !== targets.hemi) {
      // Snap the last fraction so the loop stops cleanly.
      hemi.current.color.copy(targets.sky)
      hemi.current.groundColor.copy(targets.ground)
      key.current.color.copy(targets.key)
      hemi.current.intensity = targets.hemi
      key.current.intensity = targets.keyI
      rim.current.intensity = targets.rim
      env.current = targets.env
      scene.environmentIntensity = env.current
      invalidate()
    }
  })

  const l = LIGHTING[theme]
  const map = LOW_POWER ? 1024 : 2048
  return (
    <>
      <hemisphereLight ref={hemi} args={[l.sky, l.ground, l.hemi]} />
      <directionalLight
        ref={key}
        position={[3.5, 9, 5]}
        intensity={l.keyI}
        color={l.key}
        castShadow
        shadow-mapSize={[map, map]}
        shadow-camera-left={-7}
        shadow-camera-right={7}
        shadow-camera-top={7}
        shadow-camera-bottom={-7}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
      <directionalLight ref={rim} position={[-5, 4, -6]} intensity={l.rim} color="#c9d4ff" />
    </>
  )
}

/** Rendered once: its layout effect re-applies scene settings on every render, so keep it still. */
const StudioEnvironment = memo(function StudioEnvironment() {
  return (
    <Environment resolution={LOW_POWER ? 128 : 256}>
      <Lightformer form="rect" intensity={2.2} color="#ffe6c4" position={[0, 6, -4]} scale={[12, 4, 1]} />
      <Lightformer form="rect" intensity={1.2} color="#fff2e0" position={[-6, 3, 2]} rotation={[0, Math.PI / 2, 0]} scale={[8, 2, 1]} />
      <Lightformer form="ring" intensity={1.5} color="#ffd9a8" position={[5, 5, 4]} scale={3} />
    </Environment>
  )
})

// Pinch sensitivity: fraction of the current distance moved per unit of pinch delta.
const PINCH_SPEED = 0.01

/**
 * Trackpad pinch-to-zoom. Chrome, Edge and Firefox report a pinch as a wheel event with ctrlKey
 * set; camera-controls maps that to a weak lens zoom, so we take it over and dolly the camera in
 * proportion to the pinch instead. Safari reports pinches as gesture events. Plain wheel and
 * two-finger scroll still go to camera-controls unchanged.
 */
function usePinchZoom(controls: RefObject<CameraControls | null>) {
  const { gl, invalidate } = useThree()
  useEffect(() => {
    const el = gl.domElement
    let safariScale: number | null = null
    const dollyBy = (amount: number) => {
      const c = controls.current
      if (!c) return
      void c.dolly(amount, false)
      invalidate()
    }
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey || safariScale !== null) return
      e.preventDefault()
      e.stopImmediatePropagation() // keep camera-controls' own ctrl-wheel zoom out of it
      const c = controls.current
      if (c) dollyBy(-e.deltaY * PINCH_SPEED * c.distance)
    }
    type GestureEvent = Event & { scale: number }
    const onGestureStart = (e: Event) => {
      e.preventDefault()
      safariScale = 1
    }
    const onGestureChange = (e: Event) => {
      e.preventDefault()
      const c = controls.current
      const scale = (e as GestureEvent).scale
      if (!c || safariScale === null || scale <= 0) return
      const ratio = scale / safariScale
      safariScale = scale
      dollyBy(c.distance * (1 - 1 / ratio)) // new distance = old distance / ratio
    }
    const onGestureEnd = (e: Event) => {
      e.preventDefault()
      safariScale = null
    }
    // Capture phase on the canvas runs before camera-controls' listener on the same element.
    el.addEventListener('wheel', onWheel, { passive: false, capture: true })
    el.addEventListener('gesturestart', onGestureStart, { passive: false })
    el.addEventListener('gesturechange', onGestureChange, { passive: false })
    el.addEventListener('gestureend', onGestureEnd, { passive: false })
    return () => {
      el.removeEventListener('wheel', onWheel, { capture: true })
      el.removeEventListener('gesturestart', onGestureStart)
      el.removeEventListener('gesturechange', onGestureChange)
      el.removeEventListener('gestureend', onGestureEnd)
    }
  }, [gl, invalidate, controls])
}

const LIMITS = {
  flat: { minPolar: 0, maxPolar: 0.01, rotate: 0 },
  '3d': { minPolar: 0.2, maxPolar: 1.15, rotate: 0.6 },
}

function Rig({ orientation, flat }: { orientation: Color; flat: boolean }) {
  const ref = useRef<CameraControls>(null)
  const { size } = useThree()
  const aspect = size.width / Math.max(1, size.height)
  const first = useRef(true)
  const request = useRef(0)
  usePinchZoom(ref)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const animate = !first.current && !prefersReducedMotion()
    first.current = false
    const id = ++request.current
    const limits = LIMITS[flat ? 'flat' : '3d']
    // Tilt limits are applied only once the camera has arrived. Tightening them first
    // (e.g. flat view's near-vertical limit) would snap the camera before the glide starts.
    c.minPolarAngle = 0
    c.maxPolarAngle = Math.PI / 2
    c.azimuthRotateSpeed = limits.rotate
    c.polarRotateSpeed = limits.rotate
    // Pull back on narrow canvases so the whole board stays in frame.
    const k = Math.min(2.5, Math.max(1.3, 1.55 / aspect))
    const dir = orientation === 1 ? 1 : -1
    const arrived = flat
      // Aim slightly past the centre so the board sits below the floating status bar.
      ? c.setLookAt(0, 19.5 * k, -0.7 * dir + 0.001 * dir, 0, 0, -0.7 * dir, animate)
      : c.setLookAt(0, 8.1 * k, 10.2 * k * dir, 0, -0.6, -0.45 * dir, animate)
    void arrived.then(() => {
      if (request.current !== id) return // a newer move of the camera took over
      c.minPolarAngle = limits.minPolar
      c.maxPolarAngle = limits.maxPolar
    })
  }, [orientation, flat, aspect])
  return (
    <CameraControls
      ref={ref}
      makeDefault
      minDistance={9}
      maxDistance={30}
      truckSpeed={0}
      dollySpeed={0.4}
      smoothTime={0.32}
    />
  )
}

export function BoardScene(props: BoardSceneProps) {
  const { board, ply, legalMoves, lastMove, checkSquare, orientation, interactive, onMove, emphasis = [], label } = props
  const theme = useSettings((s) => s.theme)
  const labels = useSettings((s) => s.labels)
  const flat = useSettings((s) => s.view) === 'flat'
  const [reducedMotion] = useState(prefersReducedMotion)
  const flipped = orientation === -1

  const [selected, setSelected] = useState<number | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [cursor, setCursor] = useState<number | null>(null)
  const [ready, setReady] = useState(false)
  const pieces = usePieces(board, ply, lastMove)

  // Drop any selection when the position or whose turn it is changes.
  const positionKey = `${ply}|${interactive}|${board.join()}`
  const [seenPosition, setSeenPosition] = useState(positionKey)
  if (seenPosition !== positionKey) {
    setSeenPosition(positionKey)
    setSelected(null)
    setHovered(null)
  }

  const movable = useMemo(() => new Set(interactive ? legalMoves.map(([f]) => f) : []), [legalMoves, interactive])
  const targets = useMemo(
    () => (selected === null ? [] : legalMoves.filter(([f]) => f === selected).map(([, t]) => ({ sq: t, capture: board[t] !== 0 }))),
    [legalMoves, selected, board],
  )

  const choose = useCallback(
    (sq: number) => {
      if (!interactive) return
      if (selected !== null && targets.some((t) => t.sq === sq)) {
        onMove(selected, sq)
        setSelected(null)
      } else if (movable.has(sq)) {
        setSelected(sq === selected ? null : sq)
      } else {
        setSelected(null)
      }
    },
    [interactive, selected, targets, movable, onMove],
  )

  // R3F dispatches one event per intersected mesh (a piece is several meshes), so act on
  // each native click once: the nearest hit wins, whether it is a piece or the field.
  // The handler is stable (it reads the latest `choose` through a ref), so pieces can be memoised
  // and a hover or theme change does not re-render all 32 of them.
  const lastClick = useRef<Event | null>(null)
  const chooseRef = useRef(choose)
  useLayoutEffect(() => {
    chooseRef.current = choose
  })
  const pick = useCallback((e: ThreeEvent<PointerEvent>, sq: number | null) => {
    e.stopPropagation()
    if (lastClick.current === e.nativeEvent) return
    lastClick.current = e.nativeEvent
    if (sq !== null) chooseRef.current(sq)
  }, [])
  const onField = (e: ThreeEvent<PointerEvent>) => pick(e, worldToSquare(e.point.x, e.point.z))
  const onHover = (e: ThreeEvent<PointerEvent>) => {
    const sq = worldToSquare(e.point.x, e.point.z)
    setHovered(sq)
    const clickable = sq !== null && (movable.has(sq) || targets.some((t) => t.sq === sq))
    document.body.style.cursor = clickable ? 'pointer' : ''
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const deltas: Record<string, [number, number]> = {
      ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0],
    }
    if (e.key in deltas) {
      e.preventDefault()
      const [df, dr] = deltas[e.key].map((d) => d * (flipped ? -1 : 1))
      const start = cursor ?? selected ?? (orientation === 1 ? 12 : 51)
      const f = Math.min(7, Math.max(0, (start & 7) + (cursor === null ? 0 : df)))
      const r = Math.min(7, Math.max(0, (start >> 3) + (cursor === null ? 0 : dr)))
      setCursor(r * 8 + f)
    } else if ((e.key === 'Enter' || e.key === ' ') && cursor !== null) {
      e.preventDefault()
      choose(cursor)
    } else if (e.key === 'Escape') {
      setSelected(null)
      setCursor(null)
    }
  }

  const cursorPiece = cursor !== null && board[cursor] ? PIECES[Math.abs(board[cursor]) as PieceKind] : null
  const cursorText = cursor === null ? '' :
    `${squareName(cursor)}${cursorPiece ? `, ${board[cursor] > 0 ? 'White' : 'Black'} ${cursorPiece.name}` : ', empty'}` +
    (targets.some((t) => t.sq === cursor) ? ', legal move' : '')

  return (
    <div
      className="relative h-full w-full outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-inset"
      tabIndex={0}
      role="application"
      aria-label={label ?? 'Chaturanga board. Use the arrow keys to move the cursor, Enter to select a piece and again to move it, Escape to cancel.'}
      onKeyDown={onKeyDown}
      onBlur={() => setCursor(null)}
    >
      <span className="sr-only" aria-live="polite">{cursorText}</span>
      <div
        aria-hidden={ready}
        className={`pointer-events-none absolute inset-0 z-[var(--z-stage-overlay)] grid place-items-center transition-opacity duration-500 ease-out-quart ${ready ? 'opacity-0' : 'opacity-100'}`}
      >
        <p className="flex items-center gap-2.5 text-sm text-ink-muted">
          <svg viewBox="0 0 20 20" width="16" height="16" className="animate-spin text-accent [animation-duration:2.4s]" aria-hidden>
            <path d="M4 4l12 12M16 4L4 16" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
          </svg>
          Setting up the board…
        </p>
      </div>
      <Canvas shadows="percentage" frameloop="demand" dpr={[1, LOW_POWER ? 1.25 : 1.75]} camera={{ fov: 32, position: [0, 11, 9] }} gl={{ antialias: true }}
        onPointerMissed={() => setSelected(null)}
        onCreated={(state) => {
          // The scene renders on demand, so after idling (a refresh, another window) the first
          // frame's delta can be seconds long, and camera smoothing would jump straight to the
          // end of a glide. Cap each step; the clock's elapsed time still tracks real time.
          const clock = state.clock
          const rawDelta = clock.getDelta.bind(clock)
          clock.getDelta = () => Math.min(rawDelta(), MAX_FRAME_DELTA)
          if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__three = state
          requestAnimationFrame(() => requestAnimationFrame(() => setReady(true)))
        }}>
        <Rig orientation={orientation} flat={flat} />
        <Lights theme={theme} instant={reducedMotion} />
        <StudioEnvironment />
        <PieceWarmup />
        <Board flipped={flipped} />
        <Highlights
          colors={SCENE[theme].colors}
          selected={selected}
          targets={targets}
          hovered={hovered}
          lastMove={lastMove}
          checkSquare={checkSquare}
          cursor={cursor}
          emphasis={emphasis}
          animate={!reducedMotion}
          flat={flat}
        />
        {/* Click catcher over the whole field. */}
        <mesh position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]} onPointerDown={onField} onPointerMove={onHover}
          onPointerOut={() => { setHovered(null); document.body.style.cursor = '' }}>
          <planeGeometry args={[8, 8]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
        {pieces.map((p) => (
          <AnimatedPiece
            key={p.id}
            piece={p.piece}
            sq={p.sq}
            lifted={p.sq === selected}
            flat={flat}
            flipped={flipped}
            labels={labels}
            reducedMotion={reducedMotion}
            onPick={pick}
          />
        ))}
      </Canvas>
    </div>
  )
}
