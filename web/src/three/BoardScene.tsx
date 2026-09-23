/**
 * The interactive 3D board. Presentational: it is given the position and the legal moves
 * (from the engine) and reports the move the user picks through onMove.
 */

import { CameraControls, Environment, Lightformer } from '@react-three/drei'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type { Color } from '../lib/api'
import { squareName } from '../lib/board'
import { PIECES, type PieceKind } from '../lib/pieces'
import { prefersReducedMotion, useSettings, type Theme } from '../state/settings'
import { AnimatedPiece } from './AnimatedPiece'
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

const SCENE: Record<Theme, { colors: HighlightColors; env: number }> = {
  dark: { colors: { accent: '#dcb562', danger: '#e36a4c', ink: '#f1e8d8' }, env: 0.55 },
  light: { colors: { accent: '#8d6420', danger: '#b8432e', ink: '#3a2118' }, env: 0.9 },
}

function Lights({ theme }: { theme: Theme }) {
  const shadow = {
    castShadow: true,
    'shadow-mapSize': [2048, 2048] as [number, number],
    'shadow-camera-left': -7, 'shadow-camera-right': 7, 'shadow-camera-top': 7, 'shadow-camera-bottom': -7,
    'shadow-bias': -0.0004, 'shadow-normalBias': 0.02,
  }
  if (theme === 'dark') {
    return (
      <>
        <hemisphereLight args={['#ffe0b8', '#2b140c', 0.45]} />
        <directionalLight position={[3.5, 9, 5]} intensity={2.5} color="#ffd29a" {...shadow} />
        <directionalLight position={[-5, 4, -6]} intensity={0.35} color="#c9d4ff" />
      </>
    )
  }
  return (
    <>
      <hemisphereLight args={['#ffffff', '#d8c4b3', 1.0]} />
      <directionalLight position={[4, 10, 4]} intensity={2.6} color="#fff5e8" {...shadow} />
    </>
  )
}

function Rig({ orientation, flat }: { orientation: Color; flat: boolean }) {
  const ref = useRef<CameraControls>(null)
  const { size } = useThree()
  const aspect = size.width / Math.max(1, size.height)
  const first = useRef(true)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    // Pull back on narrow canvases so the whole board stays in frame.
    const k = Math.min(2.4, Math.max(1.22, 1.45 / aspect))
    const dir = orientation === 1 ? 1 : -1
    if (flat) {
      void c.setLookAt(0, 19.5 * k, 0.001 * dir, 0, 0, 0, !first.current && !prefersReducedMotion())
    } else {
      void c.setLookAt(0, 8.1 * k, 10.2 * k * dir, 0, -0.6, 0.55 * dir, !first.current && !prefersReducedMotion())
    }
    first.current = false
  }, [orientation, flat, aspect])
  return (
    <CameraControls
      ref={ref}
      makeDefault
      minPolarAngle={flat ? 0 : 0.2}
      maxPolarAngle={flat ? 0.01 : 1.15}
      minDistance={9}
      maxDistance={30}
      azimuthRotateSpeed={flat ? 0 : 0.6}
      polarRotateSpeed={flat ? 0 : 0.6}
      truckSpeed={0}
      dollySpeed={0.4}
      smoothTime={0.35}
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
  const lastClick = useRef<Event | null>(null)
  const once = (e: ThreeEvent<PointerEvent>, sq: number | null) => {
    e.stopPropagation()
    if (lastClick.current === e.nativeEvent) return
    lastClick.current = e.nativeEvent
    if (sq !== null) choose(sq)
  }
  const onPiece = (sq: number) => (e: ThreeEvent<PointerEvent>) => once(e, sq)
  const onField = (e: ThreeEvent<PointerEvent>) => once(e, worldToSquare(e.point.x, e.point.z))
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
      <Canvas shadows="percentage" dpr={[1, 2]} camera={{ fov: 32, position: [0, 11, 9] }} gl={{ antialias: true }}
        onPointerMissed={() => setSelected(null)}
        onCreated={(state) => {
          if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__three = state
          requestAnimationFrame(() => requestAnimationFrame(() => setReady(true)))
        }}>
        <Rig orientation={orientation} flat={flat} />
        <Lights theme={theme} />
        <Environment resolution={256} environmentIntensity={SCENE[theme].env}>
          <Lightformer form="rect" intensity={2.2} color="#ffe6c4" position={[0, 6, -4]} scale={[12, 4, 1]} />
          <Lightformer form="rect" intensity={1.2} color="#fff2e0" position={[-6, 3, 2]} rotation={[0, Math.PI / 2, 0]} scale={[8, 2, 1]} />
          <Lightformer form="ring" intensity={1.5} color="#ffd9a8" position={[5, 5, 4]} scale={3} />
        </Environment>
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
            onPointerDown={onPiece(p.sq)}
          />
        ))}
      </Canvas>
    </div>
  )
}
