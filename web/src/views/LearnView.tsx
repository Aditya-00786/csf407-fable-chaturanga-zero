import { ArrowLeft, ArrowRight, BookOpen, Check, Lightbulb, ListOrdered, Maximize, Minimize, RotateCcw, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { flushSync } from 'react-dom'
import { PieceGlyph } from '../components/PieceGlyph'
import { Covered, Drawer, DrawerHeader, DrawerToggle, Floating, QuickActions, QuickToggle, ToolButton } from '../components/stage'
import { Button } from '../components/ui'
import lessonsData from '../content/lessons.json'
import { api, type GameView, type OutcomeReason } from '../lib/api'
import { captures, explainOutcome, moveText, squareIndex } from '../lib/board'
import { PIECES, type PieceKind } from '../lib/pieces'
import { href } from '../lib/route'
import { playMoveSound } from '../lib/sound'
import { useFullscreen } from '../lib/useFullscreen'
import { BoardScene } from '../three/BoardScene'

interface Task {
  prompt: string
  type: 'move' | 'outcome'
  moves?: string[]
  reason?: OutcomeReason
  hint: string
}

interface Lesson {
  id: string
  group: string
  title: string
  piece: number | null
  fen: string
  body: string[]
  facts: string[]
  task: Task | null
  emphasis: string[]
  /** Where the lesson's facts come from, for authors checking it against docs/RULES.md. Not shown. */
  source: string
}

const LESSONS = lessonsData as Lesson[]

type Feedback = { kind: 'idle' } | { kind: 'success'; text: string } | { kind: 'fail'; text: string }

function useLesson(lesson: Lesson) {
  const [moves, setMoves] = useState<string[]>([])
  const [view, setView] = useState<GameView | null>(null)
  const [feedback, setFeedback] = useState<Feedback>({ kind: 'idle' })
  const [error, setError] = useState<string | null>(null)

  const load = useCallback((next: string[]) => {
    let alive = true
    api.game(next, lesson.fen).then(
      (v) => { if (alive) { setView(v); setMoves(next); setError(null) } },
      (e: Error) => alive && setError(e.message),
    )
    return () => { alive = false }
  }, [lesson.fen])

  // Reset when the lesson changes (adjusting state during render, not in an effect).
  const [loadedFen, setLoadedFen] = useState<string | null>(null)
  if (loadedFen !== lesson.fen) {
    setLoadedFen(lesson.fen)
    setFeedback({ kind: 'idle' })
    setView(null)
  }
  useEffect(() => load([]), [load])

  const reset = () => {
    setFeedback({ kind: 'idle' })
    load([])
  }

  const move = (from: number, to: number) => {
    const text = moveText(from, to)
    const next = [...moves, text]
    api.game(next, lesson.fen).then((v) => {
      setView(v)
      setMoves(next)
      const last = v.history[v.history.length - 1]
      playMoveSound(v.outcome ? 'end' : last?.check ? 'check' : last?.captured ? 'capture' : 'move')
      const task = lesson.task
      if (!task) {
        if (v.outcome) setFeedback({ kind: 'success', text: explainOutcome(v.outcome, v.side).title })
        return
      }
      const mover = -v.side
      const solved = task.type === 'move'
        ? task.moves!.includes(text)
        : v.outcome?.reason === task.reason && v.outcome?.winner === mover
      if (solved) {
        setFeedback({ kind: 'success', text: v.outcome ? explainOutcome(v.outcome, v.side).title : 'Well played.' })
      } else {
        const why = v.outcome ? explainOutcome(v.outcome, v.side).detail : 'That move is legal, but it does not complete the task.'
        setFeedback({ kind: 'fail', text: why })
        setTimeout(() => { setFeedback({ kind: 'idle' }); load([]) }, v.outcome ? 3600 : 2200)
      }
    }, (e: Error) => setError(e.message))
  }

  return { view, feedback, error, reset, move }
}

/** The lesson list, grouped, as shown in the Lessons drawer. */
function LessonNav({ current, onPick }: { current: string; onPick: () => void }) {
  const groups = useMemo(() => {
    const out: { name: string; lessons: Lesson[] }[] = []
    for (const l of LESSONS) {
      const g = out.find((x) => x.name === l.group)
      if (g) g.lessons.push(l)
      else out.push({ name: l.group, lessons: [l] })
    }
    return out
  }, [])
  return (
    <nav aria-label="Lessons" className="flex-1 space-y-6 overflow-y-auto px-4 py-5">
      {groups.map((g) => (
        <div key={g.name}>
          <p className="mb-1.5 px-2 text-[12.5px] font-medium text-ink-faint">{g.name}</p>
          <ul className="space-y-0.5">
            {g.lessons.map((l) => {
              const active = l.id === current
              return (
                <li key={l.id}>
                  <a
                    href={href('learn', l.id)}
                    onClick={onPick}
                    aria-current={active ? 'page' : undefined}
                    className={`flex h-9 items-center gap-2.5 rounded-lg px-2 text-sm whitespace-nowrap transition-colors duration-150 ${
                      active ? 'bg-accent-soft font-medium text-ink' : 'text-ink-muted hover:bg-surface-2 hover:text-ink'
                    }`}
                  >
                    {l.piece ? <PieceGlyph piece={l.piece} size={20} /> : <span className="w-5" aria-hidden />}
                    {l.title}
                  </a>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </nav>
  )
}

type LessonState = ReturnType<typeof useLesson>

/**
 * The floating card over the board: which lesson this is, what to do, and how it went. It is
 * enough to work through a lesson with both drawers closed.
 */
function LessonCard({ lesson, index, next, state, onRead }: {
  lesson: Lesson; index: number; next: Lesson | undefined; state: LessonState; onRead: () => void
}) {
  const { feedback } = state
  const [hintFor, setHintFor] = useState<string | null>(null)
  const showHint = hintFor === lesson.id
  const solved = feedback.kind === 'success'
  return (
    <div>
      <p className="text-[12px] font-medium text-ink-faint">
        Lesson {index + 1} of {LESSONS.length} · {lesson.group}
      </p>
      <p className="mt-1 flex items-center gap-2 font-serif text-[20px] leading-tight font-semibold text-ink">
        {lesson.piece && <PieceGlyph piece={lesson.piece} size={24} />}
        {lesson.title}
      </p>
      <div aria-live="polite" className="mt-2 text-[14px] leading-snug">
        {lesson.task ? (
          <p className="text-ink">{lesson.task.prompt}</p>
        ) : (
          <p className="text-ink-muted">Explore freely: move any piece, for either side.</p>
        )}
        {feedback.kind === 'success' && (
          <p className="mt-2 flex items-center gap-2 font-medium text-success"><Check size={16} className="shrink-0" /> {feedback.text}</p>
        )}
        {feedback.kind === 'fail' && (
          <p className="mt-2 flex items-start gap-2 text-danger"><X size={16} className="mt-0.5 shrink-0" /> {feedback.text} Resetting…</p>
        )}
        {lesson.task && !solved && showHint && (
          <p className="mt-2 flex items-start gap-2 text-ink-muted"><Lightbulb size={16} className="mt-0.5 shrink-0 text-accent" /> {lesson.task.hint}</p>
        )}
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] font-medium">
        <button type="button" onClick={onRead} className="inline-flex items-center gap-1.5 text-accent hover:text-accent-hover">
          <BookOpen size={15} /> Read the lesson
        </button>
        {lesson.task && !solved && !showHint && (
          <button type="button" onClick={() => setHintFor(lesson.id)} className="text-ink-muted hover:text-ink">Show a hint</button>
        )}
        {next && (solved || !lesson.task) && (
          <a href={href('learn', next.id)} className="inline-flex items-center gap-1.5 text-accent hover:text-accent-hover">
            Next: {next.title} <ArrowRight size={15} />
          </a>
        )}
      </div>
    </div>
  )
}

/** The lesson text, in the Lesson drawer. */
function LessonText({ lesson, next, solved, onReset }: { lesson: Lesson; next: Lesson | undefined; solved: boolean; onReset: () => void }) {
  const info = lesson.piece ? PIECES[lesson.piece as PieceKind] : null
  return (
    <div className="flex-1 space-y-6 overflow-y-auto p-5 sm:p-6">
      <header className="flex items-center gap-3">
        {lesson.piece && <PieceGlyph piece={lesson.piece} size={44} />}
        <div>
          <h1 className="font-serif text-[28px] leading-tight font-semibold text-ink">{lesson.title}</h1>
          {info && (
            <p className="mt-0.5 text-sm text-ink-muted">
              <span className="font-deva text-[16px] text-ink" lang="sa">{info.devanagari}</span>
              <span className="mx-2 text-ink-faint">·</span>
              {info.english} · in chess: {info.chess}
            </p>
          )}
        </div>
      </header>

      <div className="space-y-3 text-[15px] leading-relaxed text-ink-muted">
        {lesson.body.map((p, i) => <p key={i} className="max-w-[62ch]">{p}</p>)}
      </div>

      <ul className="flex flex-wrap gap-x-5 gap-y-2 text-[13.5px] text-ink">
        {lesson.facts.map((f) => (
          <li key={f} className="flex items-center gap-2">
            <svg viewBox="0 0 10 10" width="9" height="9" aria-hidden className="text-accent">
              <path d="M1.5 1.5l7 7M8.5 1.5l-7 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            {f}
          </li>
        ))}
      </ul>

      {lesson.task && (
        <div className="rounded-xl border border-line-strong bg-bg p-4">
          <p className="text-[12.5px] font-medium text-ink-faint">Your task</p>
          <p className="mt-1 font-serif text-[17px] leading-snug text-ink">{lesson.task.prompt}</p>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {next ? (
          <a href={href('learn', next.id)}
            className={`inline-flex h-10 items-center gap-2 rounded-lg px-3.5 text-sm font-medium transition-colors duration-150 ${
              solved || !lesson.task ? 'bg-accent text-accent-ink hover:bg-accent-hover' : 'border border-line-strong text-ink hover:bg-surface-2'
            }`}>
            Next: {next.title} <ArrowRight size={16} />
          </a>
        ) : (
          <a href={href('play')} className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-3.5 text-sm font-medium text-accent-ink hover:bg-accent-hover">
            Play a game <ArrowRight size={16} />
          </a>
        )}
        <Button onClick={onReset} icon={<RotateCcw size={16} />}>Reset board</Button>
      </div>
    </div>
  )
}

type DrawerName = 'lessons' | 'lesson'

const DRAWERS: Record<DrawerName, { drawer: string; toggle: string }> = {
  lessons: { drawer: 'lesson-list', toggle: 'lesson-list-toggle' },
  lesson: { drawer: 'lesson-text', toggle: 'lesson-text-toggle' },
}
const otherDrawer = (name: DrawerName): DrawerName => (name === 'lessons' ? 'lesson' : 'lessons')

/**
 * The two drawers: the lesson list on the left and the lesson text on the right, both closed by
 * default so the board has the whole screen. Esc closes the latest one opened. On narrow screens,
 * where a drawer covers the whole board, opening one closes the other.
 */
function useDrawers() {
  const [open, setOpen] = useState<Record<DrawerName, boolean>>({ lessons: false, lesson: false })
  const [latest, setLatest] = useState<DrawerName>('lesson')
  // Bumped whenever a drawer opens, so focus moves into it once it is no longer inert.
  const [focusRequest, setFocusRequest] = useState<{ name: DrawerName; n: number } | null>(null)

  const set = (name: DrawerName, value: boolean) => {
    const narrow = window.matchMedia('(max-width: 1023px)').matches
    setOpen((o) => ({ ...o, [name]: value, ...(value && narrow ? { [otherDrawer(name)]: false } : {}) }))
    if (value) {
      setLatest(name)
      setFocusRequest((r) => ({ name, n: (r?.n ?? 0) + 1 }))
    }
  }
  // A drawer's toggle is inert while the drawer covers it, so let the drawer close before focusing it.
  const close = (name: DrawerName) => {
    flushSync(() => set(name, false))
    document.getElementById(DRAWERS[name].toggle)?.focus({ preventScroll: true })
  }

  useEffect(() => {
    // preventScroll: the drawer is still off-screen as it starts to slide in (see PlayView).
    if (focusRequest) document.getElementById(DRAWERS[focusRequest.name].drawer)?.querySelector<HTMLElement>('button, a')?.focus({ preventScroll: true })
  }, [focusRequest])

  useEffect(() => {
    if (!open.lessons && !open.lesson) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const name = open[latest] ? latest : otherDrawer(latest)
      flushSync(() => setOpen((o) => ({ ...o, [name]: false })))
      document.getElementById(DRAWERS[name].toggle)?.focus({ preventScroll: true })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, latest])

  return { open, set, close }
}

export function LearnView({ lessonId }: { lessonId: string | null }) {
  const index = Math.max(0, LESSONS.findIndex((l) => l.id === lessonId))
  const lesson = LESSONS[index]
  const prev = LESSONS[index - 1]
  const next = LESSONS[index + 1]
  const state = useLesson(lesson)
  const { view, feedback, error, reset, move } = state
  const d = useDrawers()
  const [quickOpen, setQuickOpen] = useState(false)
  const [fullscreen, toggleFullscreen, canFullscreen] = useFullscreen()

  const solved = feedback.kind === 'success'
  const interactive = !!view && !view.outcome && !(lesson.task && feedback.kind !== 'idle')
  const last = view?.history[view.history.length - 1]
  const emphasis = useMemo(() => lesson.emphasis.map(squareIndex), [lesson])
  const go = (l: Lesson) => { location.hash = href('learn', l.id) }

  return (
    <div
      className="relative min-h-[440px] flex-1 overflow-clip"
      style={{ background: 'radial-gradient(ellipse 70% 60% at 50% 55%, var(--stage-glow), transparent 75%)' }}
    >
      <section aria-label="Lesson board" className="absolute inset-0">
        <BoardScene
          board={view?.board ?? new Array(64).fill(0)}
          ply={view?.ply ?? 0}
          legalMoves={interactive ? view!.legal_moves : []}
          lastMove={last ? { from: last.from, to: last.to } : null}
          captured={captures(view)}
          checkSquare={view?.in_check ? view.king_square : null}
          orientation={1}
          interactive={interactive}
          onMove={move}
          emphasis={moveEmphasis(emphasis, view)}
          label={`Lesson board: ${lesson.title}. Use the arrow keys and Enter to move.`}
        />
      </section>

      {/* Floating lesson card and toolbars keep the board full-screen. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-[var(--z-stage-overlay)] flex items-start justify-between gap-3 p-3 sm:p-4">
        <Covered by={d.open.lessons} className="flex min-w-0 flex-col items-start gap-2">
          <Floating className="p-1">
            <DrawerToggle
              id={DRAWERS.lessons.toggle}
              open={d.open.lessons}
              controls={DRAWERS.lessons.drawer}
              onClick={() => d.set('lessons', !d.open.lessons)}
              icon={<ListOrdered size={17} />}
              label="Lessons"
              title="All lessons"
              badge={<span className="rounded-md bg-surface-2 px-1.5 text-[11.5px] text-ink-muted tabular-nums">{index + 1}/{LESSONS.length}</span>}
            />
          </Floating>
          <Floating className="max-w-[min(360px,calc(100vw-2rem))] px-4 py-3">
            <LessonCard key={lesson.id} lesson={lesson} index={index} next={next} state={state} onRead={() => d.set('lesson', true)} />
          </Floating>
        </Covered>
        <Covered by={d.open.lesson} onHidden={() => setQuickOpen(false)} className="shrink-0">
          <Floating className="flex items-center gap-0.5 p-1">
            <QuickActions id="learn-quick-actions" open={quickOpen}>
              <ToolButton label="Reset board" icon={<RotateCcw size={17} />} onClick={reset} />
              <ToolButton label="Previous lesson" icon={<ArrowLeft size={17} />} onClick={() => prev && go(prev)} disabled={!prev} />
              <ToolButton label="Next lesson" icon={<ArrowRight size={17} />} onClick={() => next && go(next)} disabled={!next} />
              {canFullscreen && (
                <ToolButton label={fullscreen ? 'Exit full screen' : 'Full screen'}
                  icon={fullscreen ? <Minimize size={17} /> : <Maximize size={17} />} onClick={toggleFullscreen} />
              )}
            </QuickActions>
            <QuickToggle open={quickOpen} controls="learn-quick-actions" onClick={() => setQuickOpen((o) => !o)}
              describe="reset, previous and next lesson, full screen" />
            <DrawerToggle
              id={DRAWERS.lesson.toggle}
              open={d.open.lesson}
              controls={DRAWERS.lesson.drawer}
              onClick={() => d.set('lesson', !d.open.lesson)}
              icon={<BookOpen size={17} />}
              label="Lesson"
              title="Read the lesson"
            />
          </Floating>
        </Covered>
      </div>

      {error && (
        <div role="alert" className="absolute inset-x-0 top-28 z-[var(--z-stage-overlay)] flex justify-center px-4">
          <p className="rounded-lg border border-danger/40 bg-surface px-4 py-2 text-sm text-ink shadow-lg">{error}</p>
        </div>
      )}

      <Drawer id={DRAWERS.lessons.drawer} label="Lessons" side="left" open={d.open.lessons} className="sm:w-[300px]">
        <DrawerHeader title="Lessons" onClose={() => d.close('lessons')} closeLabel="Close the lesson list" />
        <LessonNav current={lesson.id} onPick={() => d.set('lessons', false)} />
      </Drawer>
      <Drawer id={DRAWERS.lesson.drawer} label="Lesson" side="right" open={d.open.lesson} className="sm:w-[420px]">
        <DrawerHeader
          title={<>Lesson {index + 1} <span className="font-sans text-[13px] font-medium text-ink-faint">of {LESSONS.length} · {lesson.group}</span></>}
          onClose={() => d.close('lesson')}
          closeLabel="Close the lesson"
        />
        <LessonText lesson={lesson} next={next} solved={solved} onReset={reset} />
      </Drawer>
    </div>
  )
}

// Show the lesson's emphasised squares only in the starting position.
function moveEmphasis(squares: number[], view: GameView | null) {
  return view && view.ply === 0 ? squares : []
}
