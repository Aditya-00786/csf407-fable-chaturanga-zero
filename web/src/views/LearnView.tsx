import { ArrowRight, Check, Lightbulb, RotateCcw, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { PieceGlyph } from '../components/PieceGlyph'
import { Button } from '../components/ui'
import lessonsData from '../content/lessons.json'
import { api, type GameView, type OutcomeReason } from '../lib/api'
import { captures, explainOutcome, moveText, squareIndex } from '../lib/board'
import { PIECES, type PieceKind } from '../lib/pieces'
import { href } from '../lib/route'
import { playMoveSound } from '../lib/sound'
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

function LessonNav({ current }: { current: string }) {
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
    <nav aria-label="Lessons" className="flex gap-6 overflow-x-auto border-b border-line bg-surface px-4 py-3 lg:block lg:w-64 lg:shrink-0 lg:space-y-6 lg:overflow-y-auto lg:border-r lg:border-b-0 lg:px-4 lg:py-6">
      {groups.map((g) => (
        <div key={g.name} className="shrink-0">
          <p className="mb-1.5 hidden px-2 text-[12.5px] font-medium text-ink-faint lg:block">{g.name}</p>
          <ul className="flex gap-1 lg:block lg:space-y-0.5">
            {g.lessons.map((l) => {
              const active = l.id === current
              return (
                <li key={l.id}>
                  <a
                    href={href('learn', l.id)}
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

export function LearnView({ lessonId }: { lessonId: string | null }) {
  const index = Math.max(0, LESSONS.findIndex((l) => l.id === lessonId))
  const lesson = LESSONS[index]
  const next = LESSONS[index + 1]
  const { view, feedback, error, reset, move } = useLesson(lesson)
  const [hintFor, setHintFor] = useState<string | null>(null)
  const showHint = hintFor === lesson.id

  const solved = feedback.kind === 'success'
  const interactive = !!view && !view.outcome && !(lesson.task && feedback.kind !== 'idle')
  const last = view?.history[view.history.length - 1]
  const info = lesson.piece ? PIECES[lesson.piece as PieceKind] : null
  const emphasis = useMemo(() => lesson.emphasis.map(squareIndex), [lesson])

  return (
    <div className="flex flex-col lg:min-h-0 lg:flex-1 lg:flex-row">
      <LessonNav current={lesson.id} />

      <section
        aria-label="Lesson board"
        className="relative h-[52vh] min-h-[320px] shrink-0 overflow-hidden lg:h-auto lg:flex-1"
        style={{ background: 'radial-gradient(ellipse 70% 60% at 50% 55%, var(--stage-glow), transparent 75%)' }}
      >
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
        {error && (
          <p role="alert" className="absolute top-4 left-1/2 -translate-x-1/2 rounded-lg border border-danger/40 bg-surface px-4 py-2 text-sm text-ink">
            {error}
          </p>
        )}
      </section>

      <article className="border-line bg-surface lg:w-[420px] lg:shrink-0 lg:overflow-y-auto lg:border-l">
        <div className="space-y-6 p-6 sm:p-7">
          <header>
            <p className="text-[12.5px] font-medium text-ink-faint">
              Lesson {index + 1} of {LESSONS.length} · {lesson.group}
            </p>
            <div className="mt-2 flex items-center gap-3">
              {lesson.piece && <PieceGlyph piece={lesson.piece} size={44} />}
              <div>
                <h1 className="font-serif text-[30px] leading-tight font-semibold text-ink">{lesson.title}</h1>
                {info && (
                  <p className="mt-0.5 text-sm text-ink-muted">
                    <span className="font-deva text-[16px] text-ink" lang="sa">{info.devanagari}</span>
                    <span className="mx-2 text-ink-faint">·</span>
                    {info.english} · in chess: {info.chess}
                  </p>
                )}
              </div>
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

          {lesson.task ? (
            <div
              className={`rounded-xl border p-4 transition-colors duration-200 ${
                feedback.kind === 'success' ? 'border-success/50 bg-success-soft'
                  : feedback.kind === 'fail' ? 'border-danger/50 bg-danger-soft' : 'border-line-strong bg-bg'
              }`}
            >
              <p className="text-[12.5px] font-medium text-ink-faint">Your task</p>
              <p className="mt-1 font-serif text-[18px] leading-snug text-ink">{lesson.task.prompt}</p>
              <div aria-live="polite">
                {feedback.kind === 'success' && (
                  <p className="mt-2.5 flex items-center gap-2 text-sm font-medium text-success"><Check size={16} /> {feedback.text}</p>
                )}
                {feedback.kind === 'fail' && (
                  <p className="mt-2.5 flex items-start gap-2 text-sm text-danger"><X size={16} className="mt-0.5 shrink-0" /> {feedback.text} Resetting…</p>
                )}
              </div>
              {!solved && (showHint ? (
                <p className="mt-2.5 flex items-start gap-2 text-sm text-ink-muted"><Lightbulb size={16} className="mt-0.5 shrink-0 text-accent" /> {lesson.task.hint}</p>
              ) : (
                <button type="button" onClick={() => setHintFor(lesson.id)} className="mt-2.5 text-sm font-medium text-accent hover:text-accent-hover">
                  Show a hint
                </button>
              ))}
            </div>
          ) : (
            <p className="rounded-xl border border-line bg-bg p-4 text-sm text-ink-muted">
              Explore freely: click any piece to see where it can go, and move for either side.
              {feedback.kind === 'success' && <span className="mt-2 block font-medium text-ink">{feedback.text}</span>}
            </p>
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
            <Button onClick={reset} icon={<RotateCcw size={16} />}>Reset board</Button>
          </div>

        </div>
      </article>
    </div>
  )
}

// Show the lesson's emphasised squares only in the starting position.
function moveEmphasis(squares: number[], view: GameView | null) {
  return view && view.ply === 0 ? squares : []
}
