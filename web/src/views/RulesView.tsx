import { PieceGlyph } from '../components/PieceGlyph'
import { PIECE_ORDER, PIECES } from '../lib/pieces'
import { href } from '../lib/route'

const ENDINGS = [
  { when: 'A side is left with only its Raja (bare king)', result: 'The other side wins immediately', source: 'Murray p. 61' },
  { when: 'The side to move is in check and has no legal move', result: 'Checkmate: the attacker wins', source: 'RULES.md §5' },
  { when: 'The side to move is not in check but has no legal move', result: 'Stalemate: the stalemated side wins', source: 'Murray pp. 60–61' },
  { when: 'The same position occurs three times, same side to move', result: 'Draw', source: 'Murray p. 229' },
  { when: '70 moves each with no capture and no Padati move', result: 'Draw', source: 'Shatranj 70-move rule' },
  { when: '300 plies are played', result: 'Draw (engine limit)', source: 'Engineering rule' },
]

const CONTESTED = [
  { rule: 'Stalemate', choice: 'The stalemated player wins', tally: '7 sources to 3' },
  { rule: 'Promotion', choice: 'Always to a Mantri', tally: '4 to 4, broken by the oldest two-player Indian text' },
  { rule: 'Setup', choice: 'Rajas on e1 and d8 (not facing)', tally: '5 sources to 1' },
  { rule: 'Bare king', choice: 'Baring wins, no draw for capturing back', tally: '6 sources to 2' },
  { rule: 'Raja knight leap', choice: 'Not allowed', tally: '5 sources to 3' },
  { rule: 'Gaja move', choice: 'Two squares diagonally, jumping', tally: '6 sources to 0' },
]

export function RulesView() {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-5 py-10 sm:px-8 sm:py-14">
        <header className="max-w-[62ch]">
          <h1 className="font-serif text-[40px] leading-[1.1] font-semibold tracking-[-0.015em] text-ink sm:text-[46px]">
            Rules of Chaturanga
          </h1>
          <p className="mt-4 text-[16px] leading-relaxed text-ink-muted">
            Chaturanga (<span className="font-deva text-ink" lang="sa">चतुरङ्ग</span>, "four limbs", after the four
            divisions of an ancient Indian army) is the common ancestor of chess, shogi and xiangqi. Its exact rules
            are not known for certain. These are the rules this project plays, each one chosen by counting historical
            sources. Prefer to learn by doing? <a className="font-medium text-accent underline-offset-4 hover:underline" href={href('learn')}>Try the lessons</a>.
          </p>
        </header>

        <section className="mt-12">
          <h2 className="font-serif text-[26px] font-semibold text-ink">Setup</h2>
          <p className="mt-3 max-w-[62ch] text-[15px] leading-relaxed text-ink-muted">
            An uncheckered 8×8 board, the Ashtapada. Each side lines up like chess, Ratha, Ashva, Gaja on each wing
            and the Mantri beside the Raja, but the Rajas do not face each other: White's is on e1, Black's on d8.
            Padatis fill the second and seventh ranks. White moves first.
          </p>
        </section>

        <section className="mt-12">
          <h2 className="font-serif text-[26px] font-semibold text-ink">The pieces</h2>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-left text-[14.5px]">
              <thead>
                <tr className="border-b border-line-strong text-[12.5px] text-ink-faint">
                  <th scope="col" className="py-2 pr-4 font-medium">Piece</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Chess cousin</th>
                  <th scope="col" className="py-2 font-medium">Moves</th>
                </tr>
              </thead>
              <tbody>
                {PIECE_ORDER.map((k) => {
                  const p = PIECES[k]
                  return (
                    <tr key={k} className="border-b border-line align-top">
                      <th scope="row" className="py-3 pr-4 font-normal">
                        <a href={href('learn', p.name.toLowerCase())} className="flex items-center gap-2.5 hover:text-accent">
                          <PieceGlyph piece={k} size={26} />
                          <span>
                            <span className="block font-medium text-ink">{p.name}</span>
                            <span className="font-deva text-[14px] text-ink-muted" lang="sa">{p.devanagari}</span>
                            <span className="text-ink-faint"> · {p.english}</span>
                          </span>
                        </a>
                      </th>
                      <td className="py-3 pr-4 text-ink-muted">{p.chess}</td>
                      <td className="py-3 text-ink-muted">{p.move}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-4 max-w-[62ch] text-[15px] leading-relaxed text-ink-muted">
            You may never leave your own Raja under attack. There is no castling, no pawn double step and no en passant.
          </p>
        </section>

        <section className="mt-12">
          <h2 className="font-serif text-[26px] font-semibold text-ink">How a game ends</h2>
          <p className="mt-3 max-w-[62ch] text-[15px] leading-relaxed text-ink-muted">
            The engine checks these after every move, in this order.
          </p>
          <ol className="mt-5 space-y-0">
            {ENDINGS.map((e, i) => (
              <li key={e.when} className="grid grid-cols-[2rem_1fr] gap-x-3 border-t border-line py-3.5 sm:grid-cols-[2rem_1fr_13rem]">
                <span className="font-serif text-[17px] text-accent tabular-nums">{i + 1}</span>
                <span className="text-[14.5px] text-ink">{e.when}</span>
                <span className="col-start-2 text-[14.5px] text-ink-muted sm:col-start-3">
                  {e.result}
                  <span className="block text-[12.5px] text-ink-faint">{e.source}</span>
                </span>
              </li>
            ))}
          </ol>
        </section>

        <section className="mt-12">
          <h2 className="font-serif text-[26px] font-semibold text-ink">Where the sources disagree</h2>
          <p className="mt-3 max-w-[62ch] text-[15px] leading-relaxed text-ink-muted">
            For each contested rule we counted the sources that describe Chaturanga itself and followed the majority.
            The full tally, with citations, is in the project's <code className="rounded bg-surface-2 px-1.5 py-0.5 text-[13px]">docs/RULES.md</code>.
          </p>
          <dl className="mt-5 grid gap-x-8 gap-y-5 sm:grid-cols-2">
            {CONTESTED.map((c) => (
              <div key={c.rule} className="border-t border-line pt-3">
                <dt className="text-[12.5px] font-medium text-ink-faint">{c.rule}</dt>
                <dd className="mt-1 text-[15px] text-ink">{c.choice}</dd>
                <dd className="mt-0.5 text-[13px] text-ink-muted">{c.tally}</dd>
              </div>
            ))}
          </dl>
        </section>

        <p className="mt-14 border-t border-line pt-5 text-[13px] text-ink-faint">
          Main source: H. J. R. Murray, <i>A History of Chess</i>, Oxford, 1913. ChaturangZero is a CS F407 project by Team Fable.
        </p>
      </div>
    </div>
  )
}
