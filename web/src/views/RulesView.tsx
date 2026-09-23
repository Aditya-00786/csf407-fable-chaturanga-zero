import { PieceGlyph } from '../components/PieceGlyph'
import { PIECE_ORDER, PIECES } from '../lib/pieces'
import { href } from '../lib/route'

const ENDINGS = [
  { when: 'A side is left with only its Raja (bare king)', result: 'The other side wins immediately' },
  { when: 'The side to move is in check and has no legal move', result: 'Checkmate: the attacker wins' },
  { when: 'The side to move is not in check but has no legal move', result: 'Stalemate: the stalemated side wins' },
  { when: 'The same position occurs three times, same side to move', result: 'Draw' },
  { when: '70 moves each with no capture and no Padati move', result: 'Draw' },
  { when: '300 plies are played', result: 'Draw (engine limit)' },
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
            are not known for certain; these are the rules this project plays. Prefer to learn by doing? <a className="font-medium text-accent underline-offset-4 hover:underline" href={href('learn')}>Try the lessons</a>.
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
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  )
}
