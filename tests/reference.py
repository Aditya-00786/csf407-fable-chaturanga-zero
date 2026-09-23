"""A deliberately naive, independent move generator used to cross-check the engine.

It shares no code with chaturanga.tables or chaturanga.game: moves come straight from
(file, rank) geometry, and check is detected by brute force (can any enemy move land on
the Raja?). It is slow and only supports the default "mantri" promotion rule.
"""

from chaturanga.constants import ASHVA, GAJA, MANTRI, PADATI, RAJA, RATHA

KING_STEPS = [(df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0)]
MANTRI_STEPS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
GAJA_STEPS = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
ASHVA_STEPS = [(df, dr) for df in (-2, -1, 1, 2) for dr in (-2, -1, 1, 2) if abs(df) != abs(dr)]
RATHA_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _inside(f, r):
    return 0 <= f < 8 and 0 <= r < 8


def pseudo_moves(board, side):
    """Yield (from, to) pairs that obey piece movement, ignoring own-Raja safety."""
    for sq, p in enumerate(board):
        if p * side <= 0:
            continue
        kind, f, r = abs(p), sq % 8, sq // 8

        def target_ok(tf, tr):
            return _inside(tf, tr) and board[tr * 8 + tf] * side <= 0

        if kind == PADATI:
            tr = r + side
            if _inside(f, tr) and board[tr * 8 + f] == 0:
                yield sq, tr * 8 + f
            for tf in (f - 1, f + 1):
                if _inside(tf, tr) and board[tr * 8 + tf] * side < 0:
                    yield sq, tr * 8 + tf
        elif kind == RATHA:
            for df, dr in RATHA_DIRS:
                tf, tr = f + df, r + dr
                while _inside(tf, tr):
                    q = board[tr * 8 + tf]
                    if q * side > 0:
                        break
                    yield sq, tr * 8 + tf
                    if q != 0:
                        break
                    tf, tr = tf + df, tr + dr
        else:
            steps = {RAJA: KING_STEPS, MANTRI: MANTRI_STEPS, GAJA: GAJA_STEPS, ASHVA: ASHVA_STEPS}[kind]
            for df, dr in steps:
                if target_ok(f + df, r + dr):
                    yield sq, (r + dr) * 8 + (f + df)


def in_check(board, side):
    king = board.index(side * RAJA)
    return any(to == king for _, to in pseudo_moves(board, -side))


def legal_moves(board, side):
    result = set()
    for fr, to in pseudo_moves(board, side):
        after = list(board)
        after[to], after[fr] = after[fr], 0
        if not in_check(after, side):
            result.add(fr * 64 + to)
    return result


def perft(state, depth):
    """Leaf count using the reference generator for moves and the engine only to apply them."""
    if depth == 0:
        return 1
    moves = legal_moves(state.board, state.side)
    if depth == 1:
        return len(moves)
    return sum(perft(state.apply(m), depth - 1) for m in moves)
