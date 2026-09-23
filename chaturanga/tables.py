"""Precomputed move and attack tables, plus square-attack detection."""

from .constants import ASHVA, BLACK, GAJA, MANTRI, PADATI, RAJA, RATHA, WHITE, file_of, rank_of, square


def _leaper(offsets):
    table = []
    for sq in range(64):
        f, r = file_of(sq), rank_of(sq)
        table.append(tuple(
            square(f + df, r + dr)
            for df, dr in offsets
            if 0 <= f + df < 8 and 0 <= r + dr < 8
        ))
    return tuple(table)


def _rays(directions):
    table = []
    for sq in range(64):
        rays = []
        for df, dr in directions:
            f, r = file_of(sq) + df, rank_of(sq) + dr
            ray = []
            while 0 <= f < 8 and 0 <= r < 8:
                ray.append(square(f, r))
                f, r = f + df, r + dr
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


KING_MOVES = _leaper([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
MANTRI_MOVES = _leaper([(1, 1), (1, -1), (-1, 1), (-1, -1)])
GAJA_MOVES = _leaper([(2, 2), (2, -2), (-2, 2), (-2, -2)])
ASHVA_MOVES = _leaper([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
RATHA_RAYS = _rays([(1, 0), (-1, 0), (0, 1), (0, -1)])

LEAPER_MOVES = {RAJA: KING_MOVES, MANTRI: MANTRI_MOVES, GAJA: GAJA_MOVES, ASHVA: ASHVA_MOVES}


def _pawn_tables(color):
    pushes, captures = [], []
    for sq in range(64):
        f, r = file_of(sq), rank_of(sq)
        nr = r + color
        if 0 <= nr < 8:
            pushes.append(square(f, nr))
            captures.append(tuple(square(f + df, nr) for df in (-1, 1) if 0 <= f + df < 8))
        else:
            pushes.append(-1)
            captures.append(())
    return tuple(pushes), tuple(captures)


_WHITE_PUSH, _WHITE_CAPTURES = _pawn_tables(WHITE)
_BLACK_PUSH, _BLACK_CAPTURES = _pawn_tables(BLACK)
PADATI_PUSH = {WHITE: _WHITE_PUSH, BLACK: _BLACK_PUSH}
PADATI_CAPTURES = {WHITE: _WHITE_CAPTURES, BLACK: _BLACK_CAPTURES}

# Rank a Padati promotes on.
LAST_RANK = {WHITE: 7, BLACK: 0}


def is_attacked(board, sq: int, by: int) -> bool:
    """True if any piece of colour `by` attacks square `sq`."""
    for s in ASHVA_MOVES[sq]:
        if board[s] == by * ASHVA:
            return True
    for s in GAJA_MOVES[sq]:
        if board[s] == by * GAJA:
            return True
    for s in MANTRI_MOVES[sq]:
        if board[s] == by * MANTRI:
            return True
    for s in KING_MOVES[sq]:
        if board[s] == by * RAJA:
            return True
    # A Padati of colour `by` attacks sq from the squares that a Padati of the other
    # colour standing on sq would capture towards.
    for s in PADATI_CAPTURES[-by][sq]:
        if board[s] == by * PADATI:
            return True
    for ray in RATHA_RAYS[sq]:
        for s in ray:
            p = board[s]
            if p:
                if p == by * RATHA:
                    return True
                break
    return False
