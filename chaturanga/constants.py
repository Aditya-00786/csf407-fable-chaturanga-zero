"""Piece codes, colours and square helpers.

Board squares are indexed 0..63 with sq = rank * 8 + file, so a1 = 0, h1 = 7, a8 = 56.
A board cell holds 0 (empty), +kind (White) or -kind (Black).
"""

WHITE = 1
BLACK = -1

EMPTY = 0
PADATI = 1  # foot soldier (pawn)
ASHVA = 2   # horse (knight)
GAJA = 3    # elephant (2-square diagonal jump)
RATHA = 4   # chariot (rook)
MANTRI = 5  # minister (1-square diagonal)
RAJA = 6    # king

PIECE_KINDS = (PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA)

PIECE_LETTERS = {PADATI: "P", ASHVA: "N", GAJA: "G", RATHA: "R", MANTRI: "M", RAJA: "K"}
LETTER_PIECES = {letter: kind for kind, letter in PIECE_LETTERS.items()}
PIECE_NAMES = {
    PADATI: "Padati", ASHVA: "Ashva", GAJA: "Gaja",
    RATHA: "Ratha", MANTRI: "Mantri", RAJA: "Raja",
}

# Number of each non-king piece a side starts with (used by "file_restricted" promotion).
INITIAL_COUNTS = {PADATI: 8, ASHVA: 2, GAJA: 2, RATHA: 2, MANTRI: 1}

FILES = "abcdefgh"


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return FILES[file_of(sq)] + str(rank_of(sq) + 1)


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"bad square name {name!r}")
    return square(FILES.index(name[0]), int(name[1]) - 1)


def color_name(color: int) -> str:
    return "White" if color == WHITE else "Black"
