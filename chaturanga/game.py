"""Chaturanga game state: legal moves, move application and game termination.

A move is the integer from_sq * 64 + to_sq. Promotion is implicit (it is fully
determined by the rules), so this is also the absolute-coordinate action index.

GameState is treated as immutable: apply() returns a new state. This suits MCTS,
where each tree node keeps its own state.
"""

from dataclasses import dataclass
from typing import Optional

from .config import DEFAULT_RULES, RulesConfig
from .constants import (
    BLACK, EMPTY, INITIAL_COUNTS, LETTER_PIECES, MANTRI, PADATI, PIECE_LETTERS,
    RAJA, RATHA, ASHVA, GAJA, WHITE, file_of, parse_square, rank_of, square_name,
)
from .tables import LAST_RANK, LEAPER_MOVES, PADATI_CAPTURES, PADATI_PUSH, RATHA_RAYS, is_attacked
from .zobrist import PIECE_KEYS, SIDE_KEY, compute_hash

# Back ranks, a-file to h-file.
_WHITE_BACK = (RATHA, ASHVA, GAJA, MANTRI, RAJA, GAJA, ASHVA, RATHA)
_BLACK_BACK = {
    "crosswise": (RATHA, ASHVA, GAJA, RAJA, MANTRI, GAJA, ASHVA, RATHA),  # Raja d8, Mantri e8
    "facing": (RATHA, ASHVA, GAJA, MANTRI, RAJA, GAJA, ASHVA, RATHA),     # Raja e8, Mantri d8
}

# Promotion piece by file for the "file" rules (d- and e-files give a Mantri).
_FILE_PROMOTION = (RATHA, ASHVA, GAJA, MANTRI, MANTRI, GAJA, ASHVA, RATHA)


def encode_move(from_sq: int, to_sq: int) -> int:
    return from_sq * 64 + to_sq


def decode_move(move: int) -> tuple:
    return divmod(move, 64)


def move_to_str(move: int) -> str:
    fr, to = divmod(move, 64)
    return square_name(fr) + square_name(to)


def parse_move(text: str) -> int:
    text = text.strip()
    if len(text) != 4:
        raise ValueError(f"moves are written like 'e2e3', got {text!r}")
    return encode_move(parse_square(text[:2]), parse_square(text[2:]))


@dataclass(frozen=True)
class Outcome:
    winner: Optional[int]  # WHITE, BLACK, or None for a draw
    reason: str            # bare_king, checkmate, stalemate, repetition, no_progress, max_plies

    def value_for(self, color: int) -> int:
        """+1 if `color` won, -1 if it lost, 0 for a draw."""
        if self.winner is None:
            return 0
        return 1 if self.winner == color else -1


_UNSET = object()


class GameState:
    __slots__ = (
        "board", "side", "ply", "no_progress", "history", "hash",
        "kings", "material", "rules", "_legal", "_outcome",
    )

    def __init__(self, board, side, ply, no_progress, history, hash_, kings, material, rules):
        self.board = board            # list of 64 ints; treat as read-only
        self.side = side              # colour to move
        self.ply = ply                # plies played so far
        self.no_progress = no_progress  # plies since the last capture or Padati move
        self.history = history        # hashes since the last irreversible move, incl. current
        self.hash = hash_
        self.kings = kings            # (white raja square, black raja square)
        self.material = material      # (white, black) count of non-Raja pieces
        self.rules = rules
        self._legal = None
        self._outcome = _UNSET

    # ------------------------------------------------------------------ construction

    @classmethod
    def initial(cls, rules: RulesConfig = DEFAULT_RULES) -> "GameState":
        board = [EMPTY] * 64
        for f in range(8):
            board[f] = _WHITE_BACK[f]
            board[8 + f] = PADATI
            board[48 + f] = -PADATI
            board[56 + f] = -_BLACK_BACK[rules.setup][f]
        return cls._from_board(board, WHITE, rules)

    @classmethod
    def from_pieces(cls, pieces: dict, side: int = WHITE, rules: RulesConfig = DEFAULT_RULES,
                    ply: int = 0, no_progress: int = 0) -> "GameState":
        """Build a position from {"e1": "K", "d8": "k", ...}. Upper case is White."""
        board = [EMPTY] * 64
        for name, letter in pieces.items():
            kind = LETTER_PIECES[letter.upper()]
            board[parse_square(name)] = kind if letter.isupper() else -kind
        return cls._from_board(board, side, rules, ply, no_progress)

    @classmethod
    def from_fen(cls, fen: str, rules: RulesConfig = DEFAULT_RULES) -> "GameState":
        """Parse '<rank8>/.../<rank1> <w|b> [no_progress] [ply]'.

        Letters: K Raja, M Mantri, R Ratha, G Gaja, N Ashva, P Padati (upper case White).
        """
        fields = fen.split()
        rows = fields[0].split("/")
        if len(rows) != 8:
            raise ValueError("FEN board needs 8 ranks")
        board = [EMPTY] * 64
        for i, row in enumerate(rows):
            rank, f = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    kind = LETTER_PIECES[ch.upper()]
                    board[rank * 8 + f] = kind if ch.isupper() else -kind
                    f += 1
            if f != 8:
                raise ValueError(f"FEN rank {rank + 1} does not have 8 files")
        side = WHITE if len(fields) < 2 or fields[1] == "w" else BLACK
        no_progress = int(fields[2]) if len(fields) > 2 else 0
        ply = int(fields[3]) if len(fields) > 3 else 0
        return cls._from_board(board, side, rules, ply, no_progress)

    @classmethod
    def _from_board(cls, board, side, rules, ply=0, no_progress=0):
        if board.count(RAJA) != 1 or board.count(-RAJA) != 1:
            raise ValueError("each side needs exactly one Raja")
        kings = (board.index(RAJA), board.index(-RAJA))
        material = (
            sum(1 for p in board if 0 < p < RAJA),
            sum(1 for p in board if -RAJA < p < 0),
        )
        h = compute_hash(board, side)
        return cls(board, side, ply, no_progress, (h,), h, kings, material, rules)

    def to_fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.board[rank * 8 + f]
                if p == EMPTY:
                    empty += 1
                    continue
                if empty:
                    row, empty = row + str(empty), 0
                letter = PIECE_LETTERS[abs(p)]
                row += letter if p > 0 else letter.lower()
            rows.append(row + (str(empty) if empty else ""))
        side = "w" if self.side == WHITE else "b"
        return f"{'/'.join(rows)} {side} {self.no_progress} {self.ply}"

    # ------------------------------------------------------------------ queries

    def king_square(self, color: int) -> int:
        return self.kings[0 if color == WHITE else 1]

    def material_of(self, color: int) -> int:
        return self.material[0 if color == WHITE else 1]

    def in_check(self) -> bool:
        return is_attacked(self.board, self.king_square(self.side), -self.side)

    def repetition_count(self) -> int:
        """How many times the current position has occurred (including now)."""
        return self.history.count(self.hash)

    # ------------------------------------------------------------------ move generation

    def _promotion_piece(self, to_sq: int, side: int):
        """Piece kind a Padati of `side` becomes on `to_sq`, or None if it may not go there."""
        mode = self.rules.promotion
        if mode == "mantri":
            return MANTRI
        kind = _FILE_PROMOTION[file_of(to_sq)]
        if mode == "file":
            return kind
        # file_restricted: only to a piece type the side has already lost.
        owned = self.board.count(side * kind)
        return kind if owned < INITIAL_COUNTS[kind] else None

    def pseudo_legal_moves(self) -> list:
        """Moves that follow piece movement rules, ignoring whether the own Raja is left attacked."""
        b, s = self.board, self.side
        moves = []
        last_rank = LAST_RANK[s]
        for sq in range(64):
            p = b[sq] * s
            if p <= 0:
                continue
            if p == PADATI:
                to = PADATI_PUSH[s][sq]
                targets = []
                if to >= 0 and b[to] == EMPTY:
                    targets.append(to)
                for to in PADATI_CAPTURES[s][sq]:
                    if b[to] * s < 0:
                        targets.append(to)
                for to in targets:
                    if rank_of(to) == last_rank and self._promotion_piece(to, s) is None:
                        continue
                    moves.append(sq * 64 + to)
            elif p == RATHA:
                for ray in RATHA_RAYS[sq]:
                    for to in ray:
                        q = b[to]
                        if q == EMPTY:
                            moves.append(sq * 64 + to)
                        else:
                            if q * s < 0:
                                moves.append(sq * 64 + to)
                            break
            else:
                for to in LEAPER_MOVES[p][sq]:
                    if b[to] * s <= 0:
                        moves.append(sq * 64 + to)
        return moves

    def legal_moves(self) -> list:
        if self._legal is None:
            b, s = self.board, self.side
            king = self.king_square(s)
            legal = []
            for m in self.pseudo_legal_moves():
                fr, to = divmod(m, 64)
                moved, captured = b[fr], b[to]
                b[to], b[fr] = moved, EMPTY
                safe = not is_attacked(b, to if moved == s * RAJA else king, -s)
                b[fr], b[to] = moved, captured
                if safe:
                    legal.append(m)
            self._legal = legal
        return self._legal

    # ------------------------------------------------------------------ making moves

    def apply(self, move: int) -> "GameState":
        """Return the state after `move`. The move is assumed legal (see is_legal)."""
        fr, to = divmod(move, 64)
        s = self.side
        board = self.board[:]
        moved, captured = board[fr], board[to]
        kind = moved * s

        placed = moved
        if kind == PADATI and rank_of(to) == LAST_RANK[s]:
            placed = s * self._promotion_piece(to, s)

        h = self.hash ^ SIDE_KEY ^ PIECE_KEYS[moved + 6][fr] ^ PIECE_KEYS[placed + 6][to]
        if captured:
            h ^= PIECE_KEYS[captured + 6][to]
        board[fr], board[to] = EMPTY, placed

        kings = self.kings
        if kind == RAJA:
            kings = (to, kings[1]) if s == WHITE else (kings[0], to)

        material = self.material
        if captured:
            material = (material[0], material[1] - 1) if s == WHITE else (material[0] - 1, material[1])

        if captured or kind == PADATI:
            no_progress, history = 0, (h,)
        else:
            no_progress, history = self.no_progress + 1, self.history + (h,)

        return GameState(board, -s, self.ply + 1, no_progress, history, h, kings, material, self.rules)

    def is_legal(self, move: int) -> bool:
        return move in self.legal_moves()

    def play(self, text: str) -> "GameState":
        """Apply a move written like 'e2e3'. Raises ValueError if it is not legal."""
        move = parse_move(text)
        if not self.is_legal(move):
            raise ValueError(f"illegal move {text} in position {self.to_fen()}")
        return self.apply(move)

    # ------------------------------------------------------------------ termination

    def outcome(self) -> Optional[Outcome]:
        """The result if the game is over, else None. Checks follow RULES.md section 5."""
        if self._outcome is _UNSET:
            self._outcome = self._compute_outcome()
        return self._outcome

    def is_terminal(self) -> bool:
        return self.outcome() is not None

    def _compute_outcome(self) -> Optional[Outcome]:
        s = self.side
        own_bare = self.material_of(s) == 0
        opp_bare = self.material_of(-s) == 0
        # 1. Bare king: the side left with only its Raja loses at once.
        if own_bare and opp_bare:
            return Outcome(None, "bare_king")  # only reachable from a constructed position
        if own_bare:
            return Outcome(-s, "bare_king")
        if opp_bare:
            return Outcome(s, "bare_king")
        # 2-3. No legal move: checkmate or stalemate.
        if not self.legal_moves():
            if self.in_check():
                return Outcome(-s, "checkmate")
            rule = self.rules.stalemate
            if rule == "stalemated_wins":
                return Outcome(s, "stalemate")
            if rule == "stalemating_wins":
                return Outcome(-s, "stalemate")
            return Outcome(None, "stalemate")
        # 4-6. Draws.
        if self.repetition_count() >= self.rules.repetition_limit:
            return Outcome(None, "repetition")
        if self.no_progress >= self.rules.no_progress_plies:
            return Outcome(None, "no_progress")
        if self.ply >= self.rules.max_plies:
            return Outcome(None, "max_plies")
        return None

    # ------------------------------------------------------------------ display

    def __str__(self) -> str:
        lines = []
        for rank in range(7, -1, -1):
            cells = []
            for f in range(8):
                p = self.board[rank * 8 + f]
                cells.append("." if p == EMPTY else (PIECE_LETTERS[abs(p)] if p > 0 else PIECE_LETTERS[abs(p)].lower()))
            lines.append(f"{rank + 1} | {' '.join(cells)}")
        lines.append("    a b c d e f g h")
        lines.append(f"{'White' if self.side == WHITE else 'Black'} to move, ply {self.ply}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GameState({self.to_fen()!r})"
