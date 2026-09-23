"""Rule options for the engine.

The defaults are the ruleset decided in docs/RULES.md (v1.0). The alternatives exist
so that contested historical rules can still be experimented with.
"""

from dataclasses import dataclass

SETUPS = ("crosswise", "facing")
PROMOTIONS = ("mantri", "file", "file_restricted")
STALEMATE_RULES = ("stalemated_wins", "stalemating_wins", "draw")


@dataclass(frozen=True)
class RulesConfig:
    # "crosswise": Raja e1 / d8 (RULES.md 6.6). "facing": Raja e1 / e8 (Murray).
    setup: str = "crosswise"
    # "mantri": always Mantri (RULES.md 6.2).
    # "file": the piece that started on that file; d- and e-files give a Mantri.
    # "file_restricted": as "file", but only if that piece type has already been lost;
    #   otherwise the Padati may not step onto the last rank.
    promotion: str = "mantri"
    # Who wins on stalemate (RULES.md 6.1).
    stalemate: str = "stalemated_wins"
    # The n-th occurrence of the same position (same side to move) is a draw.
    repetition_limit: int = 3
    # Plies without a capture or Padati move before the game is drawn (70 moves each).
    no_progress_plies: int = 140
    # [ENG] Hard cap on game length, so self-play always terminates.
    max_plies: int = 300

    def __post_init__(self):
        if self.setup not in SETUPS:
            raise ValueError(f"setup must be one of {SETUPS}, got {self.setup!r}")
        if self.promotion not in PROMOTIONS:
            raise ValueError(f"promotion must be one of {PROMOTIONS}, got {self.promotion!r}")
        if self.stalemate not in STALEMATE_RULES:
            raise ValueError(f"stalemate must be one of {STALEMATE_RULES}, got {self.stalemate!r}")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least 2")
        if self.no_progress_plies < 1 or self.max_plies < 1:
            raise ValueError("no_progress_plies and max_plies must be positive")


DEFAULT_RULES = RulesConfig()
