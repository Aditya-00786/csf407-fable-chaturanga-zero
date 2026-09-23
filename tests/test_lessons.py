"""Every tutorial lesson in the web app must be a legal position with a solvable task."""

import json
from pathlib import Path

import pytest

from chaturanga import GameState, parse_move

LESSONS = json.loads((Path(__file__).parent.parent / "web" / "src" / "content" / "lessons.json").read_text())


def test_lesson_ids_are_unique():
    ids = [lesson["id"] for lesson in LESSONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_lesson_position_is_live(lesson):
    s = GameState.from_fen(lesson["fen"])
    assert s.outcome() is None, "a lesson must not start with the game already over"
    assert not s.in_check(), "lessons start without check so the board is calm"
    assert s.material_of(1) > 0 and s.material_of(-1) > 0


@pytest.mark.parametrize("lesson", [lesson for lesson in LESSONS if lesson["task"]], ids=lambda lesson: lesson["id"])
def test_lesson_task_is_solvable(lesson):
    s = GameState.from_fen(lesson["fen"])
    task = lesson["task"]
    if task["type"] == "move":
        for text in task["moves"]:
            assert s.is_legal(parse_move(text)), text
    else:
        winners = [m for m in s.legal_moves()
                   if (out := s.apply(m).outcome()) and out.reason == task["reason"] and out.winner == s.side]
        assert winners, f"no move reaches {task['reason']}"


def test_stalemate_lesson_contains_the_trap():
    lesson = next(lesson for lesson in LESSONS if lesson["id"] == "stalemate")
    out = GameState.from_fen(lesson["fen"]).play("e6f7").outcome()
    assert out.reason == "stalemate" and out.winner == -1  # the stalemated Black side wins
