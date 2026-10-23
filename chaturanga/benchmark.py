"""Yardsticks for a run beyond its test games (Milestone 4): a rating ladder and a conversion test.

The run's own tests stop showing progress as it gets strong: against the generation before, it
scores about 50% by design, and Greedy, which it nearly always beats, has little more to tell.

Rating ladder. The run's networks every few generations (with the first, the latest and the best)
play each other and Random and Greedy from the opening, `games` games a pair with colours
alternating, searching as the run's test games do (its simulations, its sampled opening plies and
the search's rules). Ratings are Elo, fitted to every result at once (Bradley-Terry, a draw half a
win), with Random at 0. Each pair also counts one virtual draw, so a clean sweep gives a large gap
rather than an endless one. The 95% intervals come from the fit's curvature and are against Random;
each step from one rated generation to the next has its own, much narrower (neighbours' ratings
move together, so their difference is known better than either rating).

Conversion test. A fixed set of positions in which one side is ahead in material (a Ratha and a
Padati against a Ratha, say), placed at random once with a fixed seed. Each network plays the side
ahead against Greedy, both playing their best moves from the first. A win counts the same however
long it takes, so this measures what scores cannot: how many it wins, and in how many plies. Greedy
playing the side ahead is the reference.

Everything is kept in report/benchmark.json with every pair's results, so rating more generations
later plays only their games. Progress is written there as games finish.
"""

import json
import math
import multiprocessing as mp
import os
import random
import socket
import statistics
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .constants import BLACK, LETTER_PIECES, PADATI, PIECE_LETTERS, RAJA, WHITE, square_name
from .game import GameState
from .mcts import wins_at_once
from .tables import is_attacked

BENCHMARK_FILE = Path("report") / "benchmark.json"
BASELINES = ("random", "greedy")
VIRTUAL_DRAWS = 1  # per pair, besides its games

# The side ahead's pieces and the side behind's, besides their Rajas: a small edge, which takes a
# plan to turn into a win (promote a Padati, bring two pieces together, catch the last piece
# without stalemating it: the stalemated side wins). Chosen from 14 balances tried on main-v4's
# networks (12 positions each) as those that separate them: generation 60 won 10 to 12 of 12 and
# generation 10 at most 8, and 60 won in about half the plies 30 took. With a large edge (a Ratha
# against a Padati, say) every network from generation 10 won 39 or 40 of 40, which says nothing.
RECIPES = [
    ("An Ashva and a Padati against an Ashva", "NP", "N"),
    ("A Ratha and a Padati against a Ratha", "RP", "R"),
    ("A Ratha and an Ashva against a Ratha", "RN", "R"),
    ("A Ratha and a Gaja against a Ratha", "RG", "R"),
    ("A Ratha and two Padatis against a Ratha and a Padati", "RPP", "RP"),
]
_PADATI_SQUARES = tuple(range(8, 56))  # never on the first or last rank


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def won_positions(per_recipe: int = 8, seed: int = 4007, recipes: Optional[list] = None) -> list[dict]:
    """The conversion test's positions, the same every time: for each recipe (RECIPES unless
    given), `per_recipe` legal, unfinished positions with the side ahead to move (White in half of
    them) and no win in one."""
    rng = random.Random(seed)
    out = []
    for name, ahead, behind in recipes or RECIPES:
        found = 0
        while found < per_recipe:
            side = WHITE if found % 2 == 0 else BLACK
            board: dict[int, int] = {}
            for colour, letters in ((side, "K" + ahead), (-side, "K" + behind)):
                for letter in letters:
                    kind = LETTER_PIECES[letter]
                    free = [sq for sq in (_PADATI_SQUARES if kind == PADATI else range(64)) if sq not in board]
                    board[rng.choice(free)] = colour * kind
            squares = [0] * 64
            for sq, piece in board.items():
                squares[sq] = piece
            their_king = next(sq for sq, p in board.items() if p == -side * RAJA)
            if is_attacked(squares, their_king, side):
                continue  # the side not to move would be in check
            state = GameState.from_pieces({square_name(sq): PIECE_LETTERS[abs(p)] if p > 0 else PIECE_LETTERS[abs(p)].lower()
                                           for sq, p in board.items()}, side)
            if state.outcome() is not None or any(wins_at_once(state, m) for m in state.legal_moves()):
                continue
            out.append({"recipe": name, "fen": state.to_fen()})
            found += 1
    return out


# ---------------------------------------------------------------------- ratings

def _fit(entries: list[str], pairs: dict, anchor: str):
    """Bradley-Terry by Newton's method (a draw half a win, one virtual draw a pair) over the entries
    linked to `anchor` by games: their names, strengths in natural units (anchor 0) and covariance."""
    import numpy as np

    linked, frontier = {anchor}, [anchor]
    while frontier:
        e = frontier.pop()
        for p in pairs.values():
            if p["games"] and e in (p["a"], p["b"]):
                other = p["b"] if p["a"] == e else p["a"]
                if other not in linked:
                    linked.add(other)
                    frontier.append(other)
    names = [e for e in entries if e in linked]
    index = {e: i for i, e in enumerate(names)}
    n = len(names)
    games, points = np.zeros((n, n)), np.zeros((n, n))  # points[i, j]: i's against j
    for p in pairs.values():
        if p["games"] and p["a"] in index and p["b"] in index:
            i, j = index[p["a"]], index[p["b"]]
            g = p["games"] + VIRTUAL_DRAWS
            s = p["a_wins"] + 0.5 * p["draws"] + 0.5 * VIRTUAL_DRAWS
            games[i, j] += g
            games[j, i] += g
            points[i, j] += s
            points[j, i] += g - s
    free = [k for k in range(n) if names[k] != anchor]
    r = np.zeros(n)  # natural units: P(i beats j) = 1 / (1 + exp(r_j - r_i))
    info = np.zeros((n, n))
    for _ in range(100):
        p = 1 / (1 + np.exp(r[None, :] - r[:, None]))
        grad = (points - games * p).sum(axis=1)
        w = games * p * (1 - p)
        info = np.diag(w.sum(axis=1)) - w
        if not free:
            break
        step = np.linalg.solve(info[np.ix_(free, free)], grad[free])
        r[free] += np.clip(step, -2, 2)
        if np.abs(step).max() < 1e-10:
            break
    cov = np.zeros((n, n))
    if free:
        cov[np.ix_(free, free)] = np.linalg.inv(info[np.ix_(free, free)])
    return names, r, cov


_SCALE = 400 / math.log(10)  # natural units to Elo


def fit_elo(entries: list[str], pairs: dict, anchor: str = "random") -> dict:
    """Elo ratings fitted to the pairs' results at once, with `anchor` at 0 and 95% intervals against
    it (see _fit). Entries with no games linking them to the anchor have no rating yet."""
    import numpy as np

    names, r, cov = _fit(entries, pairs, anchor)
    index = {e: i for i, e in enumerate(names)}
    scale, se = _SCALE, np.sqrt(np.diag(cov))
    out = {}
    for e in entries:
        if e not in index:
            out[e] = None
            continue
        k = index[e]
        played = sum(p["games"] for p in pairs.values() if e in (p["a"], p["b"]))
        scored = sum((p["a_wins"] + 0.5 * p["draws"]) if p["a"] == e else (p["b_wins"] + 0.5 * p["draws"])
                     for p in pairs.values() if e in (p["a"], p["b"]))
        out[e] = {"elo": round(float(r[k] * scale), 1), "low": round(float((r[k] - 1.96 * se[k]) * scale), 1),
                  "high": round(float((r[k] + 1.96 * se[k]) * scale), 1), "games": played,
                  "score": round(scored / played, 4) if played else None}
    return out


def elo_steps(entries: list[str], pairs: dict, anchor: str = "random") -> list[dict]:
    """From each rated generation to the next: the change in Elo, with its own 95% interval."""
    names, r, cov = _fit(entries, pairs, anchor)
    gens = sorted((int(e[4:]), names.index(e)) for e in names if e.startswith("gen_"))
    out = []
    for (ga, i), (gb, j) in zip(gens, gens[1:]):
        diff = (r[j] - r[i]) * _SCALE
        margin = 1.96 * math.sqrt(max(0.0, cov[i, i] + cov[j, j] - 2 * cov[i, j])) * _SCALE
        out.append({"from": ga, "to": gb, "diff": round(float(diff), 1), "low": round(float(diff - margin), 1),
                    "high": round(float(diff + margin), 1)})
    return out


# ---------------------------------------------------------------------- what to play

def read_benchmark(run_dir) -> Optional[dict]:
    try:
        return json.loads((Path(run_dir) / BENCHMARK_FILE).read_text())
    except (OSError, ValueError):
        return None


def _write(run_dir: Path, data: dict) -> None:
    path = run_dir / BENCHMARK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**data, "updated": _now()}))
    tmp.replace(path)


def _generations(run_dir: Path) -> list[int]:
    return sorted(int(p.stem[4:]) for p in (run_dir / "checkpoints").glob("gen_*.pt"))


def _spec(run_dir: Path, entry: str) -> str:
    """What play_matches plays for an entry: a baseline's name, or the network's checkpoint."""
    return entry if entry in BASELINES else str(run_dir / "checkpoints" / f"{entry}.pt")


def pair_key(a: str, b: str) -> str:
    return f"{a} v {b}"


def plan(run_dir, games: int = 40, every: int = 10) -> dict:
    """What rating the run with these settings would do: the ladder's entries (those rated before
    stay), the games still to play in each pair, and the networks the conversion test still needs."""
    from .analysis import _checkpoints

    run_dir = Path(run_dir)
    old = read_benchmark(run_dir) or {}
    gens = _generations(run_dir)
    picked = {g for g in gens if g % max(1, every) == 0} | set(gens[-1:])
    best = _checkpoints(run_dir).get("best")
    if best:
        picked.add(best[0])
    known = [e for e in old.get("entries", []) if e not in BASELINES]
    nets = sorted(set(known) | {f"gen_{g:04d}" for g in picked})
    entries = list(BASELINES) + nets
    playable = [e for e in entries if e in BASELINES or (run_dir / "checkpoints" / f"{e}.pt").exists()]
    pairs = old.get("pairs", {})
    ladder = []
    for i, a in enumerate(playable):
        for b in playable[i + 1:]:
            have = pairs.get(pair_key(a, b), {}).get("games", 0)
            if have < games:
                ladder.append((a, b, list(range(have, games))))
    suite = won_positions()
    measured = old.get("conversion", {}) if old.get("suite") == [p["fen"] for p in suite] else {}
    converting = [e for e in playable if e != "random" and e not in measured]
    return {"entries": entries, "new": [e for e in entries if e not in old.get("entries", [])], "ladder": ladder,
            "converting": converting, "suite": suite, "kept_conversion": measured,
            "ladder_games": sum(len(ks) for _, _, ks in ladder), "conversion_games": len(converting) * len(suite)}


# ---------------------------------------------------------------------- playing it

def run_benchmark(run_dir, games: int = 40, every: int = 10, workers: int = 0, gpu_workers: int = 0,
                  log: Callable[[str], None] = print, on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Play what `plan` says is missing and refit the ratings, saving progress to
    report/benchmark.json as games finish (and telling `on_progress`). Ladder games are dealt a
    pair at a time (two networks to a batch), and the conversion test a network at a time."""
    import torch

    from .training import _init_worker, play_matches

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    sims = config.get("simulations", 50)
    opening = config.get("eval_opening_plies", 16)
    todo = plan(run_dir, games, every)
    old = read_benchmark(run_dir) or {}
    pairs = dict(old.get("pairs", {}))
    conversion = dict(todo["kept_conversion"])
    data = {"state": "running", "games": games, "every": every, "simulations": sims, "opening_plies": opening,
            "pid": os.getpid(), "host": socket.gethostname(), "started": _now(), "entries": todo["entries"],
            "pairs": pairs, "ratings": fit_elo(todo["entries"], pairs), "suite": [p["fen"] for p in todo["suite"]],
            "recipes": [p["recipe"] for p in todo["suite"]], "conversion": conversion,
            "done": 0, "total": todo["ladder_games"] + todo["conversion_games"]}
    _write(run_dir, data)

    # Tasks of at most 40 games: (kind, key, games for play_matches).
    tasks = []
    for a, b, indices in todo["ladder"]:
        seed = zlib.crc32(pair_key(a, b).encode()) & 0x7FFFFFFF
        for k in range(0, len(indices), 40):
            tasks.append(("pair", (a, b), [(_spec(run_dir, a), _spec(run_dir, b), i, seed) for i in indices[k:k + 40]]))
    for e in todo["converting"]:
        tasks.append(("convert", e, [(_spec(run_dir, e), "greedy", j, 31_337, p["fen"])
                                     for j, p in enumerate(todo["suite"])]))
    # The slowest first (two networks searching), so the end is not held up by one long task.
    tasks.sort(key=lambda t: -sum(s not in BASELINES for s in (t[2][0][0], t[2][0][1])))

    if on_progress:
        on_progress(0, data["total"])

    def keep(kind, key, rows: list[dict]) -> None:
        if kind == "pair":
            a, b = key
            p = pairs.setdefault(pair_key(a, b), {"a": a, "b": b, "games": 0, "a_wins": 0, "draws": 0, "b_wins": 0,
                                                  "reasons": {}})
            for r in rows:
                p["games"] += 1
                p["a_wins" if r["score"] == 1 else "draws" if r["score"] == 0.5 else "b_wins"] += 1
                p["reasons"][r["reason"]] = p["reasons"].get(r["reason"], 0) + 1
            data["ratings"] = fit_elo(todo["entries"], pairs)
        else:
            won = sorted(r["plies"] for r in rows if r["score"] == 1)
            ends: dict = {}
            for r in rows:
                if r["score"] != 1:
                    ends[r["reason"]] = ends.get(r["reason"], 0) + 1
            conversion[key] = {"games": len(rows), "won": len(won), "lost": sum(r["score"] == 0 for r in rows),
                               "median_plies": statistics.median(won) if won else None,
                               "mean_plies": round(statistics.mean(won), 1) if won else None,
                               "by_recipe": {name: sum(r["score"] == 1 for r, p in zip(rows, todo["suite"])
                                                       if p["recipe"] == name) for name, _, _ in RECIPES},
                               "not_won": ends}
        _write(run_dir, data)

    try:
        if tasks:
            workers = workers or max(1, (os.cpu_count() or 2) - 2)
            on_gpu = gpu_workers > 0 and torch.cuda.is_available()
            # On a GPU, fewer processes with larger batches are faster (see training.match_plan).
            procs = max(1, workers // 2) if on_gpu else workers
            ctx = mp.get_context("spawn")
            moves, finished = ctx.Value("q", 0), ctx.Value("q", 0)  # counted by the players as they go
            devices = ["cuda" if on_gpu else "cpu"] * procs
            with ctx.Pool(procs, initializer=_init_worker, initargs=(moves, finished, devices, ctx.Value("i", 0))) as pool:
                pending = [(kind, key, pool.apply_async(play_matches, (g, sims, opening))) for kind, key, g in tasks]
                while pending:
                    still = []
                    for kind, key, task in pending:
                        if task.ready():
                            keep(kind, key, task.get())
                        else:
                            still.append((kind, key, task))
                    pending = still
                    if finished.value != data["done"]:
                        data["done"] = finished.value
                        _write(run_dir, data)
                        if on_progress:
                            on_progress(data["done"], data["total"])
                    if pending:
                        time.sleep(0.5)
        data.update(state="done", done=data["total"], ratings=fit_elo(todo["entries"], pairs), finished=_now())
        _write(run_dir, data)
        log(format_benchmark(data))
    except BaseException as exc:
        data.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _write(run_dir, data)
        raise
    return data


def _label(entry: str) -> str:
    return {"random": "Random", "greedy": "Greedy"}.get(entry, f"generation {int(entry[4:])}" if entry.startswith("gen_") else entry)


def format_benchmark(d: dict) -> str:
    rows = [f"Rating ladder: {d['games']} games a pair at {d['simulations']} simulations (Elo, Random = 0, 95% interval)"]
    for e in d["entries"]:
        r = d["ratings"].get(e)
        if r:
            rows.append(f"  {_label(e):<16} {r['elo']:+7.0f}  ({r['low']:+.0f} to {r['high']:+.0f})  "
                        f"{r['games']} games, scored {r['score']:.0%}")
    for s in elo_steps(d["entries"], d["pairs"]):
        rows.append(f"  generation {s['from']} to {s['to']}: {s['diff']:+.0f} ({s['low']:+.0f} to {s['high']:+.0f})")
    rows.append(f"Conversion test: {len(d['suite'])} won positions against Greedy")
    for e in d["entries"]:
        c = d["conversion"].get(e)
        if c:
            plies = f", median {c['median_plies']:.0f} plies" if c["median_plies"] is not None else ""
            rows.append(f"  {_label(e):<16} won {c['won']} of {c['games']}{plies}")
    return "\n".join(rows)
