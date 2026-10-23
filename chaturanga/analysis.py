"""A training run's report: what it learned, read from what the run saved.

A report is made on request (the Run report's button, scripts/report.py) or, if the run was
started with auto_report, when it finishes; make_report saves it in report/report.json with the
charts. Its parts, each usable on its own:

  run_summary(run_dir)       instant, from metrics.jsonl: did the policy learn, did the value beat
                             predicting a draw on unseen games, how the scores and self-play moved;
  value_probe(run_dir)       a second or so: the value the untrained, best and final networks give
                             positions with pieces missing (does the value head see material?);
  measure_strength(run_dir)  minutes: the untrained, best and final networks play Random and Greedy
                             from the opening; progress and results go to report/strength.json.
"""

import json
import math
import multiprocessing as mp
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

STRENGTH_FILE = Path("report") / "strength.json"
REPORT_FILE = Path("report") / "report.json"

# Each pair differs by one side's material, with White to move; the second of each pair is the
# first with colours of the missing pieces swapped, so a value that sees material rates them apart.
PROBES = [
    ("The starting position", "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w 0 0"),
    ("Black without its Rathas", "1ngkmgn1/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w 0 0"),
    ("White without its Rathas", "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/1NGMKGN1 w 0 0"),
    ("Black without Ashvas and Rathas", "2gkmg2/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w 0 0"),
    ("White without Ashvas and Rathas", "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/2GMKG2 w 0 0"),
    ("Black down to its Raja and 4 Padatis", "3k4/pppp4/8/8/8/8/PPPPPPPP/RNGMKGNR w 0 0"),
]
# (White ahead, White behind) pairs among PROBES, by index.
_IMBALANCES = [(1, 2), (3, 4)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _events(run_dir: Path) -> list[dict]:
    out = []
    try:
        with open(run_dir / "metrics.jsonl") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _generations(events: list[dict]) -> dict[int, dict]:
    gens: dict[int, dict] = {}
    for e in events:
        g = e.get("generation")
        if g is None or e["event"] == "train":
            continue
        row = gens.setdefault(g, {"eval": {}})
        if e["event"] == "eval":
            row["eval"][e["opponent"]] = e
        else:
            row[e["event"]] = e
    return gens


def _checkpoints(run_dir: Path) -> dict[str, tuple[int, Path]]:
    """The untrained, best and final networks of a run, as {label: (generation, path)}."""
    ckpts = run_dir / "checkpoints"
    gens = sorted(int(p.stem[4:]) for p in ckpts.glob("gen_*.pt"))
    if not gens:
        return {}
    out = {"untrained": (0, ckpts / "gen_0000.pt")}
    try:
        import torch

        state = torch.load(ckpts / "train_state.pt", map_location="cpu", weights_only=True)
        best = int(state.get("best", gens[-1]))
        if best not in (0, gens[-1]) and (ckpts / f"gen_{best:04d}.pt").exists():
            out["best"] = (best, ckpts / f"gen_{best:04d}.pt")
    except (OSError, RuntimeError, ValueError):
        pass
    out["final"] = (gens[-1], ckpts / f"gen_{gens[-1]:04d}.pt")
    return out


# ---------------------------------------------------------------------- the summary

def run_summary(run_dir) -> dict:
    """Verdicts on the policy, the value, strength and self-play, from the run's metrics."""
    run_dir = Path(run_dir)
    events = _events(run_dir)
    gens = _generations(events)
    done = sorted(g for g, r in gens.items() if "generation" in r)
    train = {e["step"]: e for e in events if e["event"] == "train"}
    held = [(g, gens[g]["held_out"]) for g in sorted(gens) if gens[g].get("held_out", {}).get("loss") is not None]
    out: dict = {"generations": len(done)}

    if held:
        first, last = held[0][1], held[-1][1]
        last_steps = [t for t in train.values() if t["generation"] == held[-1][0]]
        floor = sum(t["policy_entropy"] for t in last_steps) / len(last_steps) if last_steps else None
        learned = last["policy_loss"] < first["policy_loss"] - 1.0
        out["policy"] = {"first": first["policy_loss"], "last": last["policy_loss"], "floor": floor,
                         "verdict": "learned" if learned else "not_learned"}
        with_base = [(g, h) for g, h in held if "value_baseline" in h]
        if with_base:
            beat = [g for g, h in with_base if h["value_loss"] < h["value_baseline"]]
            g_last, h_last = with_base[-1]
            out["value"] = {
                "generations": len(with_base), "beat_baseline": len(beat),
                "last": h_last["value_loss"], "last_baseline": h_last["value_baseline"], "last_generation": g_last,
                "verdict": ("learned" if h_last["value_loss"] < h_last["value_baseline"] and len(beat) * 2 >= len(with_base)
                            else "partly" if beat else "not_learned"),
            }

    strength = {}
    for opponent in ("greedy", "random", "previous"):
        scores = [(g, gens[g]["eval"][opponent]["score"]) for g in sorted(gens) if opponent in gens[g]["eval"]]
        if scores:
            best_g, best_s = max(scores, key=lambda p: (p[1], p[0]))
            strength[opponent] = {"first": scores[0][1], "last": scores[-1][1], "best": best_s,
                                  "best_generation": best_g, "first_generation": scores[0][0],
                                  "last_generation": scores[-1][0]}
    out["strength"] = strength

    sp = [(g, gens[g]["selfplay"]) for g in sorted(gens) if "selfplay" in gens[g] and gens[g]["selfplay"].get("games")]
    if sp:
        decisive = [1 - s["draws"] / s["games"] for _, s in sp]
        out["selfplay"] = {"games": sum(s["games"] for _, s in sp) + sum(s.get("endgames", {}).get("games", 0) for _, s in sp),
                           "decisive_first": decisive[0], "decisive_last": decisive[-1],
                           "plies_first": sp[0][1]["mean_plies"], "plies_last": sp[-1][1]["mean_plies"]}
    return out


# ---------------------------------------------------------------------- the material probe

def value_probe(run_dir) -> dict:
    """The value each of the untrained, best and final networks gives the PROBES positions (for the
    side to move, White), and how far apart it rates the even and uneven pairs."""
    import torch

    from .game import GameState
    from .net import NetworkEvaluator, load_checkpoint

    run_dir = Path(run_dir)
    networks = []
    for label, (generation, path) in _checkpoints(run_dir).items():
        net, _ = load_checkpoint(str(path))
        ev = NetworkEvaluator(net, torch.device("cpu"))
        values = [ev.evaluate(GameState.from_fen(fen))[1] for _, fen in PROBES]
        # How much better it rates being ahead than behind, averaged over the pairs (positive = sees material).
        gap = sum(values[a] - values[b] for a, b in _IMBALANCES) / len(_IMBALANCES)
        networks.append({"label": label, "generation": generation, "values": values, "material_gap": gap})
    return {"positions": [name for name, _ in PROBES], "networks": networks}


# ---------------------------------------------------------------------- the saved report

def read_report(run_dir) -> Optional[dict]:
    try:
        return json.loads((Path(run_dir) / REPORT_FILE).read_text())
    except (OSError, ValueError):
        return None


def draw_charts(run_dir) -> Optional[Path]:
    """The six-panel chart (scripts/plot_training.py) as runs/<run>/training.png, or None without
    matplotlib."""
    import subprocess
    import sys

    run_dir = Path(run_dir)
    script = Path(__file__).resolve().parent.parent / "scripts" / "plot_training.py"
    done = subprocess.run([sys.executable, str(script), "--run", run_dir.name, "--runs-dir", str(run_dir.parent)],
                          capture_output=True, text=True, timeout=180)
    return run_dir / "training.png" if done.returncode == 0 else None


def make_report(run_dir) -> dict:
    """Work out the findings and the material probe, draw the charts, and save the report."""
    run_dir = Path(run_dir)
    report = {"made": _now(), "summary": run_summary(run_dir), "probe": value_probe(run_dir),
              "charts": draw_charts(run_dir) is not None}
    report["generations"] = report["summary"]["generations"]
    path = run_dir / REPORT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report))
    tmp.replace(path)
    return report


# ---------------------------------------------------------------------- the strength check

def read_strength(run_dir) -> Optional[dict]:
    try:
        return json.loads((Path(run_dir) / STRENGTH_FILE).read_text())
    except (OSError, ValueError):
        return None


def _write_strength(run_dir: Path, data: dict) -> None:
    path = run_dir / STRENGTH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**data, "updated": _now()}))
    tmp.replace(path)


def measure_strength(run_dir, games: int = 20, workers: int = 0, simulations: Optional[int] = None,
                     log: Callable[[str], None] = print, gpu_workers: int = 0) -> dict:
    """The untrained, best and final networks each play `games` games against Random and against
    Greedy from the opening, colours alternating: all of them side by side, several networks and
    opponents to a worker process (`gpu_workers` of which run the networks on the GPU; see
    training.play_matches). Progress and results are kept in report/strength.json as they come in."""
    import torch

    from .training import _init_worker, match_plan, play_matches

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    sims = simulations or config.get("simulations", 50)
    nets = _checkpoints(run_dir)
    jobs = [(label, opponent) for label in nets for opponent in ("random", "greedy")]
    data = {"state": "running", "games": games, "simulations": sims, "pid": os.getpid(), "host": socket.gethostname(),
            "networks": [{"label": k, "generation": g} for k, (g, _) in nets.items()],
            "results": {}, "done": 0, "total": len(jobs) * games}
    _write_strength(run_dir, data)
    seed = 424_242
    tally: dict = {}

    def add(keys: list, result: list[dict]) -> None:
        for (label, opponent), r in zip(keys, result):
            t = tally.setdefault(label, {}).setdefault(opponent, {"wins": 0, "draws": 0, "losses": 0, "reasons": {}})
            t["wins" if r["score"] == 1 else "draws" if r["score"] == 0.5 else "losses"] += 1
            t["reasons"][r["reason"]] = t["reasons"].get(r["reason"], 0) + 1
            n = t["wins"] + t["draws"] + t["losses"]
            t["score"] = (t["wins"] + 0.5 * t["draws"]) / n
            data["done"] += 1
        data["results"] = tally
        _write_strength(run_dir, data)

    try:
        workers = workers or max(1, (os.cpu_count() or 2) - 2)
        on_gpu = min(gpu_workers, workers) if torch.cuda.is_available() else 0
        ctx = mp.get_context("spawn")
        devices = ["cuda"] * on_gpu + ["cpu"] * (workers - on_gpu)
        # Every network's games against both baselines, dealt into tasks played side by side.
        everything = [((str(nets[label][1]), opponent, i, seed), (label, opponent))
                      for i in range(games) for label, opponent in jobs]
        with ctx.Pool(workers, initializer=_init_worker, initargs=(None, None, devices, ctx.Value("i", 0))) as pool:
            pending = [([k for _, k in task], pool.apply_async(play_matches, ([g for g, _ in task], sims, 8)))
                       for task in match_plan(everything, workers, on_gpu > 0)]
            while pending:
                still = []
                for keys, task in pending:
                    if task.ready():
                        add(keys, task.get())
                    else:
                        still.append((keys, task))
                pending = still
                if pending:
                    time.sleep(0.25)
        data["state"] = "done"
        _write_strength(run_dir, data)
        log(format_strength(data))
    except BaseException as exc:
        data.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _write_strength(run_dir, data)
        raise
    return data


# ---------------------------------------------------------------------- text, for the terminal

def _pct(x: Optional[float]) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def format_summary(s: dict) -> str:
    lines = [f"{s['generations']} generations"]
    if "policy" in s:
        p = s["policy"]
        lines.append(f"Policy  {p['verdict'].replace('_', ' ')}: loss on unseen games {p['first']:.2f} -> {p['last']:.2f}"
                     + (f" (floor {p['floor']:.2f})" if p.get("floor") else ""))
    if "value" in s:
        v = s["value"]
        lines.append(f"Value   {v['verdict'].replace('_', ' ')}: beat predicting a draw in {v['beat_baseline']} of "
                     f"{v['generations']} generations; last {v['last']:.3f} vs {v['last_baseline']:.3f}")
    for opponent, st in s.get("strength", {}).items():
        lines.append(f"vs {opponent:<8} first {_pct(st['first'])}, best {_pct(st['best'])} (gen {st['best_generation']}), "
                     f"last {_pct(st['last'])}")
    if "selfplay" in s:
        sp = s["selfplay"]
        lines.append(f"Self-play {sp['games']} games; decisive from the opening {_pct(sp['decisive_first'])} -> "
                     f"{_pct(sp['decisive_last'])}; length {sp['plies_first']:.0f} -> {sp['plies_last']:.0f} plies")
    return "\n".join(lines)


def format_probe(p: dict) -> str:
    head = f"{'':<38}" + "".join(f"{n['label']} ({n['generation']})".rjust(18) for n in p["networks"])
    rows = [head] + [f"{name:<38}" + "".join(f"{n['values'][i]:+18.2f}" for n in p["networks"])
                     for i, name in enumerate(p["positions"])]
    rows.append(f"{'Ahead minus behind (sees material > 0)':<38}" + "".join(f"{n['material_gap']:+18.2f}" for n in p["networks"]))
    return "\n".join(rows)


def format_strength(d: dict) -> str:
    rows = [f"{d['games']} games each at {d['simulations']} simulations per move"]
    for net in d["networks"]:
        for opponent in ("random", "greedy"):
            r = d["results"].get(net["label"], {}).get(opponent)
            if r:
                rows.append(f"  {net['label']} (gen {net['generation']}) vs {opponent:<7} "
                            f"{r['wins']} W / {r['draws']} D / {r['losses']} L  ({r['score']:.0%})")
    return "\n".join(rows)
