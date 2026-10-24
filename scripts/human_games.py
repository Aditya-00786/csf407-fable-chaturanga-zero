"""The results of the games people played against the computer in the app (results/human_games/),
as Markdown tables for the docs.

    python scripts/human_games.py                 # against every opponent
    python scripts/human_games.py --milestone     # against the Milestone 4 agent only

The Results page shows the same, and adds and checks the files teammates send (docs/HUMAN_GAMES.md).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaturanga.agents import MILESTONES  # noqa: E402
from chaturanga_server.human_games import all_games, games_dir, summary  # noqa: E402


def _row(name: str, t: dict) -> str:
    score = "–" if t["score"] is None else f"{t['score']:.0%}"
    return f"| {name} | {t['games']} | {t['win']} | {t['draw']} | {t['loss']} | {score} |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=games_dir(), help="where the games are (default results/human_games/)")
    ap.add_argument("--milestone", action="store_true", help="only the games against a milestone's frozen agent")
    args = ap.parse_args()
    games = all_games(args.folder)
    if args.milestone:
        games = [g for g in games if (g.get("network") or {}).get("fingerprint") in MILESTONES]
    if not games:
        sys.exit(f"No recorded games in {args.folder}.")
    s = summary(games)
    header = "| {} | Games | Won | Drawn | Lost | People's score |\n|---|:-:|:-:|:-:|:-:|:-:|"
    print(header.format("Opponent"))
    for o in s["opponents"]:
        print(_row(o["label"], o))
    print()
    print(header.format("Player"))
    for p in s["players"]:
        print(_row(p["player"], p))
    n, players = s["total"], len(s["players"])
    print(f"\nWon, drawn and lost are the people's; resigning or leaving a game counts as a loss. "
          f"{n} {'game' if n == 1 else 'games'} by {players} {'player' if players == 1 else 'players'}.")


if __name__ == "__main__":
    main()
