# Games against people

One file per player, `<name>.jsonl`, one game per line: the games people played against the
computer in the app with recording on, each replayed and checked by the rules engine before it was
kept. The app's **Results** page adds them up; `python scripts/human_games.py` prints the same as
Markdown tables.

How teammates record and send their games, what counts, and the fields of a game:
[docs/HUMAN_GAMES.md](../../docs/HUMAN_GAMES.md).
