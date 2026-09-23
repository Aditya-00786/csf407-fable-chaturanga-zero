# ChaturangZero

**An AlphaZero-style agent that learns Chaturanga, the ancient Indian ancestor of chess, from self-play alone.**

The agent is given nothing but the rules of the game. It combines a residual neural network
(policy + value heads) with Monte Carlo Tree Search, and improves by playing against itself.
Everything is scaled down to train on a single consumer GPU or a free Google Colab instance.

> Course project for **CS F407 Artificial Intelligence**, BITS Pilani K. K. Birla Goa Campus, by **Team Fable**.

---

## Progress

**Current milestone: 2 of 4 — MCTS with an untrained network**

```
Overall   ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  25%   (1 of 4 milestones done)
```

| # | Milestone | Progress | Status |
|:-:|---|---|---|
| 1 | Rules engine, state/move encoding, unit tests | `██████████` 100% | Done |
| 2 | MCTS integrated with an untrained network | `░░░░░░░░░░`   0% | Up next |
| 3 | Self-play training pipeline (50–100 games, loss tracking) | `░░░░░░░░░░`   0% | Planned |
| 4 | Benchmarking vs random, heuristic and earlier agents | `░░░░░░░░░░`   0% | Planned |

<details>
<summary>How to update the progress bars</summary>

Each bar is 10 characters for a milestone and 40 for the overall bar. Use `█` for done and `░`
for remaining, and update the percentage and the "Current milestone" line. Update the bars in
the same commit that completes the work.
</details>

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Setting up the repo locally](#setting-up-the-repo-locally)
- [Essential commands](#essential-commands)
- [Using the engine](#using-the-engine)
- [Project structure](#project-structure)
- [The rules in brief](#the-rules-in-brief)
- [Architecture](#architecture)
- [Team](#team)
- [Contributing and commit guidelines](#contributing-and-commit-guidelines)
- [References](#references)

---

## Highlights

- **Researched ruleset.** Every rule is backed by historical sources, and contested rules are
  settled by counting references. See [docs/RULES.md](docs/RULES.md).
- **Verified rules engine.** 61 unit tests, a cross-check against an independent reference
  move generator, and a 10,000-game random-play stress test.
- **Network-ready encoding.** 15×8×8 input planes from the side to move's point of view, and a
  4,096-move action space with a legal-move mask.
- **Configurable rules.** Contested historical rules (setup, promotion, stalemate) can be switched
  in `RulesConfig` for experiments.

| Engine benchmark (random play, Apple M1) | |
|---|---|
| Single core | ~41,000 positions/s |
| 8 cores | ~208,000 positions/s |

---

## Quick start

```bash
git clone https://github.com/Aditya-00786/csf407-fable-chaturanga-zero.git
cd csf407-fable-chaturanga-zero
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

---

## Setting up the repo locally

**Requirements:** Python 3.10 or newer (developed on 3.14) and git. PyTorch is added in Milestone 2.

1. **Clone the repository.**
   ```bash
   git clone https://github.com/Aditya-00786/csf407-fable-chaturanga-zero.git
   cd csf407-fable-chaturanga-zero
   ```
2. **Create and activate a virtual environment.** This keeps the project's packages separate from the system Python.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```
3. **Install the package in editable mode**, with development tools:
   ```bash
   pip install -e ".[dev]"
   ```
   "Editable" means changes to the code under `chaturanga/` take effect without reinstalling.
4. **Check that everything works.**
   ```bash
   python -m pytest -q                # expect: 61 passed
   ```

The commands below assume the virtual environment is active. If it isn't, prefix them with
`.venv/bin/` (e.g. `.venv/bin/python -m pytest`).

---

## Essential commands

| Task | Command |
|---|---|
| Run all tests | `python -m pytest -q` |
| Run one test file | `python -m pytest tests/test_moves.py -v` |
| Run tests matching a name | `python -m pytest -k stalemate -v` |
| Stop at the first failure | `python -m pytest -x` |
| Random-play stress test (10k games) | `python scripts/random_play.py --games 10000` |
| Quick random-play benchmark (1 core) | `python scripts/random_play.py --games 200 --workers 1` |
| Random play with alternative rules | `python scripts/random_play.py --setup facing --promotion file` |
| Show all script options | `python scripts/random_play.py --help` |

`random_play.py` prints how the games ended, how long they were and how fast the engine ran.
Every move comes from the legal-move list and every game must end, so a crash or a hang means a bug.

---

## Using the engine

```python
from chaturanga import GameState, move_to_str

s = GameState.initial()                 # standard starting position
print(s)                                # ASCII board
print([move_to_str(m) for m in s.legal_moves()])

s = s.play("e2e3").play("d7d6")         # moves are written from-square + to-square
print(s.outcome())                      # None while the game is in progress

s2 = GameState.from_fen("3k4/8/8/8/8/8/4R3/4K3 w")   # set up any position
print(s2.outcome())                     # Outcome(winner=1, reason='bare_king')
```

- **Immutable states:** `GameState` never changes. `apply(move)` and `play("e2e3")` return a new
  state, which is what the search tree needs.
- **Move numbers:** a move is the integer `from_square * 64 + to_square`, with squares `a1 = 0 … h8 = 63`.
- **Neural-network input:** see `chaturanga.encoding`.
  - `encode_state(state)` returns the 15×8×8 input planes.
  - `legal_action_mask(state)` returns the 4,096-long legal-move mask.
  - `move_to_action` and `action_to_move` convert between board moves and network actions.

FEN letters: `K` Raja, `M` Mantri, `R` Ratha, `G` Gaja, `N` Ashva, `P` Padati. Upper case is White.

---

## Project structure

```
csf407-fable-chaturanga-zero/
├── chaturanga/              # rules engine (Python package)
│   ├── config.py            # RulesConfig: rule options, defaults = docs/RULES.md
│   ├── constants.py         # piece codes, colours, square helpers
│   ├── tables.py            # precomputed move tables and attack detection
│   ├── zobrist.py           # position hashing for repetition detection
│   ├── game.py              # GameState: legal moves, apply(), outcome()
│   └── encoding.py          # network input planes, action space, legal mask
├── tests/                   # pytest suite
│   ├── reference.py         # independent, deliberately naive move generator
│   └── test_*.py            # setup, moves, check, termination, encoding, fuzzing
├── scripts/
│   └── random_play.py       # random-vs-random stress test and benchmark
├── docs/
│   ├── RULES.md             # rules specification with source tallies
│   └── Handouts/            # source PDFs cited in RULES.md
├── pyproject.toml           # package metadata and dependencies
└── README.md
```

Planned for later milestones: `chaturanga_zero/` (network, MCTS, self-play, training),
`checkpoints/` and `runs/` (git-ignored training outputs).

---

## The rules in brief

The full specification, with every source and the reference counts, is in [docs/RULES.md](docs/RULES.md).

```
  8 | r n g k m g n r        K Raja (king)       1 step any direction, no castling
  7 | p p p p p p p p        M Mantri (minister) 1 step diagonally
  6 | . . . . . . . .        R Ratha (chariot)   like a rook
  5 | . . . . . . . .        G Gaja (elephant)   jumps exactly 2 squares diagonally
  4 | . . . . . . . .        N Ashva (horse)     like a knight
  3 | . . . . . . . .        P Padati (soldier)  1 step forward, captures diagonally,
  2 | P P P P P P P P                            no double step; promotes to Mantri
  1 | R N G M K G N R
      a b c d e f g h        Rajas do not face each other (e1 vs d8). White moves first.
```

| A game ends when… | Result |
|---|---|
| a side is left with only its Raja (bare king) | the other side wins immediately |
| the side to move is checkmated | the checkmating side wins |
| the side to move has no legal move but is not in check (stalemate) | the **stalemated** side wins |
| the same position occurs 3 times | draw |
| 70 moves each pass with no capture and no Padati move | draw |
| 300 plies (single moves by either side) are played (engineering cap) | draw |

---

## Architecture

```
            ┌──────────────────┐   planes, legal mask    ┌────────────────────────┐
            │  Rules engine    │ ──────────────────────▶ │  ResNet (4–6 blocks)   │
            │  chaturanga/     │                         │  policy head: 4096     │
            └──────────────────┘ ◀────────────────────── │  value head:  [-1, 1]  │
                   ▲    │           priors P(s,a), v(s)  └────────────────────────┘
          apply()  │    ▼                                           ▲
            ┌──────────────────┐   (state, π, z) samples ┌────────────────────────┐
            │  MCTS (PUCT,     │ ──────────────────────▶ │  Replay buffer +       │
            │  30–50 sims)     │                         │  training loop         │
            └──────────────────┘                         └────────────────────────┘
                       self-play games ─────▶ new network weights ─────▶ evaluation
```

- **Loss:** value MSE + policy cross-entropy + L2 regularisation.
- **Scaled-down settings:** 4–6 residual blocks, 64–128 filters, and 30–50 MCTS simulations per move.
  This keeps training feasible on an RTX 3060/4060 or free Colab.

---

## Team

**Team Fable**, CS F407 Artificial Intelligence.

| Role | Responsibilities |
|---|---|
| Environment & Logic Engineer | Board logic, tensor encoding, legal move generation |
| Deep Learning & Pipeline Architect | ResNet model, training loop, replay buffer |
| Search Engine & Evaluation Specialist | MCTS / PUCT, evaluation suite, progress plots |

---

## Contributing and commit guidelines

### Workflow

1. **Get the latest code:** `git pull` on `main` before starting.
2. **Work on a branch**, never directly on `main`:
   ```bash
   git checkout -b feat/mcts-node-expansion
   ```
   Branch prefixes: `feat/`, `fix/`, `test/`, `docs/`, `refactor/`, `exp/` (experiments).
3. **Keep commits small and focused.** One logical change per commit.
4. **Before every commit:**
   - Run `python -m pytest -q`; all tests must pass.
   - If you touched the engine, also run `python scripts/random_play.py --games 1000`.
5. **Open a pull request** into `main`. Another team member reviews it before it is merged.

### Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary in the imperative, max ~72 chars>

<optional body: what changed and why, wrapped at ~72 chars>
```

| Type | Use for |
|---|---|
| `feat` | New functionality (`feat(mcts): add PUCT selection`) |
| `fix` | Bug fixes (`fix(engine): reset no-progress counter on capture`) |
| `test` | Adding or changing tests only |
| `docs` | Documentation, README, RULES.md |
| `refactor` | Code changes that don't change behaviour |
| `perf` | Speed improvements |
| `chore` | Tooling, dependencies, config |
| `exp` | Training experiments and hyperparameter runs |

Scopes: `engine`, `encoding`, `mcts`, `net`, `train`, `eval`, `rules`, `scripts`.

### Code and rules conventions

- **Rule changes go through [docs/RULES.md](docs/RULES.md) first.** Add the source, update the
  reference tally, then change the code and tests in the same pull request.
- **Every bug fix comes with a test** that fails without the fix.
- **Match the style of the surrounding code:** type hints on public functions, and short
  docstrings that explain *why* rather than *what*.
- **Never commit** virtual environments, training checkpoints or large datasets. `.gitignore`
  covers these. The repository is private because `docs/Handouts/` contains copyrighted source PDFs;
  keep it private.

---

## References

- H. J. R. Murray, *A History of Chess*, Oxford, 1913. The main historical source for the rules.
- D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go
  through self-play", *Science* 362 (2018). The AlphaZero algorithm.
- Full list of rule sources: [docs/RULES.md §1](docs/RULES.md#1-sources).
