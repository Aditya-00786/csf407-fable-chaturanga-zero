# ChaturangZero

**An AlphaZero-style agent that learns Chaturanga, the ancient Indian ancestor of chess, from self-play alone.**

The agent is given nothing but the rules of the game. It combines a residual neural network
(policy + value heads) with Monte Carlo Tree Search, and improves by playing against itself.
Everything is scaled down to train on a single consumer GPU or a free Google Colab instance.

> Course project for **CS F407 Artificial Intelligence**, BITS Pilani K. K. Birla Goa Campus, by **Team Fable**.

---

## Progress

**Current milestone: 3 of 4 — self-play training pipeline**

```
Overall   ████████████████████░░░░░░░░░░░░░░░░░░░░  50%   (2 of 4 milestones done)
```

| # | Milestone | Progress | Status |
|:-:|---|---|---|
| 1 | Rules engine, state/move encoding, unit tests | `██████████` 100% | Done |
| 2 | MCTS integrated with an untrained network | `██████████` 100% | Done |
| 3 | Self-play training pipeline (50–100 games, loss tracking) | `░░░░░░░░░░`   0% | Up next |
| 4 | Benchmarking vs random, heuristic and earlier agents | `░░░░░░░░░░`   0% | Planned |

Also shipped: the **web app, phase 1**. It has a 3D board, three play modes, interactive lessons
and a rules reference. The untrained MCTS agent from Milestone 2 is already one of its computer
opponents; the trained agent and the live "Watch training" view follow Milestone 3.

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
- [Search and network](#search-and-network)
- [Web app](#web-app)
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
- **Verified rules engine.** A cross-check against an independent reference move generator and a
  10,000-game random-play stress test; 129 tests in all.
- **AlphaZero search.** PUCT Monte Carlo Tree Search guided by a residual policy-value network
  (6 blocks × 64 filters, about 1M parameters), with exact values for finished games, root noise,
  temperature and tree reuse for self-play. Planted bugs in the search are all caught by the tests.
- **Browser app.** A carved rosewood-and-brass 3D board in light and dark themes, with:
  - a figural piece set after Indian court chess sets: a crowned Raja, a turbaned Mantri, a temple-car
    Ratha, a caparisoned Gaja with its howdah, a Marwari Ashva, and spear-and-shield Padatis;
  - Human vs Human, Human vs Computer and Computer vs Computer;
  - 11 interactive lessons, each position checked against the engine;
  - captured pieces carried off to trays beside the board, on the capturing player's right;
  - an explanation of why every game ended.
- **Network-ready encoding.** 15×8×8 input planes from the side to move's point of view, and a
  4,096-move action space with a legal-move mask.
- **Configurable rules.** Contested historical rules (setup, promotion, stalemate) can be switched
  in `RulesConfig` for experiments.

| Benchmark (Apple M1) | |
|---|---|
| Engine, random play, single core | ~41,000 positions/s |
| Engine, random play, 8 cores | ~208,000 positions/s |
| MCTS with the network (CPU, 1 thread) | ~1,000 simulations/s, ~50 ms per 50-simulation move |

---

## Quick start

```bash
git clone https://github.com/Aditya-00786/csf407-fable-chaturanga-zero.git
cd csf407-fable-chaturanga-zero
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
(cd web && npm install && npm run build)
python -m chaturanga_server           # then open http://127.0.0.1:8000
```

---

## Setting up the repo locally

**Requirements:** Python 3.10 or newer (developed on 3.14), git, and Node.js 20 or newer for the web app
(developed on 24). PyTorch is installed with the package; it runs on the CPU, or on CUDA when
available (Colab already has it).

1. **Fork, then clone your fork.** Collaborators can't push to this repository (see
   [Workflow](#workflow)), so each of you works in your own fork. On GitHub, click **Fork** at the
   top of the repository page, then:
   ```bash
   git clone https://github.com/<your-username>/csf407-fable-chaturanga-zero.git
   cd csf407-fable-chaturanga-zero
   git remote add upstream https://github.com/Aditya-00786/csf407-fable-chaturanga-zero.git
   ```
   `origin` is now your fork (you push there) and `upstream` is this repository (you pull from it).
   A fork of this private repository stays private. The repository owner clones this repository
   directly and skips the fork.
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
   python -m pytest -q                # expect: 129 passed
   ```
5. **Install the web app's packages** (only needed for the browser app):
   ```bash
   cd web && npm install && cd ..
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
| MCTS speed only | `python scripts/mcts_benchmark.py --speed-only --threads 1` |
| Untrained MCTS vs Random and Greedy | `python scripts/mcts_benchmark.py --sims 50 --games 10 --threads 1` |
| Start the API server | `python -m chaturanga_server` (add `--reload` while developing) |
| Start the web dev server | `npm run dev --prefix web`, then open http://localhost:5173 |
| Type-check and lint the web app | `cd web && npx tsc -b && npx oxlint src` |
| Build the web app | `npm run build --prefix web` |
| Serve the built app (demo mode) | `python -m chaturanga_server`, then open http://127.0.0.1:8000 |

`random_play.py` prints how the games ended, how long they were and how fast the engine ran.
Every move comes from the legal-move list and every game must end, so a crash or a hang means a bug.
`mcts_benchmark.py` prints the search speed and match results, game by game, with colours alternating.

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

## Web app

The browser app lives in `web/`, built with React, TypeScript, Vite, Tailwind CSS, and
React Three Fiber for the 3D board.

**It never implements the rules itself.** Every move goes to the FastAPI server in
`chaturanga_server/`, which replays the game with the Python engine and returns:
- the position;
- the legal moves;
- the result.

The server keeps no game state, so undo and review are just shorter move lists.

**Running it**
- **Development** uses two terminals, and the dev server hot-reloads:
  ```bash
  python -m chaturanga_server --reload
  ```
  ```bash
  npm run dev --prefix web
  ```
  Then open http://localhost:5173. The dev server forwards `/api` requests to port 8000.
- **Demo** needs one command, after building once:
  ```bash
  npm run build --prefix web
  ```
  ```bash
  python -m chaturanga_server
  ```
  Then open http://127.0.0.1:8000. The Python server also serves the built app.

**Screens**

| Screen | What it does |
|---|---|
| **Play** | The board fills the screen. A floating card shows whose turn it is. The toolbar has **Game**, which opens a drawer (Esc closes it) with the mode, move list (click to review), captured pieces, display options (including Reduced graphics) and Rotate. Beside it, a ‹ button slides out quick actions: Undo, Full screen and New game. Opening the drawer tucks them away again. **Details** in the drawer switches between *Essential*, which shows headlines only and is the default, and *Full*. Under Full, four toggles turn individual extras on or off: move hints, move and quiet-ply counters, why the game ended, and rule sources. While reviewing, a bar at the foot of the board steps through the moves (⏮ ◀ ▶ ⏭, or ← → Home End) and replays the game at Slow, Normal or Fast; stepping back plays the move in reverse, captured pieces returning from their tray. Modes: two players (the board turns to face the player to move after each move; Rotate overrides it only until the turn next changes hands, and a switch turns it off), vs computer (greedy or random), or computer vs computer with a pace control; it waits for **Start** (in the drawer, or in the quick actions, where it takes Undo's place) and can be paused and resumed. |
| **Learn** | 11 lessons: the board, each piece, check and checkmate, bare king, stalemate and draws. Most end with a one-move task that the engine checks. Like Play, the board fills the screen: a floating card shows the lesson, its task, feedback and a hint; **Lessons** opens the lesson list from the left and **Lesson** opens the lesson text from the right (both closed by default, Esc closes them); a ‹ button slides out Reset board, previous and next lesson, and Full screen. `tests/test_lessons.py` checks that every lesson position is legal and every task can be solved. |
| **Rules** | A compact reference: setup, the pieces and how a game ends. The sources behind each rule are in `docs/RULES.md`, not in the app. |

**Camera:** drag to orbit the board. Zoom with the mouse wheel, a two-finger scroll, a trackpad pinch, or a two-finger pinch on a touch screen.

**Accessibility**
- The board can be played by keyboard: arrow keys move a cursor, Enter selects and moves, Escape cancels.
- Highlights differ in shape as well as colour, so they don't rely on colour alone.
- Reduced-motion settings are respected.
- Light and dark themes follow the system setting, with a toggle in the top bar.

**Performance on slower machines**
- **On-demand rendering:** the 3D scene draws only when something changes, such as a move, hover or camera move. It uses no GPU at all while you think.
- **Reduced graphics** (a switch in the Game drawer): turns off shadows, glossy clear-coat finishes and the pulsing check ring, and renders at a pixel ratio of 1. Machines with 4 or fewer CPU cores, or 4 GB of memory or less, start with it on. The choice is remembered.
- **Low-power defaults:** those same machines also get smaller shadow maps and a lighter reflection map if they turn Reduced graphics off.
- **Few draw calls:** each carved piece is merged into one mesh per material (body, brass). Shadows are redrawn only when a piece moves, lifts, morphs or disappears, so turning the board or orbiting costs a single pass per frame.
- **Cheap theme switching:** the page crossfades on the GPU (View Transitions API), and the 3D lighting eases across. The set of lights never changes, so switching theme never forces shaders to recompile.

**Developer aids** (development builds only)
- `?showcase` shows every piece model side by side; `?showcase&kind=3` shows one piece up close (kinds 1–6: Padati, Ashva, Gaja, Ratha, Mantri, Raja).
- `window.__game` exposes the Play store in the browser console.

Design intent and audience are captured in [PRODUCT.md](PRODUCT.md).

---

## Search and network

Milestone 2 joins the rules engine to AlphaZero's two learned-search parts. The network is
still untrained, so for now the search runs on noise plus the exact results of finished games.

```python
from chaturanga import GameState, move_to_str
from chaturanga.mcts import MCTS, MCTSConfig
from chaturanga.net import NetworkEvaluator, new_network

evaluator = NetworkEvaluator(new_network(seed=0))         # untrained 6x64 ResNet, CPU or CUDA
mcts = MCTS(evaluator, MCTSConfig(simulations=50), seed=0)

state = GameState.from_fen("7k/7p/4K2P/8/8/8/8/R7 w")
root = mcts.search(state)                                # 50 simulations from this position
print(move_to_str(mcts.choose_move(root)))               # a1a8: mate, avoiding the stalemate trap
print(MCTS.visit_counts(root), MCTS.root_value(root))    # visits per move, value for White
pi = MCTS.policy_target(root)                            # 4,096-long training target (Milestone 3)
```

- **Network** (`chaturanga/net.py`): a residual CNN over the 15 input planes. The policy head gives
  4,096 logits; the evaluator masks the illegal moves and applies a softmax. The value head gives
  the expected result for the side to move, from −1 to 1. The size is set in `NetConfig`, and
  `new_network(seed=…)` always builds the same weights for the same seed.
- **Search** (`chaturanga/mcts.py`): each simulation descends by the PUCT rule
  `Q + c_puct · P · √N / (1 + n)`, expands the first new position with the evaluator, and backs
  the value up, changing its sign at each level. Finished games are scored exactly by the engine,
  never by the network.
- **Self-play controls:** Dirichlet noise at the root (`add_noise=True`), temperature for choosing
  moves (`choose_move(root, temperature)`), and tree reuse: searching the position after the
  opponent's reply keeps the work already done on it.
- **Agent:** `make_agent("mcts")` plays with 50 simulations per move and no noise. It is the
  "MCTS" computer opponent in the web app. The API server loads PyTorch and the network in the
  background as it starts, so the first MCTS move is as quick as the rest.
- **Tests** (`tests/test_mcts.py`, `tests/test_net.py`):
  - the search finds mate, bare-king wins and the stalemate trap for both colours, with uniform
    priors and with the untrained network;
  - values have the right sign, visits add up, and only legal moves are searched;
  - root noise, temperature, tree reuse, policy targets, network shapes, seeding and
    checkpoints behave as intended.

  Nine bugs planted in the search, one at a time, each made the tests fail.

**Baseline results** (`scripts/mcts_benchmark.py --sims 50 --games 10 --threads 1`, Apple M1):

| Untrained MCTS, 50 simulations per move | Wins | Draws | Losses | Notes |
|---|:-:|:-:|:-:|---|
| vs Random | 5 | 5 | 0 | Every win is a bare king; the draws reach the 300-ply cap |
| vs Greedy | 0 | 0 | 10 | Greedy knows piece values; the untrained network does not yet |

These are the numbers self-play training (Milestone 3) has to improve on.

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
│   ├── encoding.py          # network input planes, action space, legal mask
│   ├── net.py               # policy-value ResNet, evaluator for MCTS, checkpoints
│   ├── mcts.py              # PUCT Monte Carlo Tree Search
│   └── agents.py            # players: random, greedy (2-ply material), MCTS
├── chaturanga_server/       # FastAPI app used by the web client (stateless)
├── web/                     # browser app (React + Three.js)
│   └── src/
│       ├── three/           # 3D board, procedural pieces, highlights, animation
│       ├── views/           # Play, Learn, Rules screens
│       ├── components/      # top bar, side panel, shared controls
│       ├── content/         # lessons.json (validated by tests/test_lessons.py)
│       ├── state/           # game and settings stores (Zustand)
│       └── lib/             # API client, piece data, notation, sounds
├── tests/                   # pytest suite
│   ├── reference.py         # independent, deliberately naive move generator
│   └── test_*.py            # engine, search, network, agents, API server, lessons
├── scripts/
│   ├── random_play.py       # random-vs-random stress test and benchmark
│   └── mcts_benchmark.py    # MCTS speed and matches against the baselines
├── docs/
│   ├── RULES.md             # rules specification with source tallies
│   └── Handouts/            # source PDFs cited in RULES.md (linked from there)
├── pyproject.toml           # package metadata and dependencies
├── PRODUCT.md               # product and design brief for the web app
└── README.md
```

Planned for Milestone 3: self-play, the replay buffer and the training loop, with `checkpoints/`
and `runs/` for (git-ignored) training outputs.

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
            │  Rules engine    │ ──────────────────────▶ │  ResNet (6 × 64)       │
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

Changes reach `main` only through pull requests, and the repository owner reviews and merges every
one. Collaborators work in their own fork (**fork → branch → pull request**). Repository rulesets
enforce this for every collaborator, present and future:

| | Collaborators | Owner |
|---|---|---|
| Fork, open and review pull requests, use issues | Yes | Yes |
| Create, push to or delete branches in this repository | No | Yes |
| Create or move tags | No | Yes |
| Merge pull requests into `main` | No | Yes |
| Force-push to or delete `main` | No | No (can bypass in an emergency) |

1. **Sync your fork** before starting, so you branch from the latest `main`:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```
   (The **Sync fork** button on your fork's GitHub page does the same.)
2. **Work on a branch**, never directly on `main`:
   ```bash
   git checkout -b feat/mcts-node-expansion
   ```
   Branch prefixes: `feat/`, `fix/`, `test/`, `docs/`, `refactor/`, `exp/` (experiments).
3. **Keep commits small and focused.** One logical change per commit.
4. **Before every commit:**
   - Run `python -m pytest -q`; all tests must pass.
   - If you touched the engine, also run `python scripts/random_play.py --games 1000`.
   - If you touched the search or the network, also run `python scripts/mcts_benchmark.py --games 4`.
   - If you touched `web/`, also run `cd web && npx tsc -b && npx oxlint src`.
5. **Push the branch to your fork:**
   ```bash
   git push -u origin feat/mcts-node-expansion
   ```
6. **Open a pull request** from your fork's branch into this repository's `main`, with GitHub's
   **Compare & pull request** button or:
   ```bash
   gh pr create --repo Aditya-00786/csf407-fable-chaturanga-zero --base main
   ```
   Describe what changed and how you tested it. To answer review comments, push more commits to the
   same branch; the pull request updates itself.
7. **After it is merged**, delete the branch and sync your fork again (step 1).

The owner follows the same steps from branches in this repository instead of a fork. Because
merging updates `main`, which the rulesets reserve for the owner, a merge always goes through the
owner's bypass: on GitHub, select **Merge without waiting for requirements to be met (bypass rules)**,
or run `gh pr merge <number> --merge --admin --delete-branch`. Only the repository admin (the
owner) can bypass the rulesets; collaborators don't see that option and `--admin` fails for them.

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

Scopes: `engine`, `encoding`, `mcts`, `net`, `train`, `eval`, `rules`, `scripts`, `server`, `web`.

### Code and rules conventions

- **Rule changes go through [docs/RULES.md](docs/RULES.md) first.** Add the source, update the
  reference tally, then change the code and tests in the same pull request.
- **Every bug fix comes with a test** that fails without the fix.
- **Match the style of the surrounding code:** type hints on public functions, and short
  docstrings that explain *why* rather than *what*.
- **Never commit** virtual environments, training checkpoints or large datasets. `.gitignore`
  covers these. The repository is private because [`docs/Handouts/`](docs/Handouts/) contains copyrighted source PDFs;
  keep it private.

---

## References

- H. J. R. Murray, *A History of Chess*, Oxford, 1913. The main historical source for the rules.
- D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go
  through self-play", *Science* 362 (2018). The AlphaZero algorithm.
- Full list of rule sources: [docs/RULES.md §1](docs/RULES.md#1-sources).
