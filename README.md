# ChaturangZero

**An AlphaZero-style agent that learns Chaturanga, the ancient Indian ancestor of chess, from self-play alone.**

The agent is given nothing but the rules of the game. It combines a residual neural network
(policy + value heads) with Monte Carlo Tree Search, and improves by playing against itself.
Everything is scaled down to train on a single consumer GPU or a free Google Colab instance.

> Course project for **CS F407 Artificial Intelligence**, BITS Pilani K. K. Birla Goa Campus, by **Team Fable**.

---

## Progress

**Current milestone: 4 of 4 — benchmarking**

```
Overall   ██████████████████████████████████░░░░░░  85%   (3 of 4 milestones done)
```

| # | Milestone | Progress | Status |
|:-:|---|---|---|
| 1 | Rules engine, state/move encoding, unit tests | `██████████` 100% | Done |
| 2 | MCTS integrated with an untrained network | `██████████` 100% | Done |
| 3 | Self-play training pipeline (50–100 games, loss tracking) | `██████████` 100% | Done |
| 4 | Benchmarking vs random, heuristic and earlier agents | `████░░░░░░`  40% | In progress |

Milestone 3 went well past its 50 to 100 games: the run `main-v4` trained for 60 generations and
30,000 self-play games, and its final network beats Random in every game and Greedy in 95% of its
strength check (see "Training by self-play").

Milestone 4 is under way. A rating ladder gives every 10th generation an Elo rating against
Random, Greedy and the other generations. A conversion test measures how surely and quickly each
generation finishes a won position. `main-v4`'s ratings rise from 181 for the untrained network
to about 1,055 for generations 56 and 60, and level off after generation 56 (see
[Benchmarking](#benchmarking)).

Also shipped: the **web app**. It has a 3D board, three play modes, interactive lessons and a
rules reference. Its computer opponents include the untrained MCTS agent from Milestone 2 and,
from Milestone 3, the trained agent at four levels, playing any network of a run. Its Training
page starts, follows, pauses, resumes and reports runs on this Mac, Colab, Kaggle or your own
server.

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
- [Training by self-play](#training-by-self-play) (and [in the cloud](#training-in-the-cloud))
- [Benchmarking](#benchmarking)
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
  10,000-game random-play stress test; 271 tests in all.
- **AlphaZero search.** PUCT Monte Carlo Tree Search guided by a residual policy-value network
  (6 blocks × 64 filters, about 1M parameters), with exact values for finished games, root noise,
  temperature and tree reuse for self-play. Planted bugs in the search are all caught by the tests.
  Two rules in the style of Leela Chess Zero's proven wins make the trained agent take every win in
  one and never allow one it can avoid.
- **Self-play training.** Parallel self-play, a replay buffer and AlphaZero's loss, with every
  generation evaluated, checkpointed and logged. Runs resume after any interruption, on a laptop
  or on Colab, and teammates can contribute self-play games from their own machines.
- **Benchmarking.** An Elo rating ladder of a run's generations with Random and Greedy, each step
  between generations with its own interval, and a conversion test of won endgames. Both play in
  the background, on the run's server if it has one.
- **Browser app.** A carved rosewood-and-brass 3D board in light and dark themes, with:
  - a figural piece set after Indian court chess sets: a crowned Raja, a turbaned Mantri, a temple-car
    Ratha, a caparisoned Gaja with its howdah, a Marwari Ashva, and spear-and-shield Padatis;
  - Human vs Human, Human vs Computer and Computer vs Computer, including the trained agent at four
    levels;
  - a Watch training page that follows a run live and replays its self-play games;
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
| Trained network vs Random and Greedy | `python scripts/mcts_benchmark.py --checkpoint checkpoints/best.pt --threads 1` |
| Train (20 generations of 50 games) | `python scripts/train.py` (Ctrl+C pauses) |
| Resume a paused or stopped run | `python scripts/train.py --run main` |
| Pause a run from another terminal | `python scripts/train.py --pause now` (or `after-phase`, `after-generation`; `--cancel-pause` withdraws it) |
| Train further | `python scripts/train.py --generations 40` |
| A quick test run | `python scripts/train.py --run quick --games 8 --generations 2 --eval-games 4` |
| Plot a run's curves | `python scripts/plot_training.py` (needs `pip install -e ".[train]"`) |
| A run's report (what it learned) | `python scripts/report.py --run main` (add `--strength` for the strength check) |
| Rate a run's generations (Elo ladder, conversion test) | `python scripts/report.py --run main --ladder` (`--every 10 --games 40` by default) |
| Self-play games for a teammate's run | `python scripts/selfplay.py --checkpoint gen_0005.pt --games 50 --name <you>` |
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
| **Play** | The board fills the screen. A floating card shows whose turn it is. The toolbar has **Game**, which opens a drawer (Esc closes it) with the mode, move list (click to review), captured pieces, display options (including Reduced graphics) and Rotate. Beside it, a ‹ button slides out quick actions: Undo (Start / Pause against computers), Full screen and New game. Opening the drawer tucks them away again. **Details** in the drawer switches between *Essential*, which shows headlines only and is the default, and *Full*. Under Full, four toggles turn individual extras on or off: move hints, move and quiet-ply counters, why the game ended, and rule sources. While reviewing, a bar at the foot of the board steps through the moves (⏮ ◀ ▶ ⏭, or ← → Home End) and replays the game at Slow, Normal or Fast; stepping back plays the move in reverse, captured pieces returning from their tray. Modes: two players (the board turns to face the player to move after each move; Rotate overrides it only until the turn next changes hands, and a switch turns it off), vs computer (Random, Greedy, MCTS or Trained; the trained agent has four levels, Easy, Medium, Hard and Impossible, set by its search budget), or computer vs computer with a pace control; it waits for **Start** (in the drawer, or in the quick actions, where it takes Undo's place) and can be paused and resumed. |
| **Learn** | 11 lessons: the board, each piece, check and checkmate, bare king, stalemate and draws. Most end with a one-move task that the engine checks. Like Play, the board fills the screen: a floating card shows the lesson, its task, feedback and a hint; **Lessons** opens the lesson list from the left and **Lesson** opens the lesson text from the right (both closed by default, Esc closes them); a ‹ button slides out Reset board, previous and next lesson, and Full screen. `tests/test_lessons.py` checks that every lesson position is legal and every task can be solved. |
| **Rules** | A compact reference: setup, the pieces and how a game ends. The sources behind each rule are in `docs/RULES.md`, not in the app. |
| **Training** | Follows a training run from `runs/`: what it is doing now (self-play, training or evaluation, with live progress: games finished and moves played as they happen, a bar that fills smoothly, the time taken and roughly how long is left), with a menu of runs (our own, keyboard-friendly; each run marked training, starting, paused, finished, stopped or failed), **New run** to start training, **Delete** for a run that is not training (after a confirmation; the network it gave the Trained opponent stays), **Pause** (now, after this phase or after this generation), **Resume** for a paused or stopped run, and **Train further** for a finished one, the policy and value losses (including the loss on each generation's new games before training on them, against the score of always predicting a draw), the evaluation scores, the share of decisive self-play games, and a table of every generation. **Watch** on a row replays that generation's shortest decisive self-play game on the Play board. For a run that is not training, the **Run report** section makes a report on request: what it learned, its self-play charts and a strength check (see [The run report](#the-run-report)). It refreshes every few seconds while the run is live. |

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
- **Search rules** (`MCTSConfig(take_wins=True, avoid_losses=True)`): when a move is chosen, one that
  wins at once comes first, and one after which the opponent can win at once gives way to the most
  visited move that leaves them no such reply, if there is one. The visit counts, and so the policy
  targets, stay as the search left them. Off by default; the trained agent uses them as described
  under [The search's rules](#the-searchs-rules).
- **Agent:** `make_agent("mcts")` plays with 50 simulations per move and no noise. It is the
  "MCTS" computer opponent in the web app. The API server loads PyTorch and the network in the
  background as it starts, so the first MCTS move is as quick as the rest. The trained agent,
  `make_agent("zero", level=…)`, is described under [Training by self-play](#training-by-self-play).
- **Tests** (`tests/test_mcts.py`, `tests/test_net.py`):
  - the search finds mate, bare-king wins and the stalemate trap for both colours, with uniform
    priors and with the untrained network;
  - values have the right sign, visits add up, and only legal moves are searched;
  - root noise, temperature, tree reuse, policy targets, network shapes, seeding and
    checkpoints behave as intended;
  - the search rules take a win the search passed over (mate, bare Raja, the stalemate trap) and
    turn down a move that loses at once (a Ratha left to be taken, a stalemate given), for both
    colours, keep the search's choice when every move loses, and leave the visits alone.

  Nine bugs planted in the search, one at a time, each made the tests fail.

**Baseline results** (`scripts/mcts_benchmark.py --sims 50 --games 10 --threads 1`, Apple M1):

| Untrained MCTS, 50 simulations per move | Wins | Draws | Losses | Notes |
|---|:-:|:-:|:-:|---|
| vs Random | 5 | 5 | 0 | Every win is a bare king; the draws reach the 300-ply cap |
| vs Greedy | 0 | 0 | 10 | Greedy knows piece values; the untrained network does not yet |

These are the numbers self-play training (Milestone 3) has to improve on.

---

## Training by self-play

Milestone 3 closes AlphaZero's loop: the network plays itself with MCTS, learns from those games,
and the improved network plays the next batch.

```bash
python scripts/train.py                  # run "main": 20 generations of 50 games, about 1.5 hours on an M1
python scripts/plot_training.py          # runs/main/training.png
```

Or start a run from the app: **New run** on the Training page asks for a name, the number of
generations and games per generation (and, under More settings, simulations per move and test
games), shows a rough time for this machine, and starts `scripts/train.py` in the background. A
switch decides whether the run's best network becomes the Trained opponent in Play; it starts
off when another run already provides that network, so a test run cannot replace it by
accident. One run trains at a time: starting or resuming a second one asks you to pause the first.

**One generation**
1. **Self-play.** The newest network plays 50 games against itself, one process per CPU core
   (cores − 2 by default), several games per process sharing batched network calls. Each move
   uses 50 simulations with Dirichlet noise at the root, and the move played is sampled in
   proportion to the visit counts for the first 30 plies, then at temperature 0.5 (see below for
   why neither always nor never the most visited one). Every position
   becomes a training example: the board, the search's visit shares (the policy target) and, once
   the game ends, the value target: the result for the side to move, averaged with the search's
   own estimate of it at that position (see below). 30% of the games are
   **endgame starts**: they begin from a random legal position where each side has its Raja and
   1 to 5 of its own pieces (see below).
2. **Teammates' games.** Game files waiting in `runs/<run>/incoming/` are replayed move by move and
   added if every move and visit count is legal under our rules; anything else is set aside in
   `games/rejected/`.
3. **Training.** Batches of 256 positions come from the replay buffer, the most recent 60,000
   positions (about six generations of 50 games), or about three generations when they are larger
   (600 positions per game a generation: 300,000 for 500 games). The loss is AlphaZero's: the
   value's squared error plus the policy's cross-entropy against the visit shares, with weight
   decay in the AdamW optimiser. The number of steps is set so each new position is sampled about
   four times (at most 2,000 steps): more passes over a few games make the value head memorise
   which game a position came from.
4. **Evaluation.** The new network plays 20 games each against the previous generation, Greedy and
   Random, without noise; it samples its first 16 plies in proportion to visits so the games
   differ. All of a generation's test games, against every opponent, are played side by side:
   each worker plays its share together, sending the positions waiting for the same network to it
   in one batch, and every game is played exactly as it would be alone. On a GPU the games go to
   half the workers, in larger batches. Replaying `main-v4`'s generation 40 tests (120 games) on
   the idle lab server, every way gave the same 120 games, move for move:

   | How the test games were played | Minutes |
   |---|:-:|
   | One game, one position at a time (as before) | about 29 |
   | One opponent at a time, 4 games side by side per worker, 10 workers | 17.6 |
   | Every opponent together, 12 per worker, 10 workers | 8.3 |
   | Every opponent together, 24 per worker, 5 workers (the way now) | 5.8 |

   The run report's strength check plays its games the same way, on the run's GPU workers.

Self-play always uses the newest network, as AlphaZero does. The **best** network is the one with
the highest average score against Greedy and Random (40 games, so a few lucky draws do not decide
it; later generations win ties). It is copied to `checkpoints/best.pt`, where the app's Trained
opponent picks it up without a restart. One run feeds Play at a time, and its panel's **In Play**
menu chooses which of its networks: the best or the latest generation, followed as the run trains,
or any other generation, kept until another is chosen (the menu lists them by their score against
Greedy in the run's tests). Choosing one puts it in Play at once and stops the other runs from
replacing it; "Not in Play" leaves Play with the network it has. A new run's "Use its best network
as the Trained opponent in Play" chooses its best.

**What a run keeps** (in `runs/<run>/`, git-ignored)

| Path | Contents |
|---|---|
| `config.json` | The run's settings |
| `metrics.jsonl` | One line per event: training losses every 10 steps, self-play results, the loss on each generation's new games before training on them, evaluations |
| `status.json` | What the run is doing right now; the app's Training page reads it |
| `games/gen_NNNN.jsonl.gz` | Every self-play game: the moves, the visit counts and the search's value at each move, and the result (about 40 KB per game) |
| `checkpoints/gen_NNNN.pt` | The network after each generation, plus `best.pt` and `train_state.pt` (the optimiser) |
| `incoming/` | Drop teammates' game files here |
| `control.json` | A pause request the run has not reached yet |
| `launch.json` | For a run started from the app: whether it publishes its best network, and where it trains |
| `cloud.json`, `kaggle.log` | A Kaggle run: its script on Kaggle, and the session's log once it is home |
| `games/gen_NNNN.partial.*.jsonl.gz` | Finished games of a generation still being played (merged into `gen_NNNN.jsonl.gz` when it completes) |

**Pausing and resuming.** A run can be paused three ways, from the Training page, from another
terminal (`python scripts/train.py --pause …`) or with Ctrl+C:

| Pause | What happens |
|---|---|
| **Now** (Ctrl+C too) | Stops within seconds. Every self-play game that has finished is kept (each is saved the moment it ends); games in progress are dropped. A half-done training or test is dropped too, and redone on resuming. |
| **After this phase** | Finishes the current self-play, training or testing, then stops. |
| **After this generation** | Finishes the whole generation, testing included, then stops. |

To resume, press **Resume** on the Training page, or in a terminal run:

```bash
python scripts/train.py --run main        # the run's name; add --runs-dir if it is not in runs/
```

The run picks up exactly where it stopped: it plays only the self-play games still missing (the
same games it would have played without the pause), trains a generation whose self-play is done,
or tests one that was trained. A run that ended without pausing (the terminal closed, the machine
restarted) resumes the same way. To train beyond the planned generations, add `--generations`
with the new total (or use **Train further** on a finished run); nothing caps it.

**Changing settings as it goes.** **Settings**, beside Resume and Train further, changes what the
coming generations do: games per generation (the buffer grows to match), simulations per move,
test games per opponent, the learning rate (lowered in steps late in a run, as AlphaZero did), the
temperature after the opening, the report at the end, and for a run on the server its self-play
processes and how many use the GPU. The network's size stays. Each start or resume records the
settings it trains with in `metrics.jsonl` (`"event": "start"`), so a change is on record from the
generation it applies to.

**Why self-play samples its moves.** AlphaZero sampled only its opening moves and then
played the most visited move. Our first attempt did the same,
and by generation 3 both sides had learned to shuffle pieces until a position repeated: 34 of 50
self-play games ended in threefold repetition, the value head still could not tell a missing pair
of Rathas from the starting position, and almost nothing decisive was left to learn from.
Measured with the generation 3 network over 28 games each:

| Moves after ply 30 | Decisive games | Repetition draws |
|---|:-:|:-:|
| Most visited (temperature 0) | 3 | 22 |
| Temperature 0.5 | 6 | 10 |
| In proportion to visits (temperature 1) | 10 | 2 |

So self-play went on sampling every move in proportion to visits. The policy target is the same
visit distribution either way; only the games become more varied, and more of them are won.

That had a cost, found after the third run (`main-v3`): an opponent that samples its moves often
leaves a free piece where it stands. In generation 20's self-play, material on offer for nothing
was taken only 30% of the time, so the value head learned that a loose piece is usually safe, and
Greedy, which takes every one, beat it in every game. Measured with that network over 36 games each:

| Moves after ply 30 | Decisive games | Repetition draws | Free material taken |
|---|:-:|:-:|:-:|
| Most visited (temperature 0) | 72% | 25% | 45% |
| Temperature 0.25 | 75% | 22% | 44% |
| Temperature 0.5 | 89% | 8% | 40% |
| In proportion to visits (temperature 1) | 81% | 17% | 30% |

So self-play now plays at temperature 0.5 after the opening (`--late-temperature`, default 0.5):
blunders are punished a third more often, without the repetition draws of a lower temperature.

**Measuring honestly.** A network can lower its training loss by memorising games. So before each
generation trains on its new self-play games, the loss on those unseen games (from the opening) is
logged too, and the value loss there is compared with the score of predicting a draw everywhere.
The Training page and `plot_training.py` show both.

**The first full run, and what it taught us.** Run `main` (20 generations, 1,000 self-play games
from the opening, about 2 hours on an M1 with two pauses) finished with the pipeline working and
the agent no stronger:

| | Untrained | Generation 20 |
|---|:-:|:-:|
| Policy loss on unseen games | 8.32 | 2.50 (floor: 1.93) |
| Value loss on unseen games, vs predicting a draw | 0.49 vs 0.47 | 0.40 vs 0.33 |
| vs Random, 20 games at 50 simulations | 11 W / 9 D / 0 L (78%) | 1 W / 19 D / 0 L (52%) |
| vs Greedy, 20 games | 0 / 0 / 20 (0%) | 0 / 1 / 19 (2%) |

The policy head learned to predict the search's moves, and generalised. The value head did not
learn at all: on unseen games it was worse than predicting a draw in every generation, and at
generation 20 it rated a side missing both Rathas the same as the starting position. With a value
that says "draw" everywhere, the sharper policy only narrows a search that has nothing to aim for,
which is why generation 20 converts fewer games against Random than the untrained network did.
The cause is the data: from the opening, games run about 250 plies and only about 40% are
decided, so about 400 noisy results had to teach the value of every position.

Two changes followed. **Endgame starts** (30% of self-play games, from random positions with a
Raja and 1 to 5 pieces a side) give short, mostly decided games whose uneven material shows what
pieces are worth, still from the rules alone. **Better testing**: successive generations drew
18 to 20 of 20 games by repetition, which measured nothing, so test games now sample 16 opening
plies instead of 8, Random joins the opponents, and the best network is chosen on 40 games
against Greedy and Random rather than 20 against Greedy. Two further options are kept in reserve:
training the value on the average of the result and the search's own estimate, and scoring
capped games by material (which would add hand-written knowledge).

**The second run, on the lab server.** Run `main-v2` (the same 20 generations of 50 games, now
with endgame starts and the better testing) took 34 minutes on the server with 9 self-play
processes, against about 2 hours for `main` on the Mac. It failed the same way, and showed why:

| Value loss of generation 19 (vs predicting a draw) | Games it trained on | New games |
|---|:-:|:-:|
| From the opening | 0.11 (0.31) | 0.43 (0.36) |
| Endgame starts | 0.21 (0.68) | 0.65 (0.63) |

On games in its buffer the value head was three times better than predicting a draw; on new games
it was worse. It had learned the 300 or so games in its buffer by heart: every position of a game
shares one result, and a network of this size can tell which game a position came from. Its
policy generalised (2.56 on trained games, 2.59 on new ones). The strength check showed the cost:
against Random the untrained network scored 9 W / 9 D / 2 L, the best and final networks 1 W /
19 D and 3 W / 17 D, and none of them won a game against Greedy. Half of the opening games reached
the 300-ply cap, as draws, often with a Ratha or more of material between the sides.

Two changes follow, both still from the rules alone. **Far more games**: on the server a
generation can be 500 games, with a buffer of about three generations (300,000 positions), so
the value head sees far more games than it can learn by heart. **A blended value target**: each
position's target is the average of the game's result and the search's own estimate there
(`value_mix` 0.5, a common AlphaZero variant). Positions of one game then get
different targets, which are less noisy and harder to memorise. Losses on new games are still
measured against the result alone, so the honest test is unchanged. Scoring capped games by
material is set aside, to keep the agent learning from self-play alone; stronger weight decay
(1e-4 to 1e-3) is kept in reserve.

**The third run.** Run `main-v3` (20 generations of 500 games, a 300,000-position buffer and the
blended value target) took 1 hour 25 minutes on the server with 20 self-play processes. The value
head learned: on new games it beat predicting a draw in 18 of 20 generations (0.59 against 0.87
at the end), and each generation scored 50 to 59% against the one before.

| | `main-v2`, final | `main-v3`, final |
|---|:-:|:-:|
| Value loss on new games, vs predicting a draw | 0.42 vs 0.34 | 0.59 vs 0.87 |
| vs Random, strength check (20 games) | 3 W / 17 D / 0 L | 17 W / 3 D / 0 L |
| vs Greedy | 0 W / 1 D / 19 L | 0 W / 0 D / 20 L |
| Self-play games from the opening that were decided | 39% | 88% |

Greedy still won every game, at 50, 200 and 800 simulations alike: the network loses material on
about a dozen moves a game whatever the search size, so the limit is its judgement of loose
pieces, not the search. Why, and the change that followed (a lower temperature after the opening),
is under "Why self-play samples its moves" above.

**Generations 21 to 40, and why it cannot beat Greedy.** Run `main-v3` went on to generation 40
(temperature 0.5 after the opening). Against Random it kept improving: 92% in the test games of
generations 31 to 40, from 86.5% in 11 to 20 (98% at its best, generation 29, and 18 W / 2 D in
the strength check). Against Greedy it did not: 2 wins in 400 test games in generations 31 to 40
(its "9%" peak was 7 draws in 40), and the value margin on new games narrowed (0.71 against 0.82
in generations 31 to 40, from 0.61 against 0.85 in 11 to 20).

The network does know what pieces are worth. Taking one piece off the board in 773 positions from
its games moves its value, for the side that gains, by 0.37 for a Ratha, 0.15 for an Ashva, 0.10 for
a Gaja, 0.07 for a Mantri and 0.05 for a Padati: in proportion close to Greedy's hand-set 5, 3, 1.5,
1.5 and 1, learned from self-play alone. What it lacks is tactics. Against Greedy it lost material
on about a dozen moves a game; in its own games, free material was taken a third of the time (a
free Ratha 69% of the time, a free Padati 34%). A series of experiments on the lab server, each on
generation 39 or 40, looked for the cause (the scripts are kept with their logs in the server's
`analysis/` folder):

| Experiment | Result |
|---|---|
| More search: 50, 200 and 800 simulations per move against Greedy | about 11 to 13 material-losing moves a game at every size; 0 to 12% |
| Temperature after the opening, 1 / 0.5 / 0.25 / 0, in self-play | free material taken 30 / 40 / 44 / 45% (adopted 0.5; no gain showed in the run) |
| A lower learning rate (1e-4 against 1e-3) for more training on the same games | no lower loss on new games; it only avoids drifting upward |
| A bigger network (10 x 128 against 6 x 64), trained from scratch on the same games | value 0.682 against 0.693 on new games: slightly better, not enough to explain the gap |
| The result alone as the value target (value_mix 0 against 0.5) | memorises again: 0.94 on new games, worse than predicting a draw; the blend stays |
| How the search treats untried moves (a fixed 0, or the position's own value less 0, 0.25 or 0.5) | free material taken 6 to 9 of 40 cases; no change against Greedy |
| An even share of attention for every move at the root | free material taken up to 17 of 40; still 0 to 1 wins in 40 against Greedy |
| Flattened priors at every node (temperature 2 to 3) with 800 simulations | free material taken up to 25 of 40; still about 13 material-losing moves a game, 0 to 2 wins in 20 |

In the positions where a free Ashva or Ratha was left, the network rated the position after the
capture higher in 28 of 40, but its policy gave the capture a prior of 0.003 to 0.03, so the search
seldom tried it; and it gives the opponent's captures the same neglect, so a move that leaves a piece
to be taken looks safe. A policy that does not look at captures plays games where loose pieces go
unpunished, which teaches it nothing about them: a loop that settings did not break. Greedy looks at
every reply to every move, exactly, which no affordable number of simulations matches when the
priors point elsewhere.

So the agent learned strategy from the rules alone (material values, converting won positions:
95 to 98% against Random) but not tactics. AlphaZero learned both with about 800 simulations a move,
a 20 x 256 network and millions of games; this run had 50 simulations, a 6 x 64 network and 20,000
games. Beating Greedy by self-play alone would take a run of that larger kind, many times this one's
compute; it is a stretch goal, not a setting to change.

**The fourth run: a larger network and a deeper search.** Run `main-v4` took that route: a
10 x 128 network (3.5 million parameters), 200 simulations a move, 500 games a generation, on the
lab server with self-play's network on the GPU (see "CPU and GPU together" below). Its 60
generations, 30,000 self-play games, took about 42 hours of server time over three sessions. It
learned tactics, which `main-v3` never did:

| Generations | vs Greedy | vs Random | vs the generation before | Self-play draws (at the 300-ply cap) |
|---|:-:|:-:|:-:|:-:|
| 1 to 10 | 2% | 92% | 68% | 18% (12%) |
| 11 to 20 | 20% | 96% | 58% | 12% (5%) |
| 21 to 30 | 53% | 98% | 55% | 20% (13%) |
| 31 to 40 | 69% | 98% | 53% | 26% (17%) |
| 41 to 50 | 82% | 98% | 53% | 25% (16%) |
| 51 to 60 | 89% | 99% | 51% | 29% (19%) |

| Strength check, 20 games each at 200 simulations | vs Random | vs Greedy |
|---|:-:|:-:|
| Untrained | 6 W / 14 D / 0 L | 0 W / 3 D / 17 L |
| Generation 20 | 18 W / 2 D / 0 L | 7 W / 7 D / 6 L |
| Generation 40 | 20 W / 0 D / 0 L | 16 W / 3 D / 1 L |
| Generation 60 | 20 W / 0 D / 0 L | 18 W / 2 D / 0 L |

The deeper search gave the policy targets that take loose pieces. Where its search's favourite move
could win material outright, generation 20 took a free Ashva 76% of the time and a free Ratha 82%
(`main-v3`'s generation 40: 49% and 69%), and fewer pieces were left loose to take. Its value
beat predicting a draw in 59 of 60 generations, and rates a side missing both Rathas at 0.4 to
0.8 in the other's favour and one missing its Ashvas and Rathas at about 0.99. By generation 60 it had levelled off: each
generation scored only 51% against the one before, its policy loss on new games had hardly moved
since generation 40 (2.32 to 2.28), and more self-play games ran to the 300-ply cap as both sides
defended better (24% in generation 60). The next step for a longer run is a lower learning rate,
1e-3 to 3e-4, as AlphaZero did late in training (**Settings**, below Resume).

### What the trained agent plays

These measurements were made on the lab server with `main-v4`'s networks; the scripts are kept with
their logs in the server's `analysis/` folder.

**Openings.** No opening was given to it, and it formed and changed its own preferences, as
AlphaZero's chess openings shifted during its training:

| Generation | Its favourite first moves (its policy at the start) | Self-play's commonest first move |
|---|---|:-:|
| 1 | b1a3 8%, b1c3 8% (no preference) | 9% of games |
| 20 | g2g3 22%, Gaja c1-e3 12% | 26% |
| 40 | f2f3 37%, e2e3 11% | 32% |
| 50 | h2h3 40%, f2f3 13% | 30% |
| 60 | g2g3 16%, h2h3 13%, a2a3 13%, Ashva g1-f3 12%, Gaja f1-h3 9% | 19% |

Its preferences sharpened until about generation 47 (48% of games opened with its favourite), then
spread again: generation 60 rates several setups about equally, with a value near 0 for the
starting position. They are quiet ones (edge Padati moves such as 1. h3 a6 2. a3 h6, and Ashva and
Gaja development) and, around generation 40, they gave the Raja room (f3 and e3 free f2 and e2
for White's Raja; ...c6 frees c7 for Black's). Against itself at Hard, generation 60 plays 1. Nf3
b6 2. g3 g6 3. e3 Ge6 and builds Padati walls; it also moves an Ashva and its Raja out and back,
a sign that it has no long plan in quiet positions. Modern chess openings do not apply here (no
queen, no long-range bishops, no castling, no double step); the nearest historical idea is the
shatranj *tabiya*, a set-up both sides built before contact.

**Chaturanga's fool's mate.** Scholar's and fool's mate need a queen and bishops, but a Raja
walled in by its own pieces can be smothered by an Ashva in four moves: 1. Nc3, 2. Na4, 3. Nc5
and 4. Nxb7 mate (for Black, ...Nf6-h5-f4xg2). A scripted attacker that plays straight for it
mated none of the networks tried, from the untrained one to generation 60, as either colour; the
trained ones punish it. Generation 60 as White takes the raiding Ashva the moment it lands on f4
(g3xf4); as Black it plays ...d6 and takes it on c5 (...dxc5).

**Raja safety.** In positions from its games where some moves lose at once (mate, or its last
piece taken) and others do not, the search at 200 simulations picks a safe move 87% of the time
with generation 60, up from 81% with generation 36. In Play and test games the search's rules
(below) now pick a safe move whenever there is one.

**Finishing a won game.** A win counts the same however long it takes, so the agent has no reason
to prefer the quick finish, and it often plays on, taking more pieces first; AlphaZero is known
for the same. In generation 60's self-play there were 781 positions where a move would win at
once; the move played took the win 46% of the time, and when it did not, that side still won 93%
of the games, a median of 12 plies later. Searching again as Play's levels do, it took the
immediate win 48% of the time at Hard (200 simulations) and 50% at Impossible (800), down from 55%
and 65% with generation 36. The search is not blind to the win: in generation 36 the winning move
often had a prior near 0, so it was seldom tried, but in generation 60 it gets nearly as many
visits as another move that wins later (48 against 50, say). That move's value, 0.997, is too
close to the win's 1.0 to matter, so the visits follow the priors. The search's rules now settle it.

### The search's rules

Leela Chess Zero lays proven results over its search: a move that ends the game is known exactly,
whatever the visits say. ChaturangZero now applies two rules when it chooses a move
(`chaturanga/mcts.py`):

1. **Take a win in one.** If a move wins at once (checkmate, a bare Raja, or a stalemate, which the
   stalemated side wins), play it (the most visited, if there are several).
2. **Don't allow one.** If the chosen move lets the opponent win at once, or loses by itself (it
   stalemates the opponent), play instead the most visited move that does neither, if one exists.

They need no retraining and cost about half a millisecond a move each (a 200-simulation move takes
about 55 ms of a process's time). The visit counts, and so self-play's policy targets, stay as the
search left them.

| Where | Take a win in one | Don't allow one |
|---|:-:|:-:|
| Play, Hard and Impossible | Yes | Yes |
| Play, Easy and Medium (they stray from their best move on purpose) | No | No |
| Play, MCTS (Milestone 2's untrained agent, a benchmark baseline) | No | No |
| Test games and the strength check (in `main-v4`, from generation 61) | Yes | Yes |
| Self-play (in `main-v4`, from generation 61; `take_wins` in a run's config) | Yes | No |

Measured on the lab server with generation 60. It played 100 test games against each opponent and
160 self-play games (106 from the opening, 54 from endgame positions), each set three ways with
the same seeds:

| Test games | No rules | Take wins | Both rules |
|---|:-:|:-:|:-:|
| vs Random (W / D / L) | 99 / 1 / 0 | 99 / 1 / 0 | 99 / 1 / 0 |
| Plies to win against Random (median, mean) | 103, 116 | 87, 94 | 87, 94 |
| vs Greedy (W / D / L) | 82 / 11 / 7 | 82 / 11 / 7 | 83 / 12 / 5 |
| Plies to win against Greedy (median, mean) | 190.5, 189 | 182.5, 179 | 182, 178 |
| vs generation 59, under the same rules (W / D / L) | 25 / 61 / 14 | 27 / 55 / 18 | 23 / 66 / 11 |
| Moves the rules changed (either player's) | | 116 wins | 105 wins, 69 losses avoided |
| Time for the 300 games | 6.8 min | 6.7 min | 6.8 min |

| Self-play | No rules | Take wins | Both rules |
|---|:-:|:-:|:-:|
| Games from the opening decided | 69 of 106 | 75 of 106 | 63 of 106 |
| Their median length | 219 plies | 199 | 236 |
| Their mates | 15 | 28 | 8 |
| Endgame games, median length | 38 plies | 28 | 39 |
| Moves a second | 123 | 117 | 125 |
| Moves the rules changed | | 54 wins | 33 wins, 116 losses avoided |

- **Won games end sooner.** Against Random, wins came 16 plies sooner at the median (23 on
  average) and four times as often by mate (40 of 99 against 10); against Greedy, 8 to 10 plies
  sooner. The scores hardly moved: those games were won already. Not allowing a win in one cut
  Greedy's wins from 7 to 5, which 100 games cannot tell from chance.
- **No cost in time.** Each setting took the same time; moves a second in self-play varied with
  the mix of positions, not with the rules.
- **Self-play takes wins only.** Taking them gives more decided games with more mates, so more
  positions carry a result, and a little less time per generation. Not allowing them does the
  opposite. Both sides parry every mate in one, so fewer games are decided (63 against 75), mates
  fall from 28 to 8, and more games end at the ply cap or by repetition (43 against 31). The
  network would also stop seeing its search's blunders punished, and that is how it learned to
  avoid them (from 81% to 87% safe).
- **Test scores from generation 61 include the rules**, so a run's charts may step at that point.
  Against Greedy, 100 games put generation 60 at 87.5% (82 / 11 / 7), below the 95% of its
  20-game strength check (18 / 2 / 0). With 20 games a match, scores move by several points from
  chance alone.

### The run report

A report sums up what a run learned. It is made only when asked: **Make report** in the Training
page's Run report section (for a run that is not training), or `python scripts/report.py --run
<name>` in a terminal. A run started with **Make the run report when it finishes** (off by default
in New run; `auto_report` in its config) makes its report, strength check included, as soon as its
last generation is done; a run already on the rating ladder rates its new generations there too. The
report is saved in `runs/<run>/report/report.json`; if the run trains further, **Update report**
brings it up to date.

- **Findings**, from the run's metrics: whether the policy learned to predict the search (its loss
  on unseen games, against the lowest it can reach), whether the value beat predicting a draw on
  unseen games and in how many generations, how its test scores against Greedy and Random moved,
  and how self-play changed.
- **Does the value head see material?** The value the untrained, best and final networks give the
  starting position and the same position with pieces missing on one side or the other. A value
  that understands material rates a side ahead well above the same side behind.
- **Strength check**, on request (a few minutes, in the background, one at a time and never while a
  run trains): the untrained, best and final networks each play Random and Greedy from the
  opening. This is what showed that the first run's final network was weaker than its untrained
  one. Results are kept in `runs/<run>/report/strength.json`.
- **Rating ladder and conversion test**, on request (see [Benchmarking](#benchmarking)):
  Elo ratings for the run's generations, and how surely and quickly each finishes a won game.
  Results are kept in `runs/<run>/report/benchmark.json`.
- A server run's strength check and ladder play on its server, as its training did, and the page
  follows them; if the connection drops they play on, and connecting again picks them up. Training
  waits while one plays on the same machine.
- **Self-play over the run**: how games from the opening ended, generation by generation (decided
  endings in colour, draws in grey), and their average length. These charts live in the report;
  the page's other charts follow the run live.
- **Download charts**: the six-panel figure from `plot_training.py` (losses, test scores, self-play
  results, endings and length).

**Team self-play.** Self-play is the slow part, and it parallelises across people as well as
cores. Whoever runs the training shares a checkpoint (`runs/<run>/checkpoints/gen_NNNN.pt`);
each teammate runs:

```bash
python scripts/selfplay.py --checkpoint gen_0005.pt --games 50 --name aadit
```

and sends back the `.jsonl.gz` file it writes, which goes into `runs/<run>/incoming/`. Game files
are merged, not trained networks: averaging separately trained weights does not give a better
network, but more games from the current network always help.

### Training in the cloud

**New run** asks **Where**: this Mac, Colab, Kaggle or your own server. The cloud keeps the Mac
free; it is not faster (self-play runs on the CPU, and Colab's free machines have 2 cores and
Kaggle's about 4, against the 6 used here). A server with many cores is: self-play plays one game
per core.

**Colab.** Google offers no way for another program to start a Colab notebook, so the app does
the rest around the one click that needs you. It writes the run's folder and a request into
`ChaturangZero` in your Google Drive and opens
[notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Colab, straight from GitHub; you
press **Run all** (the first time, Colab asks to connect GitHub and Google Drive). The notebook
takes the request and trains into Drive, so a disconnect loses nothing: run the cells again and
it resumes. This needs **Google Drive for desktop** on the Mac, signed in with the same Google
account: through it the app sees the run like any other (progress, Pause, which reaches Colab
through Drive, Resume, which asks Colab again, the report), and a run that publishes gives the
Trained opponent its network. The notebook clones the private repository with a read-only GitHub
token kept in Colab's Secrets; it can also be run on its own, with the settings in its third cell.

**Kaggle.** Kaggle's command-line tool can do everything, so a Kaggle run needs nothing after
**Start on Kaggle**. `scripts/kaggle_run.py`, started in the background, uploads the code and the
run so far as a private dataset, starts a private script on a Kaggle GPU, checks it every minute
and downloads the finished run into `runs/`; restarting it (Resume, or after the Mac restarts)
follows the same session instead of starting another. Kaggle sessions stop at 12 hours, so the
script pauses the run after the generation in progress at 11; Resume continues it in a new
session. Kaggle runs report their progress when they come home, not live, and stop from their page
on Kaggle. To connect, create a token on kaggle.com (Settings, API, Generate New Token) and give
it with your username in New run; it is checked with Kaggle and kept in `~/.kaggle/access_token`,
readable by you only. Install the tool with `pip install -e ".[cloud]"`.

**Your own server.** Any Linux machine you can reach with `ssh` (Python 3.10 or newer, and
`rsync`). In New run choose **Server**, give its address (or pick a host from `~/.ssh/config`), a
folder for everything (`~/sopProject/chaturangzero` by default) and its password, and press
**Connect**. The password opens one shared SSH connection (OpenSSH's ControlMaster, its socket in
`~/.chaturanga/ssh/`, readable by you only) and is not kept: it reaches ssh through a pipe, never a
file, and later commands and copies use the open connection. A server that accepts your SSH key
needs no password. `scripts/server_run.py`, started in the background, gets the folder's Python
environment ready (the first time: a few minutes, as PyTorch is a few GB; the build is chosen for
the server's NVIDIA driver, or the CPU build without a GPU), copies the code, starts the training
there at low priority, detached from the connection, and copies the run back every few seconds, so
the Training page shows it live and Pause and Resume work as on the Mac. On the server everything
stays in the folder:

```
<folder>/env/        the Python environment (pip's caches in cache/, temporary files in tmp/)
<folder>/code/       the training code, copied before each start
<folder>/runs/<run>  the runs, as in runs/ here
```

The training carries on if the Mac sleeps or the connection drops; the page then asks for the
password again, and connecting picks the run up where it is, also after the Mac has restarted. A
right password closes the prompt at once with a green "Connected", and while the app copies back
what the run made in the meantime (a while, after a long gap) the run shows as "catching up".
**Self-play processes** sets how many of the server's cores it uses (by default all but four, left
for the training and anyone else on the machine). Deleting a run deletes the server's copy too.

**CPU and GPU together.** Each self-play process runs its search on its own CPU core; its network
runs there too, or, for the processes set **on the GPU** (`--gpu-workers`), on the server's GPU,
which then plays 32 games side by side in each process so its batches are large. Measured on the
lab server (Core Ultra 9 285K, RTX 5000 Ada), in moves of self-play a second:

| Network, simulations | Where the network runs | Moves a second |
|---|---|---:|
| 6 x 64, 50 | CPU, 20 processes | 502 |
| 10 x 128, 200 | CPU, 20 processes | 25 |
| 10 x 128, 200 | GPU, 20 processes of 8 games | 66 |
| 10 x 128, 200 | GPU, 20 processes of 32 games | 168 |
| 10 x 128, 200 | GPU, 10 processes of 32 games | 182 |

The GPU is the limit: fewer processes with larger batches beat more with small ones, and adding
processes that run the network on their own core does not pay, since each adds about 1 move a
second and holds up the end of every generation with its share of the games. For a large network,
put about 10 processes on the GPU and none on the CPU; the free cores serve the training process
and others on the machine. (The small network was measured on the CPU only, where it already plays
about 500 moves a second.)

**Playing the trained agent.** In the app choose **Trained** as the opponent (or for either side
in Computers mode) and a level:

| Level | Simulations per move | Move choice |
|---|:-:|---|
| Easy | 10 | Sampled in proportion to visits: it often strays from its favourite |
| Medium | 50 | Visits sharpened (temperature 0.5): usually its favourite |
| Hard | 200 | Always the most visited move; takes any win in one and never allows one it can avoid |
| Impossible | 800 | As Hard; about a second per move |

From Python: `make_agent("zero", level="hard")`. The server reads the network from
`checkpoints/best.pt`, or from the file named by `CHATURANGA_CHECKPOINT`; the Training page reads
runs from `runs/`, or from `CHATURANGA_RUNS`.

**Tests** (`tests/test_training.py`, `tests/test_training_api.py`): self-play records are legal,
finished games that replay exactly; the same seed gives the same games, batched or one at a time;
policy targets equal the search's visit distribution for both colours; tampered records (an
illegal move, an illegal visit, a wrong result, an unfinished game) are rejected; the buffer keeps
the newest games within its capacity; a small network overfits a fixed buffer (the loss and
gradients work); a run trains, publishes, takes a teammate's games, rejects a tampered one, and
resumes with its optimiser state; the trained agent plays at every level and picks up a newly
published network; and the Training API lists runs, losses, generations and games. Cloud and
server runs (`tests/test_cloud.py`, `tests/test_remote.py`) use stand-ins for Drive, the Kaggle
tool and ssh: a server run signs in with a password that ends up in no file, trains, comes back
live, pauses, resumes, and is followed again after the connection drops or the Mac restarts.

---

## Benchmarking

Milestone 4 measures the agent against Random, Greedy (the heuristic baseline) and its own earlier
generations. A run's test games stop showing progress once it is strong: against the generation
before, it scores about 50% by design, and Greedy, which it nearly always beats, has little left to
tell. Two yardsticks take over (`chaturanga/benchmark.py`). Both are played from the Training page's
Run report (**Rating ladder**, then **Rate generations**), or with `python scripts/report.py --run
<name> --ladder`. A server run's ladder plays on its server.

### The rating ladder

- **Who plays.** Every 10th generation (a setting), with the first, the latest and the best, and
  Random and Greedy. Each pair plays 40 games (a setting) from the opening, colours alternating.
  The networks search as test games do: the run's 200 simulations, the first 16 plies sampled in
  proportion to visits so games differ, and the search's rules.
- **Ratings.** Elo, fitted to every result at once (Bradley-Terry, a draw half a win), with Random
  at 0. One virtual draw per pair keeps a clean sweep's gap large but finite. Each rating's 95%
  interval is against Random. Each step from one rated generation to the next has its own interval,
  much narrower, because neighbouring ratings move together.
- **Adding to it.** Rating later generations plays only their games against the rest. A run already
  on the ladder adds its new generations in its end-of-run report. Results are kept in
  `runs/<run>/report/benchmark.json`.

### The conversion test

Forty positions in which one side is far ahead in material, eight of each:

- a Ratha against a Padati;
- a Ratha against a Mantri;
- a Ratha and a Padati against a Gaja;
- an Ashva and two Padatis against a Padati;
- a Ratha and an Ashva against an Ashva.

They are placed at random once, with a fixed seed: legal, the side ahead to move (White in half of
them), and no win in one. Each network plays the side ahead against Greedy, both playing their
best moves from the first ply. The test counts the wins and the plies they took. The last piece
still has to be caught (a bare Raja loses) without stalemating it (the stalemated side wins).
Greedy playing the side ahead is the reference.

### main-v4 on the ladder

2,160 games (1,800 on the ladder, 360 in the conversion test) took 44 minutes on the lab server's
GPU:

| Player | Elo (95% interval) | Score | vs Greedy (W / D / L) | Conversion: won | Median plies to win |
|---|:-:|:-:|:-:|:-:|:-:|
| Generation 56 (the run's best) | 1,058 (936 to 1,180) | 82% | 36 / 2 / 2 | 40 of 40 | 22 |
| Generation 60 | 1,054 (932 to 1,175) | 82% | 34 / 6 / 0 | 39 of 40 | 17 |
| Generation 50 | 1,001 (880 to 1,121) | 77% | 30 / 8 / 2 | 40 of 40 | 18 |
| Generation 40 | 923 (804 to 1,042) | 69% | 31 / 7 / 2 | 40 of 40 | 24 |
| Generation 30 | 846 (728 to 964) | 61% | 23 / 6 / 11 | 39 of 40 | 15 |
| Greedy | 699 (583 to 814) | 47% | | 28 of 40 | 53 |
| Generation 20 | 656 (541 to 771) | 43% | 12 / 11 / 17 | 40 of 40 | 21 |
| Generation 10 | 370 (265 to 475) | 23% | 0 / 4 / 36 | 40 of 40 | 28 |
| Generation 0 (untrained) | 181 (84 to 279) | 12% | 0 / 4 / 36 | 24 of 40 | 33 |
| Random | 0 | 4% | 0 / 0 / 40 | | |

| Step | Elo gained (95% interval) |
|---|:-:|
| Generation 0 to 10 | +188 (+102 to +276) |
| 10 to 20 | +286 (+205 to +368) |
| 20 to 30 | +190 (+124 to +255) |
| 30 to 40 | +77 (+16 to +138) |
| 40 to 50 | +78 (+16 to +140) |
| 50 to 56 | +58 (−6 to +121) |
| 56 to 60 | −5 (−70 to +60) |

- **The run levelled off.** It gained about +190 to +290 Elo every 10 generations up to
  generation 30, about +78 from 30 to 50, and nothing measurable after 56: generation 60 is level
  with 56. The tests against the generation before only hinted at this (51% in generations 51 to
  60). It is the cue for the next step, a lower learning rate (AlphaZero lowered its own as
  progress slowed).
- **Generations 56 and 60 are equals.** Against each other generation 60 scored 59% (9 wins, 29
  draws, 2 losses), but 56 did a little better against the earlier networks and Greedy.
- **It passed Greedy between generations 20 and 30** (Greedy rates 699; generation 20, 656;
  generation 30, 846). Against Greedy, generation 60 scored 92.5% in these 40 games (34 / 6 / 0),
  close to the 95% of its 20-game strength check.
- **Strong networks draw each other.** Of the 40 games between generations 56 and 60, 29 were
  drawn, 14 of them by repetition and 15 at the 300-ply cap. Neither side found a way through, which
  is where longer plans would show.
- **The conversion test is solved from generation 10.** Every trained network won 39 or 40 of the
  40 positions, against 24 for the untrained one and 28 for Greedy, which often ran into the
  70-move rule. Their median plies to win (15 to 28) show no trend after generation 20. To follow
  planning in later generations, the test needs harder positions, with a smaller material edge.

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
│   ├── selfplay.py          # self-play games (batched across games), game records, training examples
│   ├── replay_buffer.py     # the most recent positions, sampled for training
│   ├── training.py          # the generation loop: self-play, train, evaluate, checkpoint, pause, resume
│   ├── endgames.py          # random reduced-material starting positions for self-play
│   ├── analysis.py          # a run's report: findings, material probe, strength check
│   ├── benchmark.py         # Milestone 4: the rating ladder (Elo) and the conversion test
│   ├── cloud.py             # training on Colab (through Google Drive) and Kaggle (through its tool)
│   ├── remote.py            # training on your own server over SSH: connection, folder, copies
│   └── agents.py            # players: random, greedy (2-ply material), MCTS, trained (four levels)
├── chaturanga_server/       # FastAPI app used by the web client (stateless)
│   ├── app.py               # game and computer-move endpoints
│   └── training_api.py      # read-only view of training runs, for the Training page
├── web/                     # browser app (React + Three.js)
│   └── src/
│       ├── three/           # 3D board, procedural pieces, highlights, animation
│       ├── views/           # Play, Learn, Rules, Training screens
│       ├── components/      # top bar, side panel, shared controls
│       ├── content/         # lessons.json (validated by tests/test_lessons.py)
│       ├── state/           # game and settings stores (Zustand)
│       └── lib/             # API client, piece data, notation, sounds
├── tests/                   # pytest suite
│   ├── reference.py         # independent, deliberately naive move generator
│   └── test_*.py            # engine, search, network, agents, API server, lessons
├── scripts/
│   ├── random_play.py       # random-vs-random stress test and benchmark
│   ├── mcts_benchmark.py    # MCTS speed and matches against the baselines (untrained or a checkpoint)
│   ├── train.py             # Milestone 3: self-play training
│   ├── selfplay.py          # self-play games for a teammate's run
│   ├── plot_training.py     # a run's loss and evaluation curves
│   ├── report.py            # a run's report in the terminal (and the app's strength check and ladder)
│   ├── kaggle_run.py        # a run on Kaggle: upload, start, follow, bring home
│   ├── server_run.py        # a run on your server: set up, start, follow live, pass pauses on
│   └── server_job.py        # a server run's strength check or ladder, played there and followed
├── notebooks/
│   └── train_colab.ipynb    # the same training on Google Colab, saved to Drive
├── docs/
│   ├── RULES.md             # rules specification with source tallies
│   └── Handouts/            # source PDFs cited in RULES.md (linked from there)
├── runs/, checkpoints/      # training runs and the published network (git-ignored)
├── pyproject.toml           # package metadata and dependencies
├── PRODUCT.md               # product and design brief for the web app
└── README.md
```

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
