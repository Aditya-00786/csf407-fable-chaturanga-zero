"""Milestone 2 benchmark: search speed, and the untrained MCTS agent against the baselines.

    python scripts/mcts_benchmark.py                       # speed + 10 games each vs Random and Greedy
    python scripts/mcts_benchmark.py --sims 50 --games 20 --opponents random
    python scripts/mcts_benchmark.py --speed-only --device mps

The MCTS agent's network is untrained, so its priors and values are noise; whatever it wins, it
wins through the search and the exact results of finished games. Colours alternate game by game.
"""

import argparse
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from chaturanga import GameState  # noqa: E402
from chaturanga.agents import MCTSAgent, make_agent  # noqa: E402
from chaturanga.constants import WHITE, color_name  # noqa: E402
from chaturanga.mcts import MCTS, MCTSConfig, UniformEvaluator  # noqa: E402
from chaturanga.net import NetConfig, NetworkEvaluator, new_network, parameter_count  # noqa: E402


def search_speed(evaluator, simulations: int, repeats: int = 3) -> float:
    """Simulations per second from the starting position (fresh tree each time)."""
    best = 0.0
    for r in range(repeats):
        mcts = MCTS(evaluator, MCTSConfig(simulations=simulations), seed=r)
        t = time.perf_counter()
        mcts.search(GameState.initial())
        best = max(best, simulations / (time.perf_counter() - t))
    return best


def play(agent_white, agent_black) -> tuple[GameState, float]:
    """One game; returns the final state and the MCTS side's mean seconds per move."""
    state, think, moves = GameState.initial(), 0.0, 0
    while not state.is_terminal():
        agent = agent_white if state.side == WHITE else agent_black
        t = time.perf_counter()
        move = agent.select_move(state)
        if isinstance(agent, MCTSAgent):
            think += time.perf_counter() - t
            moves += 1
        state = state.apply(move)
    return state, think / max(1, moves)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=50, help="simulations per move (default 50)")
    ap.add_argument("--games", type=int, default=10, help="games per opponent (default 10)")
    ap.add_argument("--opponents", nargs="+", default=["random", "greedy"])
    ap.add_argument("--blocks", type=int, default=NetConfig.blocks)
    ap.add_argument("--channels", type=int, default=NetConfig.channels)
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda (default cpu)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    net = new_network(NetConfig(blocks=args.blocks, channels=args.channels), seed=args.seed)
    evaluator = NetworkEvaluator(net, device)

    print(f"Network: {args.blocks} blocks x {args.channels} filters, {parameter_count(net):,} parameters, "
          f"device {device}, {torch.get_num_threads()} CPU threads")
    uniform = search_speed(UniformEvaluator(), 2000)
    network = search_speed(evaluator, 200)
    print(f"Search speed: {uniform:,.0f} sims/s with uniform priors (engine only), "
          f"{network:,.0f} sims/s with the network ({1000 / network:.2f} ms per simulation)")
    print(f"At {args.sims} sims per move: about {args.sims / network * 1000:.0f} ms per move")
    if args.speed_only:
        return

    for opponent_id in args.opponents:
        tally, reasons, lengths, think = Counter(), Counter(), [], []
        t0 = time.perf_counter()
        for g in range(args.games):
            mcts = MCTSAgent(seed=args.seed + g, simulations=args.sims, evaluator=evaluator)
            opponent = make_agent(opponent_id, seed=args.seed + g)
            mcts_colour = WHITE if g % 2 == 0 else -WHITE
            white, black = (mcts, opponent) if mcts_colour == WHITE else (opponent, mcts)
            final, per_move = play(white, black)
            out = final.outcome()
            result = "draw" if out.winner is None else ("win" if out.winner == mcts_colour else "loss")
            tally[result] += 1
            reasons[out.reason] += 1
            lengths.append(final.ply)
            think.append(per_move)
            print(f"  game {g + 1:>2}: MCTS as {color_name(mcts_colour):<5} -> {result:<4} "
                  f"({out.reason}, {final.ply} plies)", flush=True)
        n = args.games
        score = (tally["win"] + 0.5 * tally["draw"]) / n
        print(f"MCTS ({args.sims} sims, untrained) vs {opponent_id}: "
              f"{tally['win']} W / {tally['draw']} D / {tally['loss']} L, score {score:.0%}")
        print(f"  endings: {dict(reasons)}; mean length {statistics.mean(lengths):.0f} plies; "
              f"{statistics.mean(think) * 1000:.0f} ms per MCTS move; {time.perf_counter() - t0:.0f} s total")


if __name__ == "__main__":
    main()
