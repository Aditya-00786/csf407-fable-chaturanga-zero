"""Play random-vs-random games to stress-test the engine and measure its speed.

    python scripts/random_play.py --games 10000 --workers 8

Every move played is chosen from legal_moves(), and every game must terminate.
Prints how games ended, game lengths and throughput (useful numbers for reports).
"""

import argparse
import os
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaturanga import GameState, RulesConfig  # noqa: E402
from chaturanga.constants import color_name  # noqa: E402


def play_batch(args):
    n_games, seed, rules = args
    rng = random.Random(seed)
    results, lengths, move_gens = [], [], 0
    for _ in range(n_games):
        s = GameState.initial(rules)
        while not s.is_terminal():
            moves = s.legal_moves()
            move_gens += 1
            s = s.apply(rng.choice(moves))
        out = s.outcome()
        winner = "draw" if out.winner is None else color_name(out.winner)
        results.append((out.reason, winner))
        lengths.append(s.ply)
    return results, lengths, move_gens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=RulesConfig().max_plies)
    parser.add_argument("--setup", default="crosswise")
    parser.add_argument("--promotion", default="mantri")
    args = parser.parse_args()

    rules = RulesConfig(setup=args.setup, promotion=args.promotion, max_plies=args.max_plies)
    workers = max(1, min(args.workers, args.games))
    base, extra = divmod(args.games, workers)
    batches = [(base + (i < extra), args.seed + i, rules) for i in range(workers)]

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(play_batch, batches))
    elapsed = time.perf_counter() - start

    results = [r for p in parts for r in p[0]]
    lengths = [n for p in parts for n in p[1]]
    move_gens = sum(p[2] for p in parts)

    print(f"{len(results)} games, rules={rules}")
    print(f"time {elapsed:.1f}s with {workers} workers "
          f"({len(results) / elapsed:.1f} games/s, {move_gens / elapsed:,.0f} positions/s)")
    print(f"game length (plies): mean {statistics.mean(lengths):.1f}, "
          f"median {statistics.median(lengths):.0f}, min {min(lengths)}, max {max(lengths)}")
    print("\nhow games ended:")
    for (reason, winner), count in Counter(results).most_common():
        print(f"  {reason:<12} {winner:<6} {count:>7}  ({100 * count / len(results):.1f}%)")


if __name__ == "__main__":
    main()
