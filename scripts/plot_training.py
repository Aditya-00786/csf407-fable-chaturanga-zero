"""Plot a training run's curves (Milestone 3's loss tracking) for the report.

    python scripts/plot_training.py                 # runs/main -> runs/main/training.png
    python scripts/plot_training.py --run quick --out quick.png

Six panels: the policy and value losses per training step (with the loss on each generation's
new games before training on them, an honest check against memorising), the evaluation scores,
and the self-play results, endings and game length per generation.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

INK, MUTED, GRID = "#2b2118", "#7a6a5a", "#e6dccb"
POLICY, VALUE, TOTAL = "#b8862b", "#3d6f8f", "#2b2118"
ENDINGS = (("bare_king", "#b8862b"), ("checkmate", "#7a3b2e"), ("stalemate", "#3d6f8f"),
           ("repetition", "#9db4c0"), ("no_progress", "#c9bfb0"), ("max_plies", "#e6dccb"))


def load(run_dir: str) -> tuple[list[dict], dict]:
    events = []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    gens: dict[int, dict] = {}
    for e in events:
        if e["event"] != "train" and "generation" in e:
            row = gens.setdefault(e["generation"], {"eval": {}})
            if e["event"] == "eval":
                row["eval"][e["opponent"]] = e
            else:
                row[e["event"]] = e
    # Training interrupted by a pause is redone with the same step numbers: keep the later points.
    train = {e["step"]: e for e in events if e["event"] == "train"}
    return [train[k] for k in sorted(train)], gens


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def legend(ax, **kwargs) -> None:
    if ax.get_legend_handles_labels()[0]:  # nothing to explain before the first evaluation
        ax.legend(fontsize=8, frameon=False, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", help="image file (default runs/<run>/training.png)")
    args = ap.parse_args()
    run_dir = os.path.join(args.runs_dir, args.run)
    if not os.path.exists(os.path.join(run_dir, "metrics.jsonl")):
        sys.exit(f"no metrics in {run_dir}; has the run started?")
    train, gens = load(run_dir)
    config = json.load(open(os.path.join(run_dir, "config.json")))
    steps_per_gen = config["steps_per_generation"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), dpi=150)
    fig.patch.set_facecolor("white")
    steps = [e["step"] for e in train]
    # Each generation's new games, measured before training on them: the end of the previous generation.
    held = [(g, r["held_out"]) for g, r in sorted(gens.items()) if r.get("held_out", {}).get("loss") is not None]
    held_x = [h.get("step", (g - 1) * steps_per_gen) for g, h in held]

    for ax, key, colour, title, ylabel in ((axes[0][0], "policy_loss", POLICY, "Policy loss", "cross-entropy"),
                                           (axes[0][1], "value_loss", VALUE, "Value loss", "squared error")):
        ax.plot(steps, [e[key] for e in train], color=colour, linewidth=1.3, label="training batches")
        if held:
            ax.plot(held_x, [h[key] for _, h in held], color=colour, linestyle="none", marker="o", markersize=4.5,
                    markerfacecolor="white", label="new games, before training on them")
        if key == "value_loss" and any("value_baseline" in h for _, h in held):
            pts = [(x, h["value_baseline"]) for x, (_, h) in zip(held_x, held) if "value_baseline" in h]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, linewidth=1, linestyle="--",
                    label="predicting a draw everywhere (new games)")
        if key == "policy_loss" and train:
            ax.plot(steps, [e["policy_entropy"] for e in train], color=MUTED, linewidth=1, linestyle="--",
                    label="lowest possible (targets' entropy)")
        style(ax, title, "training step", ylabel)
        ax.set_ylim(bottom=0)
        legend(ax)

    ax = axes[0][2]
    for opponent, colour, label in (("greedy", POLICY, "vs Greedy"), ("previous", VALUE, "vs previous generation"),
                                    ("random", MUTED, "vs Random")):
        pts = [(g, r["eval"][opponent]["score"]) for g, r in sorted(gens.items()) if opponent in r["eval"]]
        if pts:
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], marker="o", markersize=4, color=colour,
                    linewidth=1.4, label=label)
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylim(-3, 103)
    style(ax, "Evaluation score", "generation", "score (%)")
    legend(ax)

    sp = [(g, r["selfplay"]) for g, r in sorted(gens.items()) if "selfplay" in r]
    xs = [g for g, _ in sp]

    ax = axes[1][0]
    if sp:
        white = [s["white_wins"] / s["games"] * 100 for _, s in sp]
        draws = [s["draws"] / s["games"] * 100 for _, s in sp]
        black = [s["black_wins"] / s["games"] * 100 for _, s in sp]
        ax.bar(xs, white, color="#d8b46a", label="White wins")
        ax.bar(xs, draws, bottom=white, color="#c9bfb0", label="draws")
        ax.bar(xs, black, bottom=[w + d for w, d in zip(white, draws)], color="#3b2a20", label="Black wins")
    ax.set_ylim(0, 100)
    style(ax, "Self-play results", "generation", "games (%)")
    legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    ax = axes[1][1]
    if sp:
        bottom = [0.0] * len(sp)
        for reason, colour in ENDINGS:
            share = [s["reasons"].get(reason, 0) / s["games"] * 100 for _, s in sp]
            if any(share):
                ax.bar(xs, share, bottom=bottom, color=colour, label=reason.replace("_", " "))
                bottom = [b + v for b, v in zip(bottom, share)]
    ax.set_ylim(0, 100)
    style(ax, "How self-play games end", "generation", "games (%)")
    legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    ax = axes[1][2]
    if sp:
        ax.plot(xs, [s["mean_plies"] for _, s in sp], marker="o", markersize=4, color=TOTAL, linewidth=1.4)
    ax.set_ylim(0, 310)
    style(ax, "Self-play game length", "generation", "mean plies (cap 300)")

    for ax in (axes[0][2], axes[1][0], axes[1][1], axes[1][2]):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f"ChaturangZero training run \"{args.run}\"", x=0.01, ha="left", fontsize=13, color=INK,
                 fontweight="bold")
    fig.tight_layout()
    out = args.out or os.path.join(run_dir, "training.png")
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
