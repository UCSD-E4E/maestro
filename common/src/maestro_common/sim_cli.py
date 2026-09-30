"""``maestro-sim``: run the offline bandit simulator and summarize it."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from maestro_common.sim import PolicyResult, SimConfig, compare


def write_csv(results: dict[str, PolicyResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "round", "labels", "best_quality"])
        for policy, result in results.items():
            for t, (labels, quality) in enumerate(zip(result.labels, result.curve), start=1):
                writer.writerow([policy, t, int(labels), f"{quality:.4f}"])


def plot(results: dict[str, PolicyResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for policy, result in results.items():
        if policy == "bandit":
            left.plot(result.labels, result.curve, color="black", linewidth=2.5, label="bandit", zorder=3)
        else:
            left.plot(result.labels, result.curve, linewidth=1, alpha=0.6, label=policy)
    left.set(xlabel="Labels", ylabel="Mean IoU of best model", title="Pre-annotation quality (mean over seeds)")
    left.legend(fontsize=7, ncol=2)

    trace = results["bandit"].posterior_trace
    rounds = range(1, len(trace) + 1)
    for arm in trace[0]:
        right.plot(rounds, [means[arm] for means in trace], label=arm)
    right.set(xlabel="Round", ylabel="Posterior mean", title="Bandit posterior means (seed 0)")
    right.legend(fontsize=7, ncol=2)
    fig.suptitle("Synthetic, illustrative learning curves", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--audit-fraction", type=float, default=0.15)
    parser.add_argument("--r", type=float, default=1.0, help="Sherlock reshaping factor")
    parser.add_argument("--gamma", type=float, default=0.9, help="per-round discount")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--target", type=float, default=0.7, help="mean IoU for labels-to-target")
    parser.add_argument("--out", type=Path, default=Path("sim-out"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    config = SimConfig(args.rounds, args.batch_size, args.audit_fraction, args.r, args.gamma)
    results = compare(config, seeds=args.seeds, target=args.target)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(results, args.out / "curves.csv")
    if not args.no_plot:
        plot(results, args.out / "curves.png")

    print(f"Synthetic, illustrative learning curves; {args.seeds} seeds; {config}")
    print(f"{'policy':<24} {'final IoU':>9} {'labels to ' + str(args.target):>16}")
    ranked = sorted(results.items(), key=lambda kv: -kv[1].final_quality)
    for policy, result in ranked:
        reach = result.labels_to_target
        print(f"{policy:<24} {result.final_quality:>9.3f} {reach if reach is not None else 'never':>16}")
    print(f"Wrote {args.out / 'curves.csv'}" + ("" if args.no_plot else f" and {args.out / 'curves.png'}"))


if __name__ == "__main__":
    main()
