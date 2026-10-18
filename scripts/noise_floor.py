"""Reproduce the noise-floor finding: PSI on clean data depends on window size.

Draws baselines and windows independently from one known distribution (no
drift by construction), histograms them the way Feast does, and reports how
often common fixed thresholds fire. Also checks the simulated threshold's
false-positive rate against its target.

Only clean data is used here. Detection power on drifted data is measured
later, under the pre-registered plan in docs/eval-plan.md.

    python scripts/noise_floor.py [--trials 2000] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from skewguard.stats import compare_numeric
from skewguard.thresholds import feast_histogram, numeric_null

N_BASELINE = 1808  # rows in the `feast init` demo baseline (2026-10-18 run)
WINDOWS = (116, 125, 1000, 5000)
FIXED = (0.1, 0.2, 0.25)
ALPHA = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", type=int, default=50, help="baselines for calibration")
    parser.add_argument("--reps", type=int, default=500, help="null draws per baseline")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    def draw(n: int) -> np.ndarray:
        # Like the demo's conv_rate: np.random.random, cast to float32.
        return rng.random(n).astype(np.float32).astype(float)

    print(f"Clean data only: baseline {N_BASELINE} rows, Uniform(0,1), 20 bins.")
    print(f"{args.trials} independent (baseline, window) pairs per window size.\n")
    header = "| window rows | median PSI | p95 PSI | " + " | ".join(f"P(PSI > {t})" for t in FIXED)
    print(
        header + f" | median simulated limit (alpha={ALPHA}) "
        f"| its false-positive rate ({args.baselines} baselines x 100 windows) |"
    )
    print("|" + "---|" * (3 + len(FIXED) + 2))

    for n in WINDOWS:
        values = []
        for _ in range(args.trials):
            be, bc = feast_histogram(draw(N_BASELINE))
            we, wc = feast_histogram(draw(n))
            values.append(compare_numeric(be, bc, we, wc)[0])
        values = np.asarray(values)

        # Calibration, as in production: each baseline is fixed and checked
        # against many fresh windows. Average over baselines, because for
        # large windows a baseline's own sampling error decides whether all
        # of its windows alarm or none do.
        limits, rates = [], []
        for b in range(args.baselines):
            be, bc = feast_histogram(draw(N_BASELINE))
            null = numeric_null(be, bc, n, reps=args.reps, seed=args.seed + b)[0]
            limit = null.quantile(1 - ALPHA)
            fresh = [compare_numeric(be, bc, *feast_histogram(draw(n)))[0] for _ in range(100)]
            limits.append(limit)
            rates.append(np.mean(np.asarray(fresh) > limit))

        fixed = " | ".join(f"{np.mean(values > t):.1%}" for t in FIXED)
        print(
            f"| {n} | {np.median(values):.3f} | {np.quantile(values, 0.95):.3f} | {fixed} "
            f"| {np.median(limits):.3f} | {np.mean(rates):.1%} |"
        )


if __name__ == "__main__":
    main()
