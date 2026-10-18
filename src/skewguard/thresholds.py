"""Sample-size-aware alert thresholds by Monte Carlo simulation.

A fixed rule such as "PSI > 0.25 means drift" ignores that PSI between two
finite samples is positive even when nothing changed, and that this noise
floor shrinks as samples grow. With Feast's 20-bin histograms, a 116-row
window drawn from the same distribution as a 1,808-row baseline exceeds 0.25
more than 5% of the time.

Instead, simulate the no-drift case for the exact window size: draw a baseline
and a window from the baseline histogram, histogram both the way Feast does,
rebin, and compute the distance. The alert threshold is a high quantile of
that null distribution, so its false-positive rate is set directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from skewguard.stats import TopN, compare_numeric, linf_bounds


@dataclass(frozen=True)
class NullDistribution:
    """Distances observed between same-distribution samples."""

    values: tuple[float, ...]

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.values, q))

    def pvalue(self, observed: float) -> float:
        """Share of null draws at least as extreme, with the +1 correction."""
        n_ge = sum(v >= observed for v in self.values)
        return (n_ge + 1) / (len(self.values) + 1)


def is_point_style(edges: Sequence[float]) -> bool:
    """True if a constant column was stored as one zero-width bin (SQL backends)."""
    return len(edges) == 2 and edges[0] == edges[1]


def feast_histogram(
    values: np.ndarray, bins: int = 20, point_style: bool = False
) -> tuple[list[float], list[float]]:
    """Histogram ``values`` the way Feast's monitoring backends do.

    ``numpy.histogram`` over the sample's own min/max (dask / Python path); a
    constant sample becomes a single zero-width bin when ``point_style`` is set,
    matching the DuckDB / Postgres / Snowflake backends.
    """
    if point_style and values.min() == values.max():
        return [float(values[0]), float(values[0])], [float(len(values))]
    counts, edges = np.histogram(values, bins=bins)
    return edges.tolist(), counts.astype(float).tolist()


def sample_histogram(
    edges: Sequence[float], counts: Sequence[float], n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` values, uniform within each bin, with bin probability ~ count."""
    e = np.asarray(edges, dtype=float)
    c = np.asarray(counts, dtype=float)
    idx = rng.choice(len(c), size=n, p=c / c.sum())
    lo, hi = e[idx], e[idx + 1]
    return lo + (hi - lo) * rng.random(n)


@lru_cache(maxsize=4096)
def _numeric_null(
    edges: tuple[float, ...],
    counts: tuple[float, ...],
    n_baseline: int,
    n_window: int,
    reps: int,
    seed: int,
    bins: int,
) -> tuple[NullDistribution, NullDistribution]:
    rng = np.random.default_rng(seed)
    point_style = is_point_style(edges)
    psis, jss = [], []
    for _ in range(reps):
        base = sample_histogram(edges, counts, n_baseline, rng)
        win = sample_histogram(edges, counts, n_window, rng)
        be, bc = feast_histogram(base, bins, point_style)
        we, wc = feast_histogram(win, bins, point_style)
        p, j, _ = compare_numeric(be, bc, we, wc)
        psis.append(p)
        jss.append(j)
    return NullDistribution(tuple(psis)), NullDistribution(tuple(jss))


def numeric_null(
    baseline_edges: Sequence[float],
    baseline_counts: Sequence[float],
    n_window: int,
    n_baseline: int | None = None,
    reps: int = 1000,
    seed: int = 0,
) -> tuple[NullDistribution, NullDistribution]:
    """Null distributions of ``(PSI, JS)`` for a window of ``n_window`` non-null rows.

    ``n_baseline`` defaults to the baseline histogram's total. Results are
    cached per (baseline, window size), so one simulation serves every
    monitoring run until the baseline changes.
    """
    if n_window < 1:
        raise ValueError("n_window must be positive")
    n_base = int(n_baseline if n_baseline is not None else sum(baseline_counts))
    if n_base < 1:
        raise ValueError("baseline is empty")
    return _numeric_null(
        tuple(float(e) for e in baseline_edges),
        tuple(float(c) for c in baseline_counts),
        n_base,
        int(n_window),
        reps,
        seed,
        len(baseline_counts) if not is_point_style(baseline_edges) else 20,
    )


@lru_cache(maxsize=4096)
def _categorical_null(
    values: tuple,
    counts: tuple[float, ...],
    other: float,
    n_baseline: int,
    n_window: int,
    reps: int,
    seed: int,
) -> NullDistribution:
    rng = np.random.default_rng(seed)
    probs = np.asarray([*counts, other], dtype=float)
    probs /= probs.sum()
    top_n = len(values)

    def draw(n: int) -> TopN:
        sample = rng.multinomial(n, probs)
        order = np.argsort(-sample[:-1], kind="stable")[:top_n]
        kept = {values[i]: float(sample[i]) for i in order if sample[i] > 0}
        return TopN(kept, other=float(n - sum(kept.values())))

    return NullDistribution(
        tuple(linf_bounds(draw(n_baseline), draw(n_window))[0] for _ in range(reps))
    )


def categorical_null(
    baseline: TopN,
    n_window: int,
    reps: int = 1000,
    seed: int = 0,
) -> NullDistribution:
    """Null distribution of the L-infinity lower bound for a window of ``n_window`` rows.

    Values in the baseline's ``other`` bucket are simulated as one lump that
    never enters the top-N, which slightly understates their noise.
    """
    if n_window < 1:
        raise ValueError("n_window must be positive")
    if baseline.total <= 0:
        raise ValueError("baseline is empty")
    values = tuple(baseline.counts)
    return _categorical_null(
        values,
        tuple(baseline.counts[v] for v in values),
        baseline.other,
        int(baseline.total),
        int(n_window),
        reps,
        seed,
    )
