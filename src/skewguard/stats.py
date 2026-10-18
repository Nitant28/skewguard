"""Distances between binned feature distributions.

Pure Python with no dependencies: every function takes plain lists, so each
result can be checked by hand and the module works with or without Feast.

Histogram conventions follow Feast v0.67.0 monitoring rows
(``feast/monitoring/metrics_calculator.py``, ``infra/offline_stores/*.py``):

* numeric: ``{"bins": edges, "counts": counts}`` with ``len(edges) == len(counts) + 1``.
  Edges come from each window's own min/max, so two windows rarely share edges.
  A constant column is ``{"bins": [v, v], "counts": [n]}`` on the SQL backends
  (DuckDB, Postgres, ...) but 20 bins over ``[v - 0.5, v + 0.5]`` on the
  file/dask backend, which calls ``numpy.histogram``.
* categorical: ``{"values": [{"value": v, "count": c}, ...], "other_count": n}``
  holding only the top-N values (N = 10 by default).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rebinned:
    """Histogram mass moved onto new edges.

    ``counts`` has one value per target bin. Mass outside the target range is
    kept in ``below`` / ``above`` instead of being dropped: a sudden jump there
    is itself a signal (a x100 unit bug moves nearly everything ``above``).
    """

    counts: list[float]
    below: float = 0.0
    above: float = 0.0

    @property
    def out_of_range(self) -> float:
        return self.below + self.above

    @property
    def total(self) -> float:
        return sum(self.counts) + self.out_of_range


def _validate_histogram(edges: Sequence[float], counts: Sequence[float]) -> None:
    if len(edges) != len(counts) + 1:
        raise ValueError(
            f"need len(edges) == len(counts) + 1, got {len(edges)} edges, {len(counts)} counts"
        )
    _validate_edges(edges)
    if any(c < 0 for c in counts):
        raise ValueError("histogram counts must be non-negative")


def _validate_edges(edges: Sequence[float]) -> None:
    if len(edges) < 2:
        raise ValueError("a histogram needs at least two edges")
    if any(not math.isfinite(e) for e in edges):
        raise ValueError("histogram edges must be finite")
    if any(b < a for a, b in zip(edges, edges[1:])):
        raise ValueError("histogram edges must be non-decreasing")


def _overlap(a: float, b: float, lo: float, hi: float) -> float:
    return max(0.0, min(b, hi) - max(a, lo))


def _locate(edges: Sequence[float], x: float) -> int:
    """Index of the target bin holding point ``x``; -1 if below, len(edges)-1 if above.

    Bins are ``[l, r)`` except the last, which is closed, like ``numpy.histogram``.
    """
    n_bins = len(edges) - 1
    if x < edges[0]:
        return -1
    if x > edges[-1]:
        return n_bins
    if x == edges[-1]:
        return n_bins - 1
    return bisect_right(edges, x) - 1


def rebin(
    src_edges: Sequence[float],
    src_counts: Sequence[float],
    dst_edges: Sequence[float],
) -> Rebinned:
    """Move histogram mass from ``src_edges`` onto ``dst_edges``.

    Mass is assumed uniform inside each source bin, so source bin ``[a, b)`` with
    count ``c`` gives target bin ``[l, r)`` the share ``c * overlap / (b - a)``.
    A zero-width source bin (``a == b``) is a point mass at ``a``.

    Runs in O(len(src) * len(dst)); with Feast's 20 bins that is 400 steps.
    """
    _validate_histogram(src_edges, src_counts)
    _validate_edges(dst_edges)

    n_dst = len(dst_edges) - 1
    lo, hi = dst_edges[0], dst_edges[-1]
    counts = [0.0] * n_dst
    below = above = 0.0

    for a, b, c in zip(src_edges, src_edges[1:], src_counts):
        if c == 0:
            continue
        if a == b:
            idx = _locate(dst_edges, a)
            if idx < 0:
                below += c
            elif idx >= n_dst:
                above += c
            else:
                counts[idx] += c
            continue

        width = b - a
        below += c * max(0.0, min(b, lo) - a) / width
        above += c * max(0.0, b - max(a, hi)) / width
        for j in range(n_dst):
            share = _overlap(a, b, dst_edges[j], dst_edges[j + 1])
            if share > 0:
                counts[j] += c * share / width

    return Rebinned(counts=counts, below=below, above=above)


def _proportions(counts: Sequence[float], eps: float = 0.0) -> list[float]:
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = sum(counts)
    if total <= 0:
        raise ValueError("cannot compare an empty distribution")
    return [max(c / total, eps) for c in counts]


def psi(
    expected_counts: Sequence[float],
    actual_counts: Sequence[float],
    eps: float = 1e-4,
) -> float:
    """Population stability index: ``sum((a - e) * ln(a / e))``.

    Proportions are floored at ``eps`` so an empty bin does not give ``log(0)``.
    Every term is >= 0, the value is 0 only for identical distributions, and
    swapping the arguments gives the same value.
    """
    if len(expected_counts) != len(actual_counts):
        raise ValueError("distributions must have the same number of bins")
    e = _proportions(expected_counts, eps)
    a = _proportions(actual_counts, eps)
    return sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e))


def js_divergence(p_counts: Sequence[float], q_counts: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1].

    Unlike PSI it needs no ``eps``: the mixture ``m`` is positive wherever
    either input is, so empty bins are handled exactly.
    """
    if len(p_counts) != len(q_counts):
        raise ValueError("distributions must have the same number of bins")
    p = _proportions(p_counts)
    q = _proportions(q_counts)

    def kl_to_mixture(x: list[float], y: list[float]) -> float:
        return sum(xi * math.log2(2 * xi / (xi + yi)) for xi, yi in zip(x, y) if xi > 0)

    value = 0.5 * kl_to_mixture(p, q) + 0.5 * kl_to_mixture(q, p)
    return min(1.0, max(0.0, value))


def compare_numeric(
    baseline_edges: Sequence[float],
    baseline_counts: Sequence[float],
    window_edges: Sequence[float],
    window_counts: Sequence[float],
    eps: float = 1e-4,
) -> tuple[float, float, Rebinned]:
    """PSI and JS of a window against a baseline, on the baseline's edges.

    The window is rebinned onto the baseline edges, then both histograms get an
    extra bin on each side so mass outside the baseline range counts as drift
    rather than disappearing. Returns ``(psi, js, rebinned_window)``.
    """
    _validate_histogram(baseline_edges, baseline_counts)
    r = rebin(window_edges, window_counts, baseline_edges)
    expected = [0.0, *baseline_counts, 0.0]
    actual = [r.below, *r.counts, r.above]
    return psi(expected, actual, eps), js_divergence(expected, actual), r


@dataclass(frozen=True)
class TopN:
    """A truncated categorical distribution as Feast stores it.

    ``counts`` holds the top-N values; ``other`` is the summed count of every
    value outside them. A value missing from ``counts`` has an unknown count
    between 0 and ``cap``.
    """

    counts: Mapping[Hashable, float]
    other: float = 0.0
    total: float = field(init=False)

    def __post_init__(self) -> None:
        if self.other < 0 or any(c < 0 for c in self.counts.values()):
            raise ValueError("counts must be non-negative")
        object.__setattr__(self, "total", float(sum(self.counts.values()) + self.other))

    @property
    def cap(self) -> float:
        """Largest count a value outside the top-N could have."""
        if not self.counts:
            return self.other
        return min(self.other, min(self.counts.values()))

    def freq_bounds(self, value: Hashable) -> tuple[float, float]:
        if self.total <= 0:
            raise ValueError("cannot compare an empty distribution")
        if value in self.counts:
            f = self.counts[value] / self.total
            return f, f
        return 0.0, self.cap / self.total

    @classmethod
    def from_feast(cls, histogram: Mapping) -> TopN:
        entries = histogram.get("values") or []
        return cls(
            counts={e["value"]: float(e["count"]) for e in entries},
            other=float(histogram.get("other_count") or 0),
        )


def linf_bounds(baseline: TopN, window: TopN) -> tuple[float, float]:
    """Bounds on the L-infinity distance between two category frequency vectors.

    Top-N truncation hides the exact frequency of values that fell into
    ``other``, so the exact distance is unknown; this returns the smallest and
    largest distance consistent with what was stored. When both sides list every
    value (``other == 0``), the bounds coincide.
    """
    lower = upper = 0.0
    for value in set(baseline.counts) | set(window.counts):
        bl, bh = baseline.freq_bounds(value)
        wl, wh = window.freq_bounds(value)
        lower = max(lower, bl - wh, wl - bh)
        upper = max(upper, bh - wl, wh - bl)
    return lower, upper


def new_category_mass(baseline: TopN, window: TopN) -> float:
    """Fraction of the window in values the baseline provably never saw.

    Only provable when the baseline listed every value (``other == 0``);
    otherwise an unseen value might be hiding in the baseline's ``other``.
    """
    if baseline.other > 0 or window.total <= 0:
        return 0.0
    unseen = sum(c for v, c in window.counts.items() if v not in baseline.counts)
    return unseen / window.total


def null_rate_pvalue(
    baseline_nulls: int,
    baseline_rows: int,
    window_nulls: int,
    window_rows: int,
) -> float:
    """One-sided p-value that the window's null rate exceeds the baseline's.

    Two-proportion z-test with a pooled variance. Returns 1.0 when either side
    is empty or the pooled rate is 0 or 1, since there is then no evidence.
    """
    if baseline_rows <= 0 or window_rows <= 0:
        return 1.0
    p_base = baseline_nulls / baseline_rows
    p_win = window_nulls / window_rows
    pooled = (baseline_nulls + window_nulls) / (baseline_rows + window_rows)
    if pooled <= 0 or pooled >= 1:
        return 1.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / baseline_rows + 1 / window_rows))
    z = (p_win - p_base) / se
    return 0.5 * math.erfc(z / math.sqrt(2))
