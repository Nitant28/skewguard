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


def simulate_reference(
    edges: Sequence[float],
    counts: Sequence[float],
    n_baseline: int,
    n_window: int,
    reps: int,
    seed: int,
    bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Straightforward no-drift simulation: draw every value, histogram, compare.

    Costs O(reps * n) and runs in a Python loop. ``_numeric_null`` uses
    ``sample_own_histograms`` instead; this version is the reference it is
    tested against.
    """
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
    return np.asarray(psis), np.asarray(jss)


def sample_own_histograms(
    edges: Sequence[float],
    counts: Sequence[float],
    n: int,
    reps: int,
    rng: np.random.Generator,
    bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``reps`` samples of ``n`` values and return the ``numpy.histogram`` of each.

    Same distribution as ``feast_histogram(sample_histogram(...))``, but without
    materialising the ``n`` values, so the cost is O(reps * bins^2) for any ``n``:

    1. Split ``n`` across the source bins (multinomial).
    2. Draw the sample minimum from the first occupied bin and the maximum from
       the last (order statistics of uniforms: ``1 - U**(1/c)`` and ``U**(1/c)``).
       These fix the sample's own edges, ``linspace(min, max, bins + 1)``.
    3. Every other value is uniform on its source bin, clipped to ``[min, max]``,
       so its own-bin counts are multinomial in the overlap fractions.

    Returns ``(own_edges, own_counts)`` with shapes ``(reps, bins + 1)`` and
    ``(reps, bins)``.
    """
    e = np.asarray(edges, dtype=float)
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    a, b = e[:-1], e[1:]
    rows = np.arange(reps)

    c = rng.multinomial(n, p, size=reps)  # (reps, B) values per source bin
    occupied = c > 0
    first = occupied.argmax(axis=1)
    last = len(p) - 1 - occupied[:, ::-1].argmax(axis=1)
    same = first == last

    c_first = c[rows, first]
    vmin = a[first] + (b[first] - a[first]) * (1 - rng.random(reps) ** (1 / c_first))
    # The maximum is the largest of the other values in the last occupied bin.
    m_last = np.where(same, c_first - 1, c[rows, last])
    has_max = m_last > 0
    lo_last = np.where(same, vmin, a[last])
    u_max = rng.random(reps) ** (1 / np.maximum(m_last, 1))
    vmax = np.where(has_max, lo_last + (b[last] - lo_last) * u_max, vmin)

    # Each source bin's remaining values, and the interval they are uniform on.
    is_first = np.arange(len(p))[None, :] == first[:, None]
    is_last = np.arange(len(p))[None, :] == last[:, None]
    remaining = c - is_first - (is_last & has_max[:, None])
    lo = np.where(is_first, vmin[:, None], a[None, :])
    hi = np.where(is_last, vmax[:, None], b[None, :])

    # numpy.histogram widens a zero-width range to [v - 0.5, v + 0.5].
    flat = vmin == vmax
    own_lo, own_hi = np.where(flat, vmin - 0.5, vmin), np.where(flat, vmax + 0.5, vmax)
    own = np.linspace(own_lo, own_hi, bins + 1, axis=1)  # (reps, bins + 1)

    width = hi - lo
    right = np.minimum(hi[:, :, None], own[:, None, 1:])
    left = np.maximum(lo[:, :, None], own[:, None, :-1])
    overlap = np.clip(right - left, 0.0, None)  # (reps, source bin, own bin)
    probs = np.where(
        width[:, :, None] > 0, overlap / np.where(width > 0, width, 1.0)[:, :, None], 0.0
    )
    # Degenerate intervals (zero width or no remaining values): put any mass in
    # the bin holding ``lo`` so every probability vector sums to one.
    holder = np.clip(((lo - own_lo[:, None]) / (own_hi - own_lo)[:, None] * bins), 0, bins - 1)
    degenerate = probs.sum(axis=2) <= 0
    probs[degenerate] = 0.0
    r_idx, j_idx = np.nonzero(degenerate)
    probs[r_idx, j_idx, holder[r_idx, j_idx].astype(int)] = 1.0
    probs /= probs.sum(axis=2, keepdims=True)

    own_counts = rng.multinomial(remaining, probs).sum(axis=1).astype(float)
    own_counts[:, 0] += 1  # the minimum sits on the first edge
    own_counts[rows[has_max], bins - 1] += 1  # the maximum on the last (closed) edge
    single = ~has_max & flat  # n == 1: numpy puts the value in the middle bin
    own_counts[single] = 0.0
    own_counts[single, bins // 2] = 1.0
    return own, own_counts


def compare_batch(
    base_edges: np.ndarray,
    base_counts: np.ndarray,
    win_edges: np.ndarray,
    win_counts: np.ndarray,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise ``stats.compare_numeric`` for non-degenerate histograms: ``(psi, js)``.

    Every source bin must have positive width, which always holds for
    ``numpy.histogram`` output.
    """
    sa, sb = win_edges[:, :-1, None], win_edges[:, 1:, None]
    dl, dr = base_edges[:, None, :-1], base_edges[:, None, 1:]
    share = np.clip(np.minimum(sb, dr) - np.maximum(sa, dl), 0.0, None) / (sb - sa)
    moved = np.einsum("rs,rsd->rd", win_counts, share)
    lo, hi = base_edges[:, :1, None], base_edges[:, -1:, None]
    w = win_counts[:, :, None]
    below = (w * np.clip(np.minimum(sb, lo) - sa, 0.0, None) / (sb - sa)).sum(axis=(1, 2))
    above = (w * np.clip(sb - np.maximum(sa, hi), 0.0, None) / (sb - sa)).sum(axis=(1, 2))

    zeros = np.zeros((len(base_counts), 1))
    expected = np.hstack([zeros, base_counts, zeros])
    actual = np.hstack([below[:, None], moved, above[:, None]])
    pe = expected / expected.sum(axis=1, keepdims=True)
    pa = actual / actual.sum(axis=1, keepdims=True)

    fe, fa = np.maximum(pe, eps), np.maximum(pa, eps)
    psi_values = ((fa - fe) * np.log(fa / fe)).sum(axis=1)

    m = (pe + pa) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_e = np.where(pe > 0, pe * np.log2(pe / m), 0.0).sum(axis=1)
        kl_a = np.where(pa > 0, pa * np.log2(pa / m), 0.0).sum(axis=1)
    js_values = np.clip(0.5 * kl_e + 0.5 * kl_a, 0.0, 1.0)
    return psi_values, js_values


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
    if is_point_style(edges):
        # A constant baseline resamples to the same constant: every draw is identical.
        zeros = (0.0,) * reps
        return NullDistribution(zeros), NullDistribution(zeros)
    rng = np.random.default_rng(seed)
    psis, jss = [], []
    chunk = 2000  # bounds memory at ~chunk * bins^2 floats
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        be, bc = sample_own_histograms(edges, counts, n_baseline, size, rng, bins)
        we, wc = sample_own_histograms(edges, counts, n_window, size, rng, bins)
        p, j = compare_batch(be, bc, we, wc)
        psis.extend(p.tolist())
        jss.extend(j.tolist())
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
