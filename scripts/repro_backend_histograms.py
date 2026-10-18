"""Reproduce F3: Feast's monitoring backends build different histograms from the same data.

Runs the same column through three code paths in Feast v0.67.0:
  * ``MetricsCalculator.compute_numeric`` (the Python fallback),
  * ``dask._dask_compute_numeric_metrics`` (the default file offline store),
  * ``duckdb._duckdb_numeric_stats`` (the DuckDB offline store; Postgres,
    Snowflake, etc. use the same zero-width rule),
then compares the stored histograms. Feast's own tests call these same helpers
(``tests/unit/monitoring/test_compute_correctness.py``).

    python scripts/repro_backend_histograms.py
"""

from __future__ import annotations

import duckdb
import numpy as np
import pyarrow as pa
from feast.infra.offline_stores.dask import _dask_compute_numeric_metrics
from feast.infra.offline_stores.duckdb import _duckdb_numeric_stats
from feast.monitoring.metrics_calculator import MetricsCalculator

BINS = 20


def histograms(values: list[float]) -> dict[str, dict]:
    arr = pa.array(values, type=pa.float64())
    out = {
        "python": MetricsCalculator(histogram_bins=BINS).compute_numeric(arr)["histogram"],
        "dask": _dask_compute_numeric_metrics(pa.chunked_array([arr]), BINS)["histogram"],
    }
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (v DOUBLE)")
    conn.executemany("INSERT INTO t VALUES (?)", [[v] for v in values])
    out["duckdb"] = _duckdb_numeric_stats(conn, "t", ["v"], "1=1", BINS)[0]["histogram"]
    conn.close()
    return out


def show(title: str, hists: dict[str, dict]) -> None:
    print(f"\n== {title}")
    for name, h in hists.items():
        e, c = h["bins"], h["counts"]
        nz = [(i, int(x)) for i, x in enumerate(c) if x]
        print(f"  {name:<7} {len(e)} edges [{e[0]:g} .. {e[-1]:g}], non-zero bins {nz}")


def main() -> None:
    show("constant column: 100 x 7.0", histograms([7.0] * 100))

    # Integer-valued features (counts, IDs, trips) put many values exactly on edges.
    rng = np.random.default_rng(0)
    trials = mismatched = 0
    example = None
    for hi in range(1, 200):
        values = rng.integers(0, hi + 1, size=500).astype(float).tolist()
        values[:2] = [0.0, float(hi)]  # pin the range so edges are k * hi / 20
        h = histograms(values)
        trials += 1
        if h["duckdb"]["counts"] != h["dask"]["counts"]:
            mismatched += 1
            if example is None:
                example = (hi, h)
    print(
        f"\n== integer-valued columns 0..hi, hi = 1..199: duckdb vs dask counts differ "
        f"in {mismatched} of {trials} datasets"
    )
    if example:
        hi, h = example
        diff = [
            (i, int(a), int(b))
            for i, (a, b) in enumerate(zip(h["dask"]["counts"], h["duckdb"]["counts"]))
            if a != b
        ]
        print(f"   first case hi={hi}: (bin, dask, duckdb) = {diff}")


if __name__ == "__main__":
    main()
