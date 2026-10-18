<!--
DRAFT for feast-dev/feast (bug_report template). Not posted. Post it from your
own GitHub account after rechecking that no one has filed it since 2026-10-18
(searched: "histogram constant", "histogram bins monitoring", "monitoring histogram edges").
Suggested title: Monitoring histograms differ across offline stores for the same data
Labels per template: kind/bug, priority/p2
-->

## Expected Behavior

`feast monitor run` should store the same numeric histogram for the same data
whichever offline store computes it. `TestCrossBackendConsistency` already
asserts this for mean, null rate and top categories.

## Current Behavior

The histograms differ in two cases.

**1. Constant columns (min == max)**
- The SQL backends store one zero-width bin: DuckDB
  (`duckdb.py` `_duckdb_numeric_histogram`), Postgres (`_sql_numeric_histogram`),
  and the other SQL stores.
- The file/dask store and the Python fallback (`MetricsCalculator`) call
  `np.histogram`. numpy widens a zero-width range to `[v - 0.5, v + 0.5]` and
  returns 20 bins.

```
values = [7.0] * 100
python  21 edges [6.5 .. 7.5], non-zero bins [(10, 100)]
dask    21 edges [6.5 .. 7.5], non-zero bins [(10, 100)]
duckdb   2 edges [7 .. 7],     non-zero bins [(0, 100)]
```

**2. Values exactly on an interior bin edge**
- DuckDB buckets with `FLOOR((x - min) / bin_width) + 1`. In floating point,
  `33 / 2.2 = 14.999999999999998`, so 33 lands in bin 14.
- numpy's edges are `linspace(min, max, 21)`, and numpy corrects its index
  against those edges, so 33 lands in bin 15.

This mostly affects integer-valued features (counts, trips). With 500 random
integers in `0..hi` and the range pinned to `[0, hi]`, DuckDB and dask counts
differed for 5 of the 199 values of `hi` tested (`hi = 1..199`). First case:
`hi = 44`, bins 14 and 15 hold 18/27 on dask and 27/18 on DuckDB.

## Why it matters

The histogram is the input to any future drift comparison (#6341), and to the
baseline overlay the UI's `HistogramChart` already supports. Switching offline
stores, or comparing metrics computed by different stores, shows a distribution
change that isn't real.

## Steps to reproduce

The script calls the same private helpers as
`tests/unit/monitoring/test_compute_correctness.py`:
`MetricsCalculator.compute_numeric`, `_dask_compute_numeric_metrics` and
`_duckdb_numeric_stats`.
<!-- link: https://github.com/<you>/skewguard/blob/main/scripts/repro_backend_histograms.py -->

### Specifications

- Version: 0.67.0 (commit 2dfb9e6)
- Platform: Windows 11, Python 3.12; DuckDB 1.5.6
- Subsystem: `feast/monitoring`, `infra/offline_stores/{dask,duckdb,...}`

## Possible Solution

1. **Constant columns (small, ready):** make the two Python paths use the SQL
   convention through one helper in `monitoring_utils.numeric_histogram`. The
   SQL form is exact, seven SQL backends already produce it, and the UI already
   renders it. I have a patch with a parametrized cross-backend histogram test
   (golden data and a constant column); the new tests fail without the fix and
   pass with it. Happy to open the PR if this direction is OK.
2. **Edge rounding (needs a decision):** compute the bucket in SQL as numpy does,
   `FLOOR((x - min) * bins / (max - min))`, then correct it against the explicit
   edges `min + i * ((max - min) / bins)`, which are the same floating-point
   operations as `np.linspace`. That makes the SQL backends match exactly. It
   touches every SQL backend, so I'd like a maintainer's view before starting.
   Alternatively, document that bin assignment at exact edges may differ.
