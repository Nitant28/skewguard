# Evaluation plan (pre-registered)

This plan is committed **before** any detection-power numbers are computed. The
commit hash that adds this file is the evidence that the faults, metrics,
thresholds and splits were not chosen after seeing results. Any later change goes
in a new commit with a "Deviations" entry at the bottom explaining why.

So far only clean-data (no-drift) behaviour has been measured
(`scripts/noise_floor.py`, `docs/noise_floor_output.md`). That run motivated the
hypothesis below; it doesn't test it.

## Hypotheses

**H1 (primary).** At equal false-positive rate on clean data, sample-size-aware
thresholds (`Policy()`, alpha = 0.01 per check) detect at least as many injected
faults as fixed PSI 0.25, across window sizes from 100 to 5,000 rows. They detect
strictly more of the gradual-shift faults at windows of 1,000 rows or more.

**H2.** The fixed PSI 0.25 rule has a false-positive rate above 5% on clean windows
of 150 rows or fewer.

**Falsified if:** the tuned policy's detection rate is lower than fixed PSI 0.25's
at matched false-positive rate (paired over the same incidents), with the 95%
bootstrap interval of the difference entirely below zero.

## Workload

- Data: NYC TLC Yellow Taxi trip records, 2024-01 to 2024-06 (public parquet).
  Features: `trip_distance`, `fare_amount`, `passenger_count`, `payment_type`
  (categorical), `PULocationID` (categorical, top-10 truncated), and
  `tpep_pickup_datetime` as the event timestamp.
- One Feast feature view backed by a FileSource, dask offline store, SQLite online store.
- Baseline: 2024-01-01 to 2024-02-29, **excluding** the evaluation windows (fixes F4).
- Tuning range: 2024-03 (choosing `min_rows` and `max_unseen_mass` only).
- Evaluation range: 2024-04 to 2024-06, untouched until the final run.
- Window sizes: daily windows sub-sampled to 100, 300, 1,000 and 5,000 rows.

## Injected faults (labelled; 3 severities each)

| Fault | Low | Medium | High |
|---|---|---|---|
| Unit change (multiply a numeric feature) | x1.1 | x10 | x100 |
| Null burst (set share of rows to null) | +2 pp | +10 pp | +50 pp |
| Stale source (stop appending events) | 1 x TTL | 3 x TTL | 7 x TTL |
| New / re-encoded category | 1% of rows | 5% | 25% |
| Timezone shift in event timestamps | +1 h | +5 h | +12 h |
| Gradual shift (mean drift per day, in baseline sd) | 0.02 | 0.05 | 0.1 |
| Type change (numeric logged as string) | 1% | 10% | 100% |

Each fault × severity × window size is injected into 30 evaluation-range days,
chosen with seed 20261018. Clean days with no injection measure false positives,
including natural weekly seasonality.

## Detectors compared

1. No detector (status quo): detection rate 0 by definition. This is the baseline
   for "time to notice".
2. Fixed null-rate rule: alert if the null rate rises more than 10 percentage
   points (mirrors the Feast UI health badge).
3. Fixed PSI 0.25 plus fixed L-inf 0.1: `Policy(fixed_threshold=...)`.
4. skewguard default: `Policy()`, simulated thresholds, alpha = 0.01 per check.

## Metrics

- Detection rate per fault × severity × window size (an incident is detected if any
  check alerts on an injected day).
- False-positive rate on clean days, per window size.
- Time to detect, for gradual shift and stale source: days from onset to first alert.
- Detector runtime per feature per run (p50/p99), including the null simulation
  with a cold and a warm cache.

## Analysis

- 95% intervals by bootstrap over incidents (10,000 resamples, seed 0).
- Paired comparison of detectors 3 and 4 on identical incidents. For overall
  detection: a McNemar test, Holm-corrected across the 4 window sizes.
- Matched-FPR comparison: also report detector 3 re-tuned on March data to the
  same false-positive rate that detector 4 achieves, so H1 is not won by FPR alone.
- Negative results are reported as found.

## Known limitations going in

- The null simulation assumes values are uniform within each baseline bin, and
  ignores baseline/window overlap (F4).
- Categorical simulation treats the baseline's `other` bucket as one lump.
- Per-check alpha is not corrected across features. With 20 features and 4
  checks, the family-wise false-alarm rate on a clean day is up to about 55%.
  Both the per-check and the Bonferroni-adjusted numbers will be reported.

## Deviations

(none yet)
