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

### 2026-10-19: choices the plan left open, fixed while building the harness

These were made before any detection-power number was computed. The March run
that smoke-tested the harness was checked for completeness and row sanity only.
Code: `src/skewguard/inject.py`, `scripts/eval_harness.py`.

1. **TTL and freshness (differs from a literal `Policy()`).** The feature view's
   TTL is 1 day. Detectors 3 and 4 both set `max_staleness` to the TTL, because
   `Policy()` turns the freshness check off and the stale-source fault is graded in
   TTL units. Both get the same setting, so the 3-vs-4 comparison is unaffected.
2. **Detector 3** uses PSI 0.25 on the three numeric features and L-inf 0.1 on the
   two categorical ones (a per-feature `PolicySet`), since `Policy` has a single
   `fixed_threshold`. **Detector 2** compares Feast's stored `null_rate` of the
   window and the baseline, per feature.
3. **Windows.** A day's window is a uniform sample, without replacement, of that
   day's trips, seeded per (day, size). Every fault on that day starts from the
   same sample. Feast computes it with `compute_metrics(start_date=day,
   end_date=day, granularity="daily")`, which covers 00:00:00 to 23:59:59 UTC.
   Detectors run as of the following midnight. TLC pickup times are naive local
   times and are treated as UTC.
4. **Baseline.** All 5,972,131 trips with pickup between 2024-01-01 and 2024-02-29
   in the January and February files, not sub-sampled. `compute_baseline` runs once,
   on a source holding only those rows.
5. **Injected days.** For each fault, 30 onset days are drawn once (seed 20261018,
   combined with the fault's index) and reused for every severity and window size,
   so severity and window-size comparisons are paired. A fault that targets one
   feature picks it at random for each incident: unit change, gradual shift and
   type change among the numeric features; new category among the categorical
   ones; null burst among all five.
6. **Onset and duration.**
   - Unit change, null burst, new category, timezone shift and type change start at
     00:00 UTC on the injected day and are evaluated on that day. The shares apply
     to that day's rows and are drawn from non-null values. New category recodes
     the chosen rows to one unseen value, `__new__`. Timezone shift moves event
     timestamps forward, so events pushed past 23:59:59 leave the day's window.
   - Gradual shift starts at 00:00 UTC on the onset day and grows continuously: an
     event *t* days after onset moves by rate x baseline `stddev` x *t*. It is
     followed for 14 days, cut off at the end of the range, and time to detect is
     the first `day_offset` with an alert.
   - Stale source drops every event from a random second of the onset day until
     k x TTL later (k = 1, 3, 7), with no backfill. Every day that overlaps the
     outage is evaluated (2, 4 or 8 days), cut off at the end of the range.
7. **Type change storage.** A parquet column holds one type, so a numeric column
   with any stringified value is written as a string column. Feast's dask
   backend casts the column to float before computing numeric metrics.
8. **Tuning outputs.** Each result row keeps every check's value and threshold, so
   `max_unseen_mass` can be re-thresholded from March rows. A different `min_rows`
   needs a re-run (`--min-rows`), because skipped checks store no value.
9. **Freshness stamps follow `feast monitor run`.** This was found by code review
   before any detection numbers existed. `compute_metrics` stamps a window with
   the newest event inside it, or with `None` when the window is empty
   (`monitoring_service.py:47-61`, `1038-1042`). So the freshness age could never
   exceed one day, and stale source would have been undetectable by construction.
   The scheduled `auto_compute` stamps every window with the source's overall
   MAX(event_timestamp) instead (`monitoring_service.py:154`, `206`). The harness
   emulates that. After Feast computes a window, every row's `max_event_timestamp`
   is set to the newest event the source holds by the end of that day: the clean
   sample of the day before the onset day, then what the fault left of each
   evaluated day. Feast's own stamp is kept as `feast_newest_event`. This
   changes nothing on a day that has events. A first March smoke run made before
   the fix was stopped and discarded.
10. **Analysis and tuning rules, fixed before any March detection value was read
    (2026-10-19).** The March run had been checked only for completeness and row
    counts. Code: `scripts/analyze.py`.
    - **Tuning `max_unseen_mass`.** Candidates: 0.001, 0.002, 0.005, 0.01, 0.02,
      0.05. The stored `out_of_range` and `new_categories` values of detector
      `skewguard` on March clean days are re-thresholded at each candidate. The
      choice is the smallest candidate whose per-check clean alert rate is at most
      1% at every window size; if none qualifies, 0.05. Detectors 3 and 4 use the
      same value.
    - **`min_rows` is not tuned.** It stays at 30. Every planned window samples
      at least 100 rows, so only null bursts can push non-null rows below 30, and
      the null-rate check covers those.
    - **Matched-FPR detector 3.** For each window size, both fixed thresholds
      (PSI 0.25, L-inf 0.10) are multiplied by one factor k in 0.10, 0.11, ...,
      5.00. The stored `psi` / `linf` values of detector `fixed` on March clean
      days are re-thresholded; every other check keeps its stored status. The
      choice is the k whose day-level clean FPR is closest to detector 4's March
      day-level clean FPR (ties go to the larger k). That k is applied unchanged
      to the evaluation rows and reported as `fixed_matched`.
    - **Day-level FPR** (primary): the share of clean days on which a detector
      alerts on any feature (`alert`). **Per-check FPR**: the share of clean
      feature-windows on which one check alerts. H2 is tested on the `psi` check
      of detector 3 over the three numeric features at window size 100, the only
      planned size of 150 rows or fewer.
    - **Detection** (primary, as planned): an incident is detected if `alert` is
      true on any of its evaluated days. Secondary, for faults with a target
      feature: `target_alert` (the alert names the injected feature).
    - **Intervals and tests.**
      - Detection-rate intervals: percentile bootstrap over the incidents of a
        cell, 10,000 resamples, seed 0.
      - H1, per window size, pooled over faults x severities and paired on
        identical incidents: exact McNemar (two-sided binomial on discordant
        pairs) for detector 4 vs detector 3 and vs `fixed_matched`,
        Holm-corrected across the 4 window sizes.
      - Difference intervals: paired bootstrap of detector 4 minus detector 3,
        resampling incidents within each fault x severity.
      - The gradual-shift sub-claim uses the same paired comparison, restricted
        to `gradual_shift` at 1,000 and 5,000 rows.
    - **Time to detect** (gradual shift, stale source): the first `day_offset`
      with an alert. Reported as the detected share, plus the median over
      detected incidents.
    - Rows with `feast_status` other than `ok` are excluded and counted.
      Runtime p50/p99 is per feature, split into cold (`null_cache_misses > 0`)
      and warm.
    - **Bonferroni.** The workload has 5 features (the limitation above assumed
      20). Per day there are 10 alpha-controlled tests: PSI-or-L-inf plus null
      rate, per feature. Detector `skewguard_bonf` is detector 4 with
      `false_positive_rate = 0.001`, and it runs in the evaluation range
      alongside the others. It uses 10,000 null draws instead of 1,000: at
      1,000 draws, the 0.999 quantile falls between the two largest draws and
      is too noisy to estimate.
11. **Blind spots expected by construction, recorded before any detection value
    was read (2026-10-19).** These will be reported as found, not worked around.
    - **Type change.** Feast's numeric metric path casts a numeric feature's
      column to float64 (`dask.py` `_dask_compute_numeric_metrics`). Numeric
      strings such as `"1.5"` therefore produce the same metrics as the clean
      column; this was checked directly on Feast 0.67.0. Expected detection is
      close to the clean false-positive rate. Non-numeric strings would make the
      cast fail instead, but they are not injected.
    - **New value in `PULocationID`.** The baseline keeps only the top 10 values
      plus an `other` bucket, so `new_categories` can't prove a value is new and
      reports 0. Only `linf` can react, and only once the new value reaches the
      window's top 10. `payment_type` lists every value, so both checks apply
      to it.
    - **Stale source on its onset day.** The newest event is less than one day
      old at the next midnight, so freshness can't exceed the 1-day TTL. Only the
      later days of an outage can trip it.
