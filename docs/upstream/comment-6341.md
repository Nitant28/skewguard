<!--
DRAFT comment for https://github.com/feast-dev/feast/issues/6341
("Built-in feature drift detection with alerting"; open, unassigned as of 2026-10-18).
Not posted. Post only after the skewguard repo is public, and replace <repo-url>.
Keep it about evidence and design; don't ask to be assigned in the first comment.
-->

I've been prototyping drift detection on top of the stored monitoring metrics
(`feast monitor run`, v0.67.0) and hit two issues that any built-in version
will need to handle. Sharing measurements in case they're useful for the design.

**1. Stored histograms can't be compared bin by bin.** Each window calls
`np.histogram(values, bins=20)` over its own min/max (`metrics_calculator.py:114`,
`dask.py:856`). On the `feast init` demo, conv_rate's baseline edges start at
0.0002 and the daily window's at 0.0167. Comparing bins means first rebinning
the window onto the baseline's edges. I do this assuming values are uniform
within each bin, and I keep mass outside the baseline range as its own bin; a
x100 unit bug shows up there clearly.

**2. A fixed PSI threshold depends on window size.** Clean data, 20 bins,
1,808-row baseline (the demo's size), 2,000 independent pairs per row:

| window rows | median PSI | P(PSI > 0.2) | P(PSI > 0.25) |
|---|---|---|---|
| 116 | 0.153 | 23.2% | 8.6% |
| 1,000 | 0.029 | 0.0% | 0.0% |
| 5,000 | 0.015 | 0.0% | 0.0% |

So "PSI > 0.2" (suggested above) alerts on about 1 in 4 clean daily windows of
~100 rows. At 1,000+ rows it is more than 5x the noise and misses real shifts.
What has worked for me is simulating the no-drift distribution for the actual
window size from the baseline histogram and alerting above its 99th percentile.
That measured 0.4–1.1% false positives against a 1% target, from 116 to 5,000
rows, averaged over 50 baselines. The simulation never materialises rows (it
samples bin counts plus the sample's min and max), so it costs about 50 ms per
feature whatever the window size, and it can be cached per baseline.

Two smaller things I found along the way:
- The baseline covers all history, so it includes the window being compared.
  That biases distances low. A baseline range that excludes recent data would help.
- The histogram format differs between backends for constant columns (filed
  separately as #<issue>).

Prototype with tests and the reproduction scripts: <repo-url>. It reads metrics
only through `MonitoringService`, so it could become a module under
`feast/monitoring/` or stay external. Which direction would maintainers prefer?
I'm happy to write a short design doc first.
