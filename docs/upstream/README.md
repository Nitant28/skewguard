# Upstream plan (feast-dev/feast)

Everything here is a draft. Nothing has been posted. Recheck issue status before
each step; it was last checked on 2026-10-18.

| # | Step | Status | Artifact |
|---|---|---|---|
| 1 | File the histogram-consistency bug | draft ready | [issue-histogram-consistency.md](issue-histogram-consistency.md) |
| 2 | After a maintainer replies, open the fix PR (constant columns) | patch ready, tested locally | [feast-constant-histogram.patch](feast-constant-histogram.patch), local branch `fix/monitoring-constant-histogram` in `../feast-src` |
| 3 | Comment on #6341 (drift detection) with evidence and a design question | draft ready; needs the repo URL | [comment-6341.md](comment-6341.md) |
| 4 | Test PR #7002 (online TTL → OUTSIDE_MAX_AGE) using `scripts/repro_stale_online_read.py` and report results | not started | issue #6821 / PR #7002 belong to someone else; review only |

## The patch (step 2)

- `monitoring_utils.numeric_histogram`: one helper for both Python paths. A
  constant column becomes `{"bins": [v, v], "counts": [n], "bin_width": 0.0}`,
  matching the SQL backends.
- `metrics_calculator.py` and `dask.py` call it. Two now-unused `numpy` imports are removed.
- Tests: histogram asserts in both single-value tests, a constant-column test,
  and a parametrized `test_numeric_histogram_matches_across_backends` (golden
  data and constant column).
- Docs: one paragraph in `docs/how-to-guides/feature-monitoring.md`.
- Verified on 2026-10-18:
  - `tests/unit/monitoring`: 69 passed with the patch, 66 before (+3 new). The
    same 10 failures happen either way, in BigQuery, Redshift and Spark tests,
    because their cloud SDKs aren't installed locally.
  - Without the source change, 4 of the new or extended tests fail.
  - `ruff check`, `ruff format --check` and `mypy` pass on every changed file.

Opening the PR (Feast requires DCO sign-off and conventional-commit titles):

```bash
cd feast-src
git commit -s -m "fix: Store constant-column monitoring histograms consistently across backends"
# fork feast-dev/feast on GitHub, then:
git remote add fork https://github.com/<you>/feast.git
git push fork fix/monitoring-constant-histogram
```

PR title: `fix: Store constant-column monitoring histograms consistently across backends`.
Body: follow `.github/pull_request_template.md`, with `Fixes #<issue from step 1>`.
