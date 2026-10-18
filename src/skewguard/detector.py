"""Compare one feature's window metrics against its baseline.

Inputs are Feast monitoring rows: the dicts returned by
``MonitoringService.get_baseline`` / ``get_feature_metrics``, with the columns
listed in ``feast/monitoring/monitoring_utils.py::FEATURE_METRICS_COLUMNS``.
Nothing here imports Feast, so every check can be tested with plain dicts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from skewguard.policy import Policy
from skewguard.stats import (
    TopN,
    compare_numeric,
    linf_bounds,
    new_category_mass,
    null_rate_pvalue,
)
from skewguard.thresholds import categorical_null, numeric_null


class Status(str, Enum):
    OK = "ok"
    ALERT = "alert"
    SKIP = "skip"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    value: float | None = None
    threshold: float | None = None
    note: str = ""


@dataclass(frozen=True)
class FeatureReport:
    feature_view: str
    feature: str
    granularity: str
    metric_date: str | None
    source: str
    rows: int
    checks: tuple[Check, ...]

    @property
    def status(self) -> Status:
        statuses = {c.status for c in self.checks}
        if Status.ALERT in statuses:
            return Status.ALERT
        return Status.OK if Status.OK in statuses else Status.SKIP

    @property
    def alerts(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status is Status.ALERT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_view": self.feature_view,
            "feature": self.feature,
            "granularity": self.granularity,
            "metric_date": self.metric_date,
            "source": self.source,
            "rows": self.rows,
            "status": self.status.value,
            "checks": [{**c.__dict__, "status": c.status.value} for c in self.checks],
        }


def _histogram(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    h = row.get("histogram")
    if isinstance(h, str):
        h = json.loads(h)
    return h or None


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _date_str(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, (date, datetime)) else str(value)


def _non_null(row: Mapping[str, Any]) -> int:
    return int(row.get("row_count") or 0) - int(row.get("null_count") or 0)


def check_freshness(window: Mapping[str, Any], policy: Policy, now: datetime) -> Check:
    if policy.max_staleness is None:
        return Check("freshness", Status.SKIP, note="no max_staleness set")
    newest = _as_datetime(window.get("max_event_timestamp"))
    if newest is None:
        return Check("freshness", Status.SKIP, note="no max_event_timestamp")
    age = (now - newest).total_seconds()
    limit = policy.max_staleness.total_seconds()
    status = Status.ALERT if age > limit else Status.OK
    return Check("freshness", status, age, limit, f"newest event {age / 3600:.1f}h old")


def check_null_rate(
    baseline: Mapping[str, Any], window: Mapping[str, Any], policy: Policy
) -> Check:
    b_rows, w_rows = int(baseline.get("row_count") or 0), int(window.get("row_count") or 0)
    b_nulls, w_nulls = int(baseline.get("null_count") or 0), int(window.get("null_count") or 0)
    if w_rows == 0:
        return Check("null_rate", Status.SKIP, note="empty window")
    p = null_rate_pvalue(b_nulls, b_rows, w_nulls, w_rows)
    status = Status.ALERT if p < policy.false_positive_rate else Status.OK
    b_rate = b_nulls / b_rows if b_rows else 0.0
    note = f"null rate {b_rate:.2%} -> {w_nulls / w_rows:.2%}, p={p:.2g}"
    return Check("null_rate", status, p, policy.false_positive_rate, note)


def _numeric_checks(
    base_hist: Mapping[str, Any], win_hist: Mapping[str, Any], n_window: int, policy: Policy
) -> list[Check]:
    psi_value, js_value, r = compare_numeric(
        base_hist["bins"], base_hist["counts"], win_hist["bins"], win_hist["counts"]
    )
    unseen = r.out_of_range / r.total if r.total else 0.0
    checks = [
        Check(
            "out_of_range",
            Status.ALERT if unseen > policy.max_unseen_mass else Status.OK,
            unseen,
            policy.max_unseen_mass,
            f"{r.below / r.total:.1%} below, {r.above / r.total:.1%} above baseline range",
        )
    ]
    value = psi_value if policy.statistic == "psi" else js_value
    if policy.fixed_threshold is not None:
        threshold, note = policy.fixed_threshold, "fixed threshold"
    else:
        null_psi, null_js = numeric_null(
            base_hist["bins"], base_hist["counts"], n_window, reps=policy.reps, seed=policy.seed
        )
        null = null_psi if policy.statistic == "psi" else null_js
        threshold = null.quantile(1 - policy.false_positive_rate)
        note = f"p={null.pvalue(value):.3g} vs {policy.reps} no-drift draws of {n_window} rows"
    status = Status.ALERT if value > threshold else Status.OK
    checks.append(Check(policy.statistic, status, value, threshold, note))
    return checks


def _categorical_checks(
    base_hist: Mapping[str, Any], win_hist: Mapping[str, Any], n_window: int, policy: Policy
) -> list[Check]:
    base, win = TopN.from_feast(base_hist), TopN.from_feast(win_hist)
    unseen = new_category_mass(base, win)
    lower, upper = linf_bounds(base, win)
    checks = [
        Check(
            "new_categories",
            Status.ALERT if unseen > policy.max_unseen_mass else Status.OK,
            unseen,
            policy.max_unseen_mass,
            "share of window in values the baseline never had",
        )
    ]
    if policy.fixed_threshold is not None:
        threshold, note = policy.fixed_threshold, "fixed threshold"
    else:
        null = categorical_null(base, n_window, reps=policy.reps, seed=policy.seed)
        threshold = null.quantile(1 - policy.false_positive_rate)
        note = f"p={null.pvalue(lower):.3g}; distance in [{lower:.3f}, {upper:.3f}]"
    status = Status.ALERT if lower > threshold else Status.OK
    checks.append(Check("linf", status, lower, threshold, note))
    return checks


def check_feature(
    baseline: Mapping[str, Any] | None,
    window: Mapping[str, Any],
    policy: Policy | None = None,
    now: datetime | None = None,
) -> FeatureReport:
    """Run every applicable check for one feature and window."""
    policy = policy or Policy()
    now = now or datetime.now(timezone.utc)
    checks = [check_freshness(window, policy, now)]

    if baseline is None:
        checks.append(Check("baseline", Status.SKIP, note="no baseline yet"))
    elif baseline.get("feature_type") != window.get("feature_type"):
        checks.append(
            Check(
                "type",
                Status.ALERT,
                note=f"{baseline.get('feature_type')} -> {window.get('feature_type')}",
            )
        )
    else:
        checks.append(check_null_rate(baseline, window, policy))
        base_hist, win_hist = _histogram(baseline), _histogram(window)
        n_window = _non_null(window)
        if base_hist is None or win_hist is None:
            checks.append(Check("distribution", Status.SKIP, note="no non-null values"))
        elif n_window < policy.min_rows:
            checks.append(
                Check(
                    "distribution",
                    Status.SKIP,
                    note=f"{n_window} rows < min_rows {policy.min_rows}",
                )
            )
        elif window.get("feature_type") == "categorical":
            checks.extend(_categorical_checks(base_hist, win_hist, n_window, policy))
        else:
            checks.extend(_numeric_checks(base_hist, win_hist, n_window, policy))

    return FeatureReport(
        feature_view=str(window.get("feature_view_name")),
        feature=str(window.get("feature_name")),
        granularity=str(window.get("granularity")),
        metric_date=_date_str(window.get("metric_date")),
        source=str(window.get("data_source_type")),
        rows=int(window.get("row_count") or 0),
        checks=tuple(checks),
    )
