"""The only module that knows Feast: read monitoring rows and pair them up.

Built against Feast v0.67.0, where ``feast monitor run`` stores per-feature rows
keyed by (project, feature_view_name, feature_name, metric_date, granularity,
data_source_type) and marks one batch row per feature as ``is_baseline``
(``feast/monitoring/monitoring_service.py``).

* drift: batch baseline vs. latest batch window (``source="batch"``)
* training-serving skew: batch baseline vs. latest serving-log window
  (``source="log"``); log rows are already mapped back to the originating
  feature view and feature name by Feast's ``_save_log_metrics``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from skewguard.detector import FeatureReport, check_feature
from skewguard.policy import PolicySet

Row = Mapping[str, Any]


def _key(row: Row) -> tuple[str, str]:
    return str(row["feature_view_name"]), str(row["feature_name"])


def _recency(row: Row) -> tuple[str, str]:
    return str(row.get("metric_date") or ""), str(row.get("computed_at") or "")


def pair_rows(baselines: Iterable[Row], windows: Iterable[Row]) -> list[tuple[Row | None, Row]]:
    """Pair each feature's latest window with its latest baseline row.

    Feast keeps only one row per feature flagged ``is_baseline`` (recomputing
    clears the old flag), but picking the newest guards against stale copies.
    """
    latest_base: dict[tuple[str, str], Row] = {}
    for row in baselines:
        k = _key(row)
        if k not in latest_base or _recency(row) > _recency(latest_base[k]):
            latest_base[k] = row

    latest_win: dict[tuple[str, str], Row] = {}
    for row in windows:
        k = _key(row)
        if k not in latest_win or _recency(row) > _recency(latest_win[k]):
            latest_win[k] = row

    return [(latest_base.get(k), latest_win[k]) for k in sorted(latest_win)]


def collect_pairs(
    store: Any,
    granularity: str = "daily",
    source: str = "batch",
    feature_view: str | None = None,
) -> list[tuple[Row | None, Row]]:
    """Read baseline and window rows for ``store`` (a ``feast.FeatureStore``)."""
    from feast.monitoring.monitoring_service import MonitoringService

    if source not in ("batch", "log"):
        raise ValueError("source must be 'batch' or 'log'")
    svc = MonitoringService(store)
    baselines = svc.get_baseline(
        project=store.project, feature_view_name=feature_view, data_source_type="batch"
    )
    windows = svc.get_feature_metrics(
        project=store.project,
        feature_view_name=feature_view,
        granularity=granularity,
        data_source_type=source,
    )
    return pair_rows(baselines, windows)


def run(
    repo_path: str = ".",
    granularity: str = "daily",
    source: str = "batch",
    feature_view: str | None = None,
    policies: PolicySet | None = None,
    now: datetime | None = None,
) -> list[FeatureReport]:
    """Check every monitored feature in a Feast repo."""
    from feast import FeatureStore

    policies = policies or PolicySet()
    store = FeatureStore(repo_path=repo_path)
    return [
        check_feature(base, win, policies.for_feature(*_key(win)), now)
        for base, win in collect_pairs(store, granularity, source, feature_view)
    ]
