"""Build monitoring rows shaped like Feast v0.67.0's without importing Feast.

``numeric_row`` / ``categorical_row`` mirror
``feast/monitoring/metrics_calculator.py::compute_numeric / compute_categorical``
(20 numpy bins over the sample's own range; top-10 categories plus other_count).
``tests/test_feast_adapter.py`` checks them against the real MetricsCalculator.
"""

from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pytest

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _meta(feature, feature_type, rows, nulls, **extra):
    return {
        "feature_view_name": "fv",
        "feature_name": feature,
        "feature_type": feature_type,
        "granularity": "daily",
        "data_source_type": "batch",
        "metric_date": "2026-10-17",
        "max_event_timestamp": NOW.isoformat(),
        "row_count": rows,
        "null_count": nulls,
        "null_rate": nulls / rows if rows else 0.0,
        **extra,
    }


def numeric_row(values, nulls=0, feature="x", bins=20, **extra):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    hist = {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "bin_width": float(edges[1] - edges[0]),
    }
    return _meta(feature, "numeric", len(values) + nulls, nulls, histogram=hist, **extra)


def categorical_row(values, nulls=0, feature="c", top_n=10, **extra):
    ranked = sorted(Counter(values).items(), key=lambda kv: kv[1], reverse=True)
    hist = {
        "values": [{"value": v, "count": c} for v, c in ranked[:top_n]],
        "other_count": sum(c for _, c in ranked[top_n:]),
        "unique_count": len(ranked),
    }
    return _meta(feature, "categorical", len(values) + nulls, nulls, histogram=hist, **extra)


@pytest.fixture
def rng():
    return np.random.default_rng(7)
