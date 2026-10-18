from skewguard.feast_adapter import pair_rows


def _row(view, feat, date, computed="t0", **extra):
    return {
        "feature_view_name": view,
        "feature_name": feat,
        "metric_date": date,
        "computed_at": computed,
        **extra,
    }


def test_pair_rows_takes_latest_window_and_baseline():
    baselines = [_row("v", "a", "2026-10-01"), _row("v", "a", "2026-10-10", tag="new")]
    windows = [
        _row("v", "a", "2026-10-16"),
        _row("v", "a", "2026-10-17", tag="latest"),
        _row("v", "b", "2026-10-17"),
    ]
    pairs = pair_rows(baselines, windows)
    assert [(b and b.get("tag"), w["feature_name"], w.get("tag")) for b, w in pairs] == [
        ("new", "a", "latest"),
        (None, "b", None),
    ]


def test_pair_rows_breaks_same_day_ties_by_computed_at():
    windows = [_row("v", "a", "2026-10-17", "t1"), _row("v", "a", "2026-10-17", "t2", tag="rerun")]
    assert pair_rows([], windows)[0][1]["tag"] == "rerun"
