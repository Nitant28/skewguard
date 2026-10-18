from datetime import timedelta

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from skewguard.inject import (  # noqa: E402
    FAULTS,
    NEW_CATEGORY,
    SEVERITIES,
    SEVERITY_LEVELS,
    applicable_columns,
    gradual_shift,
    inject,
    new_category,
    null_burst,
    stale_source,
    timezone_shift,
    type_change,
    unit_change,
)

TS = "ts"
START = pd.Timestamp("2024-03-01", tz="UTC")
ONSET = pd.Timestamp("2024-03-02", tz="UTC")


@pytest.fixture
def events():
    """Two days of events, one per minute, with a few nulls in each column."""
    n = 2 * 24 * 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            TS: START + pd.to_timedelta(np.arange(n), unit="min"),
            "fare": rng.gamma(2.0, 7.0, n),
            "count": rng.integers(0, 6, n),
            "kind": rng.choice(["1", "2", "3", "4"], n).astype(object),
        }
    )
    df.loc[::97, "fare"] = np.nan
    df.loc[::89, "kind"] = None
    return df


def after(df):
    return df[TS] >= ONSET


def test_plan_table_is_restated_exactly():
    assert SEVERITIES == {
        "unit_change": {"low": 1.1, "medium": 10, "high": 100},
        "null_burst": {"low": 0.02, "medium": 0.10, "high": 0.50},
        "stale_source": {"low": 1, "medium": 3, "high": 7},
        "new_category": {"low": 0.01, "medium": 0.05, "high": 0.25},
        "timezone_shift": {"low": 1, "medium": 5, "high": 12},
        "gradual_shift": {"low": 0.02, "medium": 0.05, "high": 0.1},
        "type_change": {"low": 0.01, "medium": 0.10, "high": 1.0},
    }


def _inject(df, fault, severity, seed=0):
    column = {
        "unit_change": "fare",
        "null_burst": "kind",
        "new_category": "kind",
        "gradual_shift": "fare",
        "type_change": "fare",
    }.get(fault)
    return inject(
        df,
        TS,
        fault,
        severity,
        column=column,
        seed=seed,
        onset=ONSET,
        ttl=timedelta(hours=1),
        baseline_sd=10.0,
    )


@pytest.mark.parametrize("severity", SEVERITY_LEVELS)
@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_leaves_input_untouched_and_is_deterministic(events, fault, severity):
    before = events.copy(deep=True)
    out, label = _inject(events, fault, severity, seed=3)
    pd.testing.assert_frame_equal(events, before)
    assert out is not events
    again, label_again = _inject(events, fault, severity, seed=3)
    pd.testing.assert_frame_equal(out, again)
    assert label == label_again
    assert (label.fault, label.severity) == (fault, severity)
    assert label.magnitude == SEVERITIES[fault][severity]
    assert label.rows_affected > 0
    assert label.to_dict()["onset"] == ONSET.isoformat()


@pytest.mark.parametrize("fault", FAULTS)
def test_rows_before_onset_are_untouched(events, fault):
    out, _ = _inject(events, fault, "high")
    early = events[~after(events)]
    pd.testing.assert_frame_equal(out.loc[early.index].astype(object), early.astype(object))


def test_unit_change_multiplies_only_the_target_column(events):
    out, label = unit_change(events, TS, "fare", 10.0, onset=ONSET)
    post = after(events)
    np.testing.assert_allclose(out.loc[post, "fare"], events.loc[post, "fare"] * 10)
    pd.testing.assert_frame_equal(out.drop(columns="fare"), events.drop(columns="fare"))
    assert out["fare"].isna().equals(events["fare"].isna())
    assert label.rows_affected == events.loc[post, "fare"].notna().sum()


def test_unit_change_on_integers_returns_floats(events):
    out, _ = unit_change(events, TS, "count", 1.1)
    np.testing.assert_allclose(out["count"], events["count"] * 1.1)


@pytest.mark.parametrize("share", [0.02, 0.10, 0.50])
@pytest.mark.parametrize("column", ["fare", "kind", "count"])
def test_null_burst_raises_null_rate_by_the_stated_points(events, column, share):
    out, label = null_burst(events, TS, column, share, seed=1, onset=ONSET)
    post = after(events)
    rise = out.loc[post, column].isna().mean() - events.loc[post, column].isna().mean()
    assert rise == pytest.approx(share, abs=0.5 / post.sum())
    assert label.rows_affected == round(share * post.sum())
    pd.testing.assert_frame_equal(out.drop(columns=column), events.drop(columns=column))
    assert out.loc[~post, column].isna().equals(events.loc[~post, column].isna())


def test_null_burst_seed_changes_which_rows(events):
    a, _ = null_burst(events, TS, "fare", 0.1, seed=1)
    b, _ = null_burst(events, TS, "fare", 0.1, seed=2)
    assert not a["fare"].isna().equals(b["fare"].isna())


def test_null_burst_caps_at_the_non_null_rows(events):
    out, label = null_burst(events, TS, "fare", 1.0, seed=0)
    assert out["fare"].isna().all()
    assert label.rows_affected == events["fare"].notna().sum()


@pytest.mark.parametrize("multiple", [1, 3, 7])
def test_stale_source_drops_events_inside_the_outage(events, multiple):
    ttl = timedelta(hours=2)
    out, label = stale_source(events, TS, ONSET, ttl, multiple)
    end = ONSET + ttl * multiple
    gone = (events[TS] >= ONSET) & (events[TS] < end)
    assert not ((out[TS] >= ONSET) & (out[TS] < end)).any()
    pd.testing.assert_frame_equal(out, events[~gone])
    assert label.rows_affected == gone.sum() == multiple * 120
    assert (label.onset, label.end) == (ONSET, end)


def test_stale_source_accepts_naive_cutoff_for_utc_column(events):
    out, _ = stale_source(events, TS, "2024-03-02 00:00", timedelta(hours=1), 1)
    assert len(out) == len(events) - 60


@pytest.mark.parametrize("share", [0.01, 0.05, 0.25])
def test_new_category_recodes_the_stated_share(events, share):
    out, label = new_category(events, TS, "kind", share, seed=0, onset=ONSET)
    post = after(events)
    new = out["kind"] == NEW_CATEGORY
    assert not new[~post].any()
    assert new[post].mean() == pytest.approx(share, abs=0.5 / post.sum())
    assert label.rows_affected == new.sum()
    assert (out.loc[~new, "kind"].fillna("-") == events.loc[~new, "kind"].fillna("-")).all()


def test_categorical_faults_keep_arrow_string_columns(events):
    events = events.assign(kind=events["kind"].astype("string"))
    burst, _ = null_burst(events, TS, "kind", 0.5, seed=0)
    recoded, _ = new_category(events, TS, "kind", 0.5, seed=0)
    assert burst["kind"].dtype == recoded["kind"].dtype == events["kind"].dtype
    assert burst["kind"].isna().mean() == pytest.approx(events["kind"].isna().mean() + 0.5, 1e-3)
    assert (recoded["kind"] == NEW_CATEGORY).sum() / len(events) == pytest.approx(0.5, 1e-3)


def test_new_category_rejects_numeric_or_existing_values(events):
    with pytest.raises(ValueError, match="numeric"):
        new_category(events, TS, "fare", 0.1, seed=0)
    with pytest.raises(ValueError, match="already occurs"):
        new_category(events, TS, "kind", 0.1, seed=0, value="1")


@pytest.mark.parametrize("hours", [1, 5, 12])
def test_timezone_shift_moves_only_timestamps_after_onset(events, hours):
    out, label = timezone_shift(events, TS, hours, onset=ONSET)
    post = after(events)
    delta = (out[TS] - events[TS]).dt.total_seconds()
    assert (delta[post] == hours * 3600).all()
    assert (delta[~post] == 0).all()
    pd.testing.assert_frame_equal(out.drop(columns=TS), events.drop(columns=TS))
    assert label.rows_affected == post.sum()


def test_gradual_shift_grows_by_rate_sd_per_day(events):
    out, _ = gradual_shift(events, TS, "fare", 0.05, 10.0, onset=START)
    delta = (out["fare"] - events["fare"]).to_numpy()
    days = (events[TS] - START).dt.total_seconds().to_numpy() / 86400
    ok = ~np.isnan(delta)
    np.testing.assert_allclose(delta[ok], 0.05 * 10.0 * days[ok])
    assert out["fare"].isna().equals(events["fare"].isna())
    pd.testing.assert_frame_equal(out.drop(columns="fare"), events.drop(columns="fare"))


@pytest.mark.parametrize("share", [0.01, 0.10, 1.0])
def test_type_change_stringifies_the_stated_share(events, share):
    out, label = type_change(events, TS, "fare", share, seed=0, onset=ONSET)
    post = after(events)
    is_str = out["fare"].map(lambda v: isinstance(v, str))
    assert not is_str[~post].any()
    if share == 1.0:
        assert is_str[post].sum() == events.loc[post, "fare"].notna().sum()
    else:
        assert is_str[post].mean() == pytest.approx(share, abs=0.5 / post.sum())
    assert label.rows_affected == is_str.sum()
    parsed = pd.to_numeric(out["fare"], errors="raise")
    pd.testing.assert_series_equal(parsed, events["fare"], check_dtype=False)


def test_inject_validates_arguments(events):
    with pytest.raises(ValueError, match="unknown fault"):
        inject(events, TS, "bit_flip", "low", column="fare")
    with pytest.raises(ValueError, match="unknown severity"):
        inject(events, TS, "unit_change", "extreme", column="fare")
    with pytest.raises(ValueError, match="target column"):
        inject(events, TS, "unit_change", "low")
    with pytest.raises(ValueError, match="ttl"):
        inject(events, TS, "stale_source", "low")
    with pytest.raises(ValueError, match="baseline_sd"):
        inject(events, TS, "gradual_shift", "low", column="fare")


def test_applicable_columns():
    num, cat = ["a", "b"], ["c"]
    assert applicable_columns("unit_change", num, cat) == ["a", "b"]
    assert applicable_columns("new_category", num, cat) == ["c"]
    assert applicable_columns("null_burst", num, cat) == ["a", "b", "c"]
    assert applicable_columns("stale_source", num, cat) == []
