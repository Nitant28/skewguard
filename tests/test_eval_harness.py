"""scripts/eval_harness.py: schedule, storage and guards; one synthetic end-to-end run."""

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_harness.py"


@pytest.fixture(scope="module")
def h():
    spec = importlib.util.spec_from_file_location("eval_harness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_harness"] = module
    spec.loader.exec_module(module)
    return module


MARCH = [date(2024, 3, 1) + timedelta(days=i) for i in range(31)]


def synthetic_events(h, first, last, per_day):
    rng = np.random.default_rng(first.toordinal())
    days = (last - first).days + 1
    n = per_day * days
    seconds = np.sort(rng.integers(0, 86400 * days, n))
    passengers = rng.integers(0, 6, n).astype(float)
    passengers[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame(
        {
            h.TS: (h._day_start(first) + pd.to_timedelta(seconds, unit="s")).astype(
                "datetime64[us, UTC]"
            ),
            "trip_distance": rng.gamma(1.5, 2.0, n),
            "fare_amount": rng.gamma(2.0, 8.0, n),
            "passenger_count": passengers,
            "payment_type": pd.array(rng.choice(["1", "2", "3", "4"], n), dtype="string"),
            "PULocationID": pd.array(rng.choice([str(i) for i in range(1, 40)], n), dtype="string"),
        }
    )


@pytest.mark.parametrize("fault", ["unit_change", "stale_source", "new_category"])
def test_incidents_are_30_seeded_days_with_valid_targets(h, fault):
    incidents = h.incidents_for(fault, MARCH)
    assert incidents == h.incidents_for(fault, MARCH)
    days = [i.onset_day for i in incidents]
    assert len(set(days)) == len(days) == h.DAYS_PER_CELL
    assert days == sorted(days) and set(days) <= set(MARCH)
    allowed = {
        "unit_change": set(h.NUMERIC),
        "new_category": set(h.CATEGORICAL),
        "stale_source": {None},
    }[fault]
    assert {i.target for i in incidents} <= allowed
    for i in incidents:
        assert i.onset.date() == i.onset_day
        if fault != "stale_source":
            assert i.onset == h._day_start(i.onset_day)


def test_faults_get_different_days(h):
    a = [i.onset_day for i in h.incidents_for("unit_change", MARCH)]
    b = [i.onset_day for i in h.incidents_for("null_burst", MARCH)]
    assert a != b


def test_incident_days_follow_gradual_and_stale_faults(h):
    last = MARCH[-1]
    early = h.Incident("gradual_shift", 0, MARCH[0], h._day_start(MARCH[0]), "fare_amount")
    days, planned = h.incident_days(early, "low", last)
    assert planned == h.GRADUAL_HORIZON and days == MARCH[: h.GRADUAL_HORIZON]
    late = h.Incident("gradual_shift", 0, MARCH[-3], h._day_start(MARCH[-3]), "fare_amount")
    assert h.incident_days(late, "high", last) == (MARCH[-3:], h.GRADUAL_HORIZON)

    noon = h._day_start(MARCH[0]) + timedelta(hours=12)
    stale = h.Incident("stale_source", 0, MARCH[0], noon, None)
    assert h.incident_days(stale, "low", last) == (MARCH[:2], 2)
    assert h.incident_days(stale, "high", last) == (MARCH[:8], 8)
    single = h.Incident("unit_change", 0, MARCH[5], h._day_start(MARCH[5]), "fare_amount")
    assert h.incident_days(single, "medium", last) == ([MARCH[5]], 1)


def test_cells_put_clean_windows_first(h):
    cells = h.cells([100, 300], ["unit_change", "null_burst"])
    assert cells[:2] == [("none", "none", 100), ("none", "none", 300)]
    assert len(cells) == 2 + 2 * 3 * 2


def test_arrow_columns_follow_parquet_typing(h):
    import pyarrow as pa

    num = h._arrow_column(pd.Series([1.5, np.nan]), numeric=True)
    assert num.type == pa.float64() and num.null_count == 1
    mixed = h._arrow_column(pd.Series([1.5, "2.5", np.nan], dtype=object), numeric=True)
    assert mixed.type == pa.string() and mixed.to_pylist() == ["1.5", "2.5", None]
    cat = h._arrow_column(pd.Series(["a", None], dtype="string"), numeric=False)
    assert cat.type == pa.string() and cat.to_pylist() == ["a", None]


def test_day_samples_are_seeded_and_stay_inside_the_day(h):
    events = synthetic_events(h, MARCH[0], MARCH[2], per_day=2000)
    index = h.DayIndex(events)
    sample = index.sample(MARCH[1], 300)
    assert len(sample) == 300
    assert (sample[h.TS].dt.date == MARCH[1]).all()
    assert sample[h.TS].is_monotonic_increasing
    fresh = h.DayIndex(events).sample(MARCH[1], 300)
    pd.testing.assert_frame_equal(sample, fresh)
    assert len(index.sample(MARCH[1], 10_000)) == len(index.rows(MARCH[1]))


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--range", "eval"],
        ["run", "--start", "2024-03-25", "--end", "2024-04-02"],
        ["run", "--start", "2024-02-20", "--end", "2024-03-02"],
    ],
)
def test_run_refuses_evaluation_and_baseline_dates(h, argv, tmp_path, monkeypatch):
    monkeypatch.setattr(h, "RESULTS", tmp_path / "results")
    with pytest.raises(SystemExit, match="evaluation range|after the baseline"):
        h.main(argv)
    assert not (tmp_path / "results").exists()


def test_windows_are_stamped_with_the_source_newest_event(h):
    windows = {f: {"max_event_timestamp": None, "row_count": 0} for f in h.FEATURES}
    newest = pd.Timestamp("2024-03-01 11:00", tz="UTC")
    assert h.stamp_newest_event(windows, newest) is None
    assert {r["max_event_timestamp"] for r in windows.values()} == {newest.isoformat()}
    assert h.newest_event(pd.DataFrame({h.TS: []})) is None


def _freshness(row):
    return json.loads(row["checks"])["fare_amount"]["freshness"][0]


@pytest.mark.integration
def test_synthetic_run_end_to_end_and_resume(h, tmp_path, monkeypatch, capsys):
    pytest.importorskip("feast")
    monkeypatch.setattr(h, "REPO", tmp_path / "repo")
    monkeypatch.setattr(h, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(
        h,
        "load_events",
        lambda months, first, last: synthetic_events(
            h, first, last, 3000 if first.month < 3 else 8000
        ),
    )
    argv = ["run", "--start", "2024-03-01", "--end", "2024-03-04", "--sizes", "300"]
    argv += ["--faults", "unit_change", "stale_source", "type_change"]
    h.main(argv)
    out = pd.read_parquet(tmp_path / "results" / "custom-20240301-20240304.parquet")

    assert set(out["detector"]) == set(h.DETECTORS)
    assert (out.groupby(["fault", "severity", "day", "incident"]).size() == 4).all()
    assert (out["feast_status"] == "ok").all()
    clean = out[out["fault"] == "none"]
    assert len(clean) == 4 * 4 and (clean["rows_window"] == 300).all()
    assert not out.loc[out["detector"] == "none", "alert"].any()

    sg = out[out["detector"] == "skewguard"]
    assert (sg.loc[sg["fault"] == "none"].apply(_freshness, axis=1) == "ok").all()
    # A 3 x TTL outage: fresh on the onset day, stale on the days fully inside it.
    stale = sg[(sg["fault"] == "stale_source") & (sg["severity"] == "medium")]
    assert len(stale) == 4 + 3 + 2 + 1
    assert (stale.loc[stale["day_offset"] == 0].apply(_freshness, axis=1) == "ok").all()
    inside = stale[stale["day_offset"].isin([1, 2])]
    assert (inside["rows_window"] == 0).all()
    assert (inside.apply(_freshness, axis=1) == "alert").all()
    assert inside["feast_newest_event"].isna().all()  # what compute_metrics itself stores

    capsys.readouterr()
    h.main(argv)  # every cell is done: nothing is recomputed
    assert "rows in" not in capsys.readouterr().out
