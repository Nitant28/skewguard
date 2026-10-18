import json
from datetime import timedelta

import pytest
from conftest import NOW, categorical_row, numeric_row

from skewguard.detector import Status, check_feature
from skewguard.policy import Policy

POLICY = Policy(reps=300, max_staleness=timedelta(hours=26))


def statuses(report):
    return {c.name: c.status for c in report.checks}


def test_clean_window_passes(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(0, 1, 300))
    report = check_feature(base, win, POLICY, NOW)
    assert report.status is Status.OK, report.checks


def test_unit_change_x100_alerts_on_range_and_shape(rng):
    base = numeric_row(rng.uniform(0, 1, 2000))
    win = numeric_row(rng.uniform(0, 1, 300) * 100)
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["out_of_range"] is Status.ALERT
    assert s["psi"] is Status.ALERT


def test_null_burst_alerts(rng):
    base = numeric_row(rng.normal(0, 1, 2000), nulls=20)
    win = numeric_row(rng.normal(0, 1, 210), nulls=90)
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["null_rate"] is Status.ALERT
    assert s["psi"] is Status.OK


def test_mean_shift_alerts_on_large_window(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(0.5, 1, 1000))
    assert statuses(check_feature(base, win, POLICY, NOW))["psi"] is Status.ALERT


def test_stale_source_alerts_on_freshness(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    old = (NOW - timedelta(days=3)).isoformat()
    win = numeric_row(rng.normal(0, 1, 300), max_event_timestamp=old)
    report = check_feature(base, win, POLICY, NOW)
    assert statuses(report)["freshness"] is Status.ALERT
    assert [c.name for c in report.alerts] == ["freshness"]


def test_freshness_skipped_without_policy(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(0, 1, 300))
    assert statuses(check_feature(base, win, Policy(reps=50), NOW))["freshness"] is Status.SKIP


def test_new_category_alerts():
    base = categorical_row(["a"] * 600 + ["b"] * 400)
    win = categorical_row(["a"] * 60 + ["b"] * 30 + ["NEW"] * 10)
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["new_categories"] is Status.ALERT


def test_category_frequency_shift_alerts():
    base = categorical_row(["a"] * 500 + ["b"] * 500)
    win = categorical_row(["a"] * 350 + ["b"] * 150)
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["linf"] is Status.ALERT
    assert s["new_categories"] is Status.OK


def test_clean_categorical_window_passes(rng):
    pool = ["a"] * 5 + ["b"] * 3 + ["c"] * 2
    base = categorical_row(list(rng.choice(pool, 3000)))
    win = categorical_row(list(rng.choice(pool, 400)))
    assert check_feature(base, win, POLICY, NOW).status is Status.OK


def test_type_change_alerts(rng):
    base = numeric_row(rng.normal(0, 1, 100))
    win = categorical_row(["1.0", "2.0"] * 50)
    report = check_feature(base, win, POLICY, NOW)
    assert statuses(report)["type"] is Status.ALERT


def test_missing_baseline_is_a_state_not_a_crash(rng):
    win = numeric_row(rng.normal(0, 1, 300))
    report = check_feature(None, win, Policy(reps=50), NOW)
    assert report.status is Status.SKIP
    assert any(c.name == "baseline" for c in report.checks)


def test_small_window_skips_distribution_checks(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(5, 1, 10))
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["distribution"] is Status.SKIP


def test_all_null_window_still_checks_null_rate(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row([], nulls=100) | {"histogram": None}
    s = statuses(check_feature(base, win, POLICY, NOW))
    assert s["null_rate"] is Status.ALERT
    assert s["distribution"] is Status.SKIP


def test_fixed_threshold_mode_uses_given_limit(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(0, 1, 300))
    report = check_feature(base, win, Policy(fixed_threshold=0.25), NOW)
    psi_check = next(c for c in report.checks if c.name == "psi")
    assert psi_check.threshold == 0.25


def test_js_statistic_option(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(1.0, 1, 500))
    report = check_feature(base, win, Policy(statistic="js", reps=200), NOW)
    assert statuses(report)["js"] is Status.ALERT


def test_histogram_as_json_string_is_accepted(rng):
    base = numeric_row(rng.normal(0, 1, 2000))
    win = numeric_row(rng.normal(0, 1, 300))
    win["histogram"] = json.dumps(win["histogram"])
    assert check_feature(base, win, POLICY, NOW).status is Status.OK


def test_report_serializes(rng):
    base = numeric_row(rng.normal(0, 1, 500))
    win = numeric_row(rng.normal(0, 1, 100))
    d = check_feature(base, win, POLICY, NOW).to_dict()
    assert d["feature"] == "x" and d["status"] in {"ok", "alert", "skip"}
    json.dumps(d)


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_policy_rejects_bad_false_positive_rate(bad):
    with pytest.raises(ValueError):
        Policy(false_positive_rate=bad)
