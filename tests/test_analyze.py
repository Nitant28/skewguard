"""scripts/analyze.py: the Deviation 10 rules on synthetic results with known answers."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"


@pytest.fixture(scope="module")
def a():
    spec = importlib.util.spec_from_file_location("analyze", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze"] = module
    spec.loader.exec_module(module)
    return module


def _checks(psi=0.1, oor=0.0, psi_threshold=0.25, status=None):
    """One numeric feature's stored checks: [status, value, threshold]."""
    psi_status = status or ("alert" if psi > psi_threshold else "ok")
    oor_status = "alert" if oor > 0.01 else "ok"
    return {
        "freshness": ["ok", 3600.0, 86400.0],
        "null_rate": ["ok", 0.5, 0.01],
        "out_of_range": [oor_status, oor, 0.01],
        "psi": [psi_status, psi, psi_threshold],
    }


def _row(
    detector,
    fault="none",
    size=100,
    day=0,
    incident=-1,
    offset=0,
    target="",
    checks=None,
    severity="none",
):
    checks = checks or {}
    alerts = [
        f"{f}:{n}"
        for f, cs in checks.items()
        for n, v in cs.items()
        if isinstance(v, list) and v[0] == "alert"
    ]
    return {
        "run": "t",
        "day": f"2024-03-{day + 1:02d}",
        "onset_day": "",
        "day_offset": offset,
        "incident_days": 1,
        "incident": incident,
        "fault": fault,
        "severity": severity,
        "magnitude": 1.0,
        "window_size": size,
        "target": target,
        "rows_sampled": size,
        "rows_window": size,
        "feast_status": "ok",
        "feast_ms": 1.0,
        "newest_event": "",
        "feast_newest_event": "",
        "label": "",
        "detector": detector,
        "alert": bool(alerts),
        "alerts": ",".join(alerts),
        "n_features_alerting": len({x.split(":")[0] for x in alerts}),
        "checks": json.dumps(checks) if checks else "",
        "runtime_ms": 1.0,
        "feature_runtime_ms": [0.5, 0.5],
        "null_cache_misses": 0,
        "target_alert": (any(x.startswith(target + ":") for x in alerts) if target else None),
    }


def _march(sizes=(100, 1000)):
    """30 clean days per size. Out-of-range values put the 1% rule at 0.005;
    skewguard alerts on 3 of 30 days, fixed PSI ratios are spread so k is unique."""
    rows = []
    for s in sizes:
        for d in range(30):
            oor = 0.02 if d == 0 else (0.004 if d < 3 else 0.0)  # >0.005 only on day 0
            sg = {"x": _checks(psi=0.1, oor=oor, psi_threshold=0.2 if d >= 3 else 0.05)}
            fx = {"x": _checks(psi=0.01 * (d + 1), oor=oor, psi_threshold=0.25)}
            rows.append(_row("skewguard", size=s, day=d, checks=sg))
            rows.append(_row("fixed", size=s, day=d, checks=fx))
    return pd.DataFrame(rows)


def test_tune_picks_smallest_unseen_threshold_meeting_one_percent(a):
    df = _march()
    t = a.tune(df, a.check_table(df))
    # 1 of 30 clean windows exceeds 0.005 (3.3%) but none exceeds 0.02 -> 0.02 qualifies,
    # 0.01 does not (the 0.02 value exceeds it on day 0).
    assert t["max_unseen_mass"] == 0.02
    assert t["unseen_clean_rates"]["0.01"]["out_of_range"]["100"] == pytest.approx(1 / 30)


def test_tune_matches_fixed_fpr_to_skewguard(a):
    df = _march()
    t = a.tune(df, a.check_table(df))
    # skewguard alerts (PSI 0.1 > 0.05) on days 0-2: day FPR 10%
    assert t["skewguard_day_fpr"]["100"] == pytest.approx(0.1)
    # fixed ratios are 0.04 * (d + 1); alerting on exactly 3 of 30 days needs
    # 0.04 * 27 <= k < 0.04 * 28, and ties go to the larger k
    k = t["fixed_matched_factor"]["100"]
    assert 1.08 <= k < 1.12
    assert t["fixed_matched_day_fpr"]["100"] == pytest.approx(0.1)


def test_alerting_rethresholds_only_requested_checks(a):
    df = pd.DataFrame([_row("fixed", checks={"x": _checks(psi=0.3, oor=0.02)})])
    c = a.check_table(df)
    base = a.alerting(c).set_index("check")["alert"]
    assert base["psi"] and base["out_of_range"]
    loose = a.alerting(c, unseen=0.05, scale=2.0).set_index("check")["alert"]
    assert not loose["psi"] and not loose["out_of_range"]
    assert not loose["freshness"] and not loose["null_rate"]  # stored status kept


def test_mcnemar_and_holm_known_values(a):
    assert a.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert a.mcnemar_exact(5, 5) == 1.0
    assert a.mcnemar_exact(0, 0) == 1.0
    assert a.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_intervals_cover_the_mean(a):
    lo, hi = a.bootstrap_ci([1] * 20 + [0] * 10)
    assert lo < 2 / 3 < hi
    assert a.bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert all(math.isnan(v) for v in a.bootstrap_ci([]))
    x = np.array([1, 1, 1, 0] * 10, dtype=bool)
    lo, hi = a.paired_diff_ci(x, np.zeros(40, dtype=bool), np.array(["s"] * 40))
    assert lo > 0 and hi <= 1


def test_incidents_any_day_detects_and_records_first_offset(a):
    hit = {"x": _checks(psi=0.9)}
    rows = [
        _row(
            "skewguard",
            "gradual_shift",
            1000,
            0,
            incident=0,
            offset=0,
            target="x",
            severity="low",
            checks={"x": _checks()},
        ),
        _row(
            "skewguard",
            "gradual_shift",
            1000,
            1,
            incident=0,
            offset=1,
            target="x",
            severity="low",
            checks=hit,
        ),
        _row(
            "skewguard",
            "gradual_shift",
            1000,
            0,
            incident=1,
            offset=0,
            target="x",
            severity="low",
            checks={"x": _checks()},
        ),
    ]
    inc = a.incidents(pd.DataFrame(rows)).set_index("incident")
    assert inc.loc[0, "detected"] and inc.loc[0, "first_offset"] == 1
    assert inc.loc[0, "target_detected"]
    assert not inc.loc[1, "detected"] and math.isnan(inc.loc[1, "first_offset"])


def _eval_run():
    rows = []
    for s in (100, 1000):
        for d in range(10):  # clean days
            rows.append(
                _row("fixed", size=s, day=d, checks={"x": _checks(psi=0.3 if d == 0 else 0.1)})
            )
            rows.append(
                _row("skewguard", size=s, day=d, checks={"x": _checks(psi=0.1, psi_threshold=0.4)})
            )
            rows.append(_row("none", size=s, day=d))
        for i in range(30):  # skewguard catches all, fixed catches a third
            fx = {"x": _checks(psi=0.3 if i % 3 == 0 else 0.1)}
            sg = {"x": _checks(psi=0.5, psi_threshold=0.4)}
            for det, chk in (("fixed", fx), ("skewguard", sg), ("none", None)):
                rows.append(
                    _row(
                        det,
                        "gradual_shift",
                        s,
                        i % 10,
                        incident=i,
                        target="x",
                        severity="medium",
                        checks=chk,
                    )
                )
    return pd.DataFrame(rows)


def test_report_renders_every_section_and_verdict(a):
    df = _eval_run()
    checks = a.check_table(df)
    tuning = {"max_unseen_mass": 0.01, "fixed_matched_factor": {"100": 1.0, "1000": 1.0}}
    text = a.report(df, checks, {"config": {"range": "eval", "max_unseen_mass": 0.01}}, 0, tuning)
    for heading in (
        "## H2",
        "## False-positive rate",
        "## Detection rate",
        "## H1",
        "## Time to detect",
        "## Detector runtime",
    ):
        assert heading in text
    assert "skewguard ahead" in text  # 100% vs 33% on 30 paired incidents
    assert "falsified" not in text
    assert "Warning" not in text


def test_report_flags_untuned_run(a):
    df = _eval_run()
    tuning = {"max_unseen_mass": 0.02, "fixed_matched_factor": {"100": 1.0}}
    text = a.report(df, a.check_table(df), {"config": {"max_unseen_mass": 0.01}}, 0, tuning)
    assert "did not use the tuned" in text


def test_cli_round_trip_on_cell_directory(a, tmp_path):
    run = tmp_path / "tune-run"
    run.mkdir()
    _march().to_parquet(run / "none__none__100.parquet", index=False)
    (run / "manifest.json").write_text(json.dumps({"git_head": "abc", "config": {}}))
    a.main(["tune", str(run)])
    tuning = json.loads((tmp_path / "tune-run.tuning.json").read_text())
    assert tuning["max_unseen_mass"] == 0.02 and tuning["source"]["git_head"] == "abc"

    ev = tmp_path / "eval-run"
    ev.mkdir()
    _eval_run().to_parquet(ev / "cells.parquet", index=False)
    a.main(["report", str(ev), "--tuning", str(tmp_path / "tune-run.tuning.json")])
    text = (tmp_path / "eval-run.RESULTS.md").read_text()
    assert text.startswith("# skewguard evaluation results")


def test_cli_round_trip_on_combined_file(a, tmp_path):
    _march().to_parquet(tmp_path / "tune-x.parquet", index=False)
    (tmp_path / "tune-x.manifest.json").write_text(json.dumps({"git_head": "def"}))
    a.main(["tune", str(tmp_path / "tune-x.parquet")])
    tuning = json.loads((tmp_path / "tune-x.tuning.json").read_text())
    assert tuning["source"]["git_head"] == "def"

    _eval_run().to_parquet(tmp_path / "eval-x.parquet", index=False)
    a.main(
        [
            "report",
            str(tmp_path / "eval-x.parquet"),
            "--tuning",
            str(tmp_path / "tune-x.tuning.json"),
        ]
    )
    assert (tmp_path / "eval-x.RESULTS.md").exists()


def test_load_run_counts_feast_failures(a, tmp_path):
    df = _march()
    df.loc[:4, "feast_status"] = "feast_error"
    df.to_parquet(tmp_path / "r.parquet", index=False)
    rows, manifest, excluded = a.load_run(tmp_path / "r.parquet")
    assert excluded == 5 and len(rows) == len(df) - 5 and manifest == {}
