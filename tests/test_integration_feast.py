"""End-to-end against a real Feast repo: feast init -> apply -> monitor run."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import categorical_row, numeric_row

pytest.importorskip("feast", reason="needs the feast extra")
pa = pytest.importorskip("pyarrow")

from skewguard.detector import Status  # noqa: E402

pytestmark = pytest.mark.integration


def test_test_helpers_match_feast_metrics_calculator():
    from feast.monitoring.metrics_calculator import MetricsCalculator

    calc = MetricsCalculator()
    values = np.random.default_rng(0).normal(size=500)
    ours = numeric_row(values)["histogram"]
    theirs = calc.compute_numeric(pa.array(values))["histogram"]
    assert ours["bins"] == pytest.approx(theirs["bins"])
    assert ours["counts"] == theirs["counts"]

    cats = [f"c{i % 13}" for i in range(400)] + ["c0"] * 50
    ours_c = categorical_row(cats)["histogram"]
    theirs_c = calc.compute_categorical(pa.array(cats))["histogram"]
    assert ours_c["other_count"] == theirs_c["other_count"]
    assert {e["value"]: e["count"] for e in ours_c["values"]} == {
        e["value"]: e["count"] for e in theirs_c["values"]
    }


def _feast(args, cwd):
    exe = Path(sys.executable).with_name("feast.exe" if sys.platform == "win32" else "feast")
    cmd = [str(exe) if exe.exists() else (shutil.which("feast") or "feast")]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd + args, cwd=cwd, check=True, capture_output=True, env=env)


@pytest.fixture(scope="module")
def demo_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("feast")
    _feast(["init", "demo"], root)
    repo = root / "demo" / "feature_repo"
    with open(repo / "feature_store.yaml", "a", encoding="utf-8") as f:
        f.write("\ndata_quality_monitoring:\n  auto_baseline: true\n")
    _feast(["apply"], repo)
    _feast(["monitor", "run"], repo)
    return repo


def test_run_on_feast_demo_repo(demo_repo):
    from skewguard.feast_adapter import run
    from skewguard.policy import Policy, PolicySet

    reports = run(str(demo_repo), policies=PolicySet(Policy(reps=200)))
    names = {(r.feature_view, r.feature) for r in reports}
    assert ("driver_hourly_stats", "conv_rate") in names
    for r in reports:
        assert r.source == "batch" and r.granularity == "daily"
        assert r.status in (Status.OK, Status.ALERT, Status.SKIP)
        assert not any(c.name == "baseline" for c in r.checks), "every feature has a baseline"


def test_feast_windows_do_not_share_baseline_edges(demo_repo):
    """Finding 1: each window computes its own histogram edges."""
    from feast import FeatureStore
    from feast.monitoring.monitoring_service import MonitoringService

    store = FeatureStore(repo_path=str(demo_repo))
    svc = MonitoringService(store)
    kw = dict(
        project=store.project, feature_view_name="driver_hourly_stats", feature_name="conv_rate"
    )
    base = svc.get_baseline(**kw)[0]["histogram"]["bins"]
    day = svc.get_feature_metrics(granularity="daily", **kw)[0]["histogram"]["bins"]
    assert len(base) == len(day) == 21
    assert base != day
