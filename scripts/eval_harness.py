"""Evaluation harness for docs/eval-plan.md.

Injects the plan's labelled faults into NYC TLC Yellow Taxi trips, computes
Feast's own monitoring metrics for every window (``MonitoringService``,
dask offline store), runs the plan's four detectors on them, and writes one
row per (day, fault, severity, window size, detector).

    python scripts/eval_harness.py download 2024-01 2024-02 2024-03
    python scripts/eval_harness.py run --range tune             # March: tuning only
    python scripts/eval_harness.py run --range eval --final     # April-June: the final run

Raw data and the generated Feast repo live in ``data/`` (gitignored). Results go
to ``results/<run>/``, one parquet file per (fault, severity, window size) cell,
and are combined into ``results/<run>.parquet``. Re-running the same command
skips finished cells, so an interrupted run resumes where it stopped.

Design choices the plan leaves open are listed under "Deviations" in
docs/eval-plan.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from skewguard import thresholds
from skewguard.detector import Status, check_feature
from skewguard.inject import FAULTS, SEVERITIES, SEVERITY_LEVELS, applicable_columns, inject
from skewguard.policy import Policy, PolicySet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPO = DATA / "feast_repo"
RESULTS = ROOT / "results"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"

SEED = 20261018  # the plan's seed for choosing injected days
TS = "tpep_pickup_datetime"
NUMERIC = ("trip_distance", "fare_amount", "passenger_count")
CATEGORICAL = ("payment_type", "PULocationID")
FEATURES = NUMERIC + CATEGORICAL

PROJECT = "skewguard_eval"
FV = "taxi_trips"
SOURCE_FILE = "data/source.parquet"
TTL = timedelta(days=1)

BASELINE_RANGE = (date(2024, 1, 1), date(2024, 2, 29))
RANGES = {
    "tune": (date(2024, 3, 1), date(2024, 3, 31)),
    "eval": (date(2024, 4, 1), date(2024, 6, 30)),
}
EVAL_START = RANGES["eval"][0]
WINDOW_SIZES = (100, 300, 1000, 5000)
DAYS_PER_CELL = 30
GRADUAL_HORIZON = 14  # days a gradual shift is followed after its onset

DETECTORS = ("none", "null_rule", "fixed", "skewguard")
NULL_RULE_POINTS = 0.10
FIXED_PSI, FIXED_LINF = 0.25, 0.10
CLEAN = "none"

FEATURE_STORE_YAML = f"""\
project: {PROJECT}
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: dask
entity_key_serialization_version: 3
"""

log = logging.getLogger("eval_harness")
Row = dict[str, Any]


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #


def month_file(month: str) -> Path:
    return RAW / f"yellow_tripdata_{month}.parquet"


def download(months: Sequence[str], force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for month in months:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise SystemExit(f"bad month {month!r}; use YYYY-MM")
        dest = month_file(month)
        if dest.exists() and not force:
            print(f"have {dest.name}")
            continue
        url = BASE_URL + dest.name
        tmp = dest.with_suffix(".part")
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        with open(tmp, "rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            tmp.unlink()
            raise SystemExit(f"{url} did not return a parquet file")
        tmp.replace(dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB -> {dest}")


def _as_category(s: pd.Series) -> pd.Series:
    """Integer codes as Arrow-backed strings (nulls stay null)."""
    return s.astype("Int64").astype("string[pyarrow]")


def load_events(months: Iterable[str], first: date, last: date) -> pd.DataFrame:
    """Trips from ``months`` with pickup on ``first`` .. ``last``, sorted by pickup time.

    TLC pickup times are naive local times; they are treated as UTC throughout.
    """
    frames = []
    for month in months:
        path = month_file(month)
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run: python scripts/eval_harness.py download {month}"
            )
        frames.append(pq.read_table(path, columns=[TS, *FEATURES]).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df[TS] = pd.to_datetime(df[TS]).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
    for c in NUMERIC:
        df[c] = df[c].astype(float)
    for c in CATEGORICAL:
        df[c] = _as_category(df[c])
    lo, hi = _day_start(first), _day_start(last + timedelta(days=1))
    df = df[(df[TS] >= lo) & (df[TS] < hi)]
    return df.sort_values(TS, kind="stable").reset_index(drop=True)


def months_between(first: date, last: date) -> list[str]:
    months, d = [], date(first.year, first.month, 1)
    while d <= last:
        months.append(f"{d:%Y-%m}")
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return months


def _day_start(day: date) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, day.day, tz="UTC")


def days_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


class DayIndex:
    """Fast per-day slices of a time-sorted event frame, and seeded sub-samples of them."""

    def __init__(self, events: pd.DataFrame):
        self.events = events
        self.ts = events[TS].to_numpy(dtype="datetime64[us]")  # UTC
        self._samples: dict[tuple[date, int], pd.DataFrame] = {}

    def rows(self, day: date) -> pd.DataFrame:
        lo, hi = (
            np.searchsorted(self.ts, np.datetime64(d.isoformat(), "us"))
            for d in (day, day + timedelta(days=1))
        )
        return self.events.iloc[lo:hi]

    def sample(self, day: date, size: int) -> pd.DataFrame:
        """``size`` trips drawn without replacement from ``day`` (same draw for every fault).

        Callers must not modify the returned frame; the injectors never do.
        """
        key = (day, size)
        if key not in self._samples:
            rows = self.rows(day)
            rng = np.random.default_rng([SEED, 1, day.toordinal(), size])
            take = np.sort(rng.choice(len(rows), size=min(size, len(rows)), replace=False))
            self._samples[key] = rows.iloc[take].reset_index(drop=True)
        return self._samples[key]


# --------------------------------------------------------------------------- #
#  Feast
# --------------------------------------------------------------------------- #


def _arrow_column(s: pd.Series, numeric: bool) -> pa.Array:
    """One parquet column per feature. A numeric column holding any string is
    written as strings: parquet has no mixed-type columns."""
    if isinstance(s.dtype, pd.StringDtype):
        return pa.array(s.to_numpy(dtype=object, na_value=None), type=pa.string())
    if numeric and s.dtype != object:
        return pa.array(s.to_numpy(dtype=float), type=pa.float64(), from_pandas=True)
    values = s.to_numpy(dtype=object)
    if numeric and not any(isinstance(v, str) for v in values):
        return pa.array(values.astype(float), type=pa.float64(), from_pandas=True)
    return pa.array([None if pd.isna(v) else str(v) for v in values], type=pa.string())


def write_source(frame: pd.DataFrame, repo: Path) -> None:
    columns = {
        "trip_id": pa.array(np.arange(len(frame), dtype=np.int64)),
        TS: pa.array(frame[TS].astype("datetime64[us, UTC]"), type=pa.timestamp("us", tz="UTC")),
    }
    for c in FEATURES:
        columns[c] = _arrow_column(frame[c], c in NUMERIC)
    path = repo / SOURCE_FILE
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(columns), tmp)
    _replace(tmp, path)


def _replace(src: Path, dest: Path, attempts: int = 50) -> None:
    """``src.replace(dest)``, retried while Windows briefly locks ``dest``
    (virus scanners and the search indexer open freshly written files)."""
    for attempt in range(attempts):
        try:
            src.replace(dest)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)


def feature_definitions() -> list[Any]:
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Float64, String
    from feast.value_type import ValueType

    trip = Entity(name="trip", join_keys=["trip_id"], value_type=ValueType.INT64)
    source = FileSource(name="taxi_source", path=SOURCE_FILE, timestamp_field=TS)
    view = FeatureView(
        name=FV,
        entities=[trip],
        ttl=TTL,
        schema=[Field(name=c, dtype=Float64) for c in NUMERIC]
        + [Field(name=c, dtype=String) for c in CATEGORICAL],
        source=source,
        online=False,
    )
    return [trip, view]


def open_store(repo: Path) -> Any:
    from feast import FeatureStore

    return FeatureStore(repo_path=str(repo))


def build_repo(repo: Path = REPO) -> dict[str, Row]:
    """Create the Feast repo and its Jan-Feb baseline once; return baseline rows by feature."""
    from feast.monitoring.monitoring_service import MonitoringService

    marker = repo / "baseline.json"
    if marker.exists():
        store = open_store(repo)
        rows = _baseline_rows(MonitoringService(store))
        expected = json.loads(marker.read_text(encoding="utf-8"))
        if {f: r["row_count"] for f, r in rows.items()} != expected["row_count"]:
            raise SystemExit(f"{repo} baseline differs from {marker}; delete {repo} to rebuild")
        return rows

    first, last = BASELINE_RANGE
    print(f"building Feast repo in {repo} with baseline {first} .. {last}")
    events = load_events(months_between(first, last), first, last)
    (repo / "data").mkdir(parents=True, exist_ok=True)
    (repo / "feature_store.yaml").write_text(FEATURE_STORE_YAML, encoding="utf-8")
    write_source(events, repo)
    store = open_store(repo)
    store.apply(feature_definitions())
    svc = MonitoringService(store)
    svc.compute_baseline(project=PROJECT, feature_view_name=FV)
    rows = _baseline_rows(svc)
    if set(rows) != set(FEATURES) or any(r["row_count"] != len(events) for r in rows.values()):
        raise SystemExit(f"baseline incomplete: {sorted(rows)}")
    marker.write_text(
        json.dumps(
            {
                "range": [str(first), str(last)],
                "events": len(events),
                "row_count": {f: r["row_count"] for f, r in rows.items()},
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def _baseline_rows(svc: Any) -> dict[str, Row]:
    rows = svc.get_baseline(project=PROJECT, feature_view_name=FV, data_source_type="batch")
    return {r["feature_name"]: r for r in rows}


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class FeastWindows:
    """Compute one daily window's monitoring rows through Feast, as stored by it."""

    def __init__(self, repo: Path):
        from feast.monitoring.monitoring_service import MonitoringService

        self.repo = repo
        self.svc = MonitoringService(open_store(repo))

    def compute(self, frame: pd.DataFrame, day: date) -> tuple[dict[str, Row], str, float]:
        start = _day_start(day)
        end = pd.Timestamp(day.year, day.month, day.day, 23, 59, 59, tz="UTC")  # Feast's day end
        expected = int(((frame[TS] >= start) & (frame[TS] <= end)).sum())
        write_source(frame, self.repo)
        t0 = datetime.now(timezone.utc)
        tick = time.perf_counter()
        self.svc.compute_metrics(
            project=PROJECT, feature_view_name=FV, start_date=day, end_date=day, granularity="daily"
        )
        rows = self.svc.get_feature_metrics(
            project=PROJECT,
            feature_view_name=FV,
            granularity="daily",
            data_source_type="batch",
            start_date=day,
            end_date=day,
        )
        ms = (time.perf_counter() - tick) * 1000
        fresh = {
            r["feature_name"]: r
            for r in rows
            if (_parse_ts(r["computed_at"]) or datetime.min.replace(tzinfo=timezone.utc)) >= t0
        }
        if set(fresh) != set(FEATURES):
            return fresh, "feast_error", ms
        if any(int(r["row_count"] or 0) != expected for r in fresh.values()):
            return fresh, "row_count_mismatch", ms
        return fresh, "ok", ms


def newest_event(frame: pd.DataFrame) -> pd.Timestamp | None:
    return frame[TS].max() if len(frame) else None


def stamp_newest_event(windows: dict[str, Row], newest: pd.Timestamp | None) -> Any:
    """Stamp every window row with the source's newest event, as ``feast monitor run`` does.

    ``compute_metrics`` stamps a window with the newest event inside it, or with
    None when the window is empty (``monitoring_service.py:47-61``, ``1038-1042``),
    so a stalled source never looks older than one day. The scheduled
    ``auto_compute`` stamps every window with the source's MAX(event_timestamp)
    instead (``monitoring_service.py:154``, ``206``). Returns Feast's own stamp.
    """
    feast_stamp = next(iter(windows.values()), {}).get("max_event_timestamp")
    for row in windows.values():
        row["max_event_timestamp"] = None if newest is None else newest.isoformat()
    return feast_stamp


# --------------------------------------------------------------------------- #
#  Detectors
# --------------------------------------------------------------------------- #


def policies(min_rows: int, max_unseen_mass: float) -> dict[str, PolicySet]:
    common = {"max_staleness": TTL, "min_rows": min_rows, "max_unseen_mass": max_unseen_mass}
    fixed_cat = Policy(fixed_threshold=FIXED_LINF, **common)
    return {
        "fixed": PolicySet(
            default=Policy(fixed_threshold=FIXED_PSI, **common),
            overrides={f"{FV}.{c}": fixed_cat for c in CATEGORICAL},
        ),
        "skewguard": PolicySet(default=Policy(**common)),
    }


def _null_cache_misses() -> int | None:
    """Misses of the null-simulation caches, if this thresholds version exposes them."""
    total = 0
    for name in ("_numeric_null", "_categorical_null"):
        info = getattr(getattr(thresholds, name, None), "cache_info", None)
        if info is None:
            return None
        total += info().misses
    return total


def run_detectors(
    baselines: dict[str, Row], windows: dict[str, Row], now: datetime, sets: dict[str, PolicySet]
) -> list[Row]:
    out = []
    for detector in DETECTORS:
        alerts: list[str] = []
        checks: dict[str, Any] = {}
        times = []
        misses_before = _null_cache_misses()
        for feature in FEATURES:
            tick = time.perf_counter()
            base, win = baselines[feature], windows.get(feature)
            if detector == "none" or win is None:
                pass
            elif detector == "null_rule":
                rise = float(win.get("null_rate") or 0.0) - float(base.get("null_rate") or 0.0)
                checks[feature] = {"null_rate_rise": rise}
                if rise > NULL_RULE_POINTS:
                    alerts.append(f"{feature}:null_rate")
            else:
                policy = sets[detector].for_feature(FV, feature)
                report = check_feature(base, win, policy, now)
                checks[feature] = {
                    c.name: [c.status.value, c.value, c.threshold] for c in report.checks
                }
                alerts += [f"{feature}:{c.name}" for c in report.checks if c.status is Status.ALERT]
            times.append((time.perf_counter() - tick) * 1000)
        misses_after = _null_cache_misses()
        out.append(
            {
                "detector": detector,
                "alert": bool(alerts),
                "alerts": ",".join(alerts),
                "n_features_alerting": len({a.split(":")[0] for a in alerts}),
                "checks": json.dumps(checks, default=float) if checks else "",
                "runtime_ms": float(sum(times)),
                "feature_runtime_ms": times,
                "null_cache_misses": (
                    None if misses_before is None else misses_after - misses_before
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- #
#  Schedule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Incident:
    fault: str
    index: int
    onset_day: date
    onset: pd.Timestamp
    target: str | None


def incidents_for(fault: str, days: Sequence[date]) -> list[Incident]:
    """The plan's 30 injected days for ``fault``, chosen with seed 20261018.

    The same onset days and target features are used for every severity and
    window size, so those comparisons are paired.
    """
    rng = np.random.default_rng([SEED, FAULTS.index(fault)])
    picked = np.sort(rng.choice(len(days), size=min(DAYS_PER_CELL, len(days)), replace=False))
    targets = applicable_columns(fault, list(NUMERIC), list(CATEGORICAL))
    out = []
    for i, d in enumerate(picked):
        day = days[int(d)]
        target = str(rng.choice(targets)) if targets else None
        onset = _day_start(day)
        if fault == "stale_source":  # the source stops at a random second of the day
            onset += timedelta(seconds=int(rng.integers(0, 86400)))
        out.append(Incident(fault, i, day, onset, target))
    return out


def incident_days(incident: Incident, severity: str, last: date) -> tuple[list[date], int]:
    """Days evaluated for one incident, capped at the range's last day, and the uncapped count."""
    if incident.fault == "gradual_shift":
        n = GRADUAL_HORIZON
    elif incident.fault == "stale_source":
        end = incident.onset + pd.Timedelta(TTL) * SEVERITIES["stale_source"][severity]
        n = (end.date() - incident.onset_day).days + 1
    else:
        n = 1
    days = [incident.onset_day + timedelta(days=k) for k in range(n)]
    return [d for d in days if d <= last], n


def cells(sizes: Sequence[int], faults: Sequence[str]) -> list[tuple[str, str, int]]:
    out = [(CLEAN, CLEAN, n) for n in sizes]
    out += [(f, s, n) for f in faults for s in SEVERITY_LEVELS for n in sizes]
    return out


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([SEED, *parts]).generate_state(1)[0])


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_cell(
    cell: tuple[str, str, int],
    days: Sequence[date],
    index: DayIndex,
    feast: FeastWindows,
    baselines: dict[str, Row],
    sets: dict[str, PolicySet],
    run_name: str,
) -> list[Row]:
    fault, severity, size = cell
    if fault == CLEAN:
        groups = [(None, [d], 1) for d in days]
    else:
        groups = [(i, *incident_days(i, severity, days[-1])) for i in incidents_for(fault, days)]

    rows = []
    for inc, evaluated, planned in groups:
        onset_day = evaluated[0]
        # Before the onset day the source holds clean events; afterwards, what the
        # fault left of each day. Its newest event is what the freshness check sees.
        newest = newest_event(index.sample(onset_day - timedelta(days=1), size))
        for day in evaluated:
            newest, day_rows = _run_window(
                inc, day, onset_day, planned, newest, cell, index, feast, baselines, sets, run_name
            )
            rows += day_rows
    return rows


def _run_window(
    inc: Incident | None,
    day: date,
    onset_day: date,
    planned: int,
    newest: pd.Timestamp | None,
    cell: tuple[str, str, int],
    index: DayIndex,
    feast: FeastWindows,
    baselines: dict[str, Row],
    sets: dict[str, PolicySet],
    run_name: str,
) -> tuple[pd.Timestamp | None, list[Row]]:
    """Inject, compute and check one day; returns the source's newest event after it."""
    fault, severity, size = cell
    frame = sample = index.sample(day, size)
    offset = (day - onset_day).days
    label: dict[str, Any] = {}
    if inc is not None:
        sd = baselines[inc.target]["stddev"] if inc.target else None
        frame, lab = inject(
            frame,
            TS,
            fault,
            severity,
            column=inc.target,
            seed=_seed(
                2, FAULTS.index(fault), SEVERITY_LEVELS.index(severity), size, inc.index, offset
            ),
            onset=inc.onset,
            ttl=TTL,
            baseline_sd=sd,
        )
        label = lab.to_dict()
    windows, status, feast_ms = feast.compute(frame, day)
    today = newest_event(frame)
    if today is not None and (newest is None or today > newest):
        newest = today
    feast_stamp = stamp_newest_event(windows, newest)
    now = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    probe = windows.get(inc.target) if inc and inc.target else next(iter(windows.values()), {})
    common = {
        "run": run_name,
        "day": day.isoformat(),
        "onset_day": onset_day.isoformat(),
        "day_offset": offset,
        "incident_days": planned,
        "incident": -1 if inc is None else inc.index,
        "fault": fault,
        "severity": severity,
        "magnitude": label.get("magnitude", float("nan")),
        "window_size": size,
        "target": "" if inc is None or inc.target is None else inc.target,
        "rows_sampled": len(sample),
        "rows_window": int((probe or {}).get("row_count") or 0),
        "feast_status": status,
        "feast_ms": feast_ms,
        "newest_event": None if newest is None else newest.isoformat(),
        "feast_newest_event": None if feast_stamp is None else str(feast_stamp),
        "label": json.dumps(label, default=str) if label else "",
    }
    rows = []
    for det in run_detectors(baselines, windows, now, sets):
        target_alert = None
        if inc is not None and inc.target:
            target_alert = any(a.startswith(f"{inc.target}:") for a in det["alerts"].split(","))
        rows.append({**common, **det, "target_alert": target_alert})
    return newest, rows


def run(args: argparse.Namespace) -> None:
    first, last = (
        RANGES[args.range]
        if args.range
        else (date.fromisoformat(args.start), date.fromisoformat(args.end))
    )
    if last < first:
        raise SystemExit("--end is before --start")
    if last >= EVAL_START and not args.final:
        raise SystemExit(
            f"{first} .. {last} reaches the evaluation range (from {EVAL_START}). "
            "That range stays untouched until the final run; pass --final for it."
        )
    if first <= BASELINE_RANGE[1]:
        raise SystemExit(f"the range must start after the baseline ({BASELINE_RANGE[1]})")
    sizes = tuple(args.sizes)
    faults = tuple(args.faults)
    run_name = args.name or f"{args.range or 'custom'}-{first:%Y%m%d}-{last:%Y%m%d}"
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "range": [str(first), str(last)],
        "baseline": [str(d) for d in BASELINE_RANGE],
        "sizes": list(sizes),
        "faults": list(faults),
        "detectors": list(DETECTORS),
        "seed": SEED,
        "days_per_cell": DAYS_PER_CELL,
        "gradual_horizon_days": GRADUAL_HORIZON,
        "ttl_seconds": TTL.total_seconds(),
        "min_rows": args.min_rows,
        "max_unseen_mass": args.max_unseen_mass,
        "null_rule_points": NULL_RULE_POINTS,
        "fixed_psi": FIXED_PSI,
        "fixed_linf": FIXED_LINF,
        "freshness_stamp": "source_max",  # see stamp_newest_event
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_digest"] != digest:
            raise SystemExit(
                f"{out_dir} was started with a different configuration; use --name for a new run"
            )
    else:
        import feast

        manifest = {
            "config": config,
            "config_digest": digest,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git_head": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "plan_commit": _git("log", "-1", "--format=%H", "--", "docs/eval-plan.md"),
            "feast_version": feast.__version__,
            "python": sys.version.split()[0],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    baselines = build_repo(REPO)
    days = days_between(first, last)
    before = first - timedelta(days=1)  # the source already holds the day before the range
    print(f"loading trips {before} .. {last}")
    index = DayIndex(load_events(months_between(before, last), before, last))
    feast = FeastWindows(REPO)
    sets = policies(args.min_rows, args.max_unseen_mass)

    todo = cells(sizes, faults)
    started = time.perf_counter()
    for i, cell in enumerate(todo, 1):
        path = out_dir / f"{cell[0]}__{cell[1]}__{cell[2]}.parquet"
        if path.exists():
            continue
        tick = time.perf_counter()
        rows = run_cell(cell, days, index, feast, baselines, sets, run_name)
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        _replace(tmp, path)
        print(
            f"[{i}/{len(todo)}] {cell[0]}/{cell[1]}/n={cell[2]}: {len(rows)} rows "
            f"in {time.perf_counter() - tick:.1f}s (total {time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    parts = sorted(out_dir.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined.to_parquet(RESULTS / f"{run_name}.parquet", index=False)
    print(
        f"wrote {len(combined)} rows from {len(parts)} cells to {RESULTS / f'{run_name}.parquet'}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eval_harness")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch TLC Yellow Taxi monthly parquet into data/raw")
    dl.add_argument("months", nargs="+", help="YYYY-MM, e.g. 2024-01")
    dl.add_argument("--force", action="store_true")

    r = sub.add_parser("run", help="inject, compute Feast metrics, run detectors")
    which = r.add_mutually_exclusive_group(required=True)
    which.add_argument("--range", choices=sorted(RANGES))
    which.add_argument("--start", help="first day (YYYY-MM-DD); needs --end")
    r.add_argument("--end", help="last day (YYYY-MM-DD)")
    r.add_argument("--final", action="store_true", help="allow the evaluation range")
    r.add_argument("--name", help="results/<name>/ (default: from the range)")
    r.add_argument("--sizes", type=int, nargs="+", default=list(WINDOW_SIZES))
    r.add_argument("--faults", nargs="+", default=list(FAULTS), choices=FAULTS)
    r.add_argument("--min-rows", type=int, default=Policy.min_rows)
    r.add_argument("--max-unseen-mass", type=float, default=Policy.max_unseen_mass)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    if args.command == "download":
        download(args.months, args.force)
    else:
        if args.start and not args.end:
            parser.error("--start needs --end")
        run(args)


if __name__ == "__main__":
    main()
