"""Labelled fault injection for the evaluation in ``docs/eval-plan.md``.

Each fault function takes a DataFrame of raw events and the name of its
event-timestamp column, and returns ``(corrupted_copy, Label)``. The input is
never modified. A fault starts at ``onset`` (default: the earliest event), so
rows before it are untouched. Any random choice comes from ``seed`` alone: the
same frame, arguments and seed always give the same output.

``inject`` looks magnitudes up in ``SEVERITIES``, which restates the plan's table.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_LEVELS = ("low", "medium", "high")

SEVERITIES: dict[str, dict[str, float]] = {
    "unit_change": {"low": 1.1, "medium": 10.0, "high": 100.0},  # multiply by
    "null_burst": {"low": 0.02, "medium": 0.10, "high": 0.50},  # share of rows nulled
    "stale_source": {"low": 1.0, "medium": 3.0, "high": 7.0},  # outage length, x TTL
    "new_category": {"low": 0.01, "medium": 0.05, "high": 0.25},  # share of rows
    "timezone_shift": {"low": 1.0, "medium": 5.0, "high": 12.0},  # hours
    "gradual_shift": {"low": 0.02, "medium": 0.05, "high": 0.10},  # baseline sd per day
    "type_change": {"low": 0.01, "medium": 0.10, "high": 1.0},  # share of rows
}
FAULTS = tuple(SEVERITIES)

NUMERIC_FAULTS = frozenset({"unit_change", "gradual_shift", "type_change"})
CATEGORICAL_FAULTS = frozenset({"new_category"})
FRAME_FAULTS = frozenset({"stale_source", "timezone_shift"})
"""Faults that act on whole events (timestamps), not on one feature column."""

NEW_CATEGORY = "__new__"


@dataclass(frozen=True)
class Label:
    """What was injected, where, and how much of the frame it touched."""

    fault: str
    magnitude: float
    column: str | None
    onset: pd.Timestamp | None
    rows_in: int
    rows_out: int
    rows_affected: int
    end: pd.Timestamp | None = None
    """Only for ``stale_source``: when the source starts appending again."""
    severity: str | None = None
    """Set by ``inject``; ``None`` when a fault function is called directly."""
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for key in ("onset", "end"):
            d[key] = None if d[key] is None else d[key].isoformat()
        return d


def applicable_columns(fault: str, numeric: list[str], categorical: list[str]) -> list[str]:
    """The feature columns ``fault`` can target (empty for whole-event faults)."""
    if fault in FRAME_FAULTS:
        return []
    if fault in NUMERIC_FAULTS:
        return list(numeric)
    if fault in CATEGORICAL_FAULTS:
        return list(categorical)
    return [*numeric, *categorical]


def _like(series: pd.Series, value: Any) -> pd.Timestamp:
    """``value`` as a Timestamp comparable with ``series`` (UTC if one side is naive)."""
    ts = pd.Timestamp(value)
    tz = getattr(series.dtype, "tz", None)
    if tz is not None and ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(tz)
    if tz is None and ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _from_onset(
    df: pd.DataFrame, ts_col: str, onset: Any
) -> tuple[np.ndarray, pd.Timestamp | None]:
    ts = df[ts_col]
    if onset is None:
        return np.ones(len(df), dtype=bool), (ts.min() if len(ts) else None)
    start = _like(ts, onset)
    return (ts >= start).to_numpy(), start


def _pick(candidates: np.ndarray, share: float, n_ref: int, seed: int) -> np.ndarray:
    """Positions of ``round(share * n_ref)`` rows drawn without replacement from a mask."""
    pool = np.flatnonzero(candidates)
    k = min(len(pool), int(round(share * n_ref)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=k, replace=False))


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    dtype = df[column].dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        raise ValueError(f"{column!r} is not numeric ({dtype})")


def _set(col: pd.Series, positions: np.ndarray, value: Any) -> pd.Series:
    """Copy of ``col`` with ``value`` at ``positions``, widening the dtype if needed."""
    dtype = col.dtype
    if value is None and pd.api.types.is_integer_dtype(dtype) and isinstance(dtype, np.dtype):
        col = col.astype(float)
    elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
        col = col.astype(object)
    elif value is not None and pd.api.types.is_numeric_dtype(dtype):
        col = col.astype(object)
    else:
        col = col.copy()
    col.iloc[positions] = value
    return col


def unit_change(
    df: pd.DataFrame, ts_col: str, column: str, factor: float, *, onset: Any = None
) -> tuple[pd.DataFrame, Label]:
    """Multiply ``column`` by ``factor`` from ``onset`` on, as after a units bug.

    The column comes back as float.
    """
    _require_numeric(df, column)
    post, start = _from_onset(df, ts_col, onset)
    values = df[column].astype(float).to_numpy(copy=True)
    hit = post & ~np.isnan(values)
    values[hit] *= factor
    out = df.copy()
    out[column] = values
    label = Label("unit_change", factor, column, start, len(df), len(out), int(hit.sum()))
    return out, label


def null_burst(
    df: pd.DataFrame, ts_col: str, column: str, share: float, *, seed: int, onset: Any = None
) -> tuple[pd.DataFrame, Label]:
    """Null ``share`` of the rows from ``onset`` on (non-null rows only).

    The null rate after ``onset`` rises by ``share`` exactly, up to rounding,
    unless fewer non-null rows remain.
    """
    post, start = _from_onset(df, ts_col, onset)
    pos = _pick(post & df[column].notna().to_numpy(), share, int(post.sum()), seed)
    out = df.copy()
    out[column] = _set(df[column], pos, None)
    label = Label("null_burst", share, column, start, len(df), len(out), len(pos), seed=seed)
    return out, label


def stale_source(
    df: pd.DataFrame, ts_col: str, cutoff: Any, ttl: timedelta, multiple: float
) -> tuple[pd.DataFrame, Label]:
    """Drop events in ``[cutoff, cutoff + multiple * ttl)``: the source stopped appending.

    Dropped events are lost, not backfilled. Rows keep their original index.
    """
    ts = df[ts_col]
    start = _like(ts, cutoff)
    end = start + pd.Timedelta(ttl) * multiple
    drop = ((ts >= start) & (ts < end)).to_numpy()
    out = df.loc[~drop].copy()
    label = Label(
        "stale_source", multiple, None, start, len(df), len(out), int(drop.sum()), end=end
    )
    return out, label


def new_category(
    df: pd.DataFrame,
    ts_col: str,
    column: str,
    share: float,
    *,
    seed: int,
    onset: Any = None,
    value: str = NEW_CATEGORY,
) -> tuple[pd.DataFrame, Label]:
    """Recode ``share`` of the rows from ``onset`` on to ``value``, a category never seen."""
    if pd.api.types.is_numeric_dtype(df[column].dtype):
        raise ValueError(f"{column!r} is numeric; new_category needs a categorical column")
    if (df[column] == value).any():
        raise ValueError(f"{value!r} already occurs in {column!r}")
    post, start = _from_onset(df, ts_col, onset)
    pos = _pick(post & df[column].notna().to_numpy(), share, int(post.sum()), seed)
    out = df.copy()
    out[column] = _set(df[column], pos, value)
    label = Label("new_category", share, column, start, len(df), len(out), len(pos), seed=seed)
    return out, label


def timezone_shift(
    df: pd.DataFrame, ts_col: str, hours: float, *, onset: Any = None
) -> tuple[pd.DataFrame, Label]:
    """Add ``hours`` to the event timestamps from ``onset`` on, as a wrong-zone producer would."""
    post, start = _from_onset(df, ts_col, onset)
    out = df.copy()
    out[ts_col] = df[ts_col] + pd.to_timedelta(np.where(post, hours, 0.0), unit="h")
    label = Label("timezone_shift", hours, None, start, len(df), len(out), int(post.sum()))
    return out, label


def gradual_shift(
    df: pd.DataFrame,
    ts_col: str,
    column: str,
    rate: float,
    baseline_sd: float,
    *,
    onset: Any = None,
) -> tuple[pd.DataFrame, Label]:
    """Add a mean drift of ``rate`` baseline standard deviations per day since ``onset``.

    The drift grows continuously: an event ``t`` days after onset moves by
    ``rate * baseline_sd * t``.
    """
    _require_numeric(df, column)
    post, start = _from_onset(df, ts_col, onset)
    values = df[column].astype(float).to_numpy(copy=True)
    hit = post & ~np.isnan(values)
    if start is not None:
        days = (df[ts_col] - start).dt.total_seconds().to_numpy(dtype=float) / 86400
        values[hit] += rate * baseline_sd * days[hit]
    out = df.copy()
    out[column] = values
    label = Label("gradual_shift", rate, column, start, len(df), len(out), int(hit.sum()))
    return out, label


def type_change(
    df: pd.DataFrame, ts_col: str, column: str, share: float, *, seed: int, onset: Any = None
) -> tuple[pd.DataFrame, Label]:
    """Replace ``share`` of the values from ``onset`` on with their string form.

    The column comes back with object dtype, mixing floats and strings; a
    ``share`` of 1 turns every non-null value after ``onset`` into a string.
    """
    _require_numeric(df, column)
    post, start = _from_onset(df, ts_col, onset)
    pos = _pick(post & df[column].notna().to_numpy(), share, int(post.sum()), seed)
    values = df[column].to_numpy(dtype=object, copy=True)
    values[pos] = [str(v) for v in values[pos]]
    out = df.copy()
    out[column] = pd.Series(values, index=df.index, dtype=object)
    label = Label("type_change", share, column, start, len(df), len(out), len(pos), seed=seed)
    return out, label


def inject(
    df: pd.DataFrame,
    ts_col: str,
    fault: str,
    severity: str,
    *,
    column: str | None = None,
    seed: int = 0,
    onset: Any = None,
    ttl: timedelta | None = None,
    baseline_sd: float | None = None,
) -> tuple[pd.DataFrame, Label]:
    """Inject ``fault`` at one of the plan's severities (``low`` / ``medium`` / ``high``).

    ``stale_source`` needs ``ttl`` and starts its outage at ``onset``;
    ``gradual_shift`` needs the target's ``baseline_sd``. Every fault except
    ``stale_source`` and ``timezone_shift`` needs a target ``column``.
    """
    if fault not in SEVERITIES:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITY_LEVELS}")
    m = SEVERITIES[fault][severity]
    if fault not in FRAME_FAULTS and column is None:
        raise ValueError(f"{fault} needs a target column")

    if fault == "unit_change":
        out, label = unit_change(df, ts_col, column, m, onset=onset)
    elif fault == "null_burst":
        out, label = null_burst(df, ts_col, column, m, seed=seed, onset=onset)
    elif fault == "stale_source":
        if ttl is None:
            raise ValueError("stale_source needs ttl")
        cutoff = onset if onset is not None else df[ts_col].min()
        out, label = stale_source(df, ts_col, cutoff, ttl, m)
    elif fault == "new_category":
        out, label = new_category(df, ts_col, column, m, seed=seed, onset=onset)
    elif fault == "timezone_shift":
        out, label = timezone_shift(df, ts_col, m, onset=onset)
    elif fault == "gradual_shift":
        if baseline_sd is None:
            raise ValueError("gradual_shift needs baseline_sd")
        out, label = gradual_shift(df, ts_col, column, m, baseline_sd, onset=onset)
    else:
        out, label = type_change(df, ts_col, column, m, seed=seed, onset=onset)
    return out, dataclasses.replace(label, severity=severity)
