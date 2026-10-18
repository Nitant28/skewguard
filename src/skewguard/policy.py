"""Per-feature alert policies, loadable from YAML.

Example::

    default:
      false_positive_rate: 0.01
      max_staleness: 26h
    features:
      driver_hourly_stats:            # every feature in the view
        max_staleness: 2h
      driver_hourly_stats.conv_rate:  # one feature
        min_rows: 100
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Policy:
    false_positive_rate: float = 0.01
    """Per-check alert rate on clean data; sets the null-distribution quantile."""
    statistic: str = "psi"
    """Numeric distance to alert on: ``psi`` or ``js``."""
    fixed_threshold: float | None = None
    """Use this fixed threshold instead of simulating one (e.g. the common PSI 0.25)."""
    min_rows: int = 30
    """Skip distance checks when the window has fewer non-null rows."""
    max_unseen_mass: float = 0.01
    """Alert when more than this share of the window lies outside the baseline's
    support: beyond its numeric range, or in categories it never had."""
    max_staleness: timedelta | None = None
    """Alert when the newest event is older than this; ``None`` disables the check."""
    reps: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.false_positive_rate < 1:
            raise ValueError("false_positive_rate must be in (0, 1)")
        if self.statistic not in ("psi", "js"):
            raise ValueError("statistic must be 'psi' or 'js'")
        if self.min_rows < 1 or self.reps < 10:
            raise ValueError("min_rows must be >= 1 and reps >= 10")


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_duration(value: Any) -> timedelta | None:
    """``"90m"``, ``"26h"``, ``"2d"``, or a number of seconds."""
    if value is None or isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=value)
    match = _DURATION.match(str(value))
    if not match:
        raise ValueError(f"cannot parse duration {value!r}; use e.g. '90m', '26h', '2d'")
    return timedelta(**{_UNITS[match.group(2)]: float(match.group(1))})


def _merge(base: Policy, data: Mapping[str, Any]) -> Policy:
    known = {f.name for f in dataclasses.fields(Policy)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    values = dict(data)
    if "max_staleness" in values:
        values["max_staleness"] = parse_duration(values["max_staleness"])
    return dataclasses.replace(base, **values)


@dataclass(frozen=True)
class PolicySet:
    default: Policy = field(default_factory=Policy)
    overrides: Mapping[str, Policy] = field(default_factory=dict)

    def for_feature(self, feature_view: str, feature: str) -> Policy:
        return (
            self.overrides.get(f"{feature_view}.{feature}")
            or self.overrides.get(feature_view)
            or self.default
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicySet:
        unknown = set(data) - {"default", "features"}
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
        default = _merge(Policy(), data.get("default") or {})
        features = data.get("features") or {}
        # A feature-level entry inherits from its view's entry, which inherits the default.
        views = {k: _merge(default, v or {}) for k, v in features.items() if "." not in k}
        overrides: dict[str, Policy] = dict(views)
        for key, value in features.items():
            if "." in key:
                parent = views.get(key.split(".", 1)[0], default)
                overrides[key] = _merge(parent, value or {})
        return cls(default=default, overrides=overrides)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySet:
        import yaml  # optional dependency; Feast already installs it

        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
