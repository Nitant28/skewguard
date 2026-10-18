"""``skewguard check``: run the detector over a Feast repo's monitoring metrics.

Exit status: 0 when nothing alerts, 1 when any feature alerts, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone

from skewguard.detector import FeatureReport, Status
from skewguard.policy import PolicySet


def _format_table(reports: Sequence[FeatureReport]) -> str:
    header = ("FEATURE", "WINDOW", "ROWS", "STATUS", "DETAIL")
    rows = []
    for r in reports:
        shown = r.alerts or tuple(c for c in r.checks if c.status is Status.OK)[-1:]
        detail = "; ".join(
            f"{c.name}={c.value:.4g} (limit {c.threshold:.4g})"
            if c.value is not None and c.threshold is not None
            else f"{c.name}: {c.note}"
            for c in shown
        ) or "; ".join(f"{c.name}: {c.note}" for c in r.checks if c.status is Status.SKIP)
        window = f"{r.source}/{r.granularity}/{r.metric_date}"
        rows.append(
            (f"{r.feature_view}.{r.feature}", window, str(r.rows), r.status.value.upper(), detail)
        )
    widths = [max(len(x[i]) for x in [header, *rows]) for i in range(4)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)) + "  " + header[4]]
    for row in rows:
        lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)) + "  " + row[4])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skewguard")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check a Feast repo's monitoring metrics")
    check.add_argument("--repo", default=".", help="Feast feature repo path")
    check.add_argument("--granularity", default="daily")
    check.add_argument("--source", choices=("batch", "log"), default="batch")
    check.add_argument("--feature-view")
    check.add_argument("--policy", help="policy YAML file")
    check.add_argument("--now", help="ISO timestamp to evaluate freshness against")
    check.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    from skewguard.feast_adapter import run

    policies = PolicySet.from_yaml(args.policy) if args.policy else PolicySet()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    reports = run(args.repo, args.granularity, args.source, args.feature_view, policies, now)

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2, default=str))
    elif reports:
        print(_format_table(reports))
    else:
        print("no monitoring rows found; run `feast monitor run` first", file=sys.stderr)
    return 1 if any(r.status is Status.ALERT for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
