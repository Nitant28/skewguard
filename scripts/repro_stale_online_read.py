"""Reproduce F7: the default Python online read serves features older than their TTL as PRESENT.

Materializes only events from 10 to 3 days ago into the online store of a
`feast init` demo repo (driver_hourly_stats has ttl=1 day), then reads them
back and prints each value's age next to the status Feast reports.

    python scripts/repro_stale_online_read.py path/to/feature_repo

Modifies that repo's online store; run it against a throwaway demo repo.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from feast import FeatureStore
from feast.protos.feast.serving.ServingService_pb2 import FieldStatus


def main(repo: str) -> int:
    store = FeatureStore(repo_path=repo)
    fv = store.get_feature_view("driver_hourly_stats")
    now = datetime.now(timezone.utc)
    store.materialize(start_date=now - timedelta(days=10), end_date=now - timedelta(days=3))

    features = ["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate"]
    rows = [{"driver_id": d} for d in (1001, 1002, 1003, 1004, 1005)]
    resp = store.get_online_features(features=features, entity_rows=rows)

    names = list(resp.proto.metadata.feature_names.val)
    conv = names.index("conv_rate")
    result = resp.proto.results[conv]
    print(f"feature view ttl: {fv.ttl}; read at {now:%Y-%m-%d %H:%M} UTC\n")
    print("driver_id  event_age_hours  older_than_ttl  status")
    stale_but_present = 0
    for row, ts, status in zip(rows, result.event_timestamps, result.statuses):
        age_h = (now.timestamp() - ts.seconds) / 3600 if ts.seconds else float("nan")
        stale = ts.seconds and age_h > fv.ttl.total_seconds() / 3600
        stale_but_present += bool(stale and status == FieldStatus.PRESENT)
        print(
            f"{row['driver_id']:<10} {age_h:>15.1f}  {str(bool(stale)):<14}  "
            f"{FieldStatus.Name(status)}"
        )

    print(
        f"\n{stale_but_present} of {len(rows)} values are older than the TTL "
        "but reported PRESENT (Go/Java servers would report OUTSIDE_MAX_AGE)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
