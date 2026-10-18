"""Analyse evaluation-harness results exactly as docs/eval-plan.md specifies.

The rules come from the plan's Analysis section and its Deviation 10, which was
committed before any March detection value was read. This script applies them
mechanically and reports negative results as found.

    python scripts/analyze.py tune results/tune-20240301-20240331.parquet
    python scripts/analyze.py report results/<eval-run>.parquet \\
        --tuning results/tune-20240301-20240331.tuning.json

``tune`` reads the March run and writes ``results/<run>.tuning.json``
(``max_unseen_mass`` and the per-window-size factor for ``fixed_matched``).
``report`` writes ``results/<run>.RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLEAN = "none"
NUMERIC = ("trip_distance", "fare_amount", "passenger_count")
UNSEEN_CHECKS = ("out_of_range", "new_categories")
SCALED_CHECKS = ("psi", "linf")
UNSEEN_CANDIDATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
K_GRID = np.round(np.arange(0.10, 5.0 + 1e-9, 0.01), 2)
TARGET_RATE = 0.01
H2_MAX_ROWS = 150
H2_LIMIT = 0.05
BOOT = 10_000
TIMED_FAULTS = ("gradual_shift", "stale_source")
GRADUAL_SIZES = (1000, 5000)
ORDER = ("none", "null_rule", "fixed", "fixed_matched", "skewguard", "skewguard_bonf")


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #


def run_name(path: Path) -> str:
    """``results/<run>.parquet`` and ``results/<run>/`` both name run ``<run>``."""
    return path.stem if path.suffix == ".parquet" else path.name


def load_run(path: Path) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """One run's rows, with Feast failures dropped and counted.

    ``path`` is the combined ``results/<run>.parquet`` (its manifest is
    ``results/<run>.manifest.json``) or the harness's per-cell directory
    ``results/<run>/`` (manifest inside it).
    """
    if path.suffix == ".parquet":
        files = [path]
        manifest_path = path.with_name(f"{run_name(path)}.manifest.json")
    else:
        files = sorted(path.glob("*.parquet"))
        manifest_path = path / "manifest.json"
    if not files or not files[0].exists():
        raise SystemExit(f"no result files at {path}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ok = df["feast_status"] == "ok"
    return df[ok].reset_index(drop=True), manifest, int((~ok).sum())


def check_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (result row, feature, check) for detectors that store checks."""
    records = []
    for row, raw in df["checks"].items():
        if not raw:
            continue
        for feature, checks in json.loads(raw).items():
            for name, entry in checks.items():
                if isinstance(entry, list):
                    status, value, threshold = entry
                    records.append((row, feature, name, status, value, threshold))
    out = pd.DataFrame(records, columns=["row", "feature", "check", "status", "value", "threshold"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
#  Re-thresholding
# --------------------------------------------------------------------------- #


def alerting(
    checks: pd.DataFrame, unseen: float | None = None, scale: float | None = None
) -> pd.DataFrame:
    """``checks`` with an ``alert`` column after optional re-thresholding.

    ``unseen`` replaces the threshold of the out-of-range / new-category checks;
    ``scale`` multiplies the stored PSI / L-inf thresholds. Other checks keep
    their stored status, and checks with no value (skipped) never alert.
    """
    c = checks.copy()
    c["alert"] = c["status"] == "alert"
    if unseen is not None:
        m = c["check"].isin(UNSEEN_CHECKS) & c["value"].notna()
        c.loc[m, "alert"] = c.loc[m, "value"] > unseen
    if scale is not None:
        m = c["check"].isin(SCALED_CHECKS) & c["value"].notna()
        c.loc[m, "alert"] = c.loc[m, "value"] > c.loc[m, "threshold"] * scale
    return c


def recompute(
    df: pd.DataFrame, checks: pd.DataFrame, rows: pd.Index, **kw: float | None
) -> pd.DataFrame:
    """``alert`` and ``target_alert`` of ``rows`` after re-thresholding their checks."""
    c = alerting(checks[checks["row"].isin(rows)], **kw)
    hits = c[c["alert"]].groupby("row")["feature"].agg(set)
    out = df.loc[rows].copy()
    features = out.index.map(lambda r: hits.get(r, set()))
    out["alert"] = [bool(f) for f in features]
    out["target_alert"] = [
        (t in f) if t else None for t, f in zip(out["target"].fillna(""), features)
    ]
    return out


def tune(df: pd.DataFrame, checks: pd.DataFrame) -> dict[str, Any]:
    """Deviation 10: choose ``max_unseen_mass`` and the ``fixed_matched`` factors on March."""
    clean = df["fault"] == CLEAN
    sizes = sorted(int(s) for s in df.loc[clean, "window_size"].unique())

    sg_rows = df.index[clean & (df["detector"] == "skewguard")]
    u = checks[
        checks["row"].isin(sg_rows) & checks["check"].isin(UNSEEN_CHECKS) & checks["value"].notna()
    ].join(df["window_size"], on="row")
    rates: dict[str, dict[str, dict[str, float]]] = {}
    choice = UNSEEN_CANDIDATES[-1]
    for cand in reversed(UNSEEN_CANDIDATES):  # smallest qualifying candidate wins
        r = u.assign(hit=u["value"] > cand).groupby(["check", "window_size"])["hit"].mean()
        rates[str(cand)] = {
            chk: {str(int(s)): float(v) for s, v in r[chk].items()} for chk in r.index.levels[0]
        }
        if len(r) and (r <= TARGET_RATE).all():
            choice = cand

    target = recompute(df, checks, sg_rows, unseen=choice).groupby("window_size")["alert"].mean()

    fixed_rows = df.index[clean & (df["detector"] == "fixed")]
    c = alerting(checks[checks["row"].isin(fixed_rows)], unseen=choice)
    scaled = c["check"].isin(SCALED_CHECKS) & c["value"].notna()
    ratio = (
        (c.loc[scaled, "value"] / c.loc[scaled, "threshold"]).groupby(c.loc[scaled, "row"]).max()
    )
    base = c[~scaled & c["alert"]].groupby("row").size()
    fixed = df.loc[fixed_rows, ["window_size"]].copy()
    fixed["ratio"] = ratio.reindex(fixed.index).fillna(-np.inf)
    fixed["base"] = base.reindex(fixed.index).fillna(0) > 0

    factors, achieved = {}, {}
    for size in sizes:
        f = fixed[fixed["window_size"] == size]
        fpr = np.array([(f["base"] | (f["ratio"] > k)).mean() for k in K_GRID])
        gap = np.abs(fpr - float(target.get(size, np.nan)))
        best = int(np.flatnonzero(gap == gap.min())[-1])  # ties go to the larger k
        factors[str(size)] = float(K_GRID[best])
        achieved[str(size)] = float(fpr[best])

    return {
        "max_unseen_mass": choice,
        "unseen_clean_rates": rates,
        "skewguard_day_fpr": {str(int(s)): float(v) for s, v in target.items()},
        "fixed_matched_factor": factors,
        "fixed_matched_day_fpr": achieved,
    }


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #


def bootstrap_ci(x: Sequence[float], boot: int = BOOT, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the mean, resampling ``x`` with replacement."""
    a = np.asarray(x, dtype=float)
    if len(a) == 0:
        return math.nan, math.nan
    idx = np.random.default_rng(seed).integers(0, len(a), size=(boot, len(a)))
    means = a[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(
    a: np.ndarray, b: np.ndarray, strata: np.ndarray, boot: int = BOOT, seed: int = 0
) -> tuple[float, float]:
    """95% interval of mean(a - b), resampling pairs within each stratum."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    total = np.zeros(boot)
    for s in pd.unique(strata):
        ds = d[strata == s]
        total += ds[rng.integers(0, len(ds), size=(boot, len(ds)))].sum(axis=1)
    means = total / len(d)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------- #
#  Report pieces
# --------------------------------------------------------------------------- #


def with_matched(df: pd.DataFrame, checks: pd.DataFrame, factors: dict[str, float]) -> pd.DataFrame:
    """Append ``fixed_matched`` rows: detector 3 with its thresholds scaled per window size."""
    parts = [df]
    for size, k in factors.items():
        rows = df.index[(df["detector"] == "fixed") & (df["window_size"] == int(size))]
        if len(rows):
            m = recompute(df, checks, rows, scale=k)
            m["detector"] = "fixed_matched"
            parts.append(m)
    return pd.concat(parts, ignore_index=True)


def incidents(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (detector, fault, severity, window size, incident)."""
    f = df[df["fault"] != CLEAN].copy()
    f["target_hit"] = f["target_alert"].fillna(False).astype(bool)
    f["alert_offset"] = f["day_offset"].where(f["alert"])
    g = f.groupby(["detector", "fault", "severity", "window_size", "incident"])
    return g.agg(
        detected=("alert", "any"),
        target_detected=("target_hit", "any"),
        first_offset=("alert_offset", "min"),
        has_target=("target", lambda t: bool(t.iloc[0])),
    ).reset_index()


def fmt_rate(x: Sequence[float]) -> str:
    if len(x) == 0:
        return "-"
    lo, hi = bootstrap_ci(x)
    return f"{np.mean(x):.0%} [{lo:.0%}, {hi:.0%}]"


def detectors_in(df: pd.DataFrame) -> list[str]:
    present = set(df["detector"])
    return [d for d in ORDER if d in present] + sorted(present - set(ORDER))


def report(
    df: pd.DataFrame,
    checks: pd.DataFrame,
    manifest: dict[str, Any],
    excluded: int,
    tuning: dict[str, Any] | None,
) -> str:
    lines: list[str] = ["# skewguard evaluation results", ""]
    run_range = manifest.get("config", {}).get("range", "?")
    lines += [
        "Generated by `scripts/analyze.py` under the rules in `docs/eval-plan.md`",
        "(Analysis section and Deviation 10). Intervals are 95% percentile bootstraps",
        f"({BOOT:,} resamples, seed 0).",
        "",
        f"- Run: `{run_range}`, git `{manifest.get('git_head', '?')}`, plan commit "
        f"`{manifest.get('plan_commit', '?')}`, Feast {manifest.get('feast_version', '?')}",
        f"- Rows excluded for a Feast failure: {excluded}",
    ]
    if tuning:
        cfg = manifest.get("config", {}).get("max_unseen_mass")
        lines.append(f"- Tuned `max_unseen_mass` = {tuning['max_unseen_mass']} (run used {cfg})")
        if cfg is not None and float(cfg) != float(tuning["max_unseen_mass"]):
            lines.append("- **Warning: the run did not use the tuned `max_unseen_mass`.**")
        df = with_matched(df, checks, tuning["fixed_matched_factor"])
        lines.append(f"- `fixed_matched` factors per window size: {tuning['fixed_matched_factor']}")
    dets = detectors_in(df)
    sizes = sorted(int(s) for s in df["window_size"].unique())
    clean = df[df["fault"] == CLEAN]

    # H2 ------------------------------------------------------------------- #
    lines += ["", "## H2: fixed PSI 0.25 on small clean windows", ""]
    fixed_clean = clean.index[
        (clean["detector"] == "fixed") & (clean["window_size"] <= H2_MAX_ROWS)
    ]
    h2 = checks[
        checks["row"].isin(fixed_clean)
        & (checks["check"] == "psi")
        & checks["feature"].isin(NUMERIC)
        & checks["value"].notna()
    ]
    if len(h2):
        per_day = (h2["status"] == "alert").groupby(h2["row"]).mean()
        lo, hi = bootstrap_ci(per_day.to_numpy())
        rate = float((h2["status"] == "alert").mean())
        verdict = (
            "supported"
            if lo > H2_LIMIT
            else ("not supported" if hi <= H2_LIMIT else "inconclusive")
        )
        lines += [
            f"PSI > 0.25 on clean numeric feature-windows of <= {H2_MAX_ROWS} rows: "
            f"{rate:.1%} [{lo:.1%}, {hi:.1%}] over {len(h2)} feature-windows. "
            f"H2 (> {H2_LIMIT:.0%}): **{verdict}**."
        ]
    else:
        lines.append("No clean window of <= 150 rows in this run.")

    # Clean-day FPR -------------------------------------------------------- #
    lines += ["", "## False-positive rate on clean days (day level)", ""]
    lines += ["| detector | " + " | ".join(f"{s} rows" for s in sizes) + " |"]
    lines += ["|---|" + "---|" * len(sizes)]
    for d in dets:
        cells = [
            fmt_rate(clean[(clean["detector"] == d) & (clean["window_size"] == s)]["alert"])
            for s in sizes
        ]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")

    # Detection ------------------------------------------------------------ #
    inc = incidents(df)
    lines += ["", "## Detection rate per fault (incident detected if any alert)", ""]
    for fault in sorted(inc["fault"].unique()):
        lines += [f"### {fault}", "", "| severity | rows | " + " | ".join(dets) + " |"]
        lines += ["|---|---|" + "---|" * len(dets)]
        sub = inc[inc["fault"] == fault]
        for sev in ("low", "medium", "high"):
            for s in sizes:
                cell = sub[(sub["severity"] == sev) & (sub["window_size"] == s)]
                if cell.empty:
                    continue
                vals = [fmt_rate(cell[cell["detector"] == d]["detected"]) for d in dets]
                lines.append(f"| {sev} | {s} | " + " | ".join(vals) + " |")
        targeted = sub[sub["has_target"]]
        if not targeted.empty:
            rate = targeted.groupby("detector")["target_detected"].mean()
            lines += [
                "",
                "Secondary, alert on the injected feature (all severities and sizes): "
                + ", ".join(f"{d} {rate[d]:.0%}" for d in dets if d in rate),
            ]
        lines.append("")

    # H1 ------------------------------------------------------------------- #
    lines += ["## H1: skewguard vs fixed thresholds, paired on identical incidents", ""]
    for other in ("fixed", "fixed_matched"):
        if other not in dets or "skewguard" not in dets:
            continue
        rows, pvals = [], []
        for s in sizes:
            pair = _pair(inc[inc["window_size"] == s], "skewguard", other)
            if pair is None:
                continue
            rows.append((s, *pair))
            pvals.append(pair[-1])
        adjusted = holm(pvals) if pvals else []
        lines += [
            f"### skewguard vs {other}",
            "",
            "| rows | incidents | skewguard | other | difference [95% CI] | only skewguard "
            "| only other | McNemar p | Holm p | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (s, n, ra, rb, lo, hi, b, c, p), padj in zip(rows, adjusted):
            verdict = (
                "falsified" if hi < 0 else ("skewguard ahead" if lo > 0 else "no clear difference")
            )
            lines.append(
                f"| {s} | {n} | {ra:.1%} | {rb:.1%} | {ra - rb:+.1%} [{lo:+.1%}, {hi:+.1%}] "
                f"| {b} | {c} | {p:.3g} | {padj:.3g} | {verdict} |"
            )
        lines.append("")
        grad = inc[(inc["fault"] == "gradual_shift") & inc["window_size"].isin(GRADUAL_SIZES)]
        pair = _pair(grad, "skewguard", other)
        if pair is not None:
            n, ra, rb, lo, hi, b, c, p = pair
            lines += [
                f"Gradual shift at {', '.join(map(str, GRADUAL_SIZES))} rows: "
                f"skewguard {ra:.1%} vs {other} {rb:.1%}, difference {ra - rb:+.1%} "
                f"[{lo:+.1%}, {hi:+.1%}], McNemar p = {p:.3g}.",
                "",
            ]

    # Time to detect ------------------------------------------------------- #
    lines += ["## Time to detect", "", "| fault | severity | rows | " + " | ".join(dets) + " |"]
    lines += ["|---|---|---|" + "---|" * len(dets)]
    for fault in TIMED_FAULTS:
        sub = inc[inc["fault"] == fault]
        for sev in ("low", "medium", "high"):
            for s in sizes:
                cell = sub[(sub["severity"] == sev) & (sub["window_size"] == s)]
                if cell.empty:
                    continue
                vals = []
                for d in dets:
                    x = cell[cell["detector"] == d]
                    found = x["first_offset"].dropna()
                    vals.append(
                        f"{len(found)}/{len(x)}"
                        + (f", median day {found.median():g}" if len(found) else "")
                    )
                lines.append(f"| {fault} | {sev} | {s} | " + " | ".join(vals) + " |")

    # Runtime -------------------------------------------------------------- #
    lines += [
        "",
        "## Detector runtime per feature (ms)",
        "",
        "| detector | cache | p50 | p99 | n |",
        "|---|---|---|---|---|",
    ]
    timed = df[df["detector"].isin(["fixed", "skewguard", "skewguard_bonf"])].copy()
    timed["cache"] = np.where(timed["null_cache_misses"].fillna(0) > 0, "cold", "warm")
    timed = timed.explode("feature_runtime_ms")
    timed["feature_runtime_ms"] = pd.to_numeric(timed["feature_runtime_ms"], errors="coerce")
    for (d, cache), g in timed.groupby(["detector", "cache"]):
        v = g["feature_runtime_ms"].dropna()
        if len(v):
            lines.append(
                f"| {d} | {cache} | {v.quantile(0.5):.2f} | {v.quantile(0.99):.2f} | {len(v)} |"
            )
    return "\n".join(lines) + "\n"


def _pair(inc: pd.DataFrame, a: str, b: str):
    """Paired comparison of detectors ``a`` and ``b`` over identical incidents."""
    key = ["fault", "severity", "window_size", "incident"]
    wide = inc[inc["detector"].isin([a, b])].pivot_table(
        index=key, columns="detector", values="detected", aggfunc="first"
    )
    if a not in wide or b not in wide:
        return None
    wide = wide.dropna()
    if wide.empty:
        return None
    xa, xb = wide[a].astype(bool).to_numpy(), wide[b].astype(bool).to_numpy()
    strata = (
        wide.index.get_level_values("fault") + "/" + wide.index.get_level_values("severity")
    ).to_numpy()
    lo, hi = paired_diff_ci(xa, xb, strata)
    only_a, only_b = int((xa & ~xb).sum()), int((~xa & xb).sum())
    return (
        len(wide),
        float(xa.mean()),
        float(xb.mean()),
        lo,
        hi,
        only_a,
        only_b,
        mcnemar_exact(only_a, only_b),
    )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="analyze")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune", help="choose max_unseen_mass and fixed_matched factors (March)")
    t.add_argument("run", type=Path)
    r = sub.add_parser("report", help="write RESULTS.md for an evaluation run")
    r.add_argument("run", type=Path)
    r.add_argument("--tuning", type=Path, help="<tune-run>.tuning.json from the March run")
    r.add_argument("--out", type=Path, help="default: results/<run>.RESULTS.md")
    args = parser.parse_args(argv)

    df, manifest, excluded = load_run(args.run)
    checks = check_table(df)
    stem = args.run.parent / run_name(args.run)
    if args.command == "tune":
        result = tune(df, checks)
        result["source"] = {
            k: manifest.get(k) for k in ("git_head", "plan_commit", "config_digest")
        }
        out = stem.with_name(f"{stem.name}.tuning.json")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"max_unseen_mass = {result['max_unseen_mass']}")
        print(f"fixed_matched factors = {result['fixed_matched_factor']}")
        print(f"wrote {out}")
    else:
        tuning = json.loads(args.tuning.read_text()) if args.tuning else None
        text = report(df, checks, manifest, excluded, tuning)
        out = args.out or stem.with_name(f"{stem.name}.RESULTS.md")
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
