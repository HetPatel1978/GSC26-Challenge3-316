"""
Build sliding-window features + forward failure labels for the task-failure
early-warning task.

Task_usage only has continuous per-task history for trace days 0-10 (shards
0-172, see data/README.md) -- task_events, however, covers the full 29-day
trace, so failure labels are never truncated by the usage window even near
its right edge.

For each task active in the continuous window, usage samples (~5 min apart)
are bucketed into non-overlapping WINDOW-sized buckets. Each bucket gets
summary statistics (mean/std/max of CPU rate, memory usage, disk I/O) plus
static per-task features from its SUBMIT event (resource requests,
scheduling class/priority). The label is 1 if the task's first FAIL event
falls in (window_end, window_end + HORIZON] -- i.e. "will this task fail in
the next HORIZON minutes", a fixed-lead-time early-warning formulation.

Output: data/processed/features_window30min.parquet
        data/processed/features_manifest.json (row counts, label balance)

Usage:
    python src/features/build_features.py
    python src/features/build_features.py --window 30m --horizon-min 30
"""

import argparse
import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROC = REPO_ROOT / "data" / "processed"

# task_events event_type codes we care about (see src/ingest/schemas.py)
SUBMIT, FAIL = 0, 3


def build(window_us: int, horizon_us: int, out_path: Path, min_samples: int) -> dict:
    # start_time is Int64 microseconds, not a Datetime -- group_by_dynamic
    # needs the integer-duration suffix ("Ni") rather than a calendar unit
    # ("30m") to bucket a plain numeric index column.
    window = f"{window_us}i"
    task_events = pl.scan_parquet(PROC / "task_events.parquet").select(
        "time", "job_id", "task_index", "event_type",
        "cpu_request", "memory_request", "disk_space_request",
        "scheduling_class", "priority",
    )

    submit = (
        task_events
        .filter(pl.col("event_type") == SUBMIT)
        .sort("time")
        .group_by(["job_id", "task_index"])
        .agg(
            pl.col("cpu_request").first(),
            pl.col("memory_request").first(),
            pl.col("disk_space_request").first(),
            pl.col("scheduling_class").first(),
            pl.col("priority").first(),
        )
    )

    fail_time = (
        task_events
        .filter(pl.col("event_type") == FAIL)
        .group_by(["job_id", "task_index"])
        .agg(pl.col("time").min().alias("fail_time"))
    )

    usage = (
        pl.scan_parquet(PROC / "task_usage.parquet")
        .select(
            "job_id", "task_index", "start_time", "cpu_rate",
            "canonical_memory_usage", "maximum_memory_usage",
            "disk_io_time", "assigned_memory_usage",
        )
        .sort(["job_id", "task_index", "start_time"])
    )

    windows = (
        usage.group_by_dynamic(
            index_column="start_time",
            every=window,
            period=window,
            group_by=["job_id", "task_index"],
        )
        .agg(
            pl.len().alias("n_samples"),
            pl.col("cpu_rate").mean().alias("cpu_mean"),
            pl.col("cpu_rate").std().alias("cpu_std"),
            pl.col("cpu_rate").max().alias("cpu_max"),
            pl.col("canonical_memory_usage").mean().alias("mem_mean"),
            pl.col("canonical_memory_usage").std().alias("mem_std"),
            pl.col("canonical_memory_usage").max().alias("mem_max"),
            pl.col("maximum_memory_usage").mean().alias("mem_peak_mean"),
            pl.col("disk_io_time").mean().alias("disk_io_mean"),
            pl.col("assigned_memory_usage").mean().alias("assigned_mem_mean"),
        )
        .rename({"start_time": "window_start"})
        .with_columns((pl.col("window_start") + horizon_us).alias("window_end"))
        .filter(pl.col("n_samples") >= min_samples)
        .join(submit, on=["job_id", "task_index"], how="inner")
        .join(fail_time, on=["job_id", "task_index"], how="left")
        .with_columns(
            (
                pl.col("fail_time").is_not_null()
                & (pl.col("fail_time") > pl.col("window_end"))
                & (pl.col("fail_time") <= pl.col("window_end") + horizon_us)
            ).cast(pl.Int8).alias("label_fail_soon"),
            pl.when(pl.col("memory_request") > 0)
              .then(pl.col("mem_mean") / pl.col("memory_request"))
              .otherwise(None).alias("mem_usage_ratio"),
            pl.when(pl.col("cpu_request") > 0)
              .then(pl.col("cpu_mean") / pl.col("cpu_request"))
              .otherwise(None).alias("cpu_usage_ratio"),
            pl.col("cpu_std").fill_null(0.0),
            pl.col("mem_std").fill_null(0.0),
        )
        .sort("window_end")
    )

    windows.sink_parquet(out_path, compression="zstd")

    stats = pl.scan_parquet(out_path).select(
        pl.len().alias("n_rows"),
        pl.col("label_fail_soon").sum().alias("n_positive"),
        pl.struct(["job_id", "task_index"]).n_unique().alias("n_tasks"),
        pl.col("window_end").min().alias("t_min"),
        pl.col("window_end").max().alias("t_max"),
    ).collect().row(0, named=True)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=2,
                     help="drop windows with fewer than this many usage samples (noisy stats)")
    ap.add_argument("--out", default=str(PROC / "features_window30min.parquet"))
    args = ap.parse_args()

    window_us = args.window_min * 60 * 1_000_000
    horizon_us = args.horizon_min * 60 * 1_000_000
    out_path = Path(args.out)

    print(f"Building windowed features: window={args.window_min}min, horizon={args.horizon_min}min ...")
    stats = build(window_us, horizon_us, out_path, args.min_samples)

    manifest = {
        "window_min": args.window_min,
        "horizon_min": args.horizon_min,
        "min_samples": args.min_samples,
        "n_rows": stats["n_rows"],
        "n_tasks": stats["n_tasks"],
        "n_positive": stats["n_positive"],
        "positive_rate": round(stats["n_positive"] / stats["n_rows"], 5) if stats["n_rows"] else 0.0,
        "window_end_range_us": [stats["t_min"], stats["t_max"]],
    }
    manifest_path = PROC / "features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
