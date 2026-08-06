"""
Machine-level failure-prediction features: 30-min windows of aggregate task
activity per machine (mirrors src/features/build_features.py's task-level
windows), predicting whether the machine receives a REMOVE event (hardware
failure / decommission) in the next 30 minutes.

Windows are only built where task_usage has continuous coverage (trace days
0-10, same constraint as the task-level features) since that's what the
usage aggregates need; REMOVE labels are looked up from the FULL 29-day
machine_events table, same "never truncated" methodology as the task-level
FAIL labels.

A churn feature (count of EVICT/FAIL/KILL events among tasks scheduled on
the machine in that window) is included since the EDA found elevated task
churn preceding machine REMOVE events (notebooks/EDA_FINDINGS.md).

Output: data/processed/machine_features_window30min.parquet
        data/processed/machine_features_manifest.json

Usage:
    python src/features/machine_features.py
"""

import argparse
import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROC = REPO_ROOT / "data" / "processed"

ADD, REMOVE = 0, 1
EVICT, FAIL, KILL = 2, 3, 5

MACHINE_FEATURE_COLS = [
    "n_samples", "n_tasks",
    "cpu_sum", "cpu_mean", "cpu_std", "cpu_max",
    "mem_sum", "mem_mean", "mem_std", "mem_max",
    "disk_io_mean", "cpus", "memory",
    "cpu_utilization", "mem_utilization", "churn_events",
]
LABEL_COL = "label_remove_soon"


def build(window_us: int, horizon_us: int, out_path: Path, min_samples: int) -> dict:
    machine_events = pl.scan_parquet(PROC / "machine_events.parquet")

    capacity = (
        machine_events.filter(pl.col("event_type") == ADD)
        .sort("time")
        .group_by("machine_id")
        .agg(pl.col("cpus").first(), pl.col("memory").first())
    )
    remove_time = (
        machine_events.filter(pl.col("event_type") == REMOVE)
        .group_by("machine_id")
        .agg(pl.col("time").min().alias("remove_time"))
    )
    churn = (
        pl.scan_parquet(PROC / "task_events.parquet")
        .filter(pl.col("machine_id").is_not_null() & pl.col("event_type").is_in([EVICT, FAIL, KILL]))
        .select("time", "machine_id")
        .with_columns((pl.col("time") // window_us * window_us).alias("window_start"))
        .group_by("machine_id", "window_start")
        .agg(pl.len().alias("churn_events"))
    )

    usage = (
        pl.scan_parquet(PROC / "task_usage.parquet")
        .select("machine_id", "start_time", "job_id", "task_index", "cpu_rate",
                "canonical_memory_usage", "disk_io_time")
        .sort(["machine_id", "start_time"])
    )

    windows = (
        usage.group_by_dynamic(
            index_column="start_time", every=f"{window_us}i", period=f"{window_us}i",
            group_by=["machine_id"],
        )
        .agg(
            pl.len().alias("n_samples"),
            pl.struct(["job_id", "task_index"]).n_unique().alias("n_tasks"),
            pl.col("cpu_rate").sum().alias("cpu_sum"), pl.col("cpu_rate").mean().alias("cpu_mean"),
            pl.col("cpu_rate").std().alias("cpu_std"), pl.col("cpu_rate").max().alias("cpu_max"),
            pl.col("canonical_memory_usage").sum().alias("mem_sum"),
            pl.col("canonical_memory_usage").mean().alias("mem_mean"),
            pl.col("canonical_memory_usage").std().alias("mem_std"),
            pl.col("canonical_memory_usage").max().alias("mem_max"),
            pl.col("disk_io_time").mean().alias("disk_io_mean"),
        )
        .rename({"start_time": "window_start"})
        .with_columns((pl.col("window_start") + horizon_us).alias("window_end"))
        .filter(pl.col("n_samples") >= min_samples)
        .join(capacity, on="machine_id", how="left")
        .join(remove_time, on="machine_id", how="left")
        .join(churn, on=["machine_id", "window_start"], how="left")
        .with_columns(
            pl.col("churn_events").fill_null(0),
            pl.col("cpu_std").fill_null(0.0), pl.col("mem_std").fill_null(0.0),
            pl.when(pl.col("cpus") > 0).then(pl.col("cpu_sum") / pl.col("cpus"))
              .otherwise(0.0).alias("cpu_utilization"),
            pl.when(pl.col("memory") > 0).then(pl.col("mem_sum") / pl.col("memory"))
              .otherwise(0.0).alias("mem_utilization"),
            (
                pl.col("remove_time").is_not_null()
                & (pl.col("remove_time") > pl.col("window_end"))
                & (pl.col("remove_time") <= pl.col("window_end") + horizon_us)
            ).cast(pl.Int8).alias(LABEL_COL),
        )
        .sort("window_end")
    )

    windows.sink_parquet(out_path, compression="zstd")

    stats = pl.scan_parquet(out_path).select(
        pl.len().alias("n_rows"),
        pl.col(LABEL_COL).sum().alias("n_positive"),
        pl.col("machine_id").n_unique().alias("n_machines"),
    ).collect().row(0, named=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--out", default=str(PROC / "machine_features_window30min.parquet"))
    args = ap.parse_args()

    window_us = args.window_min * 60 * 1_000_000
    horizon_us = args.horizon_min * 60 * 1_000_000
    out_path = Path(args.out)

    print(f"Building machine-level windowed features: window={args.window_min}min, "
          f"horizon={args.horizon_min}min ...")
    stats = build(window_us, horizon_us, out_path, args.min_samples)

    manifest = {
        "window_min": args.window_min, "horizon_min": args.horizon_min,
        "min_samples": args.min_samples,
        "n_rows": stats["n_rows"], "n_machines": stats["n_machines"],
        "n_positive": stats["n_positive"],
        "positive_rate": round(stats["n_positive"] / stats["n_rows"], 5) if stats["n_rows"] else 0.0,
    }
    manifest_path = PROC / "machine_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path}\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
