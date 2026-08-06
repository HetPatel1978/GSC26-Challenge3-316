"""
Precompute a small, git-trackable sample dataset for the Streamlit
dashboard (app/dashboard.py) so it's runnable without the full ~27GB local
trace download: usage time series + XGBoost risk scores + actual
FAIL/REMOVE events for a handful of representative machines.

Output: data/dashboard_sample.parquet (machine-level, 30-min windows) and
        data/dashboard_events.parquet (FAIL/REMOVE event markers)
Both are small (a few thousand rows) and committed to git, unlike
data/processed/* (gitignored, requires the full local download to rebuild).

Usage:
    python src/eval/build_dashboard_sample.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROC = REPO_ROOT / "data" / "processed"
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import FEATURE_COLS  # noqa: E402

WINDOW_US = 30 * 60 * 1_000_000
N_EXTRA_REMOVED = 3
N_HEALTHY = 3
# machines used as EDA worked examples (notebooks/EDA_FINDINGS.md) -- keep
# them in the dashboard sample so the report and the dashboard tell the
# same story.
SEED_MACHINES = [372629475, 2274790707]


def pick_sample_machines() -> list[int]:
    me = pl.scan_parquet(PROC / "machine_events.parquet")
    removed = me.filter(pl.col("event_type") == 1).select("machine_id").unique().collect()
    all_machines = me.select("machine_id").unique().collect()
    healthy_pool = all_machines.join(removed, on="machine_id", how="anti")

    extra_removed = (
        removed.filter(~pl.col("machine_id").is_in(SEED_MACHINES))
        .sample(n=N_EXTRA_REMOVED, seed=42)["machine_id"].to_list()
    )
    healthy = healthy_pool.sample(n=N_HEALTHY, seed=42)["machine_id"].to_list()
    machines = SEED_MACHINES + extra_removed + healthy
    print(f"sample machines: {machines}")
    return machines


def build_machine_features(machine_id: int, task_events_full: pl.LazyFrame) -> pl.DataFrame | None:
    usage = (
        pl.scan_parquet(PROC / "task_usage.parquet")
        .filter(pl.col("machine_id") == machine_id)
        .select(
            "job_id", "task_index", "start_time", "cpu_rate",
            "canonical_memory_usage", "maximum_memory_usage",
            "disk_io_time", "assigned_memory_usage",
        )
        .sort(["job_id", "task_index", "start_time"])
        .collect()
    )
    if usage.height == 0:
        return None

    task_ids = usage.select("job_id", "task_index").unique()
    te = (
        task_events_full
        .join(task_ids.lazy(), on=["job_id", "task_index"], how="inner")
        .collect()
    )
    submit = (
        te.filter(pl.col("event_type") == 0)
        .sort("time")
        .group_by(["job_id", "task_index"])
        .agg(
            pl.col("cpu_request").first(), pl.col("memory_request").first(),
            pl.col("disk_space_request").first(), pl.col("scheduling_class").first(),
            pl.col("priority").first(),
        )
    )
    fail_time = (
        te.filter(pl.col("event_type") == 3)
        .group_by(["job_id", "task_index"]).agg(pl.col("time").min().alias("fail_time"))
    )

    windows = (
        usage.lazy()
        .group_by_dynamic(
            index_column="start_time", every=f"{WINDOW_US}i", period=f"{WINDOW_US}i",
            group_by=["job_id", "task_index"],
        )
        .agg(
            pl.len().alias("n_samples"),
            pl.col("cpu_rate").mean().alias("cpu_mean"), pl.col("cpu_rate").std().alias("cpu_std"),
            pl.col("cpu_rate").max().alias("cpu_max"),
            pl.col("canonical_memory_usage").mean().alias("mem_mean"),
            pl.col("canonical_memory_usage").std().alias("mem_std"),
            pl.col("canonical_memory_usage").max().alias("mem_max"),
            pl.col("maximum_memory_usage").mean().alias("mem_peak_mean"),
            pl.col("disk_io_time").mean().alias("disk_io_mean"),
            pl.col("assigned_memory_usage").mean().alias("assigned_mem_mean"),
        )
        .rename({"start_time": "window_start"})
        .with_columns((pl.col("window_start") + WINDOW_US).alias("window_end"))
        .filter(pl.col("n_samples") >= 2)
        .join(submit.lazy(), on=["job_id", "task_index"], how="inner")
        .join(fail_time.lazy(), on=["job_id", "task_index"], how="left")
        .with_columns(
            (
                pl.col("fail_time").is_not_null()
                & (pl.col("fail_time") > pl.col("window_end"))
                & (pl.col("fail_time") <= pl.col("window_end") + WINDOW_US)
            ).cast(pl.Int8).alias("label_fail_soon"),
            pl.when(pl.col("memory_request") > 0)
              .then(pl.col("mem_mean") / pl.col("memory_request")).otherwise(0.0).alias("mem_usage_ratio"),
            pl.when(pl.col("cpu_request") > 0)
              .then(pl.col("cpu_mean") / pl.col("cpu_request")).otherwise(0.0).alias("cpu_usage_ratio"),
            pl.col("cpu_std").fill_null(0.0), pl.col("mem_std").fill_null(0.0),
            pl.col("cpu_request").fill_null(0.0), pl.col("memory_request").fill_null(0.0),
            pl.col("disk_space_request").fill_null(0.0),
            pl.lit(machine_id, dtype=pl.Int64).alias("machine_id"),
        )
        .collect()
    )
    return windows


def main():
    task_events_full = pl.scan_parquet(PROC / "task_events.parquet").select(
        "time", "job_id", "task_index", "event_type",
        "cpu_request", "memory_request", "disk_space_request", "scheduling_class", "priority",
    )
    with open(REPO_ROOT / "models" / "xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    machines = pick_sample_machines()
    task_window_frames = []
    for mid in machines:
        print(f"machine {mid}: pulling usage + building windows ...")
        wf = build_machine_features(mid, task_events_full)
        if wf is not None:
            task_window_frames.append(wf)

    all_windows = pl.concat(task_window_frames)
    X = all_windows.select(FEATURE_COLS).to_numpy().astype(np.float64)
    all_windows = all_windows.with_columns(
        pl.Series("predicted_risk", xgb_model.predict_proba(X)[:, 1])
    )

    machine_ts = (
        all_windows
        .group_by(["machine_id", "window_start", "window_end"])
        .agg(
            pl.col("cpu_mean").sum().alias("cpu_sum"),
            pl.col("mem_mean").sum().alias("mem_sum"),
            pl.len().alias("n_tasks"),
            pl.col("predicted_risk").max().alias("max_predicted_risk"),
            pl.col("label_fail_soon").max().alias("any_fail_soon"),
        )
        .sort(["machine_id", "window_start"])
    )
    out_path = REPO_ROOT / "data" / "dashboard_sample.parquet"
    machine_ts.write_parquet(out_path)
    print(f"wrote {out_path} ({machine_ts.height} rows)")

    me = pl.scan_parquet(PROC / "machine_events.parquet").filter(
        pl.col("machine_id").is_in(machines)
    ).collect()
    fail_events = (
        all_windows.filter(pl.col("fail_time").is_not_null())
        .select("machine_id", "job_id", "task_index", "fail_time").unique()
        .rename({"fail_time": "time"})
        .with_columns(pl.lit("task_fail").alias("event"))
    )
    machine_events_out = (
        me.filter(pl.col("event_type") == 1)
        .select("machine_id", "time")
        .with_columns(pl.lit("machine_remove").alias("event"), pl.lit(None, dtype=pl.Int64).alias("job_id"),
                      pl.lit(None, dtype=pl.Int64).alias("task_index"))
        .select("machine_id", "job_id", "task_index", "time", "event")
    )
    events_out = pl.concat([
        fail_events.select("machine_id", "job_id", "task_index", "time", "event"),
        machine_events_out,
    ])
    events_path = REPO_ROOT / "data" / "dashboard_events.parquet"
    events_out.write_parquet(events_path)
    print(f"wrote {events_path} ({events_out.height} rows)")


if __name__ == "__main__":
    main()
