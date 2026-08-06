"""
Lead-time analysis: for every task whose actual FAIL event falls in the
held-out test period, find the FIRST test-period window where the trained
XGBoost model's predicted probability crossed its tuned decision threshold,
and measure how long before the real failure that first alert fired.

This is a task-level notion of "true positive" (did we ever warn about this
failure before it happened?) distinct from the window-level precision/
recall in the main results table (was THIS SPECIFIC 30-min window flagged
correctly?) -- a model can raise a correct early alert hours before failure
even though that early window's own label is 0 (its own 30-min horizon
doesn't contain the failure), which is exactly the scenario worth
measuring here.

Usage:
    python src/eval/lead_time.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROC = REPO_ROOT / "data" / "processed"
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import FEATURE_COLS, FEATURES_PATH, NULLABLE_COLS  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "xgboost.pkl"
METRICS_PATH = PROC / "model_metrics.json"
PLOT_PATH = REPO_ROOT / "results" / "plots" / "lead_time_distribution.png"
SUMMARY_PATH = REPO_ROOT / "results" / "lead_time.json"
US_PER_MIN = 60_000_000


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    xgb_meta = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["xgboost"]
    threshold = xgb_meta["threshold"]
    print(f"using XGBoost, tuned threshold={threshold}")

    df = pl.read_parquet(FEATURES_PATH)
    medians = {c: df[c].median() for c in NULLABLE_COLS}
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in medians.items()])
    cutoff = df["window_end"].quantile(0.8)

    test = df.filter(pl.col("window_end") > cutoff)
    X_test = test.select(FEATURE_COLS).to_numpy().astype(np.float64)
    test = test.with_columns(pl.Series("predicted_proba", model.predict_proba(X_test)[:, 1]))

    # Ground-truth universe: tasks with an actual *observed* imminent-failure
    # window in the test set (label_fail_soon == 1) -- i.e. exactly the
    # n_positive population behind the main results table's confusion
    # matrix. Using "any task whose fail_time > cutoff" instead would also
    # sweep in tasks whose last observed usage sample is days before their
    # eventual failure (fail_time is looked up from the full 29-day trace,
    # not bounded to the 10-day usage window) -- those have no real
    # "advance warning" story, just an unobserved gap, and produced lead
    # times up to 25,000+ minutes that swamped the distribution.
    failing_tasks = (
        test.filter(pl.col("label_fail_soon") == 1)
        .select("job_id", "task_index", "fail_time")
        .unique()
    )
    n_failing = failing_tasks.height
    print(f"test-period task failures (observed imminent-failure window): {n_failing:,}")

    # First window (by time) where the model crossed threshold, strictly
    # before the actual failure -- one row per detected task, restricted to
    # the ground-truth failing_tasks universe defined above.
    first_alerts = (
        test.join(failing_tasks.select("job_id", "task_index"), on=["job_id", "task_index"], how="inner")
        .filter((pl.col("predicted_proba") >= threshold) & (pl.col("window_end") < pl.col("fail_time")))
        .group_by("job_id", "task_index")
        .agg(pl.col("window_end").min().alias("first_alert_time"), pl.col("fail_time").first())
        .with_columns(
            ((pl.col("fail_time") - pl.col("first_alert_time")) / US_PER_MIN).alias("lead_time_min")
        )
    )
    n_detected = first_alerts.height
    lead_times = first_alerts["lead_time_min"].to_numpy()

    summary = {
        "model": "xgboost", "threshold": threshold,
        "n_test_period_failures": n_failing,
        "n_detected_with_advance_warning": n_detected,
        "detection_rate": round(n_detected / n_failing, 4) if n_failing else 0.0,
        "lead_time_minutes": {
            "mean": round(float(np.mean(lead_times)), 2) if n_detected else None,
            "median": round(float(np.median(lead_times)), 2) if n_detected else None,
            "min": round(float(np.min(lead_times)), 2) if n_detected else None,
            "max": round(float(np.max(lead_times)), 2) if n_detected else None,
            "p25": round(float(np.percentile(lead_times, 25)), 2) if n_detected else None,
            "p75": round(float(np.percentile(lead_times, 75)), 2) if n_detected else None,
        },
    }
    print(json.dumps(summary, indent=2))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(lead_times, bins=40, color="#3b6ea5")
    ax.axvline(summary["lead_time_minutes"]["median"], color="#a53b3b", ls="--", lw=1.2,
               label=f"median = {summary['lead_time_minutes']['median']:.1f} min")
    ax.axvline(summary["lead_time_minutes"]["mean"], color="#c98a2c", ls=":", lw=1.2,
               label=f"mean = {summary['lead_time_minutes']['mean']:.1f} min")
    ax.set_xlabel("lead time (minutes before actual FAIL event)")
    ax.set_ylabel("# tasks")
    ax.set_title(f"XGBoost advance-warning lead time "
                 f"({n_detected:,}/{n_failing:,} test-period failures detected, "
                 f"{summary['detection_rate']*100:.1f}%)")
    ax.legend()
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
