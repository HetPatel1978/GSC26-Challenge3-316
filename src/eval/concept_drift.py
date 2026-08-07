"""
Concept-drift check: does the tuned XGBoost model's performance hold up
across the held-out test period, or decay the further out it scores from
the training cutoff? Splits the test period (last 20% of the trace by
window_end) into 3 equal-width time bins and reports ROC-AUC/PR-AUC
(threshold-independent) plus precision/recall/F1 at the model's *fixed*
tuned threshold (not re-tuned per bin -- a deployed model doesn't get to
re-tune itself, so this tests whether the original operating point still
works later in time).

Usage:
    python src/eval/concept_drift.py
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
from eval.metrics import summarize  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "xgboost_tuned.pkl"
METRICS_PATH = PROC / "model_metrics.json"
PLOT_PATH = REPO_ROOT / "results" / "plots" / "concept_drift.png"
SUMMARY_PATH = REPO_ROOT / "results" / "concept_drift.json"
N_BINS = 3
US_PER_DAY = 1_000_000 * 86400


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    threshold = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["xgboost_tuned"]["threshold"]
    print(f"using XGBoost (tuned), fixed threshold={threshold}")

    df = pl.read_parquet(FEATURES_PATH)
    medians = {c: df[c].median() for c in NULLABLE_COLS}
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in medians.items()])
    cutoff = df["window_end"].quantile(0.8)

    test = df.filter(pl.col("window_end") > cutoff)
    t_min, t_max = test["window_end"].min(), test["window_end"].max()
    bin_width = (t_max - t_min) / N_BINS

    rows = []
    for i in range(N_BINS):
        lo = t_min + i * bin_width
        hi = t_max if i == N_BINS - 1 else t_min + (i + 1) * bin_width
        bin_df = test.filter((pl.col("window_end") >= lo) & (pl.col("window_end") <= hi))

        X = bin_df.select(FEATURE_COLS).to_numpy().astype(np.float64)
        y = bin_df["label_fail_soon"].to_numpy()
        proba = model.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        m = summarize(y, y_pred, proba)
        row = {
            "bin": i + 1,
            "day_range": [round(lo / US_PER_DAY, 2), round(hi / US_PER_DAY, 2)],
            "n_rows": int(bin_df.height),
            "n_positive": int(y.sum()),
            **m,
        }
        rows.append(row)
        print(f"bin {i+1} (day {row['day_range'][0]}-{row['day_range'][1]}, "
              f"{row['n_rows']:,} rows, {row['n_positive']:,} pos): "
              f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
              f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")

    roc_aucs = [r["roc_auc"] for r in rows]
    pr_aucs = [r["pr_auc"] for r in rows]
    roc_drift = round(roc_aucs[-1] - roc_aucs[0], 4)
    pr_drift = round(pr_aucs[-1] - pr_aucs[0], 4)

    summary = {
        "model": "xgboost_tuned", "threshold": threshold, "n_bins": N_BINS,
        "bins": rows,
        "roc_auc_drift_last_minus_first": roc_drift,
        "pr_auc_drift_last_minus_first": pr_drift,
        "verdict": (
            "holds up over time (no meaningful decay)"
            if abs(roc_drift) < 0.02 else
            ("degrades over time" if roc_drift < 0 else "improves over time")
        ),
    }
    print(f"\nROC-AUC drift (last bin - first bin): {roc_drift:+.4f}")
    print(f"PR-AUC drift (last bin - first bin): {pr_drift:+.4f}")
    print(f"Verdict: {summary['verdict']}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bin_labels = [f"day {r['day_range'][0]}-{r['day_range'][1]}" for r in rows]

    axes[0].plot(range(1, N_BINS + 1), roc_aucs, marker="o", color="#3b6ea5", lw=1.5)
    axes[0].set_xticks(range(1, N_BINS + 1))
    axes[0].set_xticklabels(bin_labels, fontsize=8)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("ROC-AUC over the test period")
    axes[0].set_ylim(0, 1)

    axes[1].plot(range(1, N_BINS + 1), pr_aucs, marker="o", color="#3ba55d", lw=1.5)
    axes[1].set_xticks(range(1, N_BINS + 1))
    axes[1].set_xticklabels(bin_labels, fontsize=8)
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("PR-AUC over the test period")
    axes[1].set_ylim(0, max(pr_aucs) * 1.3)

    fig.suptitle(f"Concept drift check -- XGBoost (tuned), fixed threshold={threshold}")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
