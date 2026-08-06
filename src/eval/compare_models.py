"""
Build the final model-comparison table + plot from
data/processed/model_metrics.json (populated by each model script's
save_result() call).

Usage:
    python src/eval/compare_models.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "model_metrics.json"
PLOT_PATH = REPO_ROOT / "results" / "plots" / "model_comparison.png"
TABLE_PATH = REPO_ROOT / "results" / "model_comparison.md"
# Trimmed copy, tracked in git (unlike data/processed/model_metrics.json)
# so app/dashboard.py works without the full local dataset/model reruns.
JSON_COPY_PATH = REPO_ROOT / "results" / "model_metrics.json"

# Fixed display order + a small categorical palette (never reassigned per
# metric) so a model's color is stable across every chart in the report.
MODEL_ORDER = ["zscore_anomaly_baseline", "logreg_baseline", "xgboost", "lstm"]
MODEL_LABELS = {
    "zscore_anomaly_baseline": "Z-score\n(unsupervised)",
    "logreg_baseline": "Logistic\nRegression",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
MODEL_COLORS = {
    "zscore_anomaly_baseline": "#8a8f98",
    "logreg_baseline": "#3b6ea5",
    "xgboost": "#3ba55d",
    "lstm": "#7a3ba5",
}


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    models = [m for m in MODEL_ORDER if m in results]
    missing = [m for m in MODEL_ORDER if m not in results]
    if missing:
        print(f"warning: missing results for {missing}, comparing only {models}")

    # ---- table ----
    cols = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    lines = ["| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for m in models:
        r = results[m]
        row = [MODEL_LABELS[m].replace("\n", " ")] + [f"{r.get(c, float('nan')):.4f}" for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    table_md = "\n".join(lines)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table_md + "\n", encoding="utf-8")
    print(table_md)
    print(f"\nWrote {TABLE_PATH}")

    trimmed = {m: {"label": MODEL_LABELS[m].replace("\n", " "), **{c: results[m].get(c) for c in cols}}
               for m in models}
    JSON_COPY_PATH.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    print(f"Wrote {JSON_COPY_PATH}")

    # ---- plot: two small multiples (different scales -> separate axes, no dual-axis) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    x = range(len(models))
    width = 0.25
    for i, metric in enumerate(["precision", "recall", "f1"]):
        vals = [results[m].get(metric, 0.0) for m in models]
        ax.bar([xi + (i - 1) * width for xi in x], vals,
               width=width, label=metric, color=[MODEL_COLORS[m] for m in models], alpha=[1.0, 0.65, 0.4][i])
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Precision / Recall / F1 (opacity = metric)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=a, label=l)
                        for l, a in [("precision", 1.0), ("recall", 0.65), ("f1", 0.4)]], fontsize=8)

    ax = axes[1]
    width = 0.35
    roc_vals = [results[m].get("roc_auc", 0.0) for m in models]
    pr_vals = [results[m].get("pr_auc", 0.0) for m in models]
    ax.bar([xi - width / 2 for xi in x], roc_vals, width=width, label="ROC-AUC",
           color=[MODEL_COLORS[m] for m in models])
    ax.bar([xi + width / 2 for xi in x], pr_vals, width=width, label="PR-AUC",
           color=[MODEL_COLORS[m] for m in models], alpha=0.5)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, label="random (ROC-AUC)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_title("ROC-AUC (solid) vs PR-AUC (light)")
    ax.legend(fontsize=8)

    fig.suptitle("Model comparison -- imminent task-failure prediction (30-min horizon)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
