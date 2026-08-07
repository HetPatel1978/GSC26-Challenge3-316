"""
Feature-group ablation: train XGBoost with fixed (untuned) hyperparameters,
varying only which feature group is available -- CPU-only, memory-only,
disk-only, scheduling-only, and the full feature set as a reference -- to
see which group carries the most standalone predictive signal.

Hyperparameters are held fixed across every run (matching src/models/
xgboost_model.py's untuned baseline) so any ROC-AUC difference is
attributable to the feature set, not tuning luck; same time-based split as
every other model (src/eval/dataset.py).

Usage:
    python src/eval/ablation.py
"""

import json
import sys
from pathlib import Path

import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import load_time_split, FEATURE_COLS  # noqa: E402
from eval.metrics import summarize  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results" / "ablation_table.md"
JSON_PATH = REPO_ROOT / "results" / "ablation_table.json"

FEATURE_GROUPS = {
    "cpu": ["cpu_mean", "cpu_std", "cpu_max", "cpu_request", "cpu_usage_ratio"],
    "memory": ["mem_mean", "mem_std", "mem_max", "mem_peak_mean", "assigned_mem_mean",
               "memory_request", "mem_usage_ratio"],
    "disk": ["disk_io_mean", "disk_space_request"],
    "scheduling": ["scheduling_class", "priority", "n_samples"],
    "all features": FEATURE_COLS,
}

_partition = set().union(*[v for k, v in FEATURE_GROUPS.items() if k != "all features"])
assert _partition == set(FEATURE_COLS), \
    f"feature groups must partition FEATURE_COLS exactly; mismatch: {_partition.symmetric_difference(FEATURE_COLS)}"


def make_model(device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device=device, random_state=42,
    )


def main():
    X_train_full, y_train, X_test_full, y_test, cutoff = load_time_split()
    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    device = "cuda"
    rows = []
    for group_name, cols in FEATURE_GROUPS.items():
        idx = [col_idx[c] for c in cols]
        X_train, X_test = X_train_full[:, idx], X_test_full[:, idx]

        try:
            clf = make_model(device)
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"GPU failed ({e}), falling back to CPU")
            device = "cpu"
            clf = make_model(device)
            clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        # Fixed neutral threshold: this table is about ranking quality
        # (ROC-AUC/PR-AUC) across feature sets, not a tuned operating point.
        y_pred = (proba >= 0.5).astype(int)
        m = summarize(y_test, y_pred, proba)
        rows.append({"group": group_name, "n_features": len(cols), **m})
        print(f"{group_name:12s} ({len(cols):2d} features): "
              f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f}")

    solo_groups = [r for r in rows if r["group"] != "all features"]
    best = max(solo_groups, key=lambda r: r["roc_auc"])
    full = next(r for r in rows if r["group"] == "all features")

    lines = [
        "# Feature-group ablation",
        "",
        "XGBoost trained with fixed (untuned) hyperparameters "
        f"(n_estimators=200, max_depth=6, learning_rate=0.1), varying only which",
        "feature group is available, on the same time-based split as every other",
        "model (`src/eval/dataset.py`). Threshold fixed at 0.5 (neutral) since this",
        "measures ranking quality (ROC-AUC/PR-AUC), not a tuned operating point.",
        "",
        "| Feature group | # features | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        marker = " **<- full model**" if r["group"] == "all features" else ""
        lines.append(
            f"| {r['group']} | {r['n_features']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |{marker}"
        )
    lines += [
        "",
        f"**`{best['group']}` alone recovers the most standalone signal among individual "
        f"groups** (ROC-AUC {best['roc_auc']:.4f}), vs. the full model's "
        f"{full['roc_auc']:.4f} -- {round(100 * best['roc_auc'] / full['roc_auc'], 1)}% of "
        "the full model's ranking quality from one feature group alone.",
    ]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}\nWrote {JSON_PATH}")


if __name__ == "__main__":
    main()
