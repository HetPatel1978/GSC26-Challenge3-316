"""
Machine-level failure (REMOVE event) prediction, parallel to the task-level
xgboost_model.py: same modeling approach (time-based split, undersampled
train, F1-tuned threshold, early stopping), different prediction target --
"will this machine be removed in the next 30 minutes" instead of "will this
task fail in the next 30 minutes". Results are tracked separately (not
merged into the 4-model task-failure comparison) since it's a different
label/universe -- see docs/03-BASELINE-RESULTS.md.

Usage:
    python src/models/machine_xgboost.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROC = REPO_ROOT / "data" / "processed"
sys.path.insert(0, str(REPO_ROOT / "src"))

from features.machine_features import MACHINE_FEATURE_COLS, LABEL_COL  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402

FEATURES_PATH = PROC / "machine_features_window30min.parquet"
MODEL_PATH = REPO_ROOT / "models" / "machine_xgboost.pkl"
NULLABLE_COLS = ["cpus", "memory", "cpu_utilization", "mem_utilization"]


def load_machine_split(test_frac: float = 0.2, train_negative_ratio: float = 20.0, seed: int = 42):
    df = pl.read_parquet(FEATURES_PATH)
    cutoff = df["window_end"].quantile(1 - test_frac)
    train = df.filter(pl.col("window_end") <= cutoff)
    test = df.filter(pl.col("window_end") > cutoff)

    medians = {c: train[c].median() for c in NULLABLE_COLS}
    train = train.with_columns([pl.col(c).fill_null(v) for c, v in medians.items()])
    test = test.with_columns([pl.col(c).fill_null(v) for c, v in medians.items()])

    pos = train.filter(pl.col(LABEL_COL) == 1)
    neg = train.filter(pl.col(LABEL_COL) == 0)
    n_neg_keep = min(neg.height, int(pos.height * train_negative_ratio))
    neg_sampled = neg.sample(n=n_neg_keep, seed=seed)
    train_bal = pl.concat([pos, neg_sampled]).sample(fraction=1.0, seed=seed, shuffle=True)

    X_train = train_bal.select(MACHINE_FEATURE_COLS).to_numpy().astype(np.float64)
    y_train = train_bal[LABEL_COL].to_numpy()
    X_test = test.select(MACHINE_FEATURE_COLS).to_numpy().astype(np.float64)
    y_test = test[LABEL_COL].to_numpy()

    print(
        f"train: {X_train.shape[0]:,} rows ({pos.height:,} pos / {n_neg_keep:,} neg sampled "
        f"from {neg.height:,}) | test: {X_test.shape[0]:,} rows ({int(y_test.sum()):,} pos, "
        f"{100*y_test.mean():.3f}% positive) | cutoff window_end={cutoff}"
    )
    return X_train, y_train, X_test, y_test, cutoff


def make_model(device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", early_stopping_rounds=20,
        tree_method="hist", device=device, random_state=42,
    )


def main():
    X_train, y_train, X_test, y_test, cutoff = load_machine_split()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42,
    )
    try:
        clf = make_model("cuda")
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        print(f"GPU training failed ({e}), falling back to CPU")
        clf = make_model("cpu")
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    print(f"best iteration: {clf.best_iteration} / {clf.n_estimators}")

    train_proba = clf.predict_proba(X_train)[:, 1]
    best_t, best_f1 = None, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_train, (train_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"tuned threshold (max train F1): {best_t:.2f} (train F1={best_f1:.4f})")

    test_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_t).astype(int)

    metrics = summarize(y_test, y_pred, test_proba)
    print_metrics("machine_xgboost", metrics)

    importances = dict(zip(MACHINE_FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_]))
    print("\nfeature importances:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to {MODEL_PATH}")

    save_result("machine_xgboost", metrics, extra={
        "threshold": float(best_t), "features": MACHINE_FEATURE_COLS,
        "feature_importances": importances,
        "prediction_target": "machine REMOVE event in next 30 min (vs. task FAIL for the other 4 models)",
        "best_iteration": int(clf.best_iteration), "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
