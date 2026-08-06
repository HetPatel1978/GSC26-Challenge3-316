"""
XGBoost failure-prediction model on the same 30-min window features/split
as the naive baselines (src/eval/dataset.py) -- same train/test rows, same
feature columns, so results are directly comparable.

Early stopping uses a held-out slice carved from the (already time-based,
undersampled) train split -- true generalization is still judged only on
the untouched time-based test split.

Usage:
    python src/models/xgboost_model.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import load_time_split, FEATURE_COLS  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "xgboost.pkl"


def make_model(device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        tree_method="hist",
        device=device,
        random_state=42,
    )


def main():
    X_train, y_train, X_test, y_test, cutoff = load_time_split()

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
    print_metrics("xgboost", metrics)

    importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_]))
    print("\nfeature importances:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model to {MODEL_PATH}")

    save_result("xgboost", metrics, extra={
        "threshold": float(best_t), "features": FEATURE_COLS,
        "feature_importances": importances,
        "best_iteration": int(clf.best_iteration), "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
