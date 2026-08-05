"""
Naive supervised baseline: logistic regression on the engineered window
features, predicting imminent (next-30-min) task failure. Standardized
features, class-balanced training set (see src/eval/dataset.py), decision
threshold tuned on train to maximize F1 rather than left at the default
0.5 (with a ~0.18% natural positive rate 0.5 is far too conservative).

Usage:
    python src/models/prediction/logreg_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import load_time_split, FEATURE_COLS  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402


def main():
    X_train, y_train, X_test, y_test, cutoff = load_time_split()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    train_proba = clf.predict_proba(X_train_s)[:, 1]
    best_t, best_f1 = None, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_train, (train_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"tuned threshold (max train F1): {best_t:.2f} (train F1={best_f1:.4f})")

    test_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred = (test_proba >= best_t).astype(int)

    metrics = summarize(y_test, y_pred, test_proba)
    print_metrics("logreg_baseline", metrics)

    coefs = dict(zip(FEATURE_COLS, [round(float(c), 4) for c in clf.coef_[0]]))
    print("\nfeature coefficients (standardized):")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v}")

    save_result("logreg_baseline", metrics, extra={
        "threshold": float(best_t), "features": FEATURE_COLS,
        "coefficients": coefs, "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
