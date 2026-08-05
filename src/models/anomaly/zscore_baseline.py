"""
Naive anomaly-detection baseline: per-feature z-scores against "normal"
(non-imminent-failure) training windows, flagged anomalous if the max
absolute z-score across a subset of resource-usage features exceeds a
threshold. No supervision beyond fitting the "normal" mean/std and tuning
one threshold -- this is the floor every later model (XGBoost, LSTM) needs
to clear.

Usage:
    python src/models/anomaly/zscore_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import load_time_split, FEATURE_COLS  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402

# Features to z-score: raw resource-usage signals, not static request/
# scheduling metadata (those aren't "usage drifted from normal" signals).
ZSCORE_FEATURES = ["cpu_mean", "cpu_max", "mem_mean", "mem_max", "mem_peak_mean", "disk_io_mean"]


def zscore_anomaly_flag(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, threshold: float) -> np.ndarray:
    z = np.abs((X - mu) / sigma)
    return (z.max(axis=1) > threshold).astype(int)


def main():
    X_train, y_train, X_test, y_test, cutoff = load_time_split()
    idx = [FEATURE_COLS.index(c) for c in ZSCORE_FEATURES]
    Xz_train, Xz_test = X_train[:, idx], X_test[:, idx]

    # "Normal" = non-imminent-failure training windows only.
    normal = Xz_train[y_train == 0]
    mu = normal.mean(axis=0)
    sigma = normal.std(axis=0)
    sigma[sigma == 0] = 1e-6

    best_t, best_f1 = None, -1.0
    for t in np.arange(1.0, 8.01, 0.25):
        pred = zscore_anomaly_flag(Xz_train, mu, sigma, t)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"tuned threshold (max train F1): {best_t} (train F1={best_f1:.4f})")

    y_pred = zscore_anomaly_flag(Xz_test, mu, sigma, best_t)
    y_score = np.abs((Xz_test - mu) / sigma).max(axis=1)

    metrics = summarize(y_test, y_pred, y_score)
    print_metrics("zscore_anomaly_baseline", metrics)
    save_result("zscore_anomaly_baseline", metrics, extra={
        "threshold": float(best_t), "features": ZSCORE_FEATURES, "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
