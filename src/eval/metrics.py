"""
Shared evaluation + results registry for every model in the comparison
table (naive baselines today, XGBoost/LSTM tomorrow). Every model's
results get appended to data/processed/model_metrics.json under its own
key, keyed off the same time-based test split, so the final comparison
table (step 7) just reads that one file.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "model_metrics.json"


def summarize(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_test": int(len(y_true)), "n_positive": int(y_true.sum()),
    }
    if y_score is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return metrics


def save_result(model_name: str, metrics: dict, extra: dict | None = None):
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    entry = dict(metrics)
    if extra:
        entry.update(extra)
    results[model_name] = entry
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {model_name} results to {RESULTS_PATH}")


def print_metrics(model_name: str, metrics: dict):
    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
