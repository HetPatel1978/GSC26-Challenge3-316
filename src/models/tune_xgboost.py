"""
Random search + cross-validation over XGBoost hyperparameters for the
task-failure model, using the exact same time-based train/test split as
xgboost_model.py (src/eval/dataset.py) so tuning never touches the held-out
test set. CV happens within the (already time-based, undersampled) train
split via stratified k-fold, scoring average precision (PR-AUC) since
that's the headline metric at this base rate.

The best config from CV is refit on the full train split with early
stopping (same recipe as the baseline script), evaluated once on the
untouched test split, and saved as a separate "xgboost_tuned" result --
see docs/03-BASELINE-RESULTS.md for the documented before/after comparison
against the untuned "xgboost" baseline.

Usage:
    python src/models/tune_xgboost.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import load_time_split, FEATURE_COLS  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "xgboost_tuned.pkl"
SEARCH_RESULTS_PATH = REPO_ROOT / "results" / "xgboost_tuning.json"

PARAM_DIST = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}
N_ITER = 12
N_FOLDS = 3
SEED = 42


def cv_score(params: dict, X: np.ndarray, y: np.ndarray, device: str) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        clf = xgb.XGBClassifier(
            tree_method="hist", device=device, random_state=SEED, eval_metric="aucpr", **params,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[va_idx])[:, 1]
        scores.append(average_precision_score(y[va_idx], proba))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    X_train, y_train, X_test, y_test, cutoff = load_time_split()
    device = "cuda"

    sampler = list(ParameterSampler(PARAM_DIST, n_iter=N_ITER, random_state=SEED))
    trials = []
    best_score, best_params = -1.0, None
    for i, raw_params in enumerate(sampler):
        params = {
            "n_estimators": int(raw_params["n_estimators"]),
            "max_depth": int(raw_params["max_depth"]),
            "learning_rate": round(float(raw_params["learning_rate"]), 4),
            "subsample": round(float(raw_params["subsample"]), 4),
            "colsample_bytree": round(float(raw_params["colsample_bytree"]), 4),
        }
        try:
            mean_ap, std_ap = cv_score(params, X_train, y_train, device)
        except Exception as e:
            print(f"GPU CV trial failed ({e}), falling back to CPU for the rest of the search")
            device = "cpu"
            mean_ap, std_ap = cv_score(params, X_train, y_train, device)
        trials.append({"params": params, "mean_pr_auc": round(mean_ap, 5), "std_pr_auc": round(std_ap, 5)})
        print(f"[{i+1}/{N_ITER}] {params} -> CV PR-AUC {mean_ap:.4f} +/- {std_ap:.4f}")
        if mean_ap > best_score:
            best_score, best_params = mean_ap, params

    print(f"\nbest CV PR-AUC: {best_score:.4f}, params: {best_params}")

    SEARCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEARCH_RESULTS_PATH.write_text(json.dumps({
        "n_iter": N_ITER, "n_folds": N_FOLDS,
        "best_params": best_params, "best_cv_pr_auc": round(best_score, 5),
        "trials": sorted(trials, key=lambda t: -t["mean_pr_auc"]),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {SEARCH_RESULTS_PATH}")

    # Final refit with early stopping on a held-out slice of train (same
    # recipe as xgboost_model.py) -- give early stopping room to use more
    # rounds than the CV trial did, since CV used a fixed n_estimators.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED,
    )
    final_params = dict(best_params)
    final_params["n_estimators"] = max(final_params["n_estimators"], 500)
    try:
        clf = xgb.XGBClassifier(
            tree_method="hist", device=device, random_state=SEED,
            eval_metric="aucpr", early_stopping_rounds=20, **final_params,
        )
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        print(f"GPU final fit failed ({e}), falling back to CPU")
        device = "cpu"
        clf = xgb.XGBClassifier(
            tree_method="hist", device=device, random_state=SEED,
            eval_metric="aucpr", early_stopping_rounds=20, **final_params,
        )
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    print(f"final best_iteration: {clf.best_iteration} / {final_params['n_estimators']}")

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
    print_metrics("xgboost_tuned", metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to {MODEL_PATH}")

    save_result("xgboost_tuned", metrics, extra={
        "threshold": float(best_t), "features": FEATURE_COLS,
        "best_params": final_params, "best_cv_pr_auc": round(best_score, 5),
        "best_iteration": int(clf.best_iteration), "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
