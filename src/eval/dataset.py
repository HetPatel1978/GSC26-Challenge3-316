"""
Shared loading/splitting for the windowed task-failure features
(data/processed/features_window30min.parquet, built by
src/features/build_features.py). Used by every model -- the naive
baselines today, XGBoost/LSTM tomorrow -- so the split and feature set
stay identical across the model comparison table.

Split is TIME-based (by window_end), not random: the last `test_frac` of
the trace by time is held out, so no model ever trains on data from after
the point it's evaluated at.

Positive rate is ~0.18% (imminent-failure windows are rare), so the train
split is undersampled to a fixed negative:positive ratio -- the test split
is left at its natural imbalance so evaluation reflects real deployment
conditions.
"""

from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features_window30min.parquet"

FEATURE_COLS = [
    "n_samples", "cpu_mean", "cpu_std", "cpu_max",
    "mem_mean", "mem_std", "mem_max", "mem_peak_mean",
    "disk_io_mean", "assigned_mem_mean",
    "cpu_request", "memory_request", "disk_space_request",
    "scheduling_class", "priority",
    "mem_usage_ratio", "cpu_usage_ratio",
]
LABEL_COL = "label_fail_soon"


def _fill_nulls(df: pl.DataFrame, fill_values: dict[str, float]) -> pl.DataFrame:
    return df.with_columns([pl.col(c).fill_null(v) for c, v in fill_values.items()])


NULLABLE_COLS = ["cpu_request", "memory_request", "disk_space_request", "mem_usage_ratio", "cpu_usage_ratio"]


def load_time_split(
    path: Path = FEATURES_PATH,
    test_frac: float = 0.2,
    train_negative_ratio: float = 20.0,
    seed: int = 42,
):
    """Returns (X_train, y_train, X_test, y_test, cutoff_us) as numpy arrays."""
    df = pl.read_parquet(path)

    cutoff = df["window_end"].quantile(1 - test_frac)
    train = df.filter(pl.col("window_end") <= cutoff)
    test = df.filter(pl.col("window_end") > cutoff)

    # A handful of tasks have a null resource request (SUBMIT event missing
    # that field) or a null usage ratio (request was 0). Median-impute from
    # train only so no test-split statistic leaks into training.
    medians = {c: train[c].median() for c in NULLABLE_COLS}
    train = _fill_nulls(train, medians)
    test = _fill_nulls(test, medians)

    pos = train.filter(pl.col(LABEL_COL) == 1)
    neg = train.filter(pl.col(LABEL_COL) == 0)
    n_neg_keep = min(neg.height, int(pos.height * train_negative_ratio))
    neg_sampled = neg.sample(n=n_neg_keep, seed=seed)
    train_bal = pl.concat([pos, neg_sampled]).sample(fraction=1.0, seed=seed, shuffle=True)

    X_train = train_bal.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_train = train_bal[LABEL_COL].to_numpy()
    X_test = test.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y_test = test[LABEL_COL].to_numpy()

    print(
        f"train: {X_train.shape[0]:,} rows ({pos.height:,} pos / {n_neg_keep:,} neg sampled "
        f"from {neg.height:,}) | test: {X_test.shape[0]:,} rows ({int(y_test.sum()):,} pos, "
        f"{100*y_test.mean():.3f}% positive) | cutoff window_end={cutoff}"
    )
    return X_train, y_train, X_test, y_test, cutoff
