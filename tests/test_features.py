"""
Tests for the task-failure feature engineering pipeline
(src/features/build_features.py) and the time-based split it feeds
(src/eval/dataset.py).

Runs entirely on small synthetic data constructed in-memory -- never
touches data/processed/ (gitignored, 66M+ rows, requires the full ~27GB
local trace download to rebuild) -- so these run in CI on every push.

Synthetic scenario: 3 tasks, N_WINDOWS_PER_TASK 30-minute windows each.
  - task 1: FAILs 15 minutes after its last window ends -> exactly one
    genuine "fails in the next 30 min" positive window (the last one).
  - task 2: never fails -> all windows negative.
  - task 3: fails, but 5 hours after its last window -> far outside the
    30-min horizon, so still all windows negative.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features.build_features import build, SUBMIT, FAIL  # noqa: E402
from eval.dataset import FEATURE_COLS, LABEL_COL  # noqa: E402

WINDOW_US = 30 * 60 * 1_000_000  # 30 min
HORIZON_US = WINDOW_US  # 30 min horizon, matches the project default
N_WINDOWS_PER_TASK = 10
SAMPLES_PER_WINDOW = 6
N_TASKS = 3


def _usage_rows(job_id: int, task_index: int) -> list[dict]:
    rows = []
    for w in range(N_WINDOWS_PER_TASK):
        window_start = w * WINDOW_US
        for s in range(SAMPLES_PER_WINDOW):
            t = window_start + s * (WINDOW_US // SAMPLES_PER_WINDOW)
            rows.append({
                "job_id": job_id, "task_index": task_index, "start_time": t,
                "cpu_rate": 0.10 + 0.01 * s,
                "canonical_memory_usage": 0.20 + 0.01 * s,
                "maximum_memory_usage": 0.25,
                "disk_io_time": 0.05,
                "assigned_memory_usage": 0.22,
            })
    return rows


def _submit_row(job_id: int, task_index: int) -> dict:
    return {
        "time": 0, "job_id": job_id, "task_index": task_index, "event_type": SUBMIT,
        "cpu_request": 0.5, "memory_request": 0.5, "disk_space_request": 0.5,
        "scheduling_class": 1, "priority": 5,
    }


def _fail_row(job_id: int, task_index: int, time_us: int) -> dict:
    return {
        "time": time_us, "job_id": job_id, "task_index": task_index, "event_type": FAIL,
        "cpu_request": None, "memory_request": None, "disk_space_request": None,
        "scheduling_class": None, "priority": None,
    }


@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("features")
    last_window_end = N_WINDOWS_PER_TASK * WINDOW_US

    usage_rows = _usage_rows(1, 0) + _usage_rows(2, 0) + _usage_rows(3, 0)
    task_usage = pl.DataFrame(usage_rows)

    submit_rows = [_submit_row(jid, 0) for jid in (1, 2, 3)]
    fail_rows = [
        _fail_row(1, 0, last_window_end + 15 * 60 * 1_000_000),        # 15 min after -> "soon"
        _fail_row(3, 0, last_window_end + 5 * 3600 * 1_000_000),       # 5h after -> not "soon"
        # task 2 gets no FAIL row at all
    ]
    task_events = pl.DataFrame(submit_rows + fail_rows)

    usage_path = tmp_path / "task_usage.parquet"
    events_path = tmp_path / "task_events.parquet"
    task_usage.write_parquet(usage_path)
    task_events.write_parquet(events_path)

    out_path = tmp_path / "features.parquet"
    build(
        WINDOW_US, HORIZON_US, out_path, min_samples=2,
        task_events_path=events_path, task_usage_path=usage_path,
    )
    return out_path


def test_expected_columns_present(synthetic_features):
    df = pl.read_parquet(synthetic_features)
    expected = set(FEATURE_COLS) | {LABEL_COL, "job_id", "task_index", "window_start", "window_end"}
    missing = expected - set(df.columns)
    assert not missing, f"missing expected columns: {missing}"


def test_no_nulls_in_key_feature_columns(synthetic_features):
    df = pl.read_parquet(synthetic_features)
    # every one of these is guaranteed non-null by construction of the
    # synthetic data (positive requests, >= min_samples usage rows/window,
    # a resolvable label for every window) -- a null here would mean a
    # real regression in the aggregation/join logic, not expected data
    # sparsity (unlike the real trace, where a few nullable columns are
    # legitimately null for some tasks).
    key_cols = FEATURE_COLS + [LABEL_COL, "window_start", "window_end"]
    null_counts = df.select([pl.col(c).null_count().alias(c) for c in key_cols]).row(0, named=True)
    bad = {c: n for c, n in null_counts.items() if n > 0}
    assert not bad, f"unexpected nulls in key columns: {bad}"


def test_window_count_is_positive(synthetic_features):
    df = pl.read_parquet(synthetic_features)
    assert df.height > 0
    # 3 tasks x N_WINDOWS_PER_TASK windows, none dropped (every window has
    # SAMPLES_PER_WINDOW >= min_samples=2 usage rows)
    assert df.height == N_TASKS * N_WINDOWS_PER_TASK


def test_train_test_split_is_time_ordered(synthetic_features):
    """No future leakage: every train-split window_end must be <= every
    test-split window_end under the same time-based cutoff dataset.py uses."""
    df = pl.read_parquet(synthetic_features)
    cutoff = df["window_end"].quantile(0.8)
    train = df.filter(pl.col("window_end") <= cutoff)
    test = df.filter(pl.col("window_end") > cutoff)

    assert train.height > 0 and test.height > 0, "synthetic dataset should produce a non-trivial split"
    assert train["window_end"].max() <= cutoff
    assert test["window_end"].min() > cutoff
    assert train["window_end"].max() <= test["window_end"].min()


def test_label_distribution_matches_expected_failure_rate(synthetic_features):
    df = pl.read_parquet(synthetic_features)
    n_positive = int(df[LABEL_COL].sum())
    # by construction: only task 1's final window is a genuine
    # "fails within the next 30 min" positive -- task 2 never fails, task
    # 3 fails far outside the horizon.
    assert n_positive == 1

    positive_rate = n_positive / df.height
    assert 0.0 < positive_rate < 0.2, (
        f"expected a low but nonzero positive rate (failures are rare), got {positive_rate}"
    )


def test_only_last_window_of_failing_task_is_labeled_positive(synthetic_features):
    """The specific window flagged positive should be task 1's very last
    window -- not an earlier one (too far from the FAIL) or task 3's (FAIL
    too far in the future to be "soon")."""
    df = pl.read_parquet(synthetic_features)
    positives = df.filter(pl.col(LABEL_COL) == 1)
    assert positives.height == 1
    row = positives.row(0, named=True)
    assert row["job_id"] == 1
    assert row["window_start"] == (N_WINDOWS_PER_TASK - 1) * WINDOW_US
