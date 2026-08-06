# Model Results — Task Failure Prediction

## Task definition

Predict, from a task's resource-usage history, whether it will receive a
`FAIL` event in the next 30 minutes ("imminent failure" early warning).

- **Features**: `src/features/build_features.py` buckets each task's usage
  samples (continuous coverage: trace days 0-10, see `data/README.md`) into
  non-overlapping 30-minute windows, computing mean/std/max of CPU rate,
  memory usage, disk I/O, plus static per-task features (resource requests,
  scheduling class/priority) and request-normalized usage ratios.
- **Labels**: 1 if the task's first `FAIL` event (from the full 29-day
  `task_events` table, so never truncated by the usage window) falls in
  `(window_end, window_end + 30min]`.
- **Scale**: 66,584,317 window-rows, 8,235,298 tasks, 119,824 positive
  windows (0.18% base rate).
- **Split**: time-based on `window_end` (last 20% of the trace held out as
  test) — never random — per the evaluation-rigor plan. Train split is
  undersampled to a 20:1 negative:positive ratio; test is left at its
  natural imbalance (0.22% positive) so metrics reflect real deployment
  conditions.
- **LSTM scope**: trained on a task subsample (all failing tasks + 10x as
  many random non-failing tasks — full-scale lag-sequence construction for
  training is a >20GB intermediate) but *evaluated* on the full test-period
  task universe via chunked scoring (`evaluate_full_test_set()` in
  `src/models/lstm_model.py`) — same 13,263,258-row test set as the other
  3 models, so all 4 are directly comparable.

## Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Z-score anomaly (unsupervised) | 0.0007 | 0.069 | 0.0014 | 0.362 | 0.0015 |
| Logistic regression (supervised) | 0.008 | 0.588 | 0.016 | 0.791 | 0.0055 |
| XGBoost | 0.033 | 0.262 | 0.058 | 0.911 | 0.051 |
| LSTM | 0.024 | 0.230 | 0.043 | 0.873 | 0.026 |

Full metrics (confusion matrix, tuned threshold, feature coefficients/
importances) in `data/processed/model_metrics.json` (gitignored — rerun the
model scripts in `src/models/` to regenerate) and the git-tracked trimmed
copy `results/model_metrics.json`.

## Reading the results

- **Z-score baseline underperforms random (ROC-AUC 0.36)**: global z-scoring
  across ~8M tasks with wildly different resource scales is a weak signal —
  a task's own "normal" isn't the cluster's "normal". This is the honest
  floor a naive unsupervised method sets, not a bug.
- **Logistic regression (ROC-AUC 0.79)** is a real baseline: `mem_max` and
  `cpu_mean` are the strongest positive predictors (recent peak memory,
  rising CPU), matching the resource-exhaustion pattern found in EDA
  (`notebooks/EDA_FINDINGS.md`). Precision is low (0.008) because the task
  is extremely imbalanced (0.22% positive) and this model uses a single
  fixed threshold with no per-task history.
- **XGBoost is the strongest model overall (PR-AUC 0.051, ~23x the 0.0022
  base rate)**. `scheduling_class` dominates feature importance, followed by
  the resource-request columns — the model is largely learning "which kinds
  of tasks fail" (workload class) more than "how usage is trending right
  now," which is a useful finding in its own right (see
  `results/model_metrics.json` → `xgboost.feature_importances`).
- **LSTM's rank ordering is close to XGBoost's (ROC-AUC 0.873 vs 0.911) but
  it loses clearly on PR-AUC/precision (0.026 vs 0.051) at the operating
  threshold.** This is the *corrected*, full-test-set number —
  an earlier version of this table evaluated the LSTM only on its training
  task subsample and showed a misleadingly strong PR-AUC (0.057, beating
  XGBoost); expanding evaluation to the full universe the other 3 models
  are scored on dropped that to 0.026. That gap between "scores well on a
  subsample" and "scores well on the full population" is itself a real
  finding worth keeping in the write-up, not smoothing over: a fixed
  4-timestep-history model trained on a curated subsample doesn't
  automatically generalize as well as tree-based tabular features that see
  every task's full context statically.
- PR-AUC is the metric to watch for model comparison, not accuracy/ROC-AUC
  alone — at this base rate accuracy is trivially ~99.8% even for a
  do-nothing classifier.

## Lead-time analysis

`src/eval/lead_time.py` asks a different question than the window-level
confusion matrix above: not "was this specific 30-min window flagged
correctly" but "did we ever warn about this failure before it happened, and
how far in advance?" — using the XGBoost model, for every test-period task
with an observed imminent-failure window (the same 28,620-task population
behind the results table's `n_positive`), it finds the *first* window
(chronologically, not necessarily the labeled one) where predicted
probability crossed the tuned threshold, strictly before the actual `FAIL`.

| | |
|---|---|
| Test-period failures | 28,620 |
| Detected with advance warning | 8,383 (29.3%) |
| Median lead time | 26.8 min |
| Mean lead time | 69.2 min |
| P25 / P75 | 9.6 / 44.2 min |
| Max | 2,619.6 min (~43.7h) |

![Lead time distribution](results/plots/lead_time_distribution.png)

Detection rate (29.3%) is a few points above the window-level recall
(26.2%) because a task counts as "detected" here if *any* window before its
failure crossed the threshold, not just the specific 30-min-out window —
slightly more generous, and arguably the more operationally relevant
number (an ops team doesn't care which window raised the pager, just that
one did, and how much runway they got). The long right tail (a handful of
detections beyond 500 minutes) comes from tasks with a genuine early
resource-usage ramp well before the 30-minute label horizon — the same
phenomenon as the EDA sawtooth memory-leak example.

**Methodological note**: an earlier version of this analysis counted *any*
task with `fail_time` after the test cutoff as a "test-period failure"
(166,061 of them) rather than restricting to tasks with an actual observed
pre-failure window. Since `fail_time` is looked up from the full 29-day
trace while usage data only has continuous coverage through day 10, that
swept in tasks whose last observed window was days before their eventual
failure — an unobserved gap, not a real "advance warning" — producing lead
times up to 25,000+ minutes that swamped the distribution. Restricting to
the 28,620-task population with a genuine labeled positive window (matching
the confusion matrix exactly) fixed this.
