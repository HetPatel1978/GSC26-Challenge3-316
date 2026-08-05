# Baseline Results — Aug 4-5

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

## Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Z-score anomaly (unsupervised) | 0.0007 | 0.069 | 0.0014 | 0.362 | 0.0015 |
| Logistic regression (supervised) | 0.008 | 0.588 | 0.016 | 0.791 | 0.0055 |

Full metrics (confusion matrix, tuned threshold, feature coefficients) in
`data/processed/model_metrics.json` (gitignored — regenerate with
`python src/models/anomaly/zscore_baseline.py` and
`python src/models/prediction/logreg_baseline.py`).

## Reading the results

- **Z-score baseline underperforms random (ROC-AUC 0.36)**: global z-scoring
  across ~8M tasks with wildly different resource scales is a weak signal —
  a task's own "normal" isn't the cluster's "normal". This is the honest
  floor a naive unsupervised method sets, not a bug; per-task or
  per-scheduling-class normalization is a natural next step if pursued
  further, but the sprint prioritizes moving to XGBoost/LSTM instead.
- **Logistic regression is a real baseline (ROC-AUC 0.79)**: `mem_max` and
  `cpu_mean` are the strongest positive predictors (recent peak memory,
  rising CPU), matching the resource-exhaustion pattern found in EDA
  (`notebooks/EDA_FINDINGS.md`). Precision is low (0.008) because the task
  is extremely imbalanced (0.22% positive) and this model uses a single
  fixed threshold with no per-task history — exactly what XGBoost (richer
  feature interactions) and the LSTM (actual sequence memory) are expected
  to improve on tomorrow.
- PR-AUC is the metric to watch for model comparison, not accuracy/ROC-AUC
  alone — at this base rate accuracy is trivially ~99.8% even for a
  do-nothing classifier.
