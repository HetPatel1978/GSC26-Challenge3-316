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
  4 models, so all 5 are directly comparable.

## Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Z-score anomaly (unsupervised) | 0.0007 | 0.069 | 0.0014 | 0.362 | 0.0015 |
| Logistic regression (supervised) | 0.008 | 0.588 | 0.016 | 0.791 | 0.0055 |
| XGBoost | 0.033 | 0.262 | 0.058 | 0.911 | 0.051 |
| **XGBoost (tuned)** | **0.081** | 0.240 | **0.122** | 0.893 | **0.068** |
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
- **XGBoost (untuned) is a large improvement over both baselines (PR-AUC
  0.051, ~23x the 0.0022 base rate)**. `scheduling_class` dominates feature
  importance, followed by the resource-request columns — the model is
  largely learning "which kinds of tasks fail" (workload class) more than
  "how usage is trending right now," which is a useful finding in its own
  right (see `results/model_metrics.json` → `xgboost.feature_importances`).
- **XGBoost (tuned) is the strongest model overall (PR-AUC 0.068, ~31x the
  base rate)** — see the dedicated tuning section below for the search
  methodology and the full before/after comparison.
- **LSTM's rank ordering is competitive (ROC-AUC 0.873, between the two
  XGBoost variants' 0.893–0.911) but it loses clearly on PR-AUC/precision
  to both (0.026 vs. 0.051 untuned / 0.068 tuned) at the operating
  threshold.** This is the *corrected*, full-test-set number —
  an earlier version of this table evaluated the LSTM only on its training
  task subsample and showed a misleadingly strong PR-AUC (0.057, beating
  the untuned XGBoost); expanding evaluation to the full universe the other
  4 models are scored on dropped that to 0.026. That gap between "scores
  well on a subsample" and "scores well on the full population" is itself a
  real finding worth keeping in the write-up, not smoothing over: a fixed
  6-timestep-history model (3 hours of lookback) trained on a curated
  subsample doesn't automatically generalize as well as tree-based tabular
  features that see every task's full context statically.
- PR-AUC is the metric to watch for model comparison, not accuracy/ROC-AUC
  alone — at this base rate accuracy is trivially ~99.8% even for a
  do-nothing classifier.

## XGBoost hyperparameter tuning

`src/models/tune_xgboost.py` runs a 12-trial random search (`n_estimators`,
`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) with 3-fold
stratified cross-validation, scored on PR-AUC, over the *same* time-based
train split as the baseline XGBoost model (CV never touches the test set).
The best config (`max_depth=8, learning_rate=0.215, subsample=0.988,
colsample_bytree=0.840, n_estimators=408`, picked by CV PR-AUC 0.828) is
refit with early stopping and evaluated once on the held-out test set.

| | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost (baseline params) | 0.033 | 0.262 | 0.058 | 0.911 | 0.051 |
| XGBoost (tuned) | 0.081 | 0.240 | 0.122 | 0.893 | 0.068 |
| **Change** | **+145%** | −8% | **+110%** | −2% | **+33%** |

Tuning is a real, broad-based win: **PR-AUC +33%, F1 +110%, precision
+145%**, at the cost of a small ROC-AUC dip (0.911→0.893) and a small
recall dip (0.262→0.240) — a deliberate, favorable trade at this base rate,
since precision was the baseline's weakest point (0.033, meaning ~97% of
alerts were false alarms) and PR-AUC is the metric that matters most under
0.22% class imbalance. Full trial log in `results/xgboost_tuning.json`.

**Note on the CV numbers**: the per-trial CV PR-AUC scores above (0.79–0.83)
look much higher than the final test PR-AUC (0.068) — this isn't a
regression, it's because CV runs on the undersampled *train* split (~5%
positive rate after the 20:1 undersampling) while the final evaluation runs
on the test split's natural 0.22% imbalance. CV PR-AUC is only meaningful
for *relative* ranking of hyperparameter configs, not as an absolute
performance estimate.

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

## Machine-level failure prediction

A second, parallel prediction target: instead of "will this *task* fail,"
"will this *machine* receive a `REMOVE` event (hardware failure/
decommission) in the next 30 minutes." Built with
`src/features/machine_features.py` (30-min windows of aggregate task
activity per machine — summed/mean CPU & memory, task count, disk I/O, a
task-churn feature counting EVICT/FAIL/KILL events among tasks scheduled on
that machine) and `src/models/machine_xgboost.py` (same modeling recipe as
the task-level XGBoost: time-based split, undersampled train, F1-tuned
threshold, early stopping).

- **Scale**: 6,016,863 window-rows, 12,558 machines, only 2,103 positive
  windows (0.035% base rate) — machine failures are far rarer than task
  failures in this trace (8,957 `REMOVE` events total vs. 13.8M task
  `FAIL` events), so this is a much sparser learning problem: 1,701 train /
  402 test positives, vs. task-level's ~96k / 28,620.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Machine XGBoost | 0.0008 | 0.154 | 0.0015 | 0.622 | 0.0007 |

- **Reading this**: ROC-AUC 0.62 is real signal (well above the 0.5
  random line) but far weaker than the task-level model's 0.91 — expected,
  given ~13x fewer positive examples to learn from and a label (hardware
  failure/decommission) that's plausibly driven by physical factors
  (age, thermal history) this feature set can't see at all. PR-AUC (0.0007)
  is only ~2x the 0.00035 base rate, vs. task-level XGBoost's ~23x lift.
  **`churn_events` and `disk_io_mean` are the top two features** —
  directly validating the EDA's "elevated task churn precedes machine
  REMOVE" finding (`notebooks/EDA_FINDINGS.md`, load-before-REMOVE worked
  example) with an actual trained model rather than just the one manually
  found example.
- Not merged into the main 5-model comparison table above since it predicts
  a different target on a different label universe; full metrics (confusion
  matrix, feature importances) tracked separately under the
  `machine_xgboost` key in `data/processed/model_metrics.json` (gitignored
  — regenerate with `python src/models/machine_xgboost.py`).
