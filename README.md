# GSC26 Challenge 3 — Team #316

**IEEE Computer Society Global Student Challenge 2026** · Challenge 3: Analyzing Resource
Usage in Data Center Computers and Predicting Events (Failures or Performance Slowdowns)

Solo entrant: Het Patel · Phase I window: Jul 13 – Aug 9, 2026, 11:59pm ET

## Problem statement

Given time-series resource-usage telemetry from a large-scale data center cluster (CPU,
memory, disk I/O), build a system that:

1. **Detects anomalies** in resource usage without relying on labels.
2. **Predicts task failure before it happens**, with an explicit prediction horizon — not
   just flagging that something is currently wrong.
3. Is evaluated with time-aware, imbalance-aware metrics (PR-AUC, not accuracy — failures
   are rare), compared across a spectrum of model complexity, and shipped as a reproducible
   pipeline + interactive demo, not a one-off notebook.

**Concretely**: predict whether a running task will receive a `FAIL` event in the next 30
minutes, using its recent resource-usage history. Four models are trained and compared
head-to-head on the identical time-based split: an unsupervised z-score anomaly detector,
logistic regression, XGBoost, and an LSTM sequence model.

Full technical rationale: [`docs/01-DEEP-DIVE.md`](docs/01-DEEP-DIVE.md) · week-by-week
plan: [`docs/02-IMPLEMENTATION-PLAN.md`](docs/02-IMPLEMENTATION-PLAN.md) · EDA writeup:
[`notebooks/EDA_FINDINGS.md`](notebooks/EDA_FINDINGS.md) · baseline results interpretation:
[`docs/03-BASELINE-RESULTS.md`](docs/03-BASELINE-RESULTS.md)

## Dataset

**Google Cluster Trace 2011-2** (v2) — 29 days, 12,583 machines, one Borg cell, ~672k jobs,
~25.4M tasks. Public on GCS bucket `clusterdata-2011-2`, downloadable over plain HTTPS with
no `gcloud`/account required. Chosen over the 2019 (v3) trace because v3 is 2.4TB of nested
JSON requiring BigQuery, while v2 is plain CSV and matches the trace used in prior published
Bi-LSTM failure-prediction papers, giving a literature baseline to compare against.

`task_usage` (the per-task resource-usage time series) only has **continuous** per-task
history for trace days 0–10 (shards 0–172) — see [`data/README.md`](data/README.md) for why
(an evenly-spaced sample looks continuous in aggregate but is actually disjoint 83-minute
snapshots; this was caught via a sawtooth artifact in an early plot). All feature
engineering below uses that continuous window so per-task time-series features are valid.

Download it yourself with `src/ingest/download_google_trace.py` — see **Setup** below.

## Approach

```
raw CSVs ──▶ typed parquet ──▶ EDA ──▶ 30-min sliding-window features ──▶ 4 models ──▶ dashboard
(ingest)     (build_parquet)   (01_eda)  (build_features, labels from      (compared on the
                                          the full-trace FAIL events)       same time split)
```

1. **Ingestion** (`src/ingest/`): stream raw CSVs into typed, compressed parquet via polars,
   with an explicit schema (raw files are headerless) and a manifest of row counts/null rates.
2. **EDA** (`notebooks/01_eda.py`): trace-wide stats, event-type distributions, usage
   distributions, failure-rate-over-time (concept-drift check), and two concrete worked
   examples — a memory-usage sawtooth ramping into a task `FAIL`, and cluster load in the
   3 days before a machine `REMOVE`.
3. **Feature engineering** (`src/features/build_features.py`): each task's usage samples are
   bucketed into non-overlapping 30-minute windows; each window gets CPU/memory/disk usage
   stats (mean/std/max) plus static per-task features (resource requests, scheduling
   class/priority) and request-normalized usage ratios. **Label**: 1 if the task's first
   `FAIL` event (looked up from the *full* 29-day `task_events` table, never truncated by the
   10-day usage window) falls in the 30 minutes after the window ends. 66.6M windows, 8.2M
   tasks, 0.18% positive rate.
4. **Split** (`src/eval/dataset.py`): time-based on window end time (last 20% of the trace
   held out) — never random, so no model ever sees the future. Train is undersampled 20:1
   negative:positive; test is left at its natural imbalance.
5. **Models** — see Results below for numbers:
   - `src/models/anomaly/zscore_baseline.py` — unsupervised, per-feature z-score vs. the
     "normal" (non-imminent-failure) training distribution.
   - `src/models/prediction/logreg_baseline.py` — logistic regression, F1-tuned threshold.
   - `src/models/xgboost_model.py` — gradient-boosted trees, GPU-accelerated, early stopping
     on a held-out slice of train.
   - `src/models/lstm_model.py` — sequence model over the last 6 windows (3h) per task,
     trained on GPU (PyTorch). *Trained/evaluated on a task subsample* (every task that ever
     fails, plus 10x as many random non-failing tasks) rather than the full 8.2M-task
     universe the tabular models use — building full lag sequences for all 66.6M windows
     would need >20GB of intermediate arrays. Noted here rather than presented as identical
     scope to the tabular models' test set.
6. **Comparison** (`src/eval/compare_models.py`): reads every model's saved metrics and
   produces the results table + comparison plot below.
7. **Dashboard** (`app/dashboard.py`): Streamlit app over a small precomputed sample (a
   handful of machines, including the two EDA worked examples) — resource usage and
   XGBoost-predicted failure risk over time, with actual `FAIL`/`REMOVE` events overlaid,
   plus the model comparison table.

## Results

Time-based test split, 30-minute imminent-failure horizon (all 4 models scored on the
tabular test set except LSTM — see the task-subsample note above):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Z-score (unsupervised) | 0.0007 | 0.0692 | 0.0014 | 0.3621 | 0.0015 |
| Logistic Regression | 0.0079 | 0.5882 | 0.0156 | 0.7911 | 0.0055 |
| XGBoost | 0.0328 | 0.2621 | 0.0584 | 0.9105 | 0.0514 |
| LSTM | 0.0582 | 0.1789 | 0.0879 | 0.8587 | 0.0569 |

![Model comparison](results/plots/model_comparison.png)

**Reading these**: PR-AUC is the metric that matters at a 0.22% base rate — accuracy would
be ~99.8% for a do-nothing classifier, and ROC-AUC alone can look deceptively good (the
z-score baseline actually scores *below* random on ROC-AUC, 0.36, because global z-scoring
across 8M tasks with wildly different resource scales is a genuinely weak signal — a task's
own "normal" isn't the cluster's "normal"). XGBoost and LSTM both land a real PR-AUC lift
(~23–26x over the base rate) over the naive baselines; `mem_max`/`cpu_mean` and
`scheduling_class` are the strongest signals, consistent with the resource-exhaustion
pattern found in EDA. Full interpretation: [`docs/03-BASELINE-RESULTS.md`](docs/03-BASELINE-RESULTS.md).

## Repo structure

```
src/
├── ingest/       # raw CSV -> typed parquet (schemas.py, build_parquet.py, downloader)
├── features/     # 30-min sliding-window feature + label construction
├── eval/         # shared time-based split, metrics registry, model comparison, dashboard-sample builder
└── models/
    ├── anomaly/prediction/  # naive baselines (z-score, logistic regression)
    ├── xgboost_model.py
    └── lstm_model.py
notebooks/        # EDA (01_eda.py) + findings + figures
app/               # Streamlit dashboard (dashboard.py)
models/            # saved model artifacts (xgboost.pkl, lstm.pt)
results/           # model comparison table + plot (git-tracked, small)
data/              # raw/processed (gitignored, rebuild locally) + small dashboard sample (tracked)
docs/              # deep dive, implementation plan, baseline results writeup
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# then install torch per your hardware (see requirements.txt comment)
```

Confirmed working on this machine: RTX 5060 Laptop GPU (Blackwell, sm_120) requires the
`cu128` torch build — `cu124` installs but throws "no kernel image available."

## How to run everything

Each step's output feeds the next; `data/processed/` and `models/*.pt`/`.pkl` are gitignored
(except the small dashboard sample and `models/xgboost.pkl`) so most of this needs to be
rerun locally after cloning.

```bash
# 1. Download the trace (~27GB; schema.csv/README + full job/task/machine events +
#    a contiguous 10-day task_usage window for time-series features)
python src/ingest/download_google_trace.py --out data/raw/google_cluster_2011 \
    --task-usage-range 0:172

# 2. Raw CSV -> typed parquet
python src/ingest/build_parquet.py

# 3. EDA (figures -> notebooks/figures/, findings -> notebooks/EDA_FINDINGS.md)
python notebooks/01_eda.py

# 4. Feature engineering (-> data/processed/features_window30min.parquet)
python src/features/build_features.py

# 5. Train + evaluate every model (each appends to data/processed/model_metrics.json)
python src/models/anomaly/zscore_baseline.py
python src/models/prediction/logreg_baseline.py
python src/models/xgboost_model.py
python src/models/lstm_model.py

# 6. Final comparison table + plot (-> results/)
python src/eval/compare_models.py

# 7. Dashboard sample data (-> data/dashboard_sample.parquet, data/dashboard_events.parquet)
python src/eval/build_dashboard_sample.py
```

## Dashboard

```bash
streamlit run app/dashboard.py
```

Runs off the small git-tracked files from step 7 above (already committed), so it works
right after cloning without needing the full local dataset or GPU — no need to rerun steps
1–7 just to see it. Pick a machine in the sidebar to see its resource usage, XGBoost
predicted failure risk over time, and actual `FAIL`/`REMOVE` event markers, alongside the
model comparison table.

## Status

Ingestion, EDA, feature engineering, and all 4 models (z-score, logistic regression,
XGBoost, LSTM) are built, trained, and compared; the dashboard is running. See
[`docs/02-IMPLEMENTATION-PLAN.md`](docs/02-IMPLEMENTATION-PLAN.md) for the original
week-by-week plan this sprint compressed.
