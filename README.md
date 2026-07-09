# GSC26 Challenge 3 — Team #316

**IEEE Computer Society Global Student Challenge 2026** · Challenge 3: Analyzing Resource Usage in Data Center Computers and Predicting Events (Failures or Performance Slowdowns)

Solo entrant: Het Patel · Phase I window: Jul 13 – Aug 9, 2026, 11:59pm ET

## Problem statement

Given time-series resource-usage telemetry from a large-scale data center / cloud cluster (CPU, memory, disk, network, and where available GPU/thermal/power), build a system that:

1. **Detects anomalies** in resource usage in near-real-time, distinguishing genuine anomalies from normal seasonal/periodic patterns.
2. **Predicts failure or performance-slowdown events before they happen**, with an explicit, reported prediction lead time — not just flagging that something is currently wrong.
3. Is evaluated with time-aware, imbalance-aware metrics (not just accuracy), and is presented as a reproducible pipeline + demo, not a one-off notebook.

## Success metrics (targets, refine as data exploration lands)

- PR-AUC / average precision as the headline classification metric (not accuracy — failures are rare).
- F2 score at a chosen, justified operating threshold.
- Reported lead-time distribution (median/mean) between first correct alert and actual failure.
- False alarms per machine-day at the chosen threshold.
- Ablation table showing the contribution of each feature group and of the unsupervised-score-as-feature hybrid trick.
- Time-based train/test split (train on earlier days, test on later days) — no random splits.

Full technical rationale: [`docs/01-DEEP-DIVE.md`](docs/01-DEEP-DIVE.md)
Week-by-week build plan: [`docs/02-IMPLEMENTATION-PLAN.md`](docs/02-IMPLEMENTATION-PLAN.md)

## Datasets

- **Primary**: Google Cluster Trace v3 (2019) — machine + task events, resource usage, failure/eviction labels.
- **Secondary**: Backblaze quarterly SMART hard-drive stats — disk failure labels.
- Download steps: see `data/README.md` (added once acquisition strategy is finalized).

## Repo structure

```
src/
├── ingest/     # raw -> clean, time-indexed parquet
├── features/   # windowing, statistical, temporal, cross-entity features
├── labels/     # prediction-horizon label construction
├── models/
│   ├── anomaly/     # Isolation Forest, LSTM-AE, STL/EWMA baseline, (stretch) TranAD
│   └── prediction/  # XGBoost/LightGBM, LSTM/GRU classifier, hybrid stacking
└── eval/       # metrics, ablations, drift analysis, cost model
notebooks/      # EDA and exploratory work
dashboard/      # Streamlit demo app
docs/           # deep dive + implementation plan
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# then install torch per your hardware (see requirements.txt comment)
```

Confirmed working on this machine: RTX 5060 Laptop GPU (Blackwell, sm_120) requires
the `cu128` torch build — `cu124` installs but throws "no kernel image available."

## Status

Week 0 (prep) — repo scaffolded, `.venv` created with core stack + GPU-enabled torch
verified, Google Cluster Trace 2011-2 subset downloaded (~27GB, see `data/README.md`).
Next: Week 1 EDA. See `docs/02-IMPLEMENTATION-PLAN.md` for the current milestone.
