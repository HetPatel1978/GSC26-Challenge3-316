# Challenge 3 Implementation Plan

Team #316 · Solo · Phase I window: **Jul 13 – Aug 9, 2026, 11:59pm ET** · Baseline activity checkpoint: **Jul 28** (bonus for activity Jul 29)

Reference: see `01-DEEP-DIVE.md` for the technical background behind every decision below.

---

## What we cut and why

This plan was written for the full 4-week window. Actual model-building work happened
in a compressed sprint (Aug 4-7), which forced real scope cuts against the list below.
Recorded here explicitly, per this doc's own §0 principle ("one primary dataset done
excellently beats two datasets done shallowly") — these are prioritization calls, not
things that were attempted and failed:

| Planned | Status | Why |
|---|---|---|
| Isolation Forest | **Cut** | The z-score baseline already fills the "simple unsupervised floor" role in the model comparison (and honestly underperforms — see `docs/03-BASELINE-RESULTS.md`); a second unsupervised method would have added another weak baseline, not new signal, for the time it cost. |
| LSTM-Autoencoder | **Cut** | Built a supervised LSTM classifier instead (`src/models/lstm_model.py`) — it fills the "deep learning comparison point" role the plan wanted, and directly optimizes the actual target (failure prediction) rather than a reconstruction-error proxy for it. |
| Two-stage hybrid (unsupervised score as a supervised feature) | **Cut** | Given the z-score baseline scores *below random* (ROC-AUC 0.36), feeding it into XGBoost as an extra feature was unlikely to help and wasn't worth the time against higher-value work (tuning, lead-time analysis, a second prediction target). |
| Ablation table | **Added** (`src/eval/ablation.py`, `results/ablation_table.md`) — cut initially under time pressure, added back once the core pipeline (5 models + tuning) was solid. |
| Concept-drift check | **Added** (`src/eval/concept_drift.py`) — same as above. |
| Cost-based metric | **Added** — a paragraph in `docs/03-BASELINE-RESULTS.md` and the README framing the operating threshold as a deliberate precision/recall trade, not a full cost-matrix simulation (no real incident-cost data available for this trace). |
| F-beta / false-alarms-per-machine-day | **Cut** | Standardized on precision/recall/F1/ROC-AUC/PR-AUC across every model instead — one consistent metric set beats a superset partially reported. |
| TranAD (stretch) | **Cut** | Was scoped as stretch-only from the start (§1); never became load-bearing to any result. |
| Backblaze secondary dataset (stretch) | **Cut** | Also stretch-only from the start; the Google Cluster Trace alone supports two distinct prediction targets (task-level and machine-level), which is where the second-dataset time went instead. |
| LightGBM | **Cut** | XGBoost was sufficient as the tree-based model; running both would have doubled tuning time for a result the literature says is usually a wash. |
| Demo video | **Cut** | The Streamlit dashboard (`app/dashboard.py`, Overview + Live Simulation tabs) is the demo artifact instead — interactive beats a recording. |

---

## 0. What "winning" actually requires (design targets, not just a checklist)

A judge is comparing you against other solo/team entries that will mostly do: download one dataset, run Isolation Forest or an LSTM, report accuracy, ship a notebook. To place top-5, you need to clearly exceed that on **3 axes**:

1. **Technical rigor** — time-aware evaluation (no leakage), imbalance-aware metrics (PR-AUC, F-beta, lead time), an ablation table, at least one cross-dataset or cross-time generalization check.
2. **Depth of problem framing** — you distinguish anomaly detection from failure *prediction*, define an explicit prediction horizon/lead time, and (if time allows) frame at least one piece as time-to-event/cost-based rather than plain binary classification.
3. **Presentation as a system, not a notebook** — a clean repo, a reproducible pipeline (raw data → features → models → evaluation → dashboard), and a short live or recorded demo (dashboard showing real-time-style scoring on held-out data). This is what makes "technically rigorous" *visible* to a judge skimming many submissions.

Scope discipline: **one primary dataset done excellently beats two datasets done shallowly.** Treat the secondary dataset (§7 of the deep dive) as a stretch goal only after the primary pipeline is solid end-to-end.

---

## 1. Decisions to lock before writing code (do this now, before Jul 13)

You don't need to wait for the window to open to think — you can prep repo scaffolding, environment, and dataset download now, then start modeling Jul 13.

| Decision | Recommendation | Rationale |
|---|---|---|
| Primary dataset | **Google Cluster Trace v2 (2011-2)** — downloaded 2026-07-08, ~27GB on disk in `data/raw/google_cluster_2011/` | Publicly downloadable via plain HTTPS (no BigQuery/gcloud needed), has both machine-failure and task-failure labels, and is the exact trace used in prior Bi-LSTM failure-prediction papers (direct literature comparison). v3 (2019) kept as a documented stretch option — see `data/README.md` |
| Secondary/stretch dataset | **Backblaze SMART disk data** | Clean, well-labeled, cheap to bolt on, gives you a second distinct failure-prediction narrative (§7.5/§7 recommendation) |
| Core problem framing | Binary "failure/degradation in next Δ" classification, Δ chosen per §4.5 (start with 30 min) | Most tractable in 4 weeks, still supports lead-time reporting |
| Anomaly detection stack | Isolation Forest + LSTM-Autoencoder (+ STL/EWMA statistical baseline) | Reliable, fast to implement, strong ablation contrast |
| Failure prediction stack | XGBoost/LightGBM as primary, LSTM/GRU classifier as deep comparison | Boosted trees are the empirically strongest, fastest to iterate; deep model for "we compared architectures" credibility |
| Stretch model | One transformer-based AD model (TranAD — public code available) | SOTA-awareness signal without betting the core pipeline on it |
| Language/stack | Python: pandas/polars, scikit-learn, XGBoost/LightGBM, PyTorch (LSTM/AE/TranAD), Prophet or statsmodels, matplotlib/plotly, Streamlit or a simple Grafana-style dashboard for the demo | Standard, defensible, fast for solo dev |
| Evaluation protocol | Time-based split (train on early days, test on later days); report PR-AUC, F2, lead time distribution, FP/machine-day, ablation table | Directly addresses the most common leakage/rigor failure mode (§8) |

---

## 2. Week-by-week plan

### Week 0 — Prep (now, Jul 7 – Jul 12, before window opens)
- [ ] `git init`, repo scaffolding: `data/`, `src/{ingest,features,models,eval}/`, `notebooks/`, `docs/`, `dashboard/`, `README.md`.
- [ ] Add organizer collaborators (bagchi, dsilv1234, japjorge, j-mckerracher) to the GitHub repo.
- [ ] Download and do a first pass over Google Cluster Trace v3 (schema, size, sampling — decide if you need BigQuery access or a downsampled local subset; cluster traces can be 10s of GB, so plan for a manageable slice, e.g., first N days or M machines, and state that scoping decision explicitly in your report).
- [ ] Download Backblaze quarterly CSV (pick 2-3 quarters covering a mix of drive models) as the secondary dataset.
- [ ] Set up environment (`requirements.txt`/`uv`/conda env), confirm PyTorch + GPU availability if you have one (affects whether LSTM/transformer training is fast or a bottleneck — plan accordingly, e.g., Google Colab/Kaggle GPU as fallback).
- [ ] Write the one-paragraph problem statement + success metrics into `README.md` so every later decision has a fixed target to check against.

### Week 1 — Jul 13–19: Data understanding, cleaning, EDA, first baseline
- [ ] Load & profile the primary dataset: schema, missingness, time range, label distribution (how rare are failures — this number drives your whole imbalance strategy).
- [ ] Build the ingestion pipeline (`src/ingest/`): raw files → clean, indexed, time-sorted parquet per machine/entity.
- [ ] EDA notebook: visualize failure patterns from deep-dive §3 (find at least one real example each of resource-exhaustion, saturation, and hardware-degradation patterns in your actual data — these become figures in your final report).
- [ ] Implement the **simplest possible baseline**: z-score/EWMA anomaly flagging on 1-2 metrics, and majority-class/logistic-regression failure classifier. This is your floor — every later model must be shown beating it.
- [ ] **Milestone**: end-to-end pipeline runs (even if crude) from raw data to a metric on held-out time. This proves the plumbing works before you invest in sophistication.

### Week 2 — Jul 20–27: Feature engineering + core models (finish *before* the Jul 28 baseline checkpoint)
- [ ] Implement full feature pipeline from deep-dive §4: windowing, statistical, temporal/trend, cross-entity/peer-relative features. Version this as a reusable module, not notebook-only code.
- [ ] Implement label construction with explicit prediction-horizon window + blackout buffer (§4.5).
- [ ] Train Isolation Forest + LSTM-Autoencoder for anomaly detection; calibrate thresholds (validation-set percentile).
- [ ] Train XGBoost/LightGBM + an LSTM/GRU classifier for failure prediction; handle imbalance (class weights first, SMOTE only if needed); do a first hyperparameter pass (Optuna or simple grid — don't over-invest here yet).
- [ ] Implement the two-stage hybrid (§6.4): unsupervised anomaly score as an extra feature into the supervised model. Compare against the standalone supervised model — this single ablation is one of your best "we understood the material" artifacts.
- [ ] **By Jul 28 (baseline activity deadline)**: have a committed, working pipeline with at least one result per model family and a visible commit history — this is very likely what "baseline activity" is checked against, so don't leave this until the last day.

### Week 3 — Jul 28 – Aug 2: Rigor, evaluation, differentiation
- [ ] Full evaluation suite (§8): PR-AUC, F-beta, ROC-AUC, confusion matrix at chosen threshold, lead-time distribution, false-alarms-per-machine-day, calibration check.
- [ ] Time-based (not random) train/test split verification; if not already done in Week 2, fix this now — it's the single most damaging mistake to leave in.
- [ ] Ablation table: feature groups on/off, hybrid vs. standalone, model family comparison.
- [ ] Concept-drift check (§6.5): train on early window, test on progressively later windows, plot performance decay — cheap, high-signal differentiator.
- [ ] Stretch (only if on schedule): bring in Backblaze as a second dataset for a cross-dataset generalization check, and/or implement TranAD as the SOTA comparison model.
- [ ] Cost-based metric (§8.2): define a simple, explicit cost matrix and report expected cost/net benefit at your operating threshold — makes results legible to non-ML judges.

### Week 4 — Aug 3–9: Dashboard, report, packaging, submission
- [ ] Build a lightweight dashboard (Streamlit/Plotly Dash or a static Grafana-style export) that replays held-out data and shows: live metric streams, anomaly flags, failure-probability score, and a simulated "alert with lead time" — this is your demo artifact.
- [ ] Record a short (3-5 min) demo walkthrough video if the competition allows/benefits from it.
- [ ] Write the final report: problem framing → data → methodology (cite §9 papers where relevant) → results (headline PR-AUC/F2/lead-time numbers + ablation table + figures from Week 1 EDA) → limitations/future work (mention concept drift, cost-based deployment considerations) → how it would be productionized (tie back to deep-dive §2 collection architecture — "here's how this plugs into a real Prometheus/DCGM pipeline").
- [ ] Clean the repo: README with setup instructions, requirements pinned, remove dead notebooks/experiments or move to an `experiments/` folder, add a clear `docs/` folder (this plan + deep dive can stay, lightly trimmed, as supporting material).
- [ ] Final review pass **by Aug 7-8**, leaving Aug 9 as buffer for submission logistics, not last-minute coding.
- [ ] Submit.

---

## 3. Risk register (things likely to eat your solo timeline — watch for these)

| Risk | Mitigation |
|---|---|
| Cluster trace is too large to process on a laptop | Scope to a subset (N days / M machines) explicitly and state the scoping decision in the report — reviewers respect a stated, reasoned scope cut far more than silent underperformance |
| Deep models (LSTM/transformer) take too long to tune solo | Treat XGBoost as your primary deliverable; deep models are comparison points, not the critical path — never let them block the evaluation/report timeline |
| Getting stuck on transformer AD implementation | Timebox it (e.g., max 3 days in Week 3); if it's not working cleanly, drop it and lean harder on the LSTM-AE + Isolation Forest story rather than shipping a broken/unconvincing SOTA model |
| Leaving evaluation rigor (time-based split, lead time, ablations) until the last week | Week 3 is dedicated specifically to this — don't let Week 2 modeling slip into Week 3's rigor time |
| No demo/dashboard, just a notebook | Budget Week 4 explicitly for this; it's disproportionately impactful for how judges perceive "systems thinking" vs. raw model count |
| Baseline activity deadline (Jul 28) missed or thin | Week 2 is scheduled to land a working, committed pipeline by Jul 27 with margin |

---

## 4. Suggested repo structure

```
GSC26-Challenge3-316/
├── README.md
├── docs/
│   ├── 01-DEEP-DIVE.md
│   └── 02-IMPLEMENTATION-PLAN.md
├── data/                # raw + processed (gitignored if large; document download steps instead)
├── src/
│   ├── ingest/          # raw -> clean parquet
│   ├── features/        # windowing, statistical, temporal, cross-entity features
│   ├── labels/           # prediction-horizon label construction
│   ├── models/
│   │   ├── anomaly/     # isolation forest, LSTM-AE, STL/EWMA baseline, (stretch) TranAD
│   │   └── prediction/  # XGBoost/LightGBM, LSTM/GRU classifier, hybrid stacking
│   └── eval/            # metrics, ablations, drift analysis, cost model
├── notebooks/           # EDA and exploratory work (kept separate from src/)
├── dashboard/           # Streamlit/Dash demo app
└── requirements.txt
```
