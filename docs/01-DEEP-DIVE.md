# Challenge 3 Deep Dive: Analyzing Resource Usage & Predicting Failures/Slowdowns in Data Centers

Team #316 · IEEE CS Global Student Challenge 2026 · Phase I: Jul 13 – Aug 9, 2026

---

## 1. The Full Metric Picture

A data center resource-usage/failure-prediction system is built on telemetry from four layers. You don't need all of them, but you need to know they exist so your feature set and your narrative ("we understand the full observability stack") is credible to judges.

### 1.1 Compute (CPU)
- **Utilization %** (user/system/iowait/steal/idle split) — steal time matters in virtualized/cloud contexts, iowait signals I/O-bound stalls not compute problems.
- **Load average** (1/5/15 min) — queueing signal independent of core count.
- **Run queue length / context switches / interrupts per second** — early signal of contention before utilization saturates.
- **CPU throttling events** (cgroup `nr_throttled`, `throttled_time`) — huge in Kubernetes/cloud: a container can be "only at 60% CPU" and still be throttled because of a low quota.
- **Frequency scaling / P-states, C-states** — power-performance tradeoff; a CPU stuck in a low P-state looks like a slowdown with normal utilization.
- **Cache misses, IPC (instructions per cycle)** — from `perf`/PMU counters; the deepest signal for "silent" performance degradation (noisy neighbor contention), rarely available in cluster traces but huge in raw hardware telemetry.

### 1.2 Memory
- **Used / free / cached / buffers**, **available memory** (the metric that actually predicts OOM risk, not "free").
- **Page faults** (minor vs major) — major faults = disk-backed = slowdown smoking gun.
- **Swap in/out rate** — leading indicator of memory pressure before OOM-kill.
- **OOM-kill events** — the actual failure event in many traces.
- **Memory fragmentation**, **slab usage**, **huge pages usage**.
- **ECC error counts (correctable/uncorrectable)** — DRAM hardware degradation, a classic precursor to hardware failure (see §3.4 and the DRAM failure-prediction literature in §9).

### 1.3 Disk / Storage I/O
- **IOPS (read/write)**, **throughput (MB/s)**, **latency (avg, p95, p99)** — tail latency is what users feel; averages hide it.
- **Queue depth**, **utilization % (time device was busy)**.
- **SMART attributes** (reallocated sectors, pending sectors, uncorrectable errors, power-on hours, temperature) — the single best-known predictor set for disk failure (Backblaze's entire public dataset is built on this).
- **Disk space used/available**, **inode exhaustion** (a classic "silent" failure mode: disk has space but is out of inodes).
- **Filesystem errors, remount-as-read-only events**.

### 1.4 Network
- **Throughput in/out**, **packet rate**, **error rate, drop rate, retransmits**.
- **Latency / RTT**, **jitter**, **connection counts, TIME_WAIT buildup**.
- **NIC errors** (CRC errors, link flaps), **buffer/queue drops**.
- **DNS resolution time, TCP handshake time** — application-layer network health.

### 1.5 Power & Thermal
- **Power draw (W)** per server/rack/PDU, **PUE (Power Usage Effectiveness)** at facility level.
- **Inlet/outlet/CPU/GPU die temperature**, **fan speed (RPM)**.
- **Thermal throttling events** — direct cause of "performance slowdown without any application-level fault."
- **Voltage regulator faults, PSU redundancy state**.

### 1.6 GPU (increasingly the center of gravity — Alibaba's 2023 trace release is GPU-centric)
- **SM utilization %, memory utilization %, memory used**.
- **GPU temperature, power draw, ECC errors (SBE/DBE)**.
- **Xid errors** (NVIDIA's GPU error/fault codes) — the single strongest GPU hardware-failure precursor signal, analogous to SMART for disks.
- **PCIe/NVLink throughput and errors**.

### 1.7 Application / Orchestration layer
- **Request rate, error rate, latency (RED method: Rate/Errors/Duration)**.
- **Task/pod scheduling latency, pending time, eviction/preemption events, restart counts**.
- **Job/task exit codes, retry counts** — this is literally the label column in Google/Alibaba traces.

### Mental model for the whole system
Metrics fall into **USE** (Utilization, Saturation, Errors — per Brendan Gregg's method, best for resources: CPU, memory, disk, network) and **RED** (Rate, Errors, Duration — best for services/requests). A comprehensive feature set touches both. Saturation and error-count metrics (queue depth, throttling, retransmits, ECC errors) are consistently the earliest predictors — utilization alone lags the actual failure.

---

## 2. How the Data Is Actually Collected

You won't be deploying real collectors for a trace-based competition, but understanding the pipeline lets you (a) correctly interpret dataset semantics, (b) simulate realistic collection artifacts (missing data, sampling gaps) if you build a synthetic component, and (c) speak fluently about production deployment in your writeup.

### 2.1 Collection architecture (the standard stack)
```
[Node] --agent/exporter--> [scrape or push] --> [TSDB] --> [query layer] --> [dashboards/alerts/ML]
```

- **Node-level agents/exporters**
  - **node_exporter** (Prometheus) — host-level CPU/mem/disk/net via `/proc`, `/sys`.
  - **cAdvisor** — per-container resource usage (cgroups), standard sidecar to Kubelet.
  - **Telegraf** (InfluxData) — plugin-based agent, huge plugin ecosystem (100+ inputs: system, disk, smart, nvidia_smi, docker, etc.), pushes to InfluxDB/Kafka/many sinks.
  - **collectd**, **Datadog agent**, **New Relic infra agent** — commercial/legacy equivalents.
  - **DCGM (NVIDIA Data Center GPU Manager)** — the GPU-equivalent of node_exporter; exposes SM util, ECC errors, Xid errors, power, temp.
  - **smartmontools/smartctl** — SMART data harvesting for disks (source of the Backblaze dataset).
  - **IPMI / Redfish** — out-of-band hardware telemetry: power, fan, thermal, PSU status, independent of OS.

- **Aggregation / transport**
  - **Prometheus** — pull-based; scrapes `/metrics` HTTP endpoints on an interval (default 15s), stores as time series with labels, PromQL for querying. The dominant open-source model at cluster scale.
  - **Grafana** — visualization only (no storage); reads from Prometheus/InfluxDB/Loki/Elasticsearch etc. and renders dashboards + alerting.
  - **Kafka** — event/log transport backbone at hyperscale (Google/Alibaba internally use custom equivalents); decouples producers (agents) from consumers (storage, ML pipelines).
  - **OpenTelemetry (OTel)** — the emerging vendor-neutral standard unifying metrics/traces/logs collection; if you want your writeup to sound current, frame your simulated collector as "OTel-compatible."

- **Storage**
  - **Time-series databases**: Prometheus TSDB, InfluxDB, VictoriaMetrics, Cortex/Thanos/Mimir (long-term, horizontally scalable Prometheus backends), TimescaleDB (Postgres extension).
  - **Log stores**: Elasticsearch/OpenSearch (ELK/EFK stack), Loki (Grafana's log system, label-indexed like Prometheus).

### 2.2 Log formats you'll encounter
- **Structured/JSON logs** — one event per line, keyed fields (timestamp, severity, service, message, trace_id). Easiest for ML.
- **Syslog (RFC 5424)** — standard Unix system log format; still common for OS/network device logs.
- **Windows Event Log (EVTX)** — structured but XML-based, common in enterprise DC context.
- **Application/job logs** (e.g., Hadoop/Spark logs, HDFS logs — the basis of the classic **BGL** (BlueGene/L) and **HDFS** log datasets used in log-anomaly-detection research like DeepLog, LogAnomaly). These are semi-structured free text needing **log parsing/templating** (Drain3 is the standard algorithm) before ML — you tokenize a raw log line into a "log template + variables," then treat template-ID sequences as a time series/sequence classification problem.
- **Traces** — OpenTelemetry/Jaeger/Zipkin span format: a DAG of timed operations, used for latency root-causing in microservices (less relevant to your challenge unless you use the AIOps 2022 microservice dataset).

### 2.3 Practical note for a trace-based competition
The datasets in §7 are already collected and cleaned CSV/JSON/protobuf dumps of exactly this pipeline's output. You should still document the collection architecture in your report (judges will reward systems thinking), but your actual pipeline will read these files directly — you are building the "query layer → ML" half of the diagram above, not the agents.

---

## 3. Common Failure & Slowdown Patterns — What They Look Like in Data

This is the section that actually differentiates a winning entry: showing you understand the *shapes* of failure, not just that you ran an algorithm.

### 3.1 Resource exhaustion (the "slow-then-cliff" pattern)
Memory leak or disk-fill: a **monotonically increasing trend** in memory-used or disk-used, often near-linear, that crosses a threshold and triggers OOM-kill or ENOSPC. In the data: a slowly rising baseline, shrinking headroom, then a discontinuity (process restart, sudden drop to zero = crash/restart).

### 3.2 Saturation / contention ("noisy neighbor")
CPU or I/O saturation without a matching rise in the "official" workload metric: utilization plateaus at ~100% while queue depth/run-queue length keeps climbing and p99 latency detaches from p50. In the data: **latency percentile divergence** (p50 flat, p99 exploding) is the single best statistical signature of contention-driven slowdown, more reliable than mean/utilization alone.

### 3.3 Cascading failure
One component's failure increases load on neighbors, which then fail too — classic in microservices (retry storms) and in distributed schedulers (task requeue floods). In the data: a **step change** in one node's metrics followed by a **time-lagged, spreading** step change in "adjacent" nodes/services — this is why cross-entity/graph features (not just per-node univariate) matter, and why correlation/lag structure across machines is a rich feature source in Google/Alibaba traces (many machines, shared job scheduler).

### 3.4 Hardware degradation (gradual precursor pattern)
Disk: **SMART reallocated-sector-count / pending-sector-count** creeping up over days-to-weeks before a hard failure. DRAM: **correctable ECC error rate** rising before an uncorrectable error / crash. GPU: recurring **Xid error codes** before a fall-off-the-bus event. These are genuinely predictive (not just correlative) with lead times of hours to weeks — this is your best chance at a "predict the failure before it happens" story with real lead time, and it's well documented in the Backblaze/Alibaba disk literature and the DRAM failure-prediction papers (§9).

### 3.5 Thermal throttling
Temperature rises → clock frequency drops → throughput drops with **no change in reported "load"** — this is the classic "slowdown that looks like nothing is wrong" pattern, and a good reason to include temperature/power features even if your headline dataset is CPU/mem-centric.

### 3.6 Straggler / long-tail task pattern (workload-specific, very present in Google/Alibaba traces)
A small fraction of tasks in a job take 10-100x longer than their peers with no obvious resource cause — a scheduling/skew problem rather than hardware failure. This is a distinct, well-studied phenomenon in the Google trace literature and a good secondary case study if you want breadth.

### 3.7 Periodic vs. real anomalies (false-positive trap)
Daily/weekly seasonality (batch jobs at 2am, traffic peaks at business hours) will look like "anomalies" to a naive detector. **Any strong entry must explicitly de-seasonalize or condition on time-of-day/day-of-week** — this is one of the most common reasons naive Isolation Forest/z-score baselines embarrass themselves on real infra data, and calling this out explicitly in your report signals maturity.

### Summary table (use this as your feature/label design checklist)

| Pattern | Signature in data | Best features | Typical lead time |
|---|---|---|---|
| Resource exhaustion | Monotonic trend + cliff | Slope/trend, time-to-threshold extrapolation | Hours–days |
| Saturation/contention | p99 detaches from p50, queue depth up | Percentile spread, queue length, run-queue | Minutes–hours |
| Cascading failure | Step change spreads across nodes with lag | Cross-node correlation, Granger-causality/lag features | Minutes |
| Hardware degradation | Slow creep in error counters (SMART/ECC/Xid) | Error-count trend, rate-of-change | Hours–weeks |
| Thermal throttling | Temp up, freq/throughput down, load flat | Temp, power, frequency vs. utilization ratio | Minutes |
| Straggler tasks | Single task >>peer duration, no resource cause | Peer-relative duration z-score | N/A (concurrent detection) |

---

## 4. Feature Engineering for Time-Series Resource Data

Structure this as a pipeline: **windowing → statistical features → temporal/frequency features → cross-entity features → label construction.**

### 4.1 Windowing
- Fixed-size sliding windows (e.g., 5-min, 15-min, 1-hour) with stride — the standard approach for tabular-ML-on-time-series (feed window features into XGBoost/Isolation Forest).
- Multi-scale windows: compute the same features at 3+ window sizes (short = reactive, long = trend) and concatenate — cheap way to capture both spikes and drifts.
- For sequence models (LSTM/Transformer): fixed-length lookback sequences, not aggregated — the model learns temporal structure itself.

### 4.2 Statistical features per window (per metric)
- Mean, median, std, min, max, range, skewness, kurtosis.
- Percentiles: p50, p90, p95, p99 (critical for latency-type metrics).
- **Rate of change / first derivative** (slope over window), **second derivative** (acceleration — catches "speeding up" degradation).
- Rolling z-score vs. a longer baseline window (adaptive thresholding).
- Coefficient of variation (std/mean) — normalizes volatility across metrics of different scale.

### 4.3 Temporal / trend / frequency features
- **Lag features**: value at t-1, t-5, t-15 (autoregressive signal).
- **Rolling trend via linear regression slope** over the window, or Theil-Sen slope (robust to outliers).
- **STL decomposition** (Seasonal-Trend-Loess) or classical decomposition → separate trend/seasonal/residual; model the residual for anomaly detection (this directly solves the §3.7 false-positive trap).
- **FFT/spectral features**: dominant frequency, spectral entropy — useful for detecting periodicity changes (a system whose daily pattern suddenly changes shape is itself a signal).
- **Autocorrelation (ACF) at key lags** (e.g., lag-24h for daily seasonality strength).
- **Time-since-last-event features**: time since last restart, last GC pause, last error — very predictive and easy to compute, often outperforms raw metric features.

### 4.4 Cross-entity / cross-metric features (this is what separates toy vs. serious solutions)
- **Correlation between metrics** (e.g., CPU vs. memory vs. network) within a window — a shift in normal correlation structure is itself anomalous even if no single metric crosses a threshold (this is exactly what autoencoders and PCA-based detectors exploit).
- **Peer-relative features**: z-score of a machine's metric vs. the fleet/cluster average at the same timestamp — flags "this node is the outlier" rather than "this node is busy" (essential, since load is often globally correlated by time-of-day).
- **Graph/topology features** if the dataset includes job→task→machine structure (Google/Alibaba do): number of co-located jobs, job priority mix, resource requests vs. actual usage (the "overcommit ratio").

### 4.5 Label engineering (the make-or-break step for supervised approaches)
- For failure prediction, don't just label the failure timestamp — construct a **prediction window** (e.g., label all samples in the 30–60 min *before* a failure as positive "pre-failure" class), so the model learns precursor patterns rather than the failure itself. This window length **is your target lead time** — choose it deliberately and report it.
- Exclude a **buffer/blackout window** immediately after a failure event (recovery transients look abnormal but aren't predictive of anything) to avoid contaminating the negative class.
- For multi-class severity (e.g., transient blip vs. sustained degradation vs. hard failure), bucket by outcome duration/recurrence — richer than binary and a good differentiator in your report.

### 4.6 Missing data & irregular sampling
Real collection pipelines drop samples (agent restarts, network partitions). Options, in order of typical rigor: forward-fill for short gaps → interpolation → explicit "missingness" indicator features (missingness itself is sometimes predictive — a node that stops reporting might be the failure) → resampling to a regular grid before feeding sequence models.

### 4.7 Scaling/normalization
Per-machine or per-metric-type normalization (RobustScaler using median/IQR, not mean/std, since failure data is inherently full of outliers you don't want to wash out). Log-transform heavy-tailed metrics (latency, queue depth, bytes) before feeding to models that assume roughly Gaussian inputs (autoencoders, LSTMs with MSE loss).

---

## 5. Models for Anomaly Detection

Anomaly detection = unsupervised/semi-supervised, "does this look unlike normal," used when you don't have (enough) labeled failures — which is the realistic case for most cluster traces.

### 5.1 Classical statistical baselines (always include — cheap, interpretable, good ablation anchor)
- **Z-score / modified z-score (MAD-based)** on residuals after seasonal decomposition.
- **EWMA / CUSUM control charts** — detect sustained shifts, robust to noise, cheap, genuinely used in production monitoring (this is what a lot of real alerting is built on — cite it as "what industry actually runs" for credibility).
- **STL + residual thresholding** — Twitter's `AnomalyDetection` package popularized this approach; still a strong, explainable baseline.

### 5.2 Isolation Forest
- Tree-based; isolates points via random recursive splits — anomalies need fewer splits to isolate. Fast, no distributional assumption, handles high-dimensional tabular features well (your engineered feature vectors from §4 are exactly its ideal input).
- **Strengths**: fast to train/score, scales to millions of rows, minimal tuning (main hyperparameter: contamination rate, n_estimators).
- **Weaknesses**: treats each window independently — no native temporal memory; struggles with anomalies that are only "weird" in a temporal-sequence sense rather than a feature-space sense (mitigate by feeding it the lag/trend features from §4.3, not just raw values).
- **Extended Isolation Forest** fixes an axis-aligned-splitting bias in the original — worth the upgrade, same API in most libraries.

### 5.3 One-Class SVM / Local Outlier Factor (LOF)
Good comparison baselines; LOF captures local density anomalies (useful when "normal" varies by regime, e.g., different times of day) but scales poorly (O(n²)-ish) — subsample or use only for a smaller labeled validation slice, not full-scale training.

### 5.4 Autoencoders (AE) and Variational Autoencoders (VAE)
- Train on (assumed mostly-normal) data to reconstruct input; **reconstruction error = anomaly score**. Works on both tabular windows (dense AE) and raw sequences (LSTM-AE, Conv-AE).
- **LSTM-Autoencoder**: encoder LSTM compresses a sequence to a latent vector, decoder LSTM reconstructs it — captures temporal dependency, the standard deep-learning entry point for time-series AD (widely used baseline in NAB/telemetry AD literature).
- **VAE**: adds a probabilistic latent space; reconstruction probability (not just error) gives a more principled, calibrated anomaly score and is more robust to a few anomalies leaking into training data.
- **Strengths**: unsupervised, multivariate-native (learns cross-metric correlation structure — directly addresses §4.4), no need for labeled failures.
- **Weaknesses**: needs enough "clean" training data; reconstruction-error thresholding still needs a calibration step (percentile of training error, or fit a distribution to validation errors and threshold at a chosen false-positive rate).

### 5.5 Prophet (and classical forecasting: ARIMA/SARIMA/ETS)
- Prophet: additive model (trend + multiple seasonalities + holiday effects) designed for business time series with strong seasonality — genuinely a good fit for daily/weekly data-center load cycles.
- **Use case**: forecast expected value + uncertainty interval; flag actual values falling **outside the prediction interval** as anomalies. This is a forecasting-based (not reconstruction-based) anomaly detection approach and pairs well as an ensemble member alongside AE/Isolation Forest.
- **Weaknesses**: univariate per metric (no cross-metric correlation capture), less effective for the sharp abrupt anomalies vs. gradual drift, can be slow to fit per-series at scale (thousands of machines) unless you batch/vectorize.
- ARIMA/SARIMA are the pre-Prophet classical alternative — more tuning, but sometimes outperform Prophet on short, well-behaved series; good as an additional baseline row in your comparison table.

### 5.6 Matrix Profile (STUMPY) — underused but excellent, worth including for differentiation
Computes the distance to the nearest-neighbor subsequence for every subsequence in a series; peaks = "discords" (never-seen-before shapes = anomalies), valleys = "motifs" (recurring patterns). Parameter-light (mainly window length), mathematically well-founded, and genuinely used in industry — including it shows breadth beyond the "obvious four" algorithms every team will use.

### 5.7 Transformer-based anomaly detection (2022–2025 state of the art)
- **Anomaly Transformer** (Xu et al., ICLR 2022) — introduces "association discrepancy" (local vs. global attention pattern divergence) as the anomaly criterion instead of plain reconstruction error; strong on multivariate benchmarks.
- **TranAD** (Tuli et al., VLDB 2022) — encoder-decoder transformer with adversarial (GAN-style) two-phase training to amplify reconstruction error on anomalies; fast inference, designed explicitly for production-scale multivariate telemetry.
- **MEMTO** (2023) — memory-guided transformer, addresses the common failure mode where transformers "reconstruct anomalies too well" (memory bank of normal prototypes constrains reconstruction).
- **Time-series foundation models** (2023–2025 wave): Chronos (Amazon), TimesFM (Google), Moirai/Uni2TS — pretrained on massive heterogeneous time-series corpora, usable zero-shot or fine-tuned for forecasting-based anomaly detection; genuinely current and a strong "we used 2025 SOTA" talking point if you have GPU budget to fine-tune one.
- **Practical verdict for a solo 4-week build**: transformers give you a strong "we're technically current" story, but they're expensive to tune correctly and easy to get wrong (overfit, "too-good reconstruction" masking anomalies) in a short timeline. Recommendation: implement LSTM-AE and Isolation Forest as your reliable core, implement **one** transformer-based model (TranAD is the most implementation-friendly, with public code) as your "advanced/SOTA" comparison point, and be honest in your report about the tradeoff. Depth of understanding shown in your writeup matters more than raw model count.

---

## 6. Models for Predictive Event Detection (Failure Prediction *Before* It Happens)

This is a genuinely different problem from anomaly detection: anomaly detection asks "is *now* weird," failure prediction asks "will there be an event at t+Δ." It needs labels (or label proxies) and a defined **prediction horizon**.

### 6.1 Framing choices (decide and state explicitly — judges will look for this)
- **Binary classification**: will a failure occur in the next Δ minutes? (most common, most tractable)
- **Time-to-event / survival analysis**: predict remaining-useful-life or hazard rate (Cox proportional hazards, or deep survival models) — more sophisticated framing, directly gives you a lead-time number, strong differentiator if you have time for it.
- **Multi-horizon**: separate models/heads for "will fail in next 5 min" vs "next 60 min" — shows you understand that lead time and precision trade off against each other.

### 6.2 Classical ML (strong, fast, defensible — your reliable backbone)
- **Gradient boosted trees: XGBoost / LightGBM / CatBoost** on the engineered feature vectors from §4. This is empirically the most consistently winning approach on tabular, imbalanced, mixed-type failure-prediction data in both industry and the papers found in research (§9 XGBoost hit near-perfect accuracy on TBF/failing-node-identification tasks) — make this your primary model, not an afterthought baseline.
- **Random Forest** — good, slightly weaker than boosted trees typically, but valuable for feature-importance sanity checks and as a fast baseline.
- **Logistic Regression with engineered features** — always include as the simplest interpretable baseline; also useful if you want calibrated probabilities cheaply.
- Handle **severe class imbalance** (failures are rare) via: class weighting, SMOTE/ADASYN oversampling of the minority class (careful: only on the training split, and preferably on feature space not raw time series to avoid creating unrealistic sequences), or threshold/operating-point tuning rather than naive resampling — and always report **precision-recall curves**, not just accuracy, given imbalance (see §8).

### 6.3 Sequence deep learning
- **LSTM / GRU classifiers** on raw or lightly-featurized windows — the standard deep baseline for task/job-failure prediction on Google-trace-style data (published Bi-LSTM work hit ~87–93% accuracy on Google cluster task/job failure prediction — a good target/reference point to cite and beat or approach).
- **Bidirectional LSTM (Bi-LSTM)** — sees both past and future within a window; can't be used in true real-time streaming (needs the whole window) but fine for offline batch scoring, which is a legitimate deployment mode to describe.
- **Temporal Convolutional Networks (TCN)** — causal dilated convolutions, often trains faster and more stably than LSTM with comparable accuracy; a good "we tried an alternative architecture and it's more efficient" note.
- **Transformer/attention-based failure classifiers** — **Time Machine** (BERT-style, two-stack transformer-decoder, 2024) explicitly predicts *both* failure occurrence and lead time jointly — directly relevant prior art to cite for your "predict before it happens" framing.

### 6.4 Hybrid / ensemble approaches (what actually wins competitions)
- **Two-stage pipeline**: unsupervised anomaly score (from §5, e.g., AE reconstruction error or Isolation Forest score) fed **as an additional feature** into a supervised gradient-boosted classifier — combines the "detect the unknown" strength of unsupervised methods with the calibration/precision of supervised methods on known failure types. This pattern (unsupervised score → supervised meta-model) shows up repeatedly in top AIOps solutions and is very achievable solo.
- **Stacking ensemble** across XGBoost + LightGBM + LSTM predictions (simple weighted average or a small meta-learner) — cheap accuracy/robustness gain, easy to justify, standard Kaggle-style technique. Cited work (AIOPS stacking ensemble on Backblaze-style disk data) uses exactly this pattern for hard-drive failure prediction.
- **Cost-sensitive evaluation**: recent 2024–2025 work explicitly argues raw precision/recall isn't enough — build a cost-benefit metric (cost of false alarm vs. cost of missed failure vs. value of lead time) and optimize your operating threshold against it. Doing this explicitly is a strong, current (2024–2025-literature-aligned) differentiator for your report (§9, McUDI/AIOps survey line of work).

### 6.5 Handling concept drift
Production failure-prediction accuracy degrades over time as workloads change ("Why does Prediction Accuracy Decrease over Time?" — Uncertain Positive Learning, 2024). Even in a static-dataset competition, discussing/demonstrating a **rolling retrain or online-learning evaluation protocol** (train on first N days, test on later days, show performance decay) is a mature, currently-relevant point that most competing teams will skip — a great differentiation section for a solo entry with limited time (it's cheap to add: just change your train/test split methodology, no new modeling).

---

## 7. Best Public Datasets

Priority-ranked for this challenge (best fit first):

### 7.1 Google Cluster Trace — **v2 (2011-2) chosen as primary; see `data/README.md` for the actual decision record**
- **v2 (2011-2)**: 29 days, ~12.5k machines, one Borg cell, plain gzip-CSV, **~41GB compressed total**, publicly downloadable over HTTPS with no Google account or gcloud SDK required (confirmed 2026-07-08 — bucket `clusterdata-2011-2` is anonymously readable via the GCS JSON API). Tables: `job_events`, `task_events` (has EVICT/FAIL/KILL/LOST event-type labels), `task_constraints`, `machine_events` (ADD/REMOVE/UPDATE — hardware failure/maintenance labels), `machine_attributes`, and `task_usage` (the actual resource-usage time series: CPU rate, memory, disk I/O, page cache, cycles-per-instruction, at ~5-min granularity). This is also the exact dataset used in the published Bi-LSTM task/job-failure-prediction work cited in §9 (~87-93% accuracy) — giving a direct literature comparison point.
- **v3 (2019)**: 8 cells, ~2.4TB compressed, JSON with nested CPU-usage histograms, companion power/energy dataset. Richer (GPU/power angle) but requires BigQuery or gsutil-based partial downloads at real scale — a good **stretch/secondary** target if the v2 pipeline is solid early, not a safe primary for a solo 4-week local-data build.
- **Practical takeaway**: v2's `task_usage` table is ~45GB decompressed across 500 hash-partitioned shards (~90MB compressed each); pulling a subset of shards (e.g. 20/500) gives a random ~4% sample of tasks spanning the *full* 29-day period (not a time-truncated slice), which is exactly the kind of scoping cut worth stating explicitly in the report per §8.3.
- Source: `github.com/google/cluster-data` (see `ClusterData2011_2.md` and `ClusterData2019.md` in that repo for the respective format docs).

### 7.2 Alibaba Cluster Trace(s)
- **cluster-trace-v2018** — 8 days, ~4000 machines, co-located online services + batch jobs, machine-level CPU/mem/disk/net usage + container resource requests — good for the "noisy neighbor / overcommit" story (§3.2).
- **cluster-trace-gpu-v2023** — ~6,200 GPUs across ~1,200 machines, AI/ML training & inference workloads, pod-level resource specs, heterogeneous GPU types, high proportion of latency-sensitive pods. **This is your best option if you want a GPU/AI-workload angle** (very on-trend for 2026 judging panels, and genuinely under-explored vs. the older CPU-centric traces most teams will default to).
- **AIOps disk-failure trace** (published for the PAKDD 2020 Alibaba AIOps competition) — millions of disks, 16+ months, SMART features + failure labels — direct plug-and-play for the classical "predict-disk-failure" framing in §3.4, closest thing to a pre-built Kaggle-style competition dataset.
- Source: `github.com/alibaba/clusterdata`.

### 7.3 Microsoft Azure Public Dataset (v1 VM traces, v2 packing traces)
- VM-level CPU utilization time series (5-min readings) over a month, plus VM lifecycle (deployment/deletion) and, in v2, bin-packing-oriented scheduling traces. Good for a **workload-forecasting + capacity-planning** angle (predicting *utilization*, not just failures) and for VM-lifecycle-based "slowdown from over-provisioning" analysis.
- Source: `github.com/Azure/AzurePublicDataset`.

### 7.4 Bitbrains (GWA-T-12, from the Grid Workloads Archive)
- ~1,750 VMs from a real distributed datacenter (fast-storage and rnd traces), CPU/memory/disk/network at 5-min intervals, long-running (months). Smaller/simpler than Google/Alibaba — a good **secondary/validation dataset** to show your method generalizes across data sources (cross-dataset generalization is a very strong differentiator few teams will attempt).

### 7.5 Backblaze Hard Drive Stats
- Daily SMART stats + failure labels for 200k+ drives, continuously published quarterly since 2013 (huge multi-year span available). The gold-standard dataset for §3.4-style hardware-degradation prediction; extremely well documented, very approachable, strong choice if you want one clean, well-labeled, classical-ML-friendly dataset to nail perfectly rather than juggling a huge multi-GB cluster trace.
- Source: `backblaze.com/cloud-storage/resources/hard-drive-test-data` (CSV downloads).

### 7.6 Log-based datasets (secondary/stretch, for a log-anomaly-detection angle)
- **HDFS log dataset** and **BGL (BlueGene/L) log dataset** (via the LogHub/loghub collection) — classic benchmarks for log-template-based anomaly detection (DeepLog-style). Only pursue if you have time left after the core metrics pipeline; it's a genuinely different data modality (semi-structured text) and adds scope risk in a 4-week solo timeline.

### 7.7 AIOps Challenge datasets (2020–2022 microservice traces)
- Multi-modal (metrics + traces + logs) microservice failure datasets from the AIOps Challenge competitions (China's CCF AIOps series) — most realistic "full observability stack" data if you want to demonstrate metric+log+trace fusion, but heavier integration lift.

### Recommended dataset strategy for a solo 4-week build
Pick **one hyperscale trace as your primary** (Google Cluster Trace v3 is the safest, best-documented, most defensible choice — or the Alibaba GPU 2023 trace if you want to differentiate on the AI-workload angle) **plus Backblaze SMART data as a secondary/validation dataset** to demonstrate the hardware-degradation-prediction story cleanly and cross-dataset generalization. This combination covers both "predict task/job/machine failure from utilization telemetry" (novel, high-scale) and "predict hardware failure from degradation signals" (classical, extremely well-labeled) — two distinct, complementary failure-prediction narratives from one coherent codebase.

---

## 8. Evaluation Metrics

Don't just report accuracy — imbalanced failure data makes accuracy nearly meaningless (predicting "no failure" always can give 99%+ accuracy). Report, at minimum:

### 8.1 Classification quality
- **Precision, Recall, F1** (and **F-beta with beta>1**, e.g., F2, if missed failures are costlier than false alarms — usually true in this domain, state your beta choice and justify it).
- **PR-AUC (average precision)** — more informative than ROC-AUC under class imbalance; report this as your headline number, not accuracy.
- **ROC-AUC** — still worth reporting for comparability with prior literature, just don't lead with it.
- **Confusion matrix at your chosen operating threshold**, plus the **precision-recall curve** (not just the AUC scalar) so judges can see the tradeoff you picked and why.

### 8.2 Prediction-specific / time-aware metrics (this is what separates "did anomaly detection" from "did failure *prediction*")
- **Lead time**: distribution (median/mean + spread) of (failure timestamp − first correct alert timestamp). Report this explicitly and prominently — it is the metric that directly answers "did you predict it before it happened," and most naive submissions will forget to report it at all.
- **Detection delay** (for anomaly detection framing): time between anomaly onset and detection.
- **False Positive Rate** and, just as important, **false alarms per unit time** (e.g., per machine-day) — operationally, "1 false alarm per 10,000 machine-hours" communicates better to a judge than a bare percentage.
- **Point-adjust / range-based evaluation**: standard time-series-AD practice is to count a whole contiguous true-anomaly segment as detected if *any* point within it is flagged (point-adjust protocol) — necessary because failures are events over an interval, not single points; but be aware this protocol has been criticized (2022 papers) for inflating scores if used naively/without care — if you use it, say so explicitly and consider also reporting the stricter point-wise numbers side by side. Being aware of and addressing this known pitfall is itself a credibility signal.
- **Cost-based / utility metric**: define an explicit cost matrix (cost of missed failure vs. false alarm vs. value of N minutes of lead time) and report an expected-cost or net-benefit number at your chosen threshold — directly aligned with the 2024–2025 AIOps literature trend noted in §6.4, and makes your results legible to a non-ML judge as "this system would save X."

### 8.3 Robustness / generalization checks (strong differentiators, cheap to add)
- **Train/test split by time, not randomly** (train on earlier days, test on later days) — random splits leak future information into training via overlapping windows and inflate scores; this is a very common and very checkable mistake, avoiding it is a quick credibility win.
- **Cross-machine or cross-dataset generalization**: train on a subset of machines/one dataset, test on held-out machines/the other dataset (§7.4's Bitbrains-as-secondary-dataset use case).
- **Ablation table**: performance with vs. without each feature group (statistical, temporal, cross-entity) and with vs. without the unsupervised-score-as-feature trick (§6.4) — shows engineering rigor, not just a final number.
- **Calibration**: reliability diagram / Brier score if you report probabilities, not just hard labels — small effort, meaningfully more sophisticated than most entries.

---

## 9. Recent Research (2023–2025) Worth Citing

- **Time Machine** (2024) — BERT-style, two-stack transformer-decoder jointly predicting cloud failure occurrence *and* lead time. Directly the closest prior art to "predict events before they happen" framing; strong citation for your intro/related-work section.
- **"Why does Prediction Accuracy Decrease over Time? Uncertain Positive Learning for Cloud Failure Prediction"** (arXiv 2402.00034, 2024) — documents and addresses concept drift in production cloud failure prediction; directly supports the rolling-retrain evaluation protocol recommended in §6.5.
- **McUDI: Model-Centric Unsupervised Degradation Indicator for Failure Prediction AIOps Solutions** (arXiv 2401.14093, 2024) — model-quality-degradation detection for AIOps pipelines themselves; good citation if you discuss maintaining your system post-deployment.
- **"Towards Generic Failure-Prediction Models in Large-Scale Distributed Computing Systems"** (MDPI Electronics, 2025) — recent survey/framework work on generalizable failure prediction across systems; useful as a related-work anchor and for terminology alignment.
- **"A Survey of AIOps for Failure Management in the Era of Large Language Models"** (arXiv 2406.11213, 2024) — broad, current survey of the whole AIOps failure-management space; excellent as your primary related-work citation to show breadth of awareness (and if you have bandwidth, an LLM-assisted log-triage add-on component would align with this trend — clearly stretch scope, not core).
- **"Exploring Error Bits for Memory Failure Prediction: An In-Depth Correlative Study"** (arXiv 2312.02855, 2023) — recent DRAM/ECC-error-based failure prediction; directly supports the §3.4 hardware-degradation feature story if you incorporate memory error data.
- **"GPU Cluster Dynamics: Insights from Alibaba's 2023 Trace Release"** (Computing, Springer, 2024) — the companion analysis paper for the cluster-trace-gpu-v2023 dataset; read this before using that dataset, it documents workload characteristics you'll want to reference.
- **"A Deep Learning Approach for Early Prediction of Task Failures in Cloud Computing Environments"** (ScienceDirect, 2025/2026) — very recent, directly on-topic task-failure-prediction work; good current-year citation.
- **TranAD** (Tuli et al., VLDB 2022) and **Anomaly Transformer** (Xu et al., ICLR 2022) — technically pre-2023 but the two transformer-AD papers everything from 2023–2025 builds on (MEMTO, Pi-Transformer, STAR, etc.); cite as your architectural foundation if you implement a transformer AD model.
- **MEMTO** (2023) — memory-guided transformer anomaly detection, addresses transformers' tendency to over-reconstruct anomalies; the most implementation-relevant of the newer transformer-AD variants if you want something more current than TranAD.
- **Time-series foundation models** — Chronos (Amazon, 2024), TimesFM (Google, 2024), Moirai/Uni2TS (Salesforce, 2024) — genuinely 2024-era, citable as "we evaluated a foundation-model zero-shot baseline" even if your production model is XGBoost/LSTM; low-cost way to demonstrate current awareness.
- **"Public Datasets for Cloud Computing: A Comprehensive Survey"** (ACM Computing Surveys, 2025) — a 2025 survey cataloguing exactly the dataset landscape in §7; cite it directly to justify your dataset choice rather than re-deriving the survey yourself.

---

## Sources consulted for this deep dive
- [Workload Failure Prediction for Data Centers (arXiv 2301.05176)](https://arxiv.org/abs/2301.05176)
- [Towards Generic Failure-Prediction Models in Large-Scale Distributed Computing Systems (MDPI)](https://www.mdpi.com/2079-9292/14/17/3386)
- [Exploring Error Bits for Memory Failure Prediction (arXiv 2312.02855)](https://arxiv.org/pdf/2312.02855)
- [A Deep Learning Approach for Early Prediction of Task Failures (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2772941926000062)
- [GPU Cluster Dynamics: Insights from Alibaba's 2023 Trace Release](https://link.springer.com/content/pdf/10.1007/s00607-024-01369-9.pdf)
- [alibaba/clusterdata (GitHub)](https://github.com/alibaba/clusterdata)
- [Public Datasets for Cloud Computing: A Comprehensive Survey (ACM)](https://dl.acm.org/doi/10.1145/3719003)
- [TranAD: Deep Transformer Networks for Anomaly Detection (arXiv 2201.07284)](https://arxiv.org/abs/2201.07284)
- [A Survey of Deep Anomaly Detection in Multivariate Time Series (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/)
- [Why does Prediction Accuracy Decrease over Time? (arXiv 2402.00034)](https://arxiv.org/pdf/2402.00034)
- [McUDI: Model-Centric Unsupervised Degradation Indicator (arXiv 2401.14093)](https://arxiv.org/pdf/2401.14093)
- [A Survey of AIOps for Failure Management in the Era of LLMs (arXiv 2406.11213)](https://arxiv.org/pdf/2406.11213)
