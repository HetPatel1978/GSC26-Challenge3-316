# Dataset: Google Cluster Trace 2011-2 (v2) — primary

**Decision (2026-07-08): switched primary dataset from v3 (2019) to v2 (2011).**
v3 is 2.4TB compressed across 8 cells in JSON with nested histograms — impractical
for a local subset without BigQuery. v2 is ~41GB compressed, plain CSV, publicly
downloadable over HTTPS with no Google account, and is the exact trace used in the
Bi-LSTM task/job-failure-prediction papers cited in `docs/01-DEEP-DIVE.md` §9 —
giving us a direct literature baseline to compare against.

## How it was downloaded

`src/ingest/download_google_trace.py` pulls directly from the public GCS bucket
`clusterdata-2011-2` via the anonymous JSON API (`storage.googleapis.com`), no
`gcloud`/`gsutil` install required. Re-run with:

```bash
python src/ingest/download_google_trace.py --out data/raw/google_cluster_2011 --task-usage-shards 20
```

It is idempotent (skips files that already exist).

## What we have locally (data/raw/google_cluster_2011/)

29 days (2011-05-01 to 2011-05-29), ~12.5k machines, one Borg cell.

| Table | Shards pulled | Size on disk (decompressed) | Contents |
|---|---|---|---|
| `machine_events` | 1/1 (all) | 2.8 MB | machine add/remove/update events — **hardware failure/maintenance labels** |
| `machine_attributes` | 1/1 (all) | 1.2 GB | machine attribute key/value pairs (platform, kernel, etc.) |
| `job_events` | 500/500 (all) | 318 MB | job submit/schedule/finish/kill/fail events |
| `task_events` | 500/500 (all) | 16 GB | task submit/schedule/**evict/fail/kill/lost**/finish events + resource *requests* — **task failure labels live here** |
| `task_constraints` | 500/500 (all) | 2.9 GB | scheduling constraints per task |
| `task_usage` | 20/500 (subsampled) | 6.7 GB | **actual resource usage time series** (CPU rate, memory, disk I/O, cache, CPI) at ~5-min intervals — shards are hash-partitioned by task, so 20/500 shards is a random ~4% sample of tasks spanning the *full* 29-day period, not a time-truncated slice |

Total: ~27 GB on disk. If more of `task_usage` is needed later (e.g. for the final
model), re-run with a higher `--task-usage-shards` value — 419 GB free disk as of
2026-07-08, so pulling significantly more (or all 500 shards, ~45GB decompressed x
some ratio) is feasible if the 20-shard sample proves insufficient.

## Schema (from `schema.csv`)

Full field list is in `schema.csv`. Key columns:

- **task_events**: time, missing info, job ID, task index, machine ID, **event type**, user, scheduling class, priority, CPU/memory/disk *request*, different-machines restriction.
- **task_usage**: start time, end time, job ID, task index, machine ID, CPU rate, canonical/assigned/maximum memory usage, unmapped/total page cache, disk I/O time, local disk space usage, maximum CPU rate, maximum disk IO time, **cycles per instruction**, **memory accesses per instruction**, sample portion, aggregation type, sampled CPU usage.
- **machine_events**: time, machine ID, event type, platform ID, CPUs, Memory.

**Event type codes** (from the trace's published documentation, not repeated in
`schema.csv` itself — record here since we'll need them for label construction):

Task/job event type: `0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL, 6=LOST, 7=UPDATE_PENDING, 8=UPDATE_RUNNING`
Machine event type: `0=ADD, 1=REMOVE, 2=UPDATE`

`EVICT`/`FAIL`/`LOST` on task_events and `REMOVE` on machine_events are the
failure-adjacent labels for the two prediction tasks described in the README
problem statement (task-failure prediction from task_usage telemetry, and
machine-failure prediction where possible).

## Secondary dataset (not yet downloaded)

Backblaze quarterly SMART hard-drive stats — see `docs/01-DEEP-DIVE.md` §7.5.
Deferred until the primary pipeline (ingest → features → labels → models) is
working end-to-end on the Google trace, per the Week-by-week plan.
