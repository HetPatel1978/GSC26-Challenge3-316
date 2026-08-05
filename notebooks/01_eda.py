"""
EDA over the processed Google Cluster Trace 2011-2 parquet tables.

Run from repo root:
    .venv/Scripts/python.exe notebooks/01_eda.py

Writes PNGs to notebooks/figures/ and a findings summary to
notebooks/EDA_FINDINGS.md (numbers + the specific task/machine IDs used as
worked examples, so feature engineering / the final report can reference
them without re-deriving).

Trace times are microseconds since an unpublished trace-relative epoch (not
a real calendar date), so all plots use "trace day" (time_us / 1e6 / 86400)
rather than absolute dates.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

US_PER_DAY = 1_000_000 * 86400

EVENT_NAMES = {
    0: "SUBMIT", 1: "SCHEDULE", 2: "EVICT", 3: "FAIL", 4: "FINISH",
    5: "KILL", 6: "LOST", 7: "UPDATE_PENDING", 8: "UPDATE_RUNNING",
}
MACHINE_EVENT_NAMES = {0: "ADD", 1: "REMOVE", 2: "UPDATE"}

findings = {}


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
print("== 1. Trace overview ==")
task_events = pl.scan_parquet(PROC / "task_events.parquet")
task_usage = pl.scan_parquet(PROC / "task_usage.parquet")
machine_events = pl.scan_parquet(PROC / "machine_events.parquet")

t_min, t_max = task_events.select(
    pl.col("time").min().alias("t_min"), pl.col("time").max().alias("t_max")
).collect().row(0)
n_days = (t_max - t_min) / US_PER_DAY
n_machines = machine_events.select(pl.col("machine_id").n_unique()).collect().item()
n_jobs = task_events.select(pl.col("job_id").n_unique()).collect().item()
n_tasks = task_events.select(pl.struct(["job_id", "task_index"]).n_unique()).collect().item()
print(f"  trace span: {n_days:.1f} days, {n_machines:,} machines, {n_jobs:,} jobs, {n_tasks:,} tasks")
findings["trace_span_days"] = round(n_days, 1)
findings["n_machines"] = n_machines
findings["n_jobs"] = n_jobs
findings["n_tasks"] = n_tasks

# event type distributions
te_counts = (
    task_events.group_by("event_type").len().sort("event_type").collect()
)
me_counts = (
    machine_events.group_by("event_type").len().sort("event_type").collect()
)
findings["task_event_counts"] = {
    EVENT_NAMES[r[0]]: r[1] for r in te_counts.rows()
}
findings["machine_event_counts"] = {
    MACHINE_EVENT_NAMES[r[0]]: r[1] for r in me_counts.rows()
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar([EVENT_NAMES[r[0]] for r in te_counts.rows()], [r[1] for r in te_counts.rows()], color="#3b6ea5")
axes[0].set_title("task_events by event type")
axes[0].set_ylabel("count")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar([MACHINE_EVENT_NAMES[r[0]] for r in me_counts.rows()], [r[1] for r in me_counts.rows()], color="#a53b3b")
axes[1].set_title("machine_events by event type")
fig.tight_layout()
savefig(fig, "01_event_type_distributions.png")

# ---------------------------------------------------------------------------
print("\n== 2. Cluster-wide resource usage over time ==")
usage_by_day = (
    task_usage
    .with_columns((pl.col("start_time") / US_PER_DAY).alias("day"))
    .group_by((pl.col("day") * 24).cast(pl.Int32).alias("hour_bucket"))
    .agg(
        pl.col("cpu_rate").sum().alias("cpu_rate_sum"),
        pl.col("canonical_memory_usage").sum().alias("mem_sum"),
        pl.len().alias("n_samples"),
    )
    .sort("hour_bucket")
    .collect()
)
hours = usage_by_day["hour_bucket"].to_numpy() / 24.0
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(hours, usage_by_day["cpu_rate_sum"], color="#3b6ea5", lw=0.8)
axes[0].set_ylabel("sum CPU rate\n(all tasks)")
axes[0].set_title("Cluster-wide resource usage over the trace (hourly)")
axes[1].plot(hours, usage_by_day["mem_sum"], color="#3ba55d", lw=0.8)
axes[1].set_ylabel("sum canonical\nmemory usage")
axes[1].set_xlabel("trace day")
fig.tight_layout()
savefig(fig, "02_cluster_usage_over_time.png")

# ---------------------------------------------------------------------------
print("\n== 3. Resource usage distributions ==")
sample = (
    task_usage
    .select("cpu_rate", "canonical_memory_usage", "maximum_cpu_rate", "maximum_memory_usage")
    .collect()
    .sample(n=2_000_000, seed=42)
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(sample["cpu_rate"].drop_nulls(), bins=100, color="#3b6ea5")
axes[0].set_title("CPU rate distribution (2M sample)")
axes[0].set_xlabel("cpu_rate (fraction of core)")
axes[0].set_yscale("log")
axes[1].hist(sample["canonical_memory_usage"].drop_nulls(), bins=100, color="#3ba55d")
axes[1].set_title("Memory usage distribution (2M sample)")
axes[1].set_xlabel("canonical_memory_usage (fraction of machine mem)")
axes[1].set_yscale("log")
fig.tight_layout()
savefig(fig, "03_usage_distributions.png")

qs = [0.5, 0.9, 0.99]
findings["cpu_rate_quantiles"] = {
    str(q): round(sample["cpu_rate"].drop_nulls().quantile(q), 4) for q in qs
}
findings["memory_usage_quantiles"] = {
    str(q): round(sample["canonical_memory_usage"].drop_nulls().quantile(q), 4) for q in qs
}

# ---------------------------------------------------------------------------
print("\n== 4. Failure rate over time (concept-drift signal) ==")
fail_by_day = (
    task_events
    .filter(pl.col("event_type").is_in([2, 3, 6]))  # EVICT, FAIL, LOST
    .with_columns((pl.col("time") / US_PER_DAY).floor().cast(pl.Int32).alias("day"))
    .group_by("day", "event_type")
    .len()
    .sort("day")
    .collect()
)
fig, ax = plt.subplots(figsize=(11, 4))
for et, label, color in [(3, "FAIL", "#a53b3b"), (2, "EVICT", "#c98a2c"), (6, "LOST", "#7a3ba5")]:
    sub = fail_by_day.filter(pl.col("event_type") == et).sort("day")
    ax.plot(sub["day"], sub["len"], marker="o", ms=3, label=label, color=color)
ax.set_xlabel("trace day")
ax.set_ylabel("event count")
ax.set_title("Task failure-adjacent events per day")
ax.legend()
fig.tight_layout()
savefig(fig, "04_failure_rate_over_time.png")

# ---------------------------------------------------------------------------
print("\n== 5. Concrete example: resource exhaustion before a task FAIL ==")
# Find FAIL events, then check whether that task has a healthy run of usage
# samples beforehand (need enough history to see a trend, not an instant fail).
fails = (
    task_events
    .filter(pl.col("event_type") == 3)
    .select("time", "job_id", "task_index", "machine_id", "memory_request", "cpu_request")
    .filter(pl.col("machine_id").is_not_null())
    .collect()
)
# Also grab each failed task's most recent SUBMIT/SCHEDULE request values (already present).
candidate = None
example_usage = None
for row in fails.sample(n=min(4000, fails.height), seed=7).iter_rows(named=True):
    jid, tidx, mid, fail_time = row["job_id"], row["task_index"], row["machine_id"], row["time"]
    if row["memory_request"] is None or row["memory_request"] <= 0:
        continue
    usage = (
        task_usage
        .filter(
            (pl.col("job_id") == jid)
            & (pl.col("task_index") == tidx)
            & (pl.col("machine_id") == mid)
            & (pl.col("end_time") <= fail_time)
        )
        .sort("start_time")
        .collect()
    )
    if usage.height < 8:
        continue
    mem_frac_request = usage["canonical_memory_usage"] / row["memory_request"]
    # Look for a clear upward trend approaching (or exceeding) the requested memory.
    if mem_frac_request[-1] > 0.8 and mem_frac_request[-1] > mem_frac_request[0] * 1.3:
        candidate = row
        example_usage = usage
        break

if candidate is not None:
    days = example_usage["start_time"].to_numpy() / US_PER_DAY
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(days, example_usage["canonical_memory_usage"], color="#3ba55d", label="memory usage")
    ax1.axhline(candidate["memory_request"], color="#3ba55d", ls="--", lw=1, label="memory request")
    ax1.set_ylabel("memory (fraction of machine)", color="#3ba55d")
    ax2 = ax1.twinx()
    ax2.plot(days, example_usage["cpu_rate"], color="#3b6ea5", alpha=0.6, label="cpu rate")
    ax2.set_ylabel("cpu rate", color="#3b6ea5")
    ax1.axvline(candidate["time"] / US_PER_DAY, color="#a53b3b", ls=":", label="FAIL event")
    ax1.set_xlabel("trace day")
    ax1.set_title(f"Job {candidate['job_id']} task {candidate['task_index']} (machine {candidate['machine_id']}): "
                   f"memory ramp before FAIL")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    savefig(fig, "05_example_resource_exhaustion_before_fail.png")
    findings["example_resource_exhaustion"] = {
        "job_id": candidate["job_id"], "task_index": candidate["task_index"],
        "machine_id": candidate["machine_id"], "fail_time_us": candidate["time"],
        "n_usage_samples": example_usage.height,
    }
    print(f"  found example: job {candidate['job_id']} task {candidate['task_index']}")
else:
    print("  no clear resource-exhaustion example found in sampled candidates")
    findings["example_resource_exhaustion"] = None

# ---------------------------------------------------------------------------
print("\n== 6. Concrete example: task churn/load before a machine REMOVE ==")
removes = (
    machine_events
    .filter(pl.col("event_type") == 1)
    .select("time", "machine_id")
    .collect()
)
machine_candidate = None
machine_usage_agg = None
for row in removes.sample(n=min(300, removes.height), seed=11).iter_rows(named=True):
    mid, remove_time = row["machine_id"], row["time"]
    window_start = remove_time - 3 * US_PER_DAY
    usage = (
        task_usage
        .filter(
            (pl.col("machine_id") == mid)
            & (pl.col("start_time") >= window_start)
            & (pl.col("start_time") <= remove_time)
        )
        .group_by("start_time")
        .agg(pl.col("cpu_rate").sum().alias("cpu_rate_sum"), pl.len().alias("n_tasks"))
        .sort("start_time")
        .collect()
    )
    if usage.height < 30:
        continue
    machine_candidate = row
    machine_usage_agg = usage
    break

if machine_candidate is not None:
    days = machine_usage_agg["start_time"].to_numpy() / US_PER_DAY
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(days, machine_usage_agg["cpu_rate_sum"], color="#3b6ea5", label="summed cpu_rate (all tasks on machine)")
    ax1.set_ylabel("summed cpu rate", color="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.plot(days, machine_usage_agg["n_tasks"], color="#c98a2c", alpha=0.7, label="concurrent tasks")
    ax2.set_ylabel("# tasks", color="#c98a2c")
    ax1.axvline(machine_candidate["time"] / US_PER_DAY, color="#a53b3b", ls=":", label="REMOVE event")
    ax1.set_xlabel("trace day")
    ax1.set_title(f"Machine {machine_candidate['machine_id']}: load in the 3 days before REMOVE")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    savefig(fig, "06_example_machine_load_before_remove.png")
    findings["example_machine_before_remove"] = {
        "machine_id": machine_candidate["machine_id"], "remove_time_us": machine_candidate["time"],
    }
    print(f"  found example: machine {machine_candidate['machine_id']}")
else:
    print("  no machine candidate with enough usage history found")
    findings["example_machine_before_remove"] = None

# ---------------------------------------------------------------------------
print("\n== 7. Request vs. actual usage (over/under-provisioning) ==")
req_vs_used = (
    task_events
    .filter((pl.col("event_type") == 0) & pl.col("memory_request").is_not_null() & (pl.col("memory_request") > 0))
    .select("job_id", "task_index", "memory_request", "cpu_request")
    .unique(subset=["job_id", "task_index"])
    .join(
        task_usage.group_by("job_id", "task_index").agg(
            pl.col("canonical_memory_usage").max().alias("peak_mem_used"),
            pl.col("cpu_rate").max().alias("peak_cpu_used"),
        ),
        on=["job_id", "task_index"],
        how="inner",
    )
    .collect()
)
mem_ratio = (req_vs_used["peak_mem_used"] / req_vs_used["memory_request"]).clip(0, 3)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(mem_ratio.drop_nulls(), bins=100, color="#7a3ba5")
ax.axvline(1.0, color="black", ls="--", lw=1, label="used = requested")
ax.set_xlabel("peak memory used / memory requested")
ax.set_ylabel("# tasks")
ax.set_title("Memory over/under-provisioning across tasks")
ax.legend()
fig.tight_layout()
savefig(fig, "07_request_vs_usage.png")
findings["pct_tasks_exceeding_memory_request"] = round(
    100 * (mem_ratio > 1.0).sum() / mem_ratio.len(), 2
)

# ---------------------------------------------------------------------------
print("\n== Writing findings ==")
findings_path = REPO_ROOT / "data" / "processed" / "eda_findings.json"
findings_path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
print(f"  wrote {findings_path.relative_to(REPO_ROOT)}")

md_lines = [
    "# EDA Findings — Google Cluster Trace 2011-2",
    "",
    f"- Trace span: **{findings['trace_span_days']} days**, {findings['n_machines']:,} machines, "
    f"{findings['n_jobs']:,} jobs, {findings['n_tasks']:,} tasks.",
    f"- Task event counts: {findings['task_event_counts']}",
    f"- Machine event counts: {findings['machine_event_counts']}",
    f"- CPU rate quantiles (p50/p90/p99): {findings['cpu_rate_quantiles']}",
    f"- Memory usage quantiles (p50/p90/p99): {findings['memory_usage_quantiles']}",
    f"- {findings['pct_tasks_exceeding_memory_request']}% of tasks exceed their requested memory at peak "
    "(scheduler over-commits memory; these are natural candidates for resource-exhaustion failures).",
    "",
    "## Worked examples (used as figures / can be reused as feature-engineering test cases)",
    f"- Resource exhaustion before FAIL: {findings.get('example_resource_exhaustion')}",
    f"- Load before machine REMOVE: {findings.get('example_machine_before_remove')}",
    "",
    "See notebooks/figures/*.png for the plots.",
]
(Path(__file__).resolve().parent / "EDA_FINDINGS.md").write_text("\n".join(md_lines), encoding="utf-8")
print("  wrote notebooks/EDA_FINDINGS.md")
print("\nDone.")
