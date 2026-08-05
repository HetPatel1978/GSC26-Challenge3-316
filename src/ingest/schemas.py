"""
Column names + dtypes for the Google Cluster Trace 2011-2 (v2) CSV tables.

Source: data/raw/google_cluster_2011/schema.csv. Files ship without headers,
so scan_csv needs explicit names/dtypes rather than inference (inference over
500 shards is slow and can flip-flop on columns that are all-null in some
shards, e.g. machine_id in early task_events rows before scheduling).

Event type codes (not in schema.csv, from the trace's published docs):
  task/job event: 0=SUBMIT 1=SCHEDULE 2=EVICT 3=FAIL 4=FINISH 5=KILL 6=LOST
                  7=UPDATE_PENDING 8=UPDATE_RUNNING
  machine event:  0=ADD 1=REMOVE 2=UPDATE
"""

import polars as pl

TASK_EVENTS = {
    "columns": [
        "time", "missing_info", "job_id", "task_index", "machine_id",
        "event_type", "user", "scheduling_class", "priority",
        "cpu_request", "memory_request", "disk_space_request",
        "different_machines_restriction",
    ],
    "dtypes": {
        "time": pl.Int64, "missing_info": pl.Int8, "job_id": pl.Int64,
        "task_index": pl.Int64, "machine_id": pl.Int64, "event_type": pl.Int8,
        "user": pl.Utf8, "scheduling_class": pl.Int8, "priority": pl.Int8,
        "cpu_request": pl.Float32, "memory_request": pl.Float32,
        "disk_space_request": pl.Float32,
        "different_machines_restriction": pl.Int8,
    },
    "categorical": ["user"],
}

JOB_EVENTS = {
    "columns": [
        "time", "missing_info", "job_id", "event_type", "user",
        "scheduling_class", "job_name", "logical_job_name",
    ],
    "dtypes": {
        "time": pl.Int64, "missing_info": pl.Int8, "job_id": pl.Int64,
        "event_type": pl.Int8, "user": pl.Utf8, "scheduling_class": pl.Int8,
        "job_name": pl.Utf8, "logical_job_name": pl.Utf8,
    },
    "categorical": ["user"],
}

MACHINE_EVENTS = {
    "columns": ["time", "machine_id", "event_type", "platform_id", "cpus", "memory"],
    "dtypes": {
        "time": pl.Int64, "machine_id": pl.Int64, "event_type": pl.Int8,
        "platform_id": pl.Utf8, "cpus": pl.Float32, "memory": pl.Float32,
    },
    "categorical": ["platform_id"],
}

MACHINE_ATTRIBUTES = {
    "columns": ["time", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"],
    "dtypes": {
        "time": pl.Int64, "machine_id": pl.Int64, "attribute_name": pl.Utf8,
        "attribute_value": pl.Utf8, "attribute_deleted": pl.Int8,
    },
    "categorical": ["attribute_name"],
}

TASK_USAGE = {
    "columns": [
        "start_time", "end_time", "job_id", "task_index", "machine_id",
        "cpu_rate", "canonical_memory_usage", "assigned_memory_usage",
        "unmapped_page_cache", "total_page_cache", "maximum_memory_usage",
        "disk_io_time", "local_disk_space_usage", "maximum_cpu_rate",
        "maximum_disk_io_time", "cpi", "mai", "sample_portion",
        "aggregation_type", "sampled_cpu_usage",
    ],
    "dtypes": {
        "start_time": pl.Int64, "end_time": pl.Int64, "job_id": pl.Int64,
        "task_index": pl.Int64, "machine_id": pl.Int64,
        "cpu_rate": pl.Float32, "canonical_memory_usage": pl.Float32,
        "assigned_memory_usage": pl.Float32, "unmapped_page_cache": pl.Float32,
        "total_page_cache": pl.Float32, "maximum_memory_usage": pl.Float32,
        "disk_io_time": pl.Float32, "local_disk_space_usage": pl.Float32,
        "maximum_cpu_rate": pl.Float32, "maximum_disk_io_time": pl.Float32,
        "cpi": pl.Float32, "mai": pl.Float32, "sample_portion": pl.Float32,
        "aggregation_type": pl.Int8, "sampled_cpu_usage": pl.Float32,
    },
    "categorical": [],
}

TABLES = {
    "task_events": TASK_EVENTS,
    "job_events": JOB_EVENTS,
    "machine_events": MACHINE_EVENTS,
    "machine_attributes": MACHINE_ATTRIBUTES,
    "task_usage": TASK_USAGE,
}
