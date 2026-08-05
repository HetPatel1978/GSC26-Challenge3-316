# EDA Findings — Google Cluster Trace 2011-2

- Trace span: **29.0 days**, 12,583 machines, 672,004 jobs, 25,424,731 tasks.
- Task event counts: {'SUBMIT': 48375166, 'SCHEDULE': 47351173, 'EVICT': 5864353, 'FAIL': 13829769, 'FINISH': 18217975, 'KILL': 10349680, 'LOST': 8754, 'UPDATE_PENDING': 8288, 'UPDATE_RUNNING': 643130}
- Machine event counts: {'ADD': 21443, 'REMOVE': 8957, 'UPDATE': 7380}
- CPU rate quantiles (p50/p90/p99): {'0.5': 0.0027, '0.9': 0.053, '0.99': 0.1941}
- Memory usage quantiles (p50/p90/p99): {'0.5': 0.0053, '0.9': 0.0652, '0.99': 0.1699}
- 3.36% of tasks exceed their requested memory at peak (scheduler over-commits memory; these are natural candidates for resource-exhaustion failures).

## Worked examples (used as figures / can be reused as feature-engineering test cases)
- Resource exhaustion before FAIL: {'job_id': 6318602032, 'task_index': 0, 'machine_id': 372629475, 'fail_time_us': 831323247845, 'n_usage_samples': 28}
- Load before machine REMOVE: {'machine_id': 2274790707, 'remove_time_us': 17015988127}

See notebooks/figures/*.png for the plots.