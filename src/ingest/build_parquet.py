"""
Ingest raw Google Cluster Trace 2011-2 CSVs into clean, typed parquet.

Raw CSVs have no header row and use "" for missing/NULL values (e.g.
machine_id is blank for a task that hasn't been scheduled yet). This script
applies the explicit schema from schemas.py, writes one parquet file per
table under data/processed/, and records row counts + null rates in
data/processed/manifest.json so downstream EDA/feature code (and the final
report) has a single source of truth for what's actually on disk.

Large tables (task_events ~16GB, task_usage ~6.7GB raw CSV) are read via
polars' streaming engine so peak memory stays well under the CSV size.

Usage:
    python src/ingest/build_parquet.py
    python src/ingest/build_parquet.py --tables machine_events,job_events
"""

import argparse
import json
import re
import time
from pathlib import Path

import polars as pl

from schemas import TABLES

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

SHARD_INDEX_RE = re.compile(r"part-(\d+)-of-\d+\.csv$")


def shard_index(path: Path) -> int:
    m = SHARD_INDEX_RE.search(path.name)
    return int(m.group(1))


def build_table(name: str, files: list[Path], out_path: Path, compression: str) -> dict:
    spec = TABLES[name]
    if not files:
        raise FileNotFoundError(f"no source files given for table {name!r}")

    t0 = time.time()
    lf = pl.scan_csv(
        [str(f) for f in files],
        has_header=False,
        new_columns=spec["columns"],
        schema_overrides=spec["dtypes"],
        # "" = normal missing field; 18446744073709551615 (2^64-1) is the
        # trace's sentinel for "time unknown" and doesn't fit in Int64.
        null_values=["", "18446744073709551615"],
    )
    lf.sink_parquet(out_path, compression=compression)
    write_s = time.time() - t0

    # Second pass over the (much smaller, columnar) parquet output to get
    # row count + null rates for the manifest -- cheap relative to the CSV read.
    stats = (
        pl.scan_parquet(out_path)
        .select(pl.len().alias("__n_rows"), pl.all().null_count())
        .collect()
    )
    n_rows = stats["__n_rows"][0]
    null_counts = {c: stats[c][0] for c in spec["columns"]}
    size_mb = out_path.stat().st_size / 1e6

    return {
        "n_source_files": len(files),
        "n_rows": n_rows,
        "size_mb": round(size_mb, 1),
        "write_seconds": round(write_s, 1),
        "null_pct": {
            c: round(100 * n / n_rows, 3) if n_rows else 0.0
            for c, n in null_counts.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw" / "google_cluster_2011"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--tables", default=",".join(TABLES.keys()))
    ap.add_argument("--compression", default="zstd")
    ap.add_argument(
        "--task-usage-max-continuous-index", type=int, default=172,
        help="task_usage shards with index <= this go into task_usage.parquet "
             "(continuous per-task history); any higher-index shards present on "
             "disk (disjoint snapshots from the earlier evenly-spaced sample) go "
             "into task_usage_snapshots.parquet instead, for later cross-time "
             "generalization checks rather than being silently mixed in.",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    manifest = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    for name in tables:
        if name not in TABLES:
            raise ValueError(f"unknown table {name!r}; choices: {list(TABLES)}")

        all_files = sorted((raw_dir / name).glob("part-*.csv"))

        if name == "task_usage":
            cutoff = args.task_usage_max_continuous_index
            continuous = [f for f in all_files if shard_index(f) <= cutoff]
            snapshots = [f for f in all_files if shard_index(f) > cutoff]
            jobs = [("task_usage", continuous, out_dir / "task_usage.parquet")]
            if snapshots:
                jobs.append(("task_usage_snapshots", snapshots, out_dir / "task_usage_snapshots.parquet"))
        else:
            jobs = [(name, all_files, out_dir / f"{name}.parquet")]

        for manifest_key, files, out_path in jobs:
            print(f"\n[{manifest_key}] scanning + writing parquet ({len(files)} files) ...")
            info = build_table(name, files, out_path, args.compression)
            manifest[manifest_key] = info
            print(
                f"[{manifest_key}] {info['n_rows']:,} rows from {info['n_source_files']} files "
                f"-> {info['size_mb']} MB parquet in {info['write_seconds']}s"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
