"""
Download a local subset of the Google Cluster Trace 2011-2 (v2).

The trace is public on GCS (bucket clusterdata-2011-2) and downloadable via
plain HTTPS with no Google account / gcloud SDK required. Full trace is
~41GB compressed, dominated by task_usage (500 shards x ~90MB). We pull:

  - schema.csv, README                    (reference)
  - machine_events/*                      (single small file, all machines)
  - machine_attributes/*                  (single file, ~140MB)
  - job_events/*                          (all 500 shards, ~150MB total)
  - task_events/*                         (all 500 shards, ~1GB total,
                                            contains FAIL/KILL/EVICT/LOST labels)
  - task_constraints/*                    (all 500 shards, ~150-200MB total)
  - task_usage/part-0000[0-N]-of-00500    (subset of shards only, ~90MB each;
                                            shards are hash-partitioned by task,
                                            so a subset of shards is a random
                                            representative sample of tasks
                                            spanning the full 29-day period)

Usage:
    python download_google_trace.py --out ../../data/raw/google_cluster_2011 \
        --task-usage-shards 20
"""

import argparse
import gzip
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request

BUCKET = "clusterdata-2011-2"
API = f"https://www.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://www.googleapis.com/download/storage/v1/b/{BUCKET}/o"

FULL_TABLES = [
    "machine_events",
    "machine_attributes",
    "job_events",
    "task_events",
    "task_constraints",
]


def _get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "gsc26-ch3"})) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{retries} after error: {e} (sleep {wait}s)")
            time.sleep(wait)


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        url = f"{API}?prefix={prefix}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        data = json.loads(_get(url))
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            return items


def download_object(name: str, dest_dir: Path, decompress: bool = False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = name.split("/")[-1]
    out_path = dest_dir / (fname[:-3] if decompress and fname.endswith(".gz") else fname)
    if out_path.exists():
        return out_path, "skipped (exists)"
    url = f"{MEDIA}/{name.replace('/', '%2F')}?alt=media"
    raw = _get(url)
    if decompress and fname.endswith(".gz"):
        raw = gzip.decompress(raw)
    out_path.write_bytes(raw)
    return out_path, f"{len(raw)/1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../data/raw/google_cluster_2011")
    ap.add_argument("--task-usage-shards", type=int, default=20,
                     help="number of task_usage shards to pull (out of 500); "
                          "~90MB each, so 20 shards ~= 1.8GB")
    ap.add_argument("--decompress", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching schema.csv and README ...")
    for name in ["schema.csv", "README"]:
        path, status = download_object(name, out)
        print(f"  {name}: {status}")

    for table in FULL_TABLES:
        print(f"\nListing {table}/ ...")
        objs = [o for o in list_objects(f"{table}/") if o["name"].endswith(".csv.gz")]
        print(f"  {len(objs)} shard(s), total {sum(int(o['size']) for o in objs)/1e6:.1f} MB compressed")
        for i, o in enumerate(objs):
            path, status = download_object(o["name"], out / table, decompress=args.decompress)
            if i % 50 == 0 or len(objs) < 5:
                print(f"  [{i+1}/{len(objs)}] {o['name']}: {status}")

    print(f"\nListing task_usage/ (sampling {args.task_usage_shards} of 500 shards) ...")
    objs = sorted(
        [o for o in list_objects("task_usage/") if o["name"].endswith(".csv.gz")],
        key=lambda o: o["name"],
    )
    subset = objs[:: max(1, len(objs) // args.task_usage_shards)][: args.task_usage_shards]
    print(f"  selected {len(subset)} shards (evenly spaced across the 500)")
    for i, o in enumerate(subset):
        path, status = download_object(o["name"], out / "task_usage", decompress=args.decompress)
        print(f"  [{i+1}/{len(subset)}] {o['name']}: {status}")

    print("\nDone. Data written to", out.resolve())


if __name__ == "__main__":
    main()
