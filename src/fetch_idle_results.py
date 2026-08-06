#!/usr/bin/env python3
"""
Fetch results for submitted idle-correlation jobs and complete their metadata
files (counts, completed_at, scheduled_duration). Run after submission once the
jobs finish.

Run:  python src/fetch_idle_results.py        # fetch all pending in dir
      python src/fetch_idle_results.py --job <job_id>
"""
from __future__ import annotations
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path.home() / "ising-bench" / "data" / "raw" / "idle_experiment"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def fetch_one(service, meta_path):
    meta = json.load(open(meta_path))
    if meta.get("counts") is not None:
        return "already-complete"
    jid = meta.get("job_id")
    if not jid:
        return "no-job-id"

    job = service.job(jid)
    status = job.status()
    if str(status) not in ("DONE", "JobStatus.DONE"):
        return f"pending ({status})"

    result = job.result()
    pub = result[0]
    # SamplerV2: counts live under the measure register (usually 'meas')
    data = pub.data
    reg = next(iter(data.__dict__)) if hasattr(data, "__dict__") else "meas"
    try:
        counts = pub.data.meas.get_counts()
    except Exception:
        counts = getattr(pub.data, reg).get_counts()

    meta["counts"] = counts
    meta["completed_at"] = utcnow()
    # scheduled_duration if scheduler_timing was enabled
    try:
        sched = pub.metadata["compilation"]["scheduler_timing"]["timing"]
        meta["scheduled_duration"] = sched
    except Exception:
        meta["scheduled_duration"] = None

    json.dump(meta, open(meta_path, "w"), indent=2)
    return "fetched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()

    files = sorted(glob.glob(str(OUT_DIR / "idle_*.json")))
    if args.job:
        files = [f for f in files if args.job in f]
    if not files:
        print("no idle metadata files found in", OUT_DIR)
        return

    for f in files:
        status = fetch_one(service, f)
        print(f"  {Path(f).name}: {status}")


if __name__ == "__main__":
    main()
