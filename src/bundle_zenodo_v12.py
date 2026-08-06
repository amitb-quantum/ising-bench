#!/usr/bin/env python3
"""
Bundle all files for the Zenodo v1.2 revision into a single zip.

Collects three groups:
  1. Original 4 IBM job-result JSONs + 4 metadata files — DOWNLOADED from the
     existing Zenodo record (they live there, not on this machine).
  2. New idle-control data + analysis outputs — collected locally by glob.
  3. Analysis scripts — copied from src/.
Plus the revised description markdown.

Reports exactly what was included and FLAGS anything missing, so you never
upload a silently-incomplete bundle.

Run:  python src/bundle_zenodo_v12.py
Output: ~/ising-bench/zenodo_v1.2_bundle.zip
"""
from __future__ import annotations
import os
import sys
import glob
import zipfile
import urllib.request
from pathlib import Path

HOME = Path.home()
ISING = HOME / "ising-bench"
IDLE_DIR = ISING / "data" / "raw" / "idle_experiment"
LOGS_DIR = ISING / "results" / "logs"
SRC_DIR = ISING / "src"
DESC = Path("/mnt/user-data/outputs/zenodo_revised_description.md")  # or wherever you saved it
OUT_ZIP = ISING / "zenodo_v1.2_bundle.zip"

ZENODO_BASE = "https://zenodo.org/records/18501679/files"
ORIGINAL_REMOTE = [
    "job-d61v0lao8gvs73f1gutg-result.json",
    "job-d62h65ns6ggc73fgqee0-result.json",
    "job-d62lmg3c4tus73fdkb9g-result.json",
    "job-d62lmurc4tus73fdkbo0-result.json",
    "meta_d61v0lao8gvs73f1gutg.json",
    "meta_temporal_d62h65ns6ggc73fgqee0.json",
    "meta_temporal_d62lmg3c4tus73fdkb9g.json",
    "meta_temporal_d62lmurc4tus73fdkbo0.json",
]

# local files: (glob_pattern, expected_min_count, label)
LOCAL_GLOBS = [
    (str(LOGS_DIR / "permutation_test.json"), 1, "permutation results"),
    (str(LOGS_DIR / "idle_correlation.json"), 1, "MI/distance results"),
    (str(IDLE_DIR / "manifest_ibm_marrakesh_*.json"), 1, "marrakesh manifest"),
    (str(IDLE_DIR / "idle_ibm_marrakesh_q12_tau*us_*.json"), 5, "marrakesh idle (5 tau)"),
    (str(IDLE_DIR / "idle_ibm_kingston_Z_idle_q12_tau*us_*.json"), 3, "kingston idle (3 tau)"),
    (str(IDLE_DIR / "idle_ibm_fez_Z_idle_q12_tau*us_*.json"), 3, "fez idle (3 tau)"),
]

SCRIPTS = [
    "analyze_idle_correlation.py",
    "permutation_test_mi.py",
    "fetch_idle_results.py",
    "run_idle_correlation_experiment.py",
    "mi_matrix.py",          # optional
    "ising_benchmark.py",    # optional
]

STAGE = ISING / "_zenodo_stage"


def download_originals(stage: Path):
    got, missing = [], []
    (stage / "original_data").mkdir(parents=True, exist_ok=True)
    for name in ORIGINAL_REMOTE:
        url = f"{ZENODO_BASE}/{name}?download=1"
        dest = stage / "original_data" / name
        try:
            print(f"  downloading {name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            if size == 0:
                raise IOError("empty file")
            print(f"ok ({size} bytes)")
            got.append(name)
        except Exception as e:
            print(f"FAILED ({e})")
            missing.append(name)
    return got, missing


def collect_local(stage: Path):
    (stage / "idle_control_data").mkdir(parents=True, exist_ok=True)
    (stage / "idle_control_results").mkdir(parents=True, exist_ok=True)
    got, warnings = [], []
    for pattern, min_n, label in LOCAL_GLOBS:
        matches = glob.glob(pattern)
        if len(matches) < min_n:
            warnings.append(f"  WARNING: {label}: expected >={min_n}, found {len(matches)}"
                            f"  (pattern: {pattern})")
        subdir = "idle_control_results" if "logs" in pattern else "idle_control_data"
        for m in matches:
            dest = stage / subdir / os.path.basename(m)
            dest.write_bytes(Path(m).read_bytes())
            got.append(os.path.basename(m))
    return got, warnings


def collect_scripts(stage: Path):
    (stage / "analysis_scripts").mkdir(parents=True, exist_ok=True)
    got, missing = [], []
    for s in SCRIPTS:
        src = SRC_DIR / s
        if src.exists():
            (stage / "analysis_scripts" / s).write_bytes(src.read_bytes())
            got.append(s)
        else:
            missing.append(s)
    return got, missing


def main():
    if STAGE.exists():
        import shutil
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    print("=== 1. Original Zenodo data (downloading) ===")
    orig_got, orig_missing = download_originals(STAGE)

    print("\n=== 2. Local idle-control files ===")
    local_got, local_warn = collect_local(STAGE)
    for w in local_warn:
        print(w)
    print(f"  collected {len(local_got)} local data/result files")

    print("\n=== 3. Analysis scripts ===")
    scr_got, scr_missing = collect_scripts(STAGE)
    print(f"  included: {scr_got}")
    if scr_missing:
        print(f"  not found (skipped): {scr_missing}")

    print("\n=== 4. Revised description ===")
    if DESC.exists():
        (STAGE / "README_v1.2_revised_description.md").write_bytes(DESC.read_bytes())
        print(f"  included {DESC.name}")
    else:
        print(f"  WARNING: description not found at {DESC} — bundle will lack the writeup")

    # zip it
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in STAGE.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(STAGE))

    # final report
    print("\n" + "=" * 60)
    print(f"BUNDLE: {OUT_ZIP}  ({OUT_ZIP.stat().st_size} bytes)")
    print("=" * 60)
    if orig_missing:
        print(f"!! {len(orig_missing)} ORIGINAL files failed to download: {orig_missing}")
        print("   (check network access to zenodo.org, or download manually)")
    if local_warn:
        print(f"!! {len(local_warn)} local-file warnings above — review before uploading")
    if not orig_missing and not local_warn:
        print("All expected files present. Bundle is complete.")
    print(f"\nContents:")
    with zipfile.ZipFile(OUT_ZIP) as z:
        for n in sorted(z.namelist()):
            print(f"  {n}")


if __name__ == "__main__":
    main()
