#!/usr/bin/env python3
"""
Bundle the Zenodo v1.3 (reproducibility correction) deposit.

Folder structure in the zip:
  original_data/        4 job-result JSONs + 4 metadata files
  phase1_reanalysis/    analysis script, rerun log, computed analysis json,
                        multiple-comparison correction json
  idle_control_data/    per-job idle result JSONs (3 backends) + manifest
  idle_control_results/ idle_correlation.json, permutation_test.json
  analysis_scripts/     all idle + reanalysis scripts
  FILES.md              auto-generated manifest describing every file
  README_v1.3.md        the revised description

Verifies presence and FLAGS anything missing. Run from anywhere.
Run:  python src/bundle_zenodo_v13.py
Out:  ~/ising-bench/zenodo_v1.3_bundle.zip
"""
from __future__ import annotations
import os, glob, zipfile, shutil
from pathlib import Path
from datetime import datetime, timezone

HOME = Path.home()
SSM = HOME / "ssm"
ISING = HOME / "ising-bench"
IDLE_DIR = ISING / "data" / "raw" / "idle_experiment"
LOGS_DIR = ISING / "results" / "logs"
SRC_DIR = ISING / "src"

# the v1.3 description — adjust if you saved it elsewhere
DESC = ISING / "zenodo_v1.3_description.md"

STAGE = ISING / "_zenodo_v13_stage"
OUT_ZIP = ISING / "zenodo_v1.3_bundle.zip"

# (source_path_or_glob, dest_subdir, expected_min, label)
PLAN = [
    # original data — these live in ~/ssm
    (str(SSM / "job-d61v0lao8gvs73f1gutg-result.json"), "original_data", 1, "Phase1 result"),
    (str(SSM / "meta_d61v0lao8gvs73f1gutg.json"), "original_data", 1, "Phase1 meta"),
    (str(SSM / "job-d62h65ns6ggc73fgqee0-result.json"), "original_data", 1, "Phase2 result"),
    (str(SSM / "meta_temporal_d62h65ns6ggc73fgqee0.json"), "original_data", 1, "Phase2 meta"),
    (str(SSM / "job-d62lmg3c4tus73fdkb9g-result.json"), "original_data", 1, "Phase3A result"),
    (str(SSM / "meta_temporal_d62lmg3c4tus73fdkb9g.json"), "original_data", 1, "Phase3A meta"),
    (str(SSM / "job-d62lmurc4tus73fdkbo0-result.json"), "original_data", 1, "Phase3B result"),
    (str(SSM / "meta_temporal_d62lmurc4tus73fdkbo0.json"), "original_data", 1, "Phase3B meta"),
    # phase 1 reanalysis
    (str(SSM / "analyze_yz_syndrome_sweep.py"), "phase1_reanalysis", 1, "Phase1 analysis script"),
    (str(SSM / "phase1_rerun.log"), "phase1_reanalysis", 1, "Phase1 rerun log"),
    (str(SSM / "yz_syndrome_analysis.json"), "phase1_reanalysis", 1, "Phase1 computed output"),
    (str(SSM / "phase1_multiple_comparison_correction.json"), "phase1_reanalysis", 1, "Phase1 correction"),
    # idle control data
    (str(IDLE_DIR / "idle_*.json"), "idle_control_data", 11, "idle result JSONs"),
    (str(IDLE_DIR / "manifest_*.json"), "idle_control_data", 1, "idle manifest(s)"),
    # idle control results
    (str(LOGS_DIR / "idle_correlation.json"), "idle_control_results", 1, "MI/distance results"),
    (str(LOGS_DIR / "permutation_test.json"), "idle_control_results", 1, "permutation results"),
    # analysis scripts
    (str(SRC_DIR / "run_idle_correlation_experiment.py"), "analysis_scripts", 1, "idle submit"),
    (str(SRC_DIR / "fetch_idle_results.py"), "analysis_scripts", 1, "idle fetch"),
    (str(SRC_DIR / "analyze_idle_correlation.py"), "analysis_scripts", 1, "idle analysis"),
    (str(SRC_DIR / "permutation_test_mi.py"), "analysis_scripts", 1, "permutation test"),
    (str(SRC_DIR / "phase1_correction.py"), "analysis_scripts", 1, "phase1 correction script"),
    (str(SRC_DIR / "reanalyze_reprepare.py"), "analysis_scripts", 1, "reprepare reanalysis (provenance check)"),
]

DESCRIPTIONS = {
    "job-d61v0lao8gvs73f1gutg-result.json": "Phase 1 spatial Y⊗Z parity sweep, raw IBM result (36 circuits, 6 modules).",
    "job-d62h65ns6ggc73fgqee0-result.json": "Phase 2 temporal sequential-memory, raw IBM result. (Claim WITHDRAWN — see README.)",
    "job-d62lmg3c4tus73fdkb9g-result.json": "Phase 3A echo-decoupling, raw IBM result. (Low significance.)",
    "job-d62lmurc4tus73fdkbo0-result.json": "Phase 3B reset/reprepare, raw IBM result. (Claim WITHDRAWN — not reproducible.)",
    "analyze_yz_syndrome_sweep.py": "Phase 1 analysis pipeline. Loads the raw result, decodes counts, computes parity consistency. Reproduces published Phase 1 numbers.",
    "phase1_rerun.log": "Console log of the Phase 1 reanalysis rerun (reproduces 4.86σ Module 0 etc.).",
    "yz_syndrome_analysis.json": "Computed Phase 1 output produced by the analysis script.",
    "phase1_multiple_comparison_correction.json": "Bonferroni + BH-FDR correction across the 6 Phase 1 modules.",
    "idle_correlation.json": "Idle study: residual MI vs physical distance and vs idle time.",
    "permutation_test.json": "Idle study: permutation-null significance + Bonferroni, per backend.",
    "run_idle_correlation_experiment.py": "Idle study: builds + submits idle/identity circuits, saves layout/metadata.",
    "fetch_idle_results.py": "Idle study: fetches completed jobs, fills counts/timing.",
    "analyze_idle_correlation.py": "Idle study: residual-MI matrix, physical-distance + tau analysis.",
    "permutation_test_mi.py": "Idle study: permutation test + multiple-comparison correction.",
    "phase1_correction.py": "Computes the Phase 1 multiple-comparison correction JSON.",
    "reanalyze_reprepare.py": "Attempted reanalysis of Phase 3B from raw register; documents the reproducibility gap.",
}


def collect():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    included, warnings = [], []
    for src, sub, min_n, label in PLAN:
        (STAGE / sub).mkdir(parents=True, exist_ok=True)
        matches = glob.glob(src)
        if len(matches) < min_n:
            warnings.append(f"  MISSING/SHORT: {label}: expected >={min_n}, found {len(matches)} ({src})")
        for mpath in matches:
            dest = STAGE / sub / os.path.basename(mpath)
            shutil.copy2(mpath, dest)
            included.append((sub, os.path.basename(mpath)))
    return included, warnings


def write_files_md(included):
    lines = ["# FILES.md — Zenodo v1.3 (reproducibility correction)\n",
             f"Generated {datetime.now(timezone.utc).isoformat()}\n",
             "\nThis deposit separates reproducible from non-reproducible claims. "
             "See README_v1.3.md for the full status of each.\n"]
    by_sub = {}
    for sub, name in included:
        by_sub.setdefault(sub, []).append(name)
    for sub in ["original_data", "phase1_reanalysis", "idle_control_data",
                "idle_control_results", "analysis_scripts"]:
        if sub not in by_sub:
            continue
        lines.append(f"\n## {sub}/\n")
        for name in sorted(by_sub[sub]):
            desc = DESCRIPTIONS.get(name, "")
            lines.append(f"- `{name}` — {desc}" if desc else f"- `{name}`")
    (STAGE / "FILES.md").write_text("\n".join(lines))


def main():
    print("=== collecting files ===")
    included, warnings = collect()
    print(f"  collected {len(included)} files")
    for w in warnings:
        print(w)

    print("\n=== description ===")
    if DESC.exists():
        shutil.copy2(DESC, STAGE / "README_v1.3.md")
        print(f"  included {DESC.name} -> README_v1.3.md")
    else:
        warnings.append(f"  MISSING: description at {DESC}")
        print(f"  !! description not found at {DESC}")

    write_files_md(included)
    print("  generated FILES.md")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in STAGE.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(STAGE))

    print("\n" + "=" * 60)
    print(f"BUNDLE: {OUT_ZIP}  ({OUT_ZIP.stat().st_size} bytes)")
    print("=" * 60)
    if warnings:
        print("!! REVIEW WARNINGS before uploading:")
        for w in warnings:
            print(w)
    else:
        print("All expected files present.")
    print("\nContents:")
    with zipfile.ZipFile(OUT_ZIP) as z:
        for n in sorted(z.namelist()):
            print(f"  {n}")


if __name__ == "__main__":
    main()
