#!/usr/bin/env python3
"""
ip_guard.py — one-time protective setup so the patent-pending Y⊗Z sequence
cannot leak through the normal bundle/upload/commit workflow.

Does three things:
  1. Writes/updates .gitignore to exclude private provenance + filled circuits.
  2. Scans any bundle staging dirs and zip files for the forbidden content,
     reporting (never auto-deleting) anything that looks like it contains the
     private sequence.
  3. Provides a reusable check_no_private_ip(paths) function the bundler imports
     and calls before zipping, so a leak aborts the bundle.

The detection is heuristic: it flags files in provenance_private/, .qasm files,
and any python file whose build_yz_stabilizer_circuit() is FILLED (i.e. does not
contain NotImplementedError). It cannot understand your gates; it just refuses
to let those files into a public bundle.

Run once:  python src/ip_guard.py --setup
Scan:      python src/ip_guard.py --scan
"""
from __future__ import annotations
import os, re, sys, glob, zipfile, argparse
from pathlib import Path

HOME = Path.home()
ISING = HOME / "ising-bench"
PRIVATE_DIR = ISING / "provenance_private"

GITIGNORE_ENTRIES = [
    "# --- IP protection: never commit/deposit these ---",
    "provenance_private/",
    "*.qasm",
    "yz_submit_FILLED.py",          # if you keep a filled copy, name it this
    "**/qasm_*.qasm",
]

# markers that indicate a python file contains a FILLED circuit (not the stub)
STUB_MARKERS = ["NotImplementedError", "Fill in build_yz_stabilizer_circuit"]
FILLED_HINTS = [r"qc\.rz\(", r"qc\.rx\(", r"qc\.cz\(", r"np\.pi\s*/\s*4"]


def setup_gitignore():
    gi = ISING / ".gitignore"
    existing = gi.read_text().splitlines() if gi.exists() else []
    added = []
    with open(gi, "a") as f:
        if existing and existing[-1].strip():
            f.write("\n")
        for e in GITIGNORE_ENTRIES:
            if e not in existing:
                f.write(e + "\n"); added.append(e)
    print(f"  .gitignore: added {len(added)} entries" if added
          else "  .gitignore: already up to date")


def file_looks_filled(path):
    """True if a .py file appears to contain a filled circuit (not the stub)."""
    try:
        txt = Path(path).read_text()
    except Exception:
        return False
    if "build_yz_stabilizer_circuit" not in txt:
        return False
    has_stub = any(m in txt for m in STUB_MARKERS)
    has_gates = sum(bool(re.search(p, txt)) for p in FILLED_HINTS) >= 2
    # filled = gate hints present AND stub guard removed
    return has_gates and not has_stub


def check_no_private_ip(paths):
    """Bundler calls this before zipping. Returns (ok, offenders)."""
    offenders = []
    for p in paths:
        name = os.path.basename(p)
        if "provenance_private" in str(p):
            offenders.append((p, "private provenance dir"))
        elif name.endswith(".qasm"):
            offenders.append((p, "raw QASM (may encode the sequence)"))
        elif name.endswith(".py") and file_looks_filled(p):
            offenders.append((p, "python file with a FILLED circuit stub"))
    return (len(offenders) == 0), offenders


def scan():
    print("=== scanning for private-IP exposure ===")
    # staging dirs and zips that get uploaded
    candidates = []
    for pat in ["_zenodo*stage/**/*", "*.zip", "**/*.qasm"]:
        candidates += glob.glob(str(ISING / pat), recursive=True)
    # also scan inside zips
    flagged = []
    for c in candidates:
        if c.endswith(".zip"):
            try:
                with zipfile.ZipFile(c) as z:
                    for n in z.namelist():
                        if n.endswith(".qasm") or "provenance_private" in n:
                            flagged.append((f"{c}::{n}", "inside zip"))
            except Exception:
                pass
        else:
            ok, off = check_no_private_ip([c])
            flagged += off
    if flagged:
        print("  !! POTENTIAL EXPOSURE — review before any upload:")
        for p, why in flagged:
            print(f"     {p}  ({why})")
    else:
        print("  clean: no private QASM or filled circuits in staging/zips")
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    if args.setup:
        PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
        setup_gitignore()
        print(f"  private dir ready: {PRIVATE_DIR}")
    if args.scan or not args.setup:
        scan()
    print("\nReminder: keep your filled build_yz_stabilizer_circuit() LOCAL only.")


if __name__ == "__main__":
    main()
