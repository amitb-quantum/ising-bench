#!/usr/bin/env python3
"""
Pairwise mutual-information (MI) matrix from quantum_central_db measurement counts.

Computes, per qubit width, the full MI matrix between all qubit pairs (by
MEASUREMENT INDEX). This is the direct, data-grounded statement of correlation
structure: off-diagonal MI > 0 means qubits carry shared information beyond
independence — the quantitative trace of the spatial-correlation / non-Markovian
finding.

IMPORTANT SCOPE NOTES (read before interpreting figures):
  * The axis is MEASUREMENT INDEX, not physical heavy-hex lattice position.
    All IBM QPUs here are heavy-hex, but the bitstring index -> physical qubit
    mapping lives in the transpiled final_layout, which is NOT in this data.
    So "MI vs index separation" is a correlation-structure curve, NOT a
    physical-distance decay curve. A dormant final_layout hook is provided to
    upgrade this to true physical distance once layouts are located.
  * MI is bias-corrected (Miller-Madow) since finite shots inflate raw MI.

No new dependencies beyond numpy/matplotlib.
Author: Quantum-Clarity LLC
"""

from __future__ import annotations
import os
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR    = "/home/manager/quantum_central_db"
RESULTS_DIR = Path.home() / "ising-bench" / "results"
N_RECENT    = 50           # most-recent jobs to scan
MAX_SAMPLES = 60000        # cap bit samples per width (memory)
SEED        = 1234
np.random.seed(SEED)


# ----------------------------------------------------------------------------- 
# Optional physical-layout hook (dormant).
# If you locate the transpiled layout for a job, return a list mapping
# measurement-index -> physical qubit id, and the index axis becomes physical.
# -----------------------------------------------------------------------------
def final_layout_for(job_meta: dict):
    """Return index->physical-qubit list, or None if unavailable.
    Currently always None: no layout data exists in quantum_central_db."""
    return None


def load_counts_by_width(data_dir: str, n_recent: int):
    files = glob.glob(os.path.join(data_dir, "job_*results_*.json"))

    loaded = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        counts = d.get("results", {}).get("counts", {})
        if not counts:
            continue
        ts = d.get("completed_at") or d.get("submitted_at") or ""
        loaded.append((ts, counts))

    loaded.sort(key=lambda x: x[0], reverse=True)
    loaded = loaded[:n_recent]

    by_width = defaultdict(list)
    for _, counts in loaded:
        w = len(next(iter(counts)))
        by_width[w].append(counts)
    return by_width


def counts_to_bit_array(counts_list, width, max_samples=MAX_SAMPLES):
    """Expand a list of count dicts into an (N, width) array of bit samples."""
    chunks = []
    for counts in counts_list:
        for bitstring, n in counts.items():
            if len(bitstring) != width:
                continue
            bits = np.array([int(b) for b in bitstring], dtype=np.int8)
            n = min(int(n), 3000)
            chunks.append(np.tile(bits, (n, 1)))
    if not chunks:
        return None
    X = np.vstack(chunks)
    if len(X) > max_samples:
        X = X[np.random.choice(len(X), max_samples, replace=False)]
    return X


def entropy_bits(col):
    """Shannon entropy (bits) of a single binary column, Miller-Madow corrected."""
    n = len(col)
    p1 = col.mean()
    p0 = 1.0 - p1
    h = 0.0
    for p in (p0, p1):
        if p > 0:
            h -= p * np.log2(p)
    # Miller-Madow bias correction: +(K-1)/(2N ln2), K = #nonzero bins
    K = int(p0 > 0) + int(p1 > 0)
    h += (K - 1) / (2 * n * np.log(2))
    return h


def mutual_information(a, b):
    """Bias-corrected MI (bits) between two binary columns."""
    n = len(a)
    # joint distribution over {00,01,10,11}
    joint = np.zeros(4)
    idx = (a.astype(int) << 1) | b.astype(int)
    for k in range(4):
        joint[k] = np.mean(idx == k)
    pa = np.array([1 - a.mean(), a.mean()])
    pb = np.array([1 - b.mean(), b.mean()])

    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = joint[(i << 1) | j]
            if pij > 0 and pa[i] > 0 and pb[j] > 0:
                mi += pij * np.log2(pij / (pa[i] * pb[j]))
    # Miller-Madow: MI bias ~ +(Kab - Ka - Kb + 1)/(2N ln2)
    Kab = int((joint > 0).sum())
    Ka = int((pa > 0).sum())
    Kb = int((pb > 0).sum())
    mi -= (Kab - Ka - Kb + 1) / (2 * n * np.log(2))
    return max(mi, 0.0)


def mi_matrix(X):
    """Full pairwise MI matrix; diagonal = single-qubit entropy."""
    w = X.shape[1]
    M = np.zeros((w, w))
    for i in range(w):
        M[i, i] = entropy_bits(X[:, i])
        for j in range(i + 1, w):
            m = mutual_information(X[:, i], X[:, j])
            M[i, j] = M[j, i] = m
    return M


def mi_vs_separation(M):
    """Average off-diagonal MI as a function of |i-j| index separation."""
    w = M.shape[0]
    seps, vals = [], []
    for s in range(1, w):
        diag_vals = [M[i, i + s] for i in range(w - s)]
        if diag_vals:
            seps.append(s)
            vals.append(np.mean(diag_vals))
    return np.array(seps), np.array(vals)


def analyze_width(width, counts_list, out_dir: Path):
    X = counts_to_bit_array(counts_list, width)
    if X is None or len(X) < 200:
        print(f"  width={width}: insufficient samples, skipped")
        return None
    M = mi_matrix(X)
    off = M[np.triu_indices(width, k=1)]
    print(f"  width={width}: {len(X)} samples, "
          f"mean off-diag MI={off.mean():.4f} bits, max={off.max():.4f} bits")

    # heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="magma", vmin=0)
    ax.set_title(f"MI matrix (bits) — width {width}\n(axis = measurement index, NOT physical qubit)")
    ax.set_xlabel("qubit index"); ax.set_ylabel("qubit index")
    fig.colorbar(im, ax=ax, label="mutual information (bits)")
    fig.tight_layout()
    p1 = out_dir / "figures" / f"mi_matrix_w{width}.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1, dpi=150); plt.close(fig)

    # separation curve for larger widths
    if width >= 5:
        seps, vals = mi_vs_separation(M)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(seps, vals, "o-", color="#2a9d8f")
        ax.set_xlabel("index separation |i-j|  (NOT physical distance)")
        ax.set_ylabel("mean MI (bits)")
        ax.set_title(f"MI vs index separation — width {width}")
        fig.tight_layout()
        p2 = out_dir / "figures" / f"mi_vs_sep_w{width}.png"
        fig.savefig(p2, dpi=150); plt.close(fig)

    return {"width": width, "n_samples": int(len(X)),
            "mean_offdiag_mi": float(off.mean()), "max_mi": float(off.max())}


def main():
    by_width = load_counts_by_width(DATA_DIR, N_RECENT)
    print("widths found:", {w: len(v) for w, v in by_width.items()})

    results = {}
    # process all widths, but report largest-first since those carry the
    # scientific weight for a heavy-hex spatial-correlation argument.
    for w in sorted(by_width, reverse=True):
        r = analyze_width(w, by_width[w], RESULTS_DIR)
        if r:
            results[w] = r

    log = RESULTS_DIR / "logs" / "mi_matrix.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(log, "w"), indent=2)
    print(f"\nfigures -> {RESULTS_DIR/'figures'}")
    print(f"metrics -> {log}")
    print("\nNOTE: index axis != physical heavy-hex position. Physical-distance")
    print("decay analysis requires transpiled final_layout (not in this data).")


if __name__ == "__main__":
    main()
