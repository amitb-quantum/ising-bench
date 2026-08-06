#!/usr/bin/env python3
"""
Residual-MI analysis for the idle/identity experiment.

Because the ideal output is all-zeros, the RESIDUAL is the measured bitstring
itself: every '1' is an error event. We compute:

  1. Per-job pairwise MI matrix over error events (Miller-Madow corrected).
  2. MI vs PHYSICAL heavy-hex distance, using the recorded final_layout +
     backend coupling map (graph distance between physical qubits).
  3. MI vs idle duration tau, to probe the ~30 us characteristic timescale.

This is the analysis the historical archive could not support: known ideal
output (so MI = error correlation, not designed entanglement) AND physical
placement (so the distance axis is real).

Run:  python src/analyze_idle_correlation.py [--coupling-json coupling.json]

If --coupling-json is omitted the script tries QiskitRuntimeService to fetch the
backend coupling map; if offline, it falls back to index separation with a
clear warning (same caveat as the archive analysis).
"""
from __future__ import annotations
import os
import json
import glob
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path.home() / "ising-bench" / "data" / "raw" / "idle_experiment"
RESULTS_DIR = Path.home() / "ising-bench" / "results"


# ---- information-theoretic primitives (Miller-Madow corrected) --------------
def entropy_bits(col):
    n = len(col); p1 = col.mean(); p0 = 1 - p1
    h = 0.0
    for p in (p0, p1):
        if p > 0:
            h -= p * np.log2(p)
    K = int(p0 > 0) + int(p1 > 0)
    return h + (K - 1) / (2 * n * np.log(2))


def mutual_information(a, b):
    n = len(a)
    idx = (a.astype(int) << 1) | b.astype(int)
    joint = np.array([np.mean(idx == k) for k in range(4)])
    pa = np.array([1 - a.mean(), a.mean()])
    pb = np.array([1 - b.mean(), b.mean()])
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = joint[(i << 1) | j]
            if pij > 0 and pa[i] > 0 and pb[j] > 0:
                mi += pij * np.log2(pij / (pa[i] * pb[j]))
    Kab = int((joint > 0).sum()); Ka = int((pa > 0).sum()); Kb = int((pb > 0).sum())
    mi -= (Kab - Ka - Kb + 1) / (2 * n * np.log(2))
    return max(mi, 0.0)


def counts_to_array(counts, width):
    chunks = []
    for bitstring, n in counts.items():
        bs = bitstring.replace(" ", "")
        if len(bs) != width:
            continue
        bits = np.array([int(c) for c in bs], dtype=np.int8)
        chunks.append(np.tile(bits, (int(n), 1)))
    return np.vstack(chunks) if chunks else None


def physical_distances(layout, coupling_edges, width):
    """Graph distance between physical qubits behind each measurement index."""
    import collections
    adj = collections.defaultdict(set)
    for a, b in coupling_edges:
        adj[a].add(b); adj[b].add(a)

    def bfs(src):
        dist = {src: 0}; q = collections.deque([src])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; q.append(y)
        return dist

    phys = {i: layout[str(i)] if str(i) in layout else layout.get(i)
            for i in range(width)}
    D = np.full((width, width), np.nan)
    for i in range(width):
        di = bfs(phys[i])
        for j in range(width):
            if phys[j] in di:
                D[i, j] = di[phys[j]]
    return D


def load_coupling(coupling_json, backend_name):
    if coupling_json and os.path.exists(coupling_json):
        return json.load(open(coupling_json))
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        be = svc.backend(backend_name)
        return list(be.coupling_map.get_edges())
    except Exception as e:
        print(f"  WARNING: no coupling map ({e}); falling back to index distance")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling-json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(DATA_DIR / "idle_*.json")))
    files = [f for f in files if json.load(open(f)).get("counts")]
    if not files:
        print("no completed idle jobs found — run fetch_idle_results.py first")
        return

    RESULTS_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
    by_tau = {}   # tau_us -> mean off-diagonal MI
    dist_mi = defaultdict(list)  # physical_distance -> [MI values across pairs/jobs]

    for f in files:
        m = json.load(open(f))
        width = m["num_qubits"]
        X = counts_to_array(m["counts"], width)
        if X is None or len(X) < 200:
            print(f"  {Path(f).name}: insufficient samples"); continue

        # MI matrix over error events (X is already the residual: ideal=0)
        M = np.zeros((width, width))
        for i in range(width):
            M[i, i] = entropy_bits(X[:, i])
            for j in range(i + 1, width):
                M[i, j] = M[j, i] = mutual_information(X[:, i], X[:, j])

        off = M[np.triu_indices(width, k=1)]
        by_tau[m["idle_us"]] = float(off.mean())
        err_rate = X.mean()
        print(f"  tau={m['idle_us']:>5}us  err_rate={err_rate:.4f}  "
              f"mean off-diag MI={off.mean():.5f}  max={off.max():.5f}")

        # heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="magma", vmin=0)
        ax.set_title(f"Residual-error MI — {m['backend']}, idle {m['idle_us']}us")
        fig.colorbar(im, ax=ax, label="MI (bits)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / f"residmi_tau{int(m['idle_us'])}.png", dpi=150)
        plt.close(fig)

        # physical-distance binning
        edges = load_coupling(args.coupling_json, m["backend"])
        if edges:
            D = physical_distances(m["final_layout"], edges, width)
            for i in range(width):
                for j in range(i + 1, width):
                    if not np.isnan(D[i, j]):
                        dist_mi[int(D[i, j])].append(M[i, j])

    # MI vs idle tau
    if by_tau:
        taus = sorted(by_tau)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(taus, [by_tau[t] for t in taus], "o-", color="#e76f51")
        ax.axvline(30, ls="--", color="gray", alpha=0.6, label="~30us timescale")
        ax.set_xlabel("idle duration tau (us)"); ax.set_ylabel("mean off-diag MI (bits)")
        ax.set_title("Error correlation vs idle time"); ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "mi_vs_tau.png", dpi=150)
        plt.close(fig)

    # MI vs physical distance (the real spatial result)
    if dist_mi:
        dists = sorted(dist_mi)
        means = [np.mean(dist_mi[d]) for d in dists]
        sems = [np.std(dist_mi[d]) / np.sqrt(len(dist_mi[d])) for d in dists]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(dists, means, yerr=sems, fmt="o-", color="#2a9d8f", capsize=3)
        ax.set_xlabel("physical heavy-hex graph distance")
        ax.set_ylabel("mean pairwise MI (bits)")
        ax.set_title("Error MI vs PHYSICAL distance (spatial correlation test)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "mi_vs_physical_distance.png", dpi=150)
        plt.close(fig)
        print("\n  spatial result: MI vs physical distance ->",
              {d: round(np.mean(dist_mi[d]), 5) for d in dists})

    log = RESULTS_DIR / "logs" / "idle_correlation.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"mi_vs_tau": by_tau,
               "mi_vs_distance": {str(d): float(np.mean(v)) for d, v in dist_mi.items()}},
              open(log, "w"), indent=2)
    print(f"\nfigures -> {RESULTS_DIR/'figures'}\nmetrics -> {log}")


if __name__ == "__main__":
    main()
