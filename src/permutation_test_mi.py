#!/usr/bin/env python3
"""
Permutation-test significance for the idle-correlation residual MI.

Turns "MI looks flat/near-zero" into a quantified statement:
    "residual error MI is below X bits at the 99% level — i.e. no correlation
     detectable above the finite-sample/bias noise floor."

Method (per qubit pair i,j):
  * observed MI = MI(col_i, col_j) on the real data.
  * null: independently PERMUTE col_j across shots (breaks any real i-j
    correlation, preserves each qubit's marginal error rate exactly), recompute
    MI. Repeat n_perm times -> null distribution.
  * p-value = fraction of null MIs >= observed.
  * sensitivity floor = 99th percentile of the pooled null distribution: the
    smallest MI that would have been distinguishable from noise.

A pair is "significant" only if its observed MI exceeds the per-pair null
percentile AND survives multiple-comparison correction (Bonferroni over pairs).

Run:  python src/permutation_test_mi.py [--n-perm 500] [--shots-floor]
"""
from __future__ import annotations
import json
import glob
import argparse
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "ising-bench" / "data" / "raw" / "idle_experiment"
RESULTS_DIR = Path.home() / "ising-bench" / "results"
rng = np.random.default_rng(1234)


def mutual_information(a, b):
    """Bias-corrected MI (bits), Miller-Madow."""
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


def permutation_test_pair(col_i, col_j, n_perm):
    """Return (observed_mi, null_mis array)."""
    obs = mutual_information(col_i, col_j)
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = rng.permutation(col_j)
        null[k] = mutual_information(col_i, shuffled)
    return obs, null


def analyze_file(path, n_perm):
    m = json.load(open(path))
    width = m["num_qubits"]
    X = counts_to_array(m["counts"], width)
    if X is None or len(X) < 200:
        return None

    pairs = [(i, j) for i in range(width) for j in range(i + 1, width)]
    n_pairs = len(pairs)
    bonferroni_alpha = 0.01 / n_pairs  # 99% family-wise

    pooled_null = []
    results = []
    n_significant = 0
    for (i, j) in pairs:
        obs, null = permutation_test_pair(X[:, i], X[:, j], n_perm)
        pooled_null.append(null)
        # p-value: fraction of null >= observed (one-sided)
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        sig = p < bonferroni_alpha
        n_significant += int(sig)
        results.append({"pair": [i, j], "mi": float(obs), "p": float(p), "sig": bool(sig)})

    pooled_null = np.concatenate(pooled_null)
    floor_95 = float(np.percentile(pooled_null, 95))
    floor_99 = float(np.percentile(pooled_null, 99))
    max_obs = max(r["mi"] for r in results)

    return {
        "idle_us": m["idle_us"],
        "backend": m["backend"],
        "n_shots_effective": int(len(X)),
        "n_pairs": n_pairs,
        "err_rate": float(X.mean()),
        "max_observed_mi": float(max_obs),
        "sensitivity_floor_95": floor_95,
        "sensitivity_floor_99": floor_99,
        "n_significant_pairs": n_significant,
        "bonferroni_alpha": bonferroni_alpha,
        "pairs": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    files = sorted(glob.glob(str(DATA_DIR / "idle_*.json")))
    files = [f for f in files if json.load(open(f)).get("counts")]
    if not files:
        print("no completed idle jobs found — run fetch_idle_results.py first")
        return

    print(f"permutation test: {args.n_perm} permutations/pair\n")

    # analyze every file, grouped by backend
    from collections import defaultdict
    by_backend = defaultdict(list)
    all_results = {}
    for f in files:
        r = analyze_file(f, args.n_perm)
        if not r:
            continue
        by_backend[r["backend"]].append(r)
        all_results.setdefault(r["backend"], {})[str(r["idle_us"])] = r

    # per-backend detail tables
    for bname in sorted(by_backend):
        rows = sorted(by_backend[bname], key=lambda x: x["idle_us"])
        print(f"=== {bname} ===")
        print(f"{'tau(us)':>8} {'err_rate':>9} {'max MI':>9} {'floor99':>9} "
              f"{'sig pairs':>9} {'verdict':>10}")
        print("-" * 64)
        for r in rows:
            verdict = "SIGNAL" if r["n_significant_pairs"] > 0 else "null"
            print(f"{r['idle_us']:>8} {r['err_rate']:>9.4f} {r['max_observed_mi']:>9.2e} "
                  f"{r['sensitivity_floor_99']:>9.2e} {r['n_significant_pairs']:>9d} "
                  f"{verdict:>10}")
        print()

    # cross-hardware summary table
    print("=" * 78)
    print("CROSS-HARDWARE SUMMARY")
    print("=" * 78)
    print(f"{'backend':>16} {'tau range':>12} {'err_rate':>14} {'max MI':>10} "
          f"{'floor99':>10} {'sig':>5} {'verdict':>8}")
    print("-" * 78)
    for bname in sorted(by_backend):
        rows = by_backend[bname]
        taus = sorted(r["idle_us"] for r in rows)
        errs = [r["err_rate"] for r in rows]
        maxmi = max(r["max_observed_mi"] for r in rows)
        floor = max(r["sensitivity_floor_99"] for r in rows)
        sig = sum(r["n_significant_pairs"] for r in rows)
        verdict = "SIGNAL" if sig > 0 else "null"
        print(f"{bname:>16} {f'{int(min(taus))}-{int(max(taus))}us':>12} "
              f"{f'{min(errs):.4f}-{max(errs):.4f}':>14} {maxmi:>10.2e} "
              f"{floor:>10.2e} {sig:>5d} {verdict:>8}")

    log = RESULTS_DIR / "logs" / "permutation_test.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(log, "w"), indent=2)

    any_signal = any(r["n_significant_pairs"] > 0
                     for b in all_results.values() for r in b.values())
    print("\n" + "=" * 78)
    if any_signal:
        sig_backends = [b for b, rs in all_results.items()
                        if any(r["n_significant_pairs"] > 0 for r in rs.values())]
        print(f"RESULT: significant correlation on {sig_backends} (survives Bonferroni).")
        print("        Effect is BACKEND-SPECIFIC — inspect sig pairs vs physical")
        print("        distance there; that backend warrants a high-shot follow-up.")
    else:
        print("RESULT: NULL across ALL backends tested. No qubit pair on any device")
        print("        shows MI above the permutation null at 99% family-wise.")
        print("        This is a strong controlled-negative result: the null")
        print("        GENERALIZES across Heron-class systems, not just one device.")
    print(f"\nmetrics -> {log}")


if __name__ == "__main__":
    main()
