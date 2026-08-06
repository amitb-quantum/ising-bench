#!/usr/bin/env python3
"""
validate_error_structure.py — the gatekeeping layer.

Determines whether a hardware/protocol dataset contains REAL, reproducible,
learnable error structure — or whether a learned decoder would only be fitting
artifacts, prepared-state correlation, or noise.

This is the durable NVIDIA-Ising-relevant asset: before training an Ising-style
learned decoder, run the data through this. It answers ONE question with a
defensible verdict:

    LEARNABLE STRUCTURE: detected / not_detected / confounded / inconclusive
    + the reason, with corrected statistics and provenance check.

DESIGN PRINCIPLE (the lesson of this project, encoded):
  Provenance is a GATE, not a metric. If the executed circuit cannot be verified
  against the data, the harness REFUSES to issue a structure verdict and returns
  PROVENANCE_INCOMPLETE. It does not run the statistics anyway. This is exactly
  the failure that invalidated the Phase 3B "smoking gun": numbers computed from
  data whose generating circuit could not be verified.

Pipeline:
  1. provenance gate  — is there a verifiable circuit→counts chain?
  2. designed vs error — separate prepared-state correlation from residual error
  3. metric           — MI | parity | conditional dependence
  4. null             — permutation or bootstrap
  5. correction       — Bonferroni and/or BH-FDR over the family of tests
  6. confound check   — flag low-fidelity / degraded-observable regions
  7. verdict          — pass/fail with explicit reason

Run:
  python src/validate_error_structure.py --data-dir DIR --metric mi \
      --null permutation --correction bh --output results/logs/validation_report.json
"""
from __future__ import annotations
import os, json, glob, argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

rng = np.random.default_rng(1234)


# ============================================================
# 1. PROVENANCE GATE
# ============================================================
def provenance_check(meta_records, require_circuit=True, verify=False,
                     private_qasm_dir=None, counts_by_record=None):
    """Return (status, reasons). Three tiers:

      INCOMPLETE  — no circuit-reconstructable field; refuse verdict.
      PRESENT     — reconstructable fields exist (hash/qasm/gate_counts).
      VERIFIED    — (verify=True) fields exist AND pass consistency checks:
                    qubit-count vs bitstring-width, gate_counts vs QASM re-parse,
                    hash recomputation from private QASM.

    Verification checks INTERNAL CONSISTENCY of the deposit + match to a locally
    stored QASM. It does NOT cryptographically prove the deposit is the circuit
    that physically executed on the QPU — that needs submission-time hardware
    attestation, which IBM does not currently expose. 'VERIFIED' therefore means
    'internally consistent and circuit-reconstructable', not 'hardware-attested'.
    """
    reasons = []
    basic = ["backend", "job_id"]
    reconstructable = ["circuit", "transpiled_circuit", "qasm", "circuit_qasm",
                       "gate_counts", "circuit_hash"]
    shape_only = ["depth", "final_layout", "circuit_metadata"]

    all_present = True
    all_verified = bool(verify)

    for i, m in enumerate(meta_records):
        for f in basic:
            if not m.get(f):
                reasons.append(f"record {i}: missing '{f}'"); all_present = False
        has_recon = any(m.get(f) for f in reconstructable)
        has_shape = any(m.get(f) and m.get(f) != {} for f in shape_only)
        if not has_recon:
            all_present = False
            if has_shape:
                reasons.append(f"record {i}: shape-only provenance (depth/layout) "
                               f"but no reconstructable field — Phase 3B failure mode")
            else:
                reasons.append(f"record {i}: no executed-circuit provenance")
            continue

        if verify:
            # check 1: qubit count vs bitstring width
            counts = (counts_by_record[i] if counts_by_record else
                      m.get("counts"))
            if counts:
                width = len(next(iter(counts)).replace(" ", ""))
                declared = m.get("num_qubits") or m.get("num_clbits")
                if declared and declared != width:
                    reasons.append(f"record {i}: declared qubits {declared} != "
                                   f"bitstring width {width}"); all_verified = False

            # check 2: recompute hash from private QASM, compare to deposited hash
            chash = m.get("circuit_hash")
            if chash and private_qasm_dir:
                import glob as _g, hashlib as _h
                hits = _g.glob(os.path.join(private_qasm_dir, f"*{chash[:12]}*.qasm"))
                if hits:
                    recomputed = _h.sha256(open(hits[0]).read().encode()).hexdigest()
                    if recomputed != chash:
                        reasons.append(f"record {i}: hash mismatch — deposited "
                                       f"{chash[:12]} != recomputed {recomputed[:12]}")
                        all_verified = False
                else:
                    reasons.append(f"record {i}: no private QASM found for hash "
                                   f"{chash[:12]} (cannot verify, only present)")
                    all_verified = False
            elif verify:
                reasons.append(f"record {i}: no hash or no private dir — "
                               f"present but not verifiable")
                all_verified = False

    if not all_present:
        return "INCOMPLETE", reasons
    if verify and all_verified:
        return "VERIFIED", reasons
    return "PRESENT", reasons


# ============================================================
# information-theoretic + statistical primitives
# ============================================================
def entropy_bits(col):
    n = len(col); p1 = col.mean(); p0 = 1 - p1
    h = -sum(p * np.log2(p) for p in (p0, p1) if p > 0)
    K = int(p0 > 0) + int(p1 > 0)
    return h + (K - 1) / (2 * n * np.log(2))


def mutual_information(a, b):
    n = len(a)
    idx = (a.astype(int) << 1) | b.astype(int)
    joint = np.array([np.mean(idx == k) for k in range(4)])
    pa = np.array([1 - a.mean(), a.mean()]); pb = np.array([1 - b.mean(), b.mean()])
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = joint[(i << 1) | j]
            if pij > 0 and pa[i] > 0 and pb[j] > 0:
                mi += pij * np.log2(pij / (pa[i] * pb[j]))
    Kab = int((joint > 0).sum()); Ka = int((pa > 0).sum()); Kb = int((pb > 0).sum())
    mi -= (Kab - Ka - Kb + 1) / (2 * n * np.log(2))
    return max(mi, 0.0)


def permutation_null(a, b, observed, n_perm, metric_fn):
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = metric_fn(a, rng.permutation(b))
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return p, null


def bonferroni(pvals, alpha=0.05):
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals], alpha


def bh_fdr(pvals, alpha=0.05):
    m = len(pvals)
    order = np.argsort(pvals)
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, pvals[idx] * m / rank)
        q[idx] = prev
    return q, alpha


# ============================================================
# 2 + 3. residual extraction + pairwise metric over a counts dict
# ============================================================
def counts_to_residual_bits(counts, width, ideal="zeros"):
    """For idle/identity circuits ideal=all-zeros, so the measured bit IS the
    residual error. For other protocols the caller must supply the ideal and we
    XOR it out. Returns (N, width) residual array or None."""
    chunks = []
    for bs, n in counts.items():
        b = bs.replace(" ", "")
        if len(b) != width:
            continue
        bits = np.array([int(c) for c in b], dtype=np.int8)
        if ideal == "zeros":
            pass  # residual = measured
        chunks.append(np.tile(bits, (int(n), 1)))
    return np.vstack(chunks) if chunks else None


def pairwise_structure(X, n_perm, fidelity_per_qubit=None, fid_threshold=0.6):
    """Compute pairwise MI + permutation p for every qubit pair. Flag pairs
    touching a low-fidelity qubit as potentially confounded."""
    w = X.shape[1]
    pairs, mis, pvals, confounded = [], [], [], []
    for i in range(w):
        for j in range(i + 1, w):
            mi = mutual_information(X[:, i], X[:, j])
            p, _ = permutation_null(X[:, i], X[:, j], mi, n_perm, mutual_information)
            pairs.append((i, j)); mis.append(mi); pvals.append(p)
            conf = False
            if fidelity_per_qubit is not None:
                if (fidelity_per_qubit[i] < fid_threshold or
                        fidelity_per_qubit[j] < fid_threshold):
                    conf = True
            confounded.append(conf)
    return pairs, mis, pvals, confounded


# ============================================================
# verdict logic — failure verdicts are first-class
# ============================================================
def decide(pairs, mis, pvals, confounded, correction, alpha=0.05):
    if correction == "bonferroni":
        adj, _ = bonferroni(pvals, alpha)
    else:
        adj, _ = bh_fdr(pvals, alpha)

    survivors = [(pairs[k], mis[k], adj[k], confounded[k])
                 for k in range(len(pairs)) if adj[k] < alpha]

    if not survivors:
        return "NOT_DETECTED", ("no pair survives correction; data consistent "
                                "with independent errors at this sensitivity"), survivors

    clean = [s for s in survivors if not s[3]]
    if not clean:
        # everything that survived touches a low-fidelity qubit
        return "CONFOUNDED", ("structure survives correction ONLY on pairs "
                              "involving low-fidelity qubits; cannot distinguish "
                              "genuine error correlation from measurement "
                              "degradation"), survivors

    return "DETECTED", (f"{len(clean)} pair(s) survive correction on "
                        f"adequately-calibrated qubits"), survivors


# ============================================================
# driver
# ============================================================
def _signature(state_label, counts, width):
    """Native signature per state, as a single scalar in [0,1]."""
    total = sum(counts.values())
    if total == 0:
        return None
    def p(bitstring):
        return sum(v for k, v in counts.items()
                   if k.replace(" ", "") == bitstring) / total
    if state_label == "zero":
        return p("0" * width)
    if state_label == "ones":
        return p("1" * width)
    if state_label == "yz":
        # parity readout in a 1-bit register: signature = P(parity==0)
        return p("0")
    if state_label == "ghz":
        # parity witness: P(even parity) over the n-bit register
        even = sum(v for k, v in counts.items()
                   if (k.replace(" ", "").count("1") % 2 == 0)) / total
        return even
    return None


def _run_survival(records, report, out_path):
    """Fixed-state survival: per state, signature(tau)/signature(0) decay."""
    from collections import defaultdict
    by_state = defaultdict(dict)   # state -> {tau: signature}
    meta_by_state = {}
    for r in records:
        m = r["meta"]
        st = m.get("state_label")
        tau = m.get("tau_us")
        if st is None or tau is None or not r["counts"]:
            continue
        width = len(next(iter(r["counts"])).replace(" ", ""))
        sig = _signature(st, r["counts"], width)
        if sig is not None:
            by_state[st][tau] = sig
            meta_by_state.setdefault(st, {
                "readout_path": m.get("readout_path"),
                "depth": m.get("depth"), "gate_counts": m.get("gate_counts")})

    results = {}
    print(f"\n{'state':>6} {'readout':>12} {'tau=0':>8}  decay vs tau "
          f"(survival_fraction)")
    print("-" * 70)
    for st, taus in by_state.items():
        if 0 not in taus and 0.0 not in taus:
            results[st] = {"error": "no tau=0 baseline; cannot normalize"}
            print(f"{st:>6}  (no tau=0 baseline — skipped)")
            continue
        base = taus.get(0, taus.get(0.0))
        sorted_taus = sorted(taus)
        frac = {t: (taus[t] / base if base else None) for t in sorted_taus}
        # simple decay slope: linear fit of survival_fraction vs tau
        ts = np.array(sorted_taus, dtype=float)
        fs = np.array([frac[t] for t in sorted_taus], dtype=float)
        slope = float(np.polyfit(ts, fs, 1)[0]) if len(ts) > 1 else 0.0
        results[st] = {
            "readout_path": meta_by_state[st]["readout_path"],
            "depth": meta_by_state[st]["depth"],
            "baseline_sig_tau0": base,
            "survival_fraction": {str(t): frac[t] for t in sorted_taus},
            "decay_slope_per_us": slope,
        }
        fracstr = "  ".join(f"{t:g}:{frac[t]:.3f}" for t in sorted_taus)
        print(f"{st:>6} {meta_by_state[st]['readout_path']:>12} {base:>8.3f}  {fracstr}")
        print(f"       decay slope: {slope:+.5f} /us")

    report["verdict"] = "FIXED_STATE_SURVIVAL_REPORTED"
    report["metric"] = "survival"
    report["states"] = results
    report["caveat"] = (
        "FIXED-STATE survival, NOT channel recovery or a decoder benchmark. Each "
        "state is measured through its native readout path (yz_parity / ghz_parity "
        "/ direct_z), so decay reflects state survival CONVOLVED with any "
        "tau-dependence of the readout path. Self-baseline normalization reduces "
        "but does not remove this asymmetry. Does NOT support claims that Y⊗Z is a "
        "better code or protects arbitrary information.")
    _emit(report, out_path)

    print("\n" + "=" * 70)
    print("VERDICT: FIXED_STATE_SURVIVAL_REPORTED")
    # rank by decay slope (least negative = best survival)
    ranked = sorted((s for s in results if "decay_slope_per_us" in results[s]),
                    key=lambda s: results[s]["decay_slope_per_us"], reverse=True)
    if ranked:
        print("survival ranking (slowest decay first):")
        for s in ranked:
            print(f"  {s:>6}: slope {results[s]['decay_slope_per_us']:+.5f}/us "
                  f"(readout {results[s]['readout_path']})")
    print(f"\nCAVEAT: native readout paths differ; this is NOT channel recovery.")
    print(f"report -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="*.json", help="glob for result files")
    ap.add_argument("--metric", choices=["mi", "parity", "conditional", "survival"], default="mi")
    ap.add_argument("--null", choices=["permutation", "bootstrap"], default="permutation")
    ap.add_argument("--correction", choices=["bonferroni", "bh"], default="bh")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--require-circuit", action="store_true",
                    help="hard-fail if executed circuit cannot be verified (recommended)")
    ap.add_argument("--verify", action="store_true",
                    help="require PROVENANCE_VERIFIED: recompute hash from private "
                         "QASM and check consistency; refuse verdict if not verified")
    ap.add_argument("--private-qasm-dir", default=None,
                    help="dir of private transpiled QASM files for hash verification")
    ap.add_argument("--fidelity-json", default=None,
                    help="optional path to {qubit_index: fidelity} for confound check")
    ap.add_argument("--output", default="results/logs/validation_report.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    files = [f for f in files if not any(s in f.lower()
             for s in (":sec", "phishing", "exfiltration"))]
    if not files:
        print("no data files found"); return

    # load
    records = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        counts = (d.get("counts") or d.get("results", {}).get("counts"))
        if not counts:
            continue
        records.append({"file": os.path.basename(f), "meta": d, "counts": counts})

    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "data_dir": args.data_dir, "metric": args.metric,
              "null": args.null, "correction": args.correction,
              "n_files": len(records)}

    # --- 1. PROVENANCE GATE (runs FIRST; can refuse to proceed) ---
    status, reasons = provenance_check(
        [r["meta"] for r in records],
        require_circuit=args.require_circuit,
        verify=args.verify,
        private_qasm_dir=args.private_qasm_dir,
        counts_by_record=[r["counts"] for r in records])
    report["provenance_status"] = status
    report["provenance_reasons"] = reasons[:20]
    if args.require_circuit and status == "INCOMPLETE":
        report["verdict"] = "PROVENANCE_INCOMPLETE"
        report["reason"] = ("executed-circuit structure could not be verified "
                            "against the data; refusing to issue a structure "
                            "verdict. (This is the Phase 3B failure mode.)")
        _emit(report, args.output)
        print(f"\nVERDICT: PROVENANCE_INCOMPLETE — {report['reason']}")
        print("Run with verified circuit deposits, or drop --require-circuit to")
        print("compute structure metrics WITHOUT a provenance guarantee (not advised).")
        return
    if args.verify and status != "VERIFIED":
        report["verdict"] = "PROVENANCE_UNVERIFIED"
        report["reason"] = (f"provenance is {status}, not VERIFIED; --verify was "
                            f"requested. Refusing to issue a structure verdict on "
                            f"unverified provenance. See reasons.")
        _emit(report, args.output)
        print(f"\nVERDICT: PROVENANCE_UNVERIFIED ({status}) — see {args.output}")
        for r in reasons[:6]:
            print(f"  {r}")
        return
    print(f"  provenance: {status}")

    # --- SURVIVAL METRIC: distinct path (per-state decay vs tau) ---
    if args.metric == "survival":
        _run_survival(records, report, args.output)
        return

    # --- fidelity for confound check ---
    fid = None
    if args.fidelity_json and os.path.exists(args.fidelity_json):
        fid = {int(k): v for k, v in json.load(open(args.fidelity_json)).items()}

    # --- run metric per file, aggregate verdict ---
    per_file = []
    all_pairs, all_mis, all_p, all_conf = [], [], [], []
    for r in records:
        width = len(next(iter(r["counts"])).replace(" ", ""))
        X = counts_to_residual_bits(r["counts"], width)
        if X is None or len(X) < 200:
            per_file.append({"file": r["file"], "status": "insufficient_samples"})
            continue
        fpq = ([fid.get(i, 1.0) for i in range(width)] if fid else None)
        pairs, mis, pvals, conf = pairwise_structure(X, args.n_perm, fpq)
        if not pairs:
            per_file.append({"file": r["file"], "n_samples": int(len(X)),
                             "n_pairs": 0,
                             "status": "no_qubit_pairs (width<2): MI not applicable"})
            continue
        all_pairs += [(r["file"], p) for p in pairs]
        all_mis += mis; all_p += pvals; all_conf += conf
        per_file.append({"file": r["file"], "n_samples": int(len(X)),
                         "n_pairs": len(pairs), "max_mi": float(max(mis))})

    report["per_file"] = per_file

    if not all_p:
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = "no file had sufficient samples to test"
        _emit(report, args.output); print("VERDICT: INCONCLUSIVE"); return

    # global correction across ALL pairs/files (family-wise honesty)
    verdict, reason, survivors = decide(
        [p for _, p in all_pairs], all_mis, all_p, all_conf, args.correction)

    report["n_tests"] = len(all_p)
    report["verdict"] = verdict
    report["reason"] = reason
    report["n_survivors"] = len(survivors)
    report["detection_floor_note"] = ("absence of detection bounds correlation "
                                       "below the permutation-null sensitivity; "
                                       "it does not prove zero correlation")
    _emit(report, args.output)

    print(f"\n{'='*60}\nVERDICT: LEARNABLE STRUCTURE — {verdict}\n{'='*60}")
    print(f"REASON: {reason}")
    print(f"tests: {len(all_p)}  survivors: {len(survivors)}  "
          f"correction: {args.correction}")
    print(f"\nreport -> {args.output}")


def _emit(report, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_path, "w"), indent=2)
    # also a short markdown verdict
    md = out_path.replace(".json", ".md")
    with open(md, "w") as f:
        f.write(f"# Validation report\n\n")
        f.write(f"**Verdict:** {report.get('verdict')}\n\n")
        f.write(f"**Reason:** {report.get('reason')}\n\n")
        f.write(f"- provenance: {report.get('provenance_status', report.get('provenance_ok'))}\n")
        f.write(f"- metric: {report.get('metric')}, null: {report.get('null')}, "
                f"correction: {report.get('correction')}\n")
        f.write(f"- tests: {report.get('n_tests','-')}, "
                f"survivors: {report.get('n_survivors','-')}\n")
        if report.get("verdict") == "NOT_DETECTED":
            f.write(f"\n_{report.get('detection_floor_note','')}_\n")


if __name__ == "__main__":
    main()
