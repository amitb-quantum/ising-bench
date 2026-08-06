#!/usr/bin/env python3
"""
Multiple-comparison correction for the Phase 1 six-module parity-consistency
scan. Reads the reproduced per-module sigmas (from yz_syndrome_analysis.json if
present, else uses the values reproduced by analyze_yz_syndrome_sweep.py),
applies Bonferroni and Benjamini-Hochberg FDR, writes a JSON record.

Run:  python src/phase1_correction.py
Out:  ~/ssm/phase1_multiple_comparison_correction.json
"""
from __future__ import annotations
import json
from math import erfc, sqrt
from pathlib import Path

SSM = Path.home() / "ssm"
ANALYSIS_JSON = SSM / "yz_syndrome_analysis.json"
OUT = SSM / "phase1_multiple_comparison_correction.json"

# Reproduced single-test sigmas (from the analyze_yz_syndrome_sweep.py rerun).
# If yz_syndrome_analysis.json contains per-module sigma, we prefer that.
FALLBACK_SIGMAS = {
    "module_0": {"qubits": [0, 1, 2],    "sigma": 4.86},
    "module_2": {"qubits": [8, 9, 10],   "sigma": 3.76},
    "module_1": {"qubits": [4, 5, 6],    "sigma": 2.96},
    "module_5": {"qubits": [17, 27, 26], "sigma": 2.35},
    "module_4": {"qubits": [16, 23, 22], "sigma": 1.89},
    "module_3": {"qubits": [12, 13, 14], "sigma": 1.05},
}


def load_sigmas():
    """Try to read sigmas from the deposited analysis JSON; else fallback."""
    if ANALYSIS_JSON.exists():
        try:
            d = json.load(open(ANALYSIS_JSON))
            # best-effort: scan for per-module sigma fields
            found = {}
            def walk(o):
                if isinstance(o, dict):
                    if "sigma" in o and ("module" in o or "qubits" in o or "center" in o):
                        key = f"module_{o.get('module', o.get('center','?'))}"
                        found[key] = {"qubits": o.get("qubits"), "sigma": float(o["sigma"])}
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(d)
            if len(found) >= 6:
                print(f"  using sigmas from {ANALYSIS_JSON.name}")
                return found
        except Exception as e:
            print(f"  could not parse {ANALYSIS_JSON.name} ({e}); using fallback")
    print("  using reproduced fallback sigmas")
    return FALLBACK_SIGMAS


def main():
    sig = load_sigmas()
    m = len(sig)

    # two-sided p-values
    recs = []
    for name, info in sig.items():
        z = info["sigma"]
        p = erfc(z / sqrt(2))
        recs.append({"module": name, "qubits": info.get("qubits"),
                     "sigma": z, "p_two_sided": p,
                     "p_bonferroni": min(1.0, p * m)})

    # BH-FDR step-up q-values
    recs_sorted = sorted(recs, key=lambda r: r["p_two_sided"])
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = min(prev, recs_sorted[i]["p_two_sided"] * m / rank)
        recs_sorted[i]["q_bh_fdr"] = q
        prev = q

    for r in recs_sorted:
        r["survives_bonferroni_0.05"] = r["p_bonferroni"] < 0.05
        r["survives_fdr_0.05"] = r["q_bh_fdr"] < 0.05

    out = {
        "test": "Phase 1 Y(d0)Z(d1)*Y(d1)Z(d2)=Y(d0)Z(d2) parity consistency",
        "n_modules": m,
        "correction": "Bonferroni and Benjamini-Hochberg FDR, two-sided, alpha=0.05",
        "modules": recs_sorted,
        "summary": {
            "survive_bonferroni": [r["module"] for r in recs_sorted if r["survives_bonferroni_0.05"]],
            "survive_fdr": [r["module"] for r in recs_sorted if r["survives_fdr_0.05"]],
        },
        "note": ("Module 0 has the lowest primary-measurement fidelity (YZ01~0.553) "
                 "of all modules; its parity violation may be partly confounded by "
                 "measurement degradation and warrants follow-up with improved readout."),
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"\n{'module':<12}{'sigma':>7}{'p_bonf':>12}{'q_FDR':>12}  survives")
    print("-" * 50)
    for r in recs_sorted:
        s = "+".join([x for x, c in [("Bonf", r["survives_bonferroni_0.05"]),
                                     ("FDR", r["survives_fdr_0.05"])] if c]) or "ns"
        print(f"{r['module']:<12}{r['sigma']:>7.2f}{r['p_bonferroni']:>12.2e}"
              f"{r['q_bh_fdr']:>12.2e}  {s}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
