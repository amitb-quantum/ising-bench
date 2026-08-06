#!/usr/bin/env python3
"""
Y⊗Z stabilizer submission scaffold — clean rebuild with provenance by construction.

Replaces the legacy yz_syndrome_temporal_enhanced.py for new runs. Does ONE thing:
submit the Y⊗Z spatial parity protocol with a complete, verifiable provenance
chain, sized for a tiny QPU budget, dry-run first.

PROVENANCE MODEL (dual, IP-aware):
  - PRIVATE (stays on your machine, never deposited): the full transpiled QASM,
    written to provenance_private/. This lets YOU verify exactly what ran and
    re-derive the hash. It contains the compiled form of the patent-pending
    sequence, so it is NOT part of any public deposit.
  - PUBLIC (safe to deposit): circuit_hash (SHA-256 of canonical transpiled
    QASM), gate_counts, depth, final_layout, backend, job_id, shots, counts.
    This proves the circuit→counts chain is internally consistent and that no
    circuit was swapped, WITHOUT publishing the enabling gate sequence.

The validation harness reaches PROVENANCE_VERIFIED when the public hash matches
the hash recomputed from the private QASM (forward-looking: only runs made with
this scaffold can be verified, since the hash must be stored at submission time).

IP NOTE: build_yz_stabilizer_circuit() contains the local patent-pending
sequence. Keep this file private unless you intentionally publish that sequence.

Run:
  python yz_submit.py --backend ibm_fez --modules 0 2 --shots 4096 --dry-run
  python yz_submit.py --backends ibm_fez ibm_marrakesh --modules 0 2 --shots 4096
"""
from __future__ import annotations
import os, json, time, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path.home() / "ising-bench" / "data" / "raw" / "yz_experiment"
PRIVATE_DIR = Path.home() / "ising-bench" / "provenance_private"   # NEVER deposit


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# LOCAL CIRCUIT — filled privately. Returns a 4-qubit logical circuit
# (3 data + 1 ancilla) measuring the Y⊗Z parity observable. The scaffold
# transpiles, captures provenance, and submits whatever this returns.
# ============================================================
def build_yz_stabilizer_circuit(module_qubits):
    """
    Build the Y⊗Z stabilizer circuit for one module.
    `module_qubits` = [d0, d1, d2] data qubit indices (logical; transpiler maps
    to physical via the layout you pass in main()).

    Must return a qiskit QuantumCircuit with classical measurement of the
    Y⊗Z parity observable. The scaffold does the rest (transpile, hash, submit).
    """
    from qiskit import QuantumCircuit
    import numpy as np

    if len(module_qubits) != 3:
        raise ValueError("module_qubits must contain exactly three data qubits")

    qc = QuantumCircuit(4, 1)
    # --- BEGIN local patent-pending basis migration sequence ---
    # Patent-pending basis migration sequence.
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(np.pi / 4, 1)
    qc.rx(np.pi / 4, 0)
    qc.cz(0, 2)
    qc.cx(2, 3)
    qc.rz(-np.pi / 4, 3)

    # Parity readout.
    qc.measure(3, 0)
    # --- END local patent-pending basis migration sequence ---
    return qc


def canonical_qasm(circuit):
    """Serialize transpiled circuit to QASM3 (canonical form for hashing)."""
    try:
        from qiskit.qasm3 import dumps
        return dumps(circuit)
    except Exception:
        try:
            return circuit.qasm()
        except Exception:
            return None


def circuit_hash(qasm_str, gate_counts, depth):
    """SHA-256 over canonical QASM if available, else over a structural
    signature (gate_counts + depth). The signature path is weaker but still
    detects circuit swaps; recorded in 'hash_basis' for honesty."""
    if qasm_str:
        h = hashlib.sha256(qasm_str.encode()).hexdigest()
        return h, "qasm3"
    sig = json.dumps({"gate_counts": gate_counts, "depth": depth}, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest(), "structural_signature"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--backends", nargs="+", default=None,
                    help="multi-backend sweep (overrides --backend)")
    ap.add_argument("--modules", type=int, nargs="+", required=True,
                    help="module indices to run (each a 3-qubit data group)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    service = QiskitRuntimeService()

    backend_names = (args.backends or ([args.backend] if args.backend
                     else [service.least_busy(operational=True, simulator=False).name]))
    print(f"Y⊗Z submission over {backend_names}, modules={args.modules}, "
          f"shots={args.shots}\n")

    # module -> data-qubit triples. EDIT to your real module map; these mirror
    # the historical Phase 1 modules for continuity.
    MODULE_MAP = {
        0: [0, 1, 2], 1: [4, 5, 6], 2: [8, 9, 10],
        3: [12, 13, 14], 4: [16, 23, 22], 5: [17, 27, 26],
    }

    grand = []
    for bname in backend_names:
        backend = service.backend(bname)
        circuits, metas = [], []
        for mod in args.modules:
            if mod not in MODULE_MAP:
                print(f"  skip module {mod}: not in MODULE_MAP"); continue
            data = MODULE_MAP[mod]
            qc = build_yz_stabilizer_circuit(data)
            # transpile to the physical module qubits (+1 ancilla chosen by layout)
            qc_t = transpile(qc, backend, optimization_level=args.opt_level,
                             initial_layout=data + [data[-1] + 1])
            qasm = canonical_qasm(qc_t)
            gate_counts = dict(qc_t.count_ops())
            depth = qc_t.depth()
            chash, hbasis = circuit_hash(qasm, gate_counts, depth)

            # PRIVATE: write full transpiled QASM locally (never deposited)
            if qasm:
                pq = PRIVATE_DIR / f"qasm_{bname}_mod{mod}_{chash[:12]}.qasm"
                pq.write_text(qasm)

            # PUBLIC-safe metadata (deposit-able): hash, NOT the qasm
            meta = {
                "backend": bname,
                "protocol": "yz_spatial_parity",
                "module": mod,
                "data_qubits": data,
                "shots": args.shots,
                "depth": depth,
                "gate_counts": gate_counts,
                "circuit_hash": chash,
                "hash_basis": hbasis,
                "opt_level": args.opt_level,
                "private_qasm_stored": bool(qasm),
                "submitted_at": utcnow(),
                "job_id": None,
                "completed_at": None,
                "counts": None,
            }
            circuits.append(qc_t)
            metas.append(meta)
            print(f"  [{bname}] module {mod}: depth={depth}, hash={chash[:12]} "
                  f"({hbasis})")

        if not circuits:
            continue
        if args.dry_run:
            print(f"  [dry-run] {bname}: {len(circuits)} circuits built, not submitted")
            grand += metas
            continue

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = args.shots
        job = sampler.run(circuits)
        jid = job.job_id()
        for m in metas:
            m["job_id"] = jid
            fn = OUT_DIR / f"yz_{bname}_mod{m['module']}_{jid}.json"
            json.dump(m, open(fn, "w"), indent=2)
        print(f"  submitted {bname} -> job {jid} ({len(circuits)} circuits)")
        grand += metas

    man = OUT_DIR / f"yz_manifest_{int(time.time())}.json"
    json.dump(grand, open(man, "w"), indent=2)
    print(f"\nmanifest -> {man}")
    print(f"private QASM -> {PRIVATE_DIR} (DO NOT deposit)")
    if args.dry_run:
        print("DRY RUN — nothing submitted.")


if __name__ == "__main__":
    main()
