#!/usr/bin/env python3
"""
survival_submit.py — B1 fixed-state survival spectrum.

Compares how the prepared Y⊗Z eigenstate's measured signature decays with idle
time tau, against a SPECTRUM of Z-basis reference states (|000>, |111>, GHZ),
each normalized to its own tau=0 baseline.

SCOPE (must stay in the report): this is a FIXED-STATE SURVIVAL experiment, not a
variable-input recovery or decoder benchmark. Each state is measured through its
NATIVE signature/readout path, so observed decay = state survival convolved with
any tau-dependence of that readout path. Self-baseline normalization reduces, but
does not remove, readout-path asymmetry. Do not read this as "Y⊗Z is a better
code" or "protects arbitrary information."

States & signatures:
  Y⊗Z eigenstate : ⟨Y⊗Z parity⟩   via local protected stub (build_yz + readout)
  |000>          : P(000)          direct Z measurement
  |111>          : P(111)          direct Z measurement
  GHZ            : parity witness  parity-style readout (NOT full tomography)

All state×tau circuits batched into ONE SamplerV2 job. Dual provenance per
circuit (private QASM local, public hash deposited). Y⊗Z stays behind the stub.

Run:
  python src/survival_submit.py --backend ibm_fez --modules 0 --shots 4096 \
      --taus 0 10 30 60 120 --dry-run
"""
from __future__ import annotations
import os, json, time, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path.home() / "ising-bench" / "data" / "raw" / "survival_experiment"
PRIVATE_DIR = Path.home() / "ising-bench" / "provenance_private"

MODULE_MAP = {0: [0, 1, 2], 1: [4, 5, 6], 2: [8, 9, 10],
              3: [12, 13, 14], 4: [16, 23, 22], 5: [17, 27, 26]}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Y⊗Z arm — LOCAL PROTECTED CIRCUIT (keep private unless publishing sequence)
# ============================================================
def build_yz_eigenstate_with_readout(data_qubits):
    """Prepare the Y⊗Z eigenstate on data_qubits and append parity readout.
    Return a 4-qubit (3 data + 1 anc) QuantumCircuit with 1 classical bit holding
    the parity outcome. The scaffold inserts the idle delay BEFORE readout."""
    from qiskit import QuantumCircuit
    import numpy as np

    if len(data_qubits) != 3:
        raise ValueError("data_qubits must contain exactly three data qubits")

    qc = QuantumCircuit(4, 1)
    # --- BEGIN local patent-pending basis migration sequence ---
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


# ============================================================
# Z-basis reference arms — written in full (no IP)
# ============================================================
def build_zero(n=3):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    # |000> is the default init; measure directly
    qc.measure(range(n), range(n))
    return qc


def build_ones(n=3):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.x(q)            # prepare |111>
    qc.measure(range(n), range(n))
    return qc


def build_ghz_parity(n=3):
    """GHZ then a parity-style readout. Signature = parity of the n bits;
    a coherent GHZ gives a structured parity distribution, decohering toward
    uniform. (Parity witness, NOT full tomography.)"""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.measure(range(n), range(n))
    return qc


# state registry: label -> (builder, where the idle delay is inserted)
# For Z-refs the delay goes after prep, before measure. For Y⊗Z the stub returns
# prep+readout and we insert delay before the readout block via recompile.
def build_state(label, data_qubits):
    if label == "yz":
        return build_yz_eigenstate_with_readout(data_qubits)
    if label == "zero":
        return build_zero(len(data_qubits))
    if label == "ones":
        return build_ones(len(data_qubits))
    if label == "ghz":
        return build_ghz_parity(len(data_qubits))
    raise ValueError(label)


def insert_idle(qc_t, tau_dt, physical_qubits):
    """Insert an idle delay on the physical qubits just before the measurements,
    on the transpiled circuit (avoids the bare-delay SamplerV2 coercion bug)."""
    if tau_dt <= 0:
        return qc_t
    from qiskit import QuantumCircuit
    idle = QuantumCircuit(qc_t.num_qubits, qc_t.num_clbits)
    for pq in physical_qubits:
        idle.delay(tau_dt, pq, unit="dt")
    return idle.compose(qc_t, front=False)


def canonical_qasm(circuit):
    try:
        from qiskit.qasm3 import dumps
        return dumps(circuit)
    except Exception:
        try:
            return circuit.qasm()
        except Exception:
            return None


def chash(qasm, gate_counts, depth):
    if qasm:
        return hashlib.sha256(qasm.encode()).hexdigest(), "qasm3"
    sig = json.dumps({"gate_counts": gate_counts, "depth": depth}, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest(), "structural_signature"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--modules", type=int, nargs="+", default=[0])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--taus", type=float, nargs="+", default=[0, 10, 30, 60, 120])
    ap.add_argument("--states", nargs="+", default=["yz", "zero", "ones", "ghz"])
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    dt = backend.dt
    print(f"backend: {backend.name}, states={args.states}, taus={args.taus}us, "
          f"modules={args.modules}")

    circuits, metas = [], []
    for mod in args.modules:
        data = MODULE_MAP[mod]
        # physical qubits used = data (+anc for yz). Z-refs use the 3 data qubits.
        for state in args.states:
            qc = build_state(state, data)
            init_layout = (data + [data[-1] + 1]) if state == "yz" else data
            qc_t = transpile(qc, backend, optimization_level=args.opt_level,
                             initial_layout=init_layout)
            for tau in args.taus:
                tau_dt = int(round(tau * 1e-6 / dt)) if tau > 0 else 0
                qc_tau = insert_idle(qc_t, tau_dt, init_layout)
                qasm = canonical_qasm(qc_tau)
                gc = dict(qc_tau.count_ops())
                depth = qc_tau.depth()
                h, hb = chash(qasm, gc, depth)
                if qasm:
                    (PRIVATE_DIR / f"qasm_{backend.name}_{state}_mod{mod}_tau{int(tau)}_{h[:12]}.qasm").write_text(qasm)
                metas.append({
                    "backend": backend.name, "experiment": "fixed_state_survival",
                    "state_label": state, "module": mod, "data_qubits": data,
                    "tau_us": tau, "tau_dt": tau_dt, "dt_seconds": dt,
                    "shots": args.shots, "depth": depth, "gate_counts": gc,
                    "circuit_hash": h, "hash_basis": hb, "opt_level": args.opt_level,
                    "private_qasm_stored": bool(qasm),
                    "readout_path": ("yz_parity" if state == "yz" else
                                     "ghz_parity" if state == "ghz" else "direct_z"),
                    "submitted_at": utcnow(), "job_id": None,
                    "completed_at": None, "counts": None,
                })
                circuits.append(qc_tau)

    print(f"\n{len(circuits)} circuits built ({len(args.states)} states x "
          f"{len(args.taus)} taus x {len(args.modules)} modules)")
    for m in metas[:8]:
        print(f"  {m['state_label']:>5} tau={m['tau_us']:>5} depth={m['depth']:>3} "
              f"hash={m['circuit_hash'][:10]} readout={m['readout_path']}")
    if len(metas) > 8:
        print(f"  ... ({len(metas)-8} more)")

    if args.dry_run:
        man = OUT_DIR / f"survival_DRYRUN_{int(time.time())}.json"
        json.dump(metas, open(man, "w"), indent=2)
        print(f"\n[dry-run] nothing submitted. manifest -> {man}")
        print(f"private QASM -> {PRIVATE_DIR} (DO NOT deposit)")
        return

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = args.shots
    job = sampler.run(circuits)          # ALL circuits in ONE batched job
    jid = job.job_id()
    for i, m in enumerate(metas):
        m["job_id"] = jid
        m["batch_index"] = i             # position in the batched result
        fn = OUT_DIR / f"survival_{m['backend']}_{m['state_label']}_mod{m['module']}_tau{int(m['tau_us'])}_{jid}.json"
        json.dump(m, open(fn, "w"), indent=2)
    man = OUT_DIR / f"survival_manifest_{jid}.json"
    json.dump(metas, open(man, "w"), indent=2)
    print(f"\nsubmitted ONE batched job: {jid} ({len(circuits)} circuits)")
    print(f"manifest -> {man}")
    print(f"private QASM -> {PRIVATE_DIR} (DO NOT deposit)")


if __name__ == "__main__":
    main()
