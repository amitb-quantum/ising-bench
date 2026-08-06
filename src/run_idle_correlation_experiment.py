#!/usr/bin/env python3
"""
Path B — controlled idle/identity correlation experiment (IBM Quantum).

Submits circuits whose IDEAL output is all-zeros: prepare |0...0>, idle for a
swept duration tau, measure. Because the ideal output is trivially known, ANY
measured '1' is an error event, and mutual information between qubits' error
events is unambiguously ERROR correlation — not designed entanglement. This is
what the historical archive could not give us.

Key design choices (see README "Decision"):
  * Real idle TIME via delay instructions, swept around the ~30 us timescale,
    so shared-bath / environmental effects can imprint. A zero-duration idle
    circuit would show nothing.
  * Contiguous heavy-hex patch placement via initial_layout, so physical
    distances are well-defined and span near + far pairs.
  * final_layout, physical qubits, and scheduled_duration captured at submit
    time and saved per job — the metadata the archive lacked.
  * Delay applied on the ISA (post-transpile) circuit to avoid the known
    SamplerV2 bare-delay coercion bug (qiskit-ibm-runtime issue #1613).

This script SUBMITS and SAVES metadata + a results-fetch hook. It is the data
collector; anal_idle_correlation.py does the science on the saved files.

Run:  python src/run_idle_correlation_experiment.py --backend ibm_torino \
          --qubits 12 --shots 8192 --tau-us 0 10 30 60 120
"""

from __future__ import annotations
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path.home() / "ising-bench" / "data" / "raw" / "idle_experiment"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def pick_contiguous_patch(backend, n):
    """Pick n physically-connected qubits from the backend coupling map, so the
    chosen region is a contiguous heavy-hex patch (defined distances)."""
    cmap = backend.coupling_map
    # BFS from a seed to collect a connected set of size n
    from collections import deque
    adj = {}
    for a, b in cmap.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = None
    for seed in sorted(adj):
        seen, q = {seed}, deque([seed])
        while q and len(seen) < n:
            x = q.popleft()
            for y in sorted(adj.get(x, ())):
                if y not in seen:
                    seen.add(y); q.append(y)
                    if len(seen) >= n:
                        break
        if len(seen) >= n:
            best = sorted(list(seen))[:n]
            break
    if best is None:
        raise RuntimeError(f"could not find {n} connected qubits on {backend.name}")
    return best


def build_idle_circuit(n_logical, tau_dt, backend, initial_layout):
    """Prepare |0..0>, idle tau, measure. Delay added post-transpile (ISA)."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    qc = QuantumCircuit(n_logical)
    qc.measure_all()  # ideal = all zeros; we insert the delay onto ISA below

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, initial_layout=initial_layout
    )
    isa = pm.run(qc)

    # Insert idle delay on the physical qubits AFTER transpile, before measure.
    # (Avoids issue #1613; delay carried on ISA circuit at hardware dt units.)
    if tau_dt > 0:
        from qiskit import QuantumCircuit as QC
        idle = QC(isa.num_qubits, isa.num_clbits)
        for pq in initial_layout:
            idle.delay(tau_dt, pq, unit="dt")
        # compose: state-prep(none) -> idle -> measures already in isa
        # rebuild: idle first on physical qubits, then the isa measures
        isa = idle.compose(isa, front=False)
    return isa


def extract_layout(isa_circuit, initial_layout):
    """Pull a measurement-index -> physical-qubit mapping from the ISA circuit."""
    layout = {}
    try:
        tl = isa_circuit.layout
        if tl is not None:
            final = tl.final_index_layout()  # list: clbit/qubit order -> physical
            layout = {i: int(p) for i, p in enumerate(final)}
    except Exception:
        layout = {i: int(p) for i, p in enumerate(initial_layout)}
    return layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="single backend; default = least busy")
    ap.add_argument("--backends", nargs="+", default=None,
                    help="multiple backends for a cross-hardware sweep "
                         "(overrides --backend), e.g. ibm_marrakesh ibm_kingston ibm_fez")
    ap.add_argument("--qubits", type=int, default=12)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--tau-us", type=float, nargs="+", default=[0, 10, 30, 60, 120],
                    help="idle durations in microseconds to sweep")
    ap.add_argument("--basis", default="Z_idle",
                    help="basis tag stored in manifest (Z_idle now; X reserved for later)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + save circuits/metadata but do NOT submit")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    service = QiskitRuntimeService()

    # resolve backend list: --backends > --backend > least busy
    if args.backends:
        backend_names = args.backends
    elif args.backend:
        backend_names = [args.backend]
    else:
        backend_names = [service.least_busy(operational=True, simulator=False).name]

    print(f"cross-hardware sweep over {len(backend_names)} backend(s): {backend_names}")
    print(f"basis={args.basis}  qubits={args.qubits}  shots={args.shots}  "
          f"tau(us)={args.tau_us}\n")

    grand_manifest = []
    for bname in backend_names:
        backend = service.backend(bname)
        dt = backend.dt
        # IMPORTANT: patch is selected PER BACKEND from its own coupling map,
        # NOT reused across devices. (Known limitation: picks first contiguous
        # patch, not the best-calibrated one — see select-by-calibration hook.)
        patch = pick_contiguous_patch(backend, args.qubits)
        print(f"[{bname}] {backend.num_qubits} qubits; patch={patch}")

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = args.shots
        sampler.options.experimental = {"execution": {"scheduler_timing": True}}

        for tau_us in args.tau_us:
            tau_dt = int(round(tau_us * 1e-6 / dt)) if tau_us > 0 else 0
            isa = build_idle_circuit(args.qubits, tau_dt, backend, patch)
            layout = extract_layout(isa, patch)

            # circuit-reconstructable provenance (clears the validation harness
            # strict provenance gate; shape-only depth/layout is NOT sufficient).
            from collections import Counter as _Counter
            gate_counts = dict(_Counter(i.operation.name for i in isa.data))
            try:
                from qiskit.qasm3 import dumps as _qasm3_dumps
                circuit_qasm = _qasm3_dumps(isa)
            except Exception:
                try:
                    circuit_qasm = isa.qasm()  # older qiskit fallback
                except Exception:
                    circuit_qasm = None

            meta = {
                "backend": bname,
                "circuit_family": "idle_identity",
                "basis": args.basis,
                "num_qubits": args.qubits,
                "shots": args.shots,
                "idle_us": tau_us,
                "idle_dt": tau_dt,
                "dt_seconds": dt,
                "logical_qubits": list(range(args.qubits)),
                "physical_qubits": patch,
                "final_layout": layout,
                "depth": isa.depth(),
                "gate_counts": gate_counts,        # reconstructable provenance
                "circuit_qasm": circuit_qasm,      # reconstructable provenance
                "submitted_at": utcnow(),
                "job_id": None,
                "completed_at": None,
                "scheduled_duration": None,
                "counts": None,
            }

            if args.dry_run:
                print(f"  [dry-run] {bname} tau={tau_us}us dt={tau_dt} depth={isa.depth()}")
                grand_manifest.append(meta)
                continue

            job = sampler.run([isa])
            meta["job_id"] = job.job_id()
            print(f"  submitted {bname} tau={tau_us}us -> job {meta['job_id']}")
            fname = OUT_DIR / (f"idle_{bname}_{args.basis}_q{args.qubits}"
                               f"_tau{int(tau_us)}us_{meta['job_id']}.json")
            json.dump(meta, open(fname, "w"), indent=2)
            grand_manifest.append(meta)

    man_path = OUT_DIR / f"manifest_multibackend_{int(time.time())}.json"
    json.dump(grand_manifest, open(man_path, "w"), indent=2)
    print(f"\nmanifest -> {man_path}")
    if args.dry_run:
        print("DRY RUN — nothing submitted. Re-run without --dry-run to submit.")
    else:
        n = len([m for m in grand_manifest if m['job_id']])
        print(f"\n{n} jobs submitted across {len(backend_names)} backends.")
        print("Fetch with fetch_idle_results.py once jobs complete.")


if __name__ == "__main__":
    main()
