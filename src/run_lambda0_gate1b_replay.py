#!/usr/bin/env python3
"""
Gate-1B Track A: deterministic Candidate-C replay.

Validity prerequisites are checked before training.
Only an anchor-conforming replay may persist the reconstructed model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
import torch

from exact_lambda0_oracle import (
    Q_X,
    enumerate_states,
    exact_population_risk,
    table_sha256,
)
from ising_benchmark import generate_surface_code_data
from run_lambda0_parity_gate import (
    CANDIDATES,
    DISTANCE,
    N,
    N_TEST,
    N_TRAIN,
    N_VAL,
    P_ERR,
    SEED,
    evaluate,
    train_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lambda0_gate1b_replay"

RUNNER = ROOT / "src" / "run_lambda0_parity_gate.py"
GENERATOR = ROOT / "src" / "ising_benchmark.py"
ORACLE = ROOT / "src" / "exact_lambda0_oracle.py"
RAW = ROOT / "adjudications" / "LAMBDA0_PARITY_GATE_V1_RAW_RESULT.json"
REFERENCE = (
    ROOT
    / "adjudications"
    / "LAMBDA0_GATE1B_EXACT_ORACLE_REFERENCE.json"
)

EXPECTED_HASHES = {
    RUNNER: "231c51d3aa9e81dba927cb7f419c6718186e9822c20ddcf3b8133d870a2836b9",
    GENERATOR: "e61f028b7f3604f3f5025e65fd06f33debea76e6e5ff81117a12776d58b5835b",
    ORACLE: "c5506eb138c93ae776254d927cb62f48f0fec6fbf7769f3666148a683e8a03d5",
    RAW: "76c752f85ec8897a9ce152ab184bc4ed0d3ce171d2ce0782d3b51a3969d68d50",
    REFERENCE: "4a3ec40b2ceef00c3b68eab9e7e93c3650914942a28b7257b66af0f795909154",
}

EXPECTED_TABLE_SHA = (
    "e27c84d604596b320125adeef0ed2effe25cfb4ffc265b8490cea36af81cfcb3"
)

EXPECTED_ANCHORS = {
    "name": "C",
    "best_epoch": 97,
    "validation_ler": "0.010600",
    "validation_bce": "0.044843",
    "final_ler": "0.009550",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fraction_text(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def check(name, actual, expected):
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "pass": actual == expected,
    }


def validity_checks():
    checks = []

    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path) if path.exists() else "MISSING"
        checks.append(
            check(
                f"sha256:{path.relative_to(ROOT)}",
                actual,
                expected,
            )
        )

    checks.extend(
        [
            check(
                "python",
                ".".join(map(str, sys.version_info[:3])),
                "3.11.10",
            ),
            check("torch", torch.__version__, "2.5.1+cu121"),
            check("numpy", np.__version__, "2.4.6"),
            check("sklearn", sklearn.__version__, "1.8.0"),
            check("torch_threads", torch.get_num_threads(), 24),
            check(
                "torch_interop_threads",
                torch.get_num_interop_threads(),
                24,
            ),
            check(
                "CUDA_VISIBLE_DEVICES",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                "",
            ),
            check("torch_cuda_available", torch.cuda.is_available(), False),
        ]
    )

    reference = json.loads(REFERENCE.read_text())

    checks.extend(
        [
            check(
                "oracle_table_sha256",
                reference["decision_table_sha256"],
                EXPECTED_TABLE_SHA,
            ),
            check(
                "oracle_bayes_ler",
                reference["bayes_ler_fraction"],
                "373445699/38443359375",
            ),
            check(
                "oracle_rows",
                len(reference["decision_table"]),
                64,
            ),
        ]
    )

    return checks


def extract_model_table(model, reference):
    syndrome_strings = [
        row["syndrome"]
        for row in reference["decision_table"]
    ]

    X = np.asarray(
        [[int(c) for c in s] for s in syndrome_strings],
        dtype=np.float32,
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        pred = (
            torch.sigmoid(logits) > 0.5
        ).cpu().numpy().astype(int)

    rows = [
        {
            "syndrome": syndrome,
            "decision": int(decision),
        }
        for syndrome, decision in zip(syndrome_strings, pred)
    ]

    return rows


def exact_model_risk(rows):
    decision_table = {
        tuple(int(c) for c in row["syndrome"]): row["decision"]
        for row in rows
    }

    states = enumerate_states(Q_X)
    return exact_population_risk(decision_table, states)


def main():
    print("=" * 72)
    print("LAMBDA=0 GATE-1B — CANDIDATE-C REPLAY")
    print("=" * 72)

    checks = validity_checks()

    for item in checks:
        mark = "PASS" if item["pass"] else "FAIL"
        print(
            f"{mark:4s} {item['name']}: "
            f"{item['actual']}"
        )

    if not all(x["pass"] for x in checks):
        print()
        print("STRUCTURAL/PROVENANCE VALIDITY: FLAG")
        print("REPLAY IDENTITY: NOT EVALUATED")
        raise SystemExit(2)

    print()
    print("STRUCTURAL/PROVENANCE VALIDITY: ACCEPT")
    print()

    # Reconstruct the original Gate-1 dataset exactly.
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y, n_stab = generate_surface_code_data(
        DISTANCE,
        N,
        P_ERR,
    )

    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xval, yval = (
        X[N_TRAIN:N_TRAIN + N_VAL],
        y[N_TRAIN:N_TRAIN + N_VAL],
    )
    Xte, yte = (
        X[N_TRAIN + N_VAL:],
        y[N_TRAIN + N_VAL:],
    )

    assert len(Xte) == N_TEST
    assert n_stab == 8

    print("Training frozen Candidate C only...")
    replay = train_candidate(
        "C",
        CANDIDATES["C"],
        Xtr,
        ytr,
        Xval,
        yval,
    )

    final_ler, final_bce, _ = evaluate(
        replay["model"],
        Xte,
        yte,
    )

    actual = {
        "name": replay["name"],
        "best_epoch": replay["best_epoch"],
        "validation_ler": format(replay["val_ler"], ".6f"),
        "validation_bce": format(replay["val_bce"], ".6f"),
        "final_ler": format(final_ler, ".6f"),
    }

    anchor_checks = {
        key: {
            "actual": actual[key],
            "expected": expected,
            "match": actual[key] == expected,
        }
        for key, expected in EXPECTED_ANCHORS.items()
    }

    print()
    print("=" * 72)
    print("REPLAY ANCHORS")
    print("=" * 72)

    for key, item in anchor_checks.items():
        mark = "MATCH" if item["match"] else "MISS"
        print(
            f"{mark:5s} {key:16s} "
            f"actual={item['actual']} "
            f"expected={item['expected']}"
        )

    identity_confirmed = all(
        x["match"] for x in anchor_checks.values()
    )

    verdict = "CONFIRM" if identity_confirmed else "ABSTAIN"

    OUT.mkdir(parents=True, exist_ok=True)

    base_payload = {
        "schema": "ising-bench.lambda0-gate1b-replay.v1",
        "validity": "ACCEPT",
        "replay_identity": verdict,
        "validity_checks": checks,
        "anchor_checks": anchor_checks,
        "diagnostics": {
            "epochs_executed": replay["epochs_executed"],
            "validation_ler_raw": replay["val_ler"],
            "validation_bce_raw": replay["val_bce"],
            "final_ler_raw": final_ler,
            "final_bce_raw": final_bce,
        },
    }

    if not identity_confirmed:
        result_path = OUT / "replay_result.json"
        result_path.write_text(
            json.dumps(base_payload, indent=2) + "\n"
        )

        print()
        print("REPLAY IDENTITY: ABSTAIN")
        print("No reconstructed model persisted.")
        print("result:", result_path)
        print("result SHA-256:", sha256_file(result_path))
        return

    reference = json.loads(REFERENCE.read_text())

    learned_rows = extract_model_table(
        replay["model"],
        reference,
    )

    learned_table_sha = table_sha256(learned_rows)

    oracle_map = {
        row["syndrome"]: row["decision"]
        for row in reference["decision_table"]
    }

    differing = [
        {
            "syndrome": row["syndrome"],
            "learned": row["decision"],
            "oracle": oracle_map[row["syndrome"]],
        }
        for row in learned_rows
        if row["decision"] != oracle_map[row["syndrome"]]
    ]

    risk = exact_model_risk(learned_rows)

    bayes = Fraction(
        reference["bayes_ler_fraction"]
    )

    excess = risk - bayes

    oracle_verdict = (
        "CONFIRM"
        if len(differing) == 0
        else "REJECT"
    )

    state_path = OUT / "candidate_c_replay_state_dict.pt"
    table_path = OUT / "candidate_c_replay_table.json"

    torch.save(
        replay["model"].state_dict(),
        state_path,
    )

    table_payload = {
        "schema": "ising-bench.lambda0-gate1b-model-table.v1",
        "rows": learned_rows,
        "decision_table_sha256": learned_table_sha,
    }

    table_path.write_text(
        json.dumps(table_payload, indent=2) + "\n"
    )

    base_payload["oracle_performance"] = {
        "verdict": oracle_verdict,
        "bayes_table_sha256": EXPECTED_TABLE_SHA,
        "learned_table_sha256": learned_table_sha,
        "table_match": len(differing) == 0,
        "differing_entries": differing,
        "differing_entry_count": len(differing),
        "exact_population_ler_fraction": fraction_text(risk),
        "exact_population_ler_float": float(risk),
        "bayes_ler_fraction": fraction_text(bayes),
        "exact_excess_ler_fraction": fraction_text(excess),
        "exact_excess_ler_float": float(excess),
    }

    base_payload["artifacts"] = {
        "state_dict": {
            "path": str(state_path.relative_to(ROOT)),
            "sha256": sha256_file(state_path),
        },
        "decision_table": {
            "path": str(table_path.relative_to(ROOT)),
            "sha256": sha256_file(table_path),
        },
    }

    result_path = OUT / "replay_result.json"
    result_path.write_text(
        json.dumps(base_payload, indent=2) + "\n"
    )

    print()
    print("=" * 72)
    print("GATE-1B TRACK A RESULT")
    print("=" * 72)
    print("REPLAY IDENTITY:       CONFIRM")
    print("ORACLE PERFORMANCE:   ", oracle_verdict)
    print("differing entries:    ", len(differing))
    print(
        "exact population LER: ",
        fraction_text(risk),
        f"({float(risk):.18f})",
    )
    print(
        "exact Bayes LER:      ",
        fraction_text(bayes),
        f"({float(bayes):.18f})",
    )
    print(
        "exact excess LER:     ",
        fraction_text(excess),
        f"({float(excess):.18f})",
    )
    print("learned table SHA-256:", learned_table_sha)
    print("state_dict SHA-256:   ", sha256_file(state_path))
    print("result:", result_path)
    print("result SHA-256:", sha256_file(result_path))


if __name__ == "__main__":
    main()
