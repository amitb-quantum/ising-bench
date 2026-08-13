#!/usr/bin/env python3
"""
Gate-1B Track B: prospectively frozen training-procedure characterization.

4 data seeds x 5 training seeds = 20 runs.

No sampled test set is used for adjudication.
Each trained model is evaluated exactly over all 64 attainable syndromes
and all 512 latent error configurations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from exact_lambda0_oracle import (
    Q_X,
    enumerate_states,
    exact_population_risk,
    table_sha256,
)
from ising_benchmark import (
    CNNDecoder,
    generate_surface_code_data,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lambda0_gate1b_panel"

PREREG = ROOT / "prereg" / "LAMBDA0_GATE1B_EXACT_ORACLE_V1.md"
GENERATOR = ROOT / "src" / "ising_benchmark.py"
ORACLE = ROOT / "src" / "exact_lambda0_oracle.py"
REFERENCE = (
    ROOT
    / "adjudications"
    / "LAMBDA0_GATE1B_EXACT_ORACLE_REFERENCE.json"
)
ENVIRONMENT = ROOT / "provenance" / "LAMBDA0_GATE1_ENVIRONMENT.txt"

EXPECTED_HASHES = {
    PREREG: "c5c2800d37f99b477aae6b6a59c0c223c54ad8a95e6547535267cb56aabb1feb",
    GENERATOR: "e61f028b7f3604f3f5025e65fd06f33debea76e6e5ff81117a12776d58b5835b",
    ORACLE: "c5506eb138c93ae776254d927cb62f48f0fec6fbf7769f3666148a683e8a03d5",
    REFERENCE: "4a3ec40b2ceef00c3b68eab9e7e93c3650914942a28b7257b66af0f795909154",
    ENVIRONMENT: "ab85529133892d14b7b1aad2b584a7cfb458c4a36acf242dc9dfa05b39cb201a",
}

EXPECTED_TABLE_SHA = (
    "e27c84d604596b320125adeef0ed2effe25cfb4ffc265b8490cea36af81cfcb3"
)

DATA_SEEDS = (31001, 31002, 31003, 31004)
TRAINING_SEEDS = (41001, 41002, 41003, 41004, 41005)

N = 100_000
N_TRAIN = 60_000
N_VAL = 20_000
DISTANCE = 3
P_ERR = 0.05

BATCH_SIZE = 256
LR = 3e-4
MAX_EPOCHS = 400
PATIENCE = 30

DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()

    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes(order="C"))

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


def validity_checks(reference):
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
            check(
                "torch_cuda_available",
                torch.cuda.is_available(),
                False,
            ),
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


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.from_numpy(X).float().to(DEVICE)
        )

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            torch.from_numpy(y).float().to(DEVICE),
        ).item()

        pred = (
            torch.sigmoid(logits) > 0.5
        ).cpu().numpy().astype(np.float32)

    ler = float(np.mean(pred != y))

    return ler, float(loss)


def train_seeded(
    Xtr,
    ytr,
    Xval,
    yval,
    training_seed,
):
    torch.manual_seed(training_seed)

    model = CNNDecoder(Xtr.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(ytr).float(),
    )

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(training_seed)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
    )

    best_state = None
    best_val_ler = float("inf")
    best_val_bce = float("inf")
    best_epoch = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            loss = loss_fn(
                model(xb),
                yb,
            )

            loss.backward()
            optimizer.step()

        val_ler, val_bce = evaluate(
            model,
            Xval,
            yval,
        )

        improved = (
            val_ler < best_val_ler
            or (
                val_ler == best_val_ler
                and val_bce < best_val_bce
            )
        )

        if improved:
            best_val_ler = val_ler
            best_val_bce = val_bce
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            "no best checkpoint recorded"
        )

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "epochs_executed": epoch,
        "validation_ler": best_val_ler,
        "validation_bce": best_val_bce,
    }


def extract_model_table(model, reference):
    syndrome_strings = [
        row["syndrome"]
        for row in reference["decision_table"]
    ]

    X = np.asarray(
        [
            [int(c) for c in syndrome]
            for syndrome in syndrome_strings
        ],
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        pred = (
            torch.sigmoid(logits) > 0.5
        ).cpu().numpy().astype(int)

    return [
        {
            "syndrome": syndrome,
            "decision": int(decision),
        }
        for syndrome, decision
        in zip(syndrome_strings, pred)
    ]


def exact_model_risk(rows, states):
    decision_table = {
        tuple(int(c) for c in row["syndrome"]):
        row["decision"]
        for row in rows
    }

    return exact_population_risk(
        decision_table,
        states,
    )


def exact_median(values):
    ordered = sorted(values)
    n = len(ordered)

    if n % 2:
        return ordered[n // 2]

    return (
        ordered[n // 2 - 1]
        + ordered[n // 2]
    ) / 2


def main():
    print("=" * 72)
    print("LAMBDA=0 GATE-1B — 20-RUN CAPABILITY PANEL")
    print("=" * 72)

    reference = json.loads(
        REFERENCE.read_text()
    )

    checks = validity_checks(reference)

    for item in checks:
        mark = "PASS" if item["pass"] else "FAIL"
        print(
            f"{mark:4s} {item['name']}: "
            f"{item['actual']}"
        )

    if not all(x["pass"] for x in checks):
        print()
        print(
            "STRUCTURAL/PROVENANCE VALIDITY: FLAG"
        )
        print("PANEL: NOT EVALUATED")
        raise SystemExit(2)

    print()
    print(
        "STRUCTURAL/PROVENANCE VALIDITY: ACCEPT"
    )

    oracle_map = {
        row["syndrome"]: row["decision"]
        for row in reference["decision_table"]
    }

    bayes = Fraction(
        reference["bayes_ler_fraction"]
    )

    states = enumerate_states(Q_X)

    runs = []

    for data_seed in DATA_SEEDS:
        print()
        print("-" * 72)
        print("DATA SEED:", data_seed)

        np.random.seed(data_seed)

        X, y, n_stab = generate_surface_code_data(
            DISTANCE,
            N,
            P_ERR,
        )

        assert n_stab == 8

        dataset_sha = sha256_arrays(X, y)

        Xtr = X[:N_TRAIN]
        ytr = y[:N_TRAIN]

        Xval = X[
            N_TRAIN:N_TRAIN + N_VAL
        ]
        yval = y[
            N_TRAIN:N_TRAIN + N_VAL
        ]

        print("dataset SHA-256:", dataset_sha)

        for training_seed in TRAINING_SEEDS:
            print(
                f"  training_seed={training_seed}",
                end=" ",
                flush=True,
            )

            trained = train_seeded(
                Xtr,
                ytr,
                Xval,
                yval,
                training_seed,
            )

            rows = extract_model_table(
                trained["model"],
                reference,
            )

            learned_table_sha = table_sha256(rows)

            differing = [
                {
                    "syndrome": row["syndrome"],
                    "learned": row["decision"],
                    "oracle": oracle_map[
                        row["syndrome"]
                    ],
                }
                for row in rows
                if row["decision"]
                != oracle_map[row["syndrome"]]
            ]

            risk = exact_model_risk(
                rows,
                states,
            )

            excess = risk - bayes

            run = {
                "data_seed": data_seed,
                "training_seed": training_seed,
                "dataset_sha256": dataset_sha,
                "best_epoch": trained["best_epoch"],
                "epochs_executed":
                    trained["epochs_executed"],
                "validation_ler":
                    trained["validation_ler"],
                "validation_bce":
                    trained["validation_bce"],
                "decision_table_sha256":
                    learned_table_sha,
                "exact_bayes_table_match":
                    len(differing) == 0,
                "differing_entry_count":
                    len(differing),
                "differing_entries": differing,
                "exact_population_ler_fraction":
                    fraction_text(risk),
                "exact_population_ler_float":
                    float(risk),
                "exact_excess_ler_fraction":
                    fraction_text(excess),
                "exact_excess_ler_float":
                    float(excess),
            }

            runs.append(run)

            print(
                f"best_epoch={trained['best_epoch']} "
                f"val_LER={trained['validation_ler']:.6f} "
                f"diff={len(differing):2d} "
                f"excess={float(excess):.9f}"
            )

    assert len(runs) == 20

    excesses = [
        Fraction(
            run["exact_excess_ler_fraction"]
        )
        for run in runs
    ]

    match_count = sum(
        1
        for run in runs
        if run["exact_bayes_table_match"]
    )

    median_excess = exact_median(excesses)
    minimum_excess = min(excesses)
    maximum_excess = max(excesses)

    aggregate = {
        "runs": 20,
        "exact_table_matches": match_count,
        "exact_table_match_fraction":
            match_count / 20,
        "median_exact_excess_ler_fraction":
            fraction_text(median_excess),
        "median_exact_excess_ler_float":
            float(median_excess),
        "minimum_exact_excess_ler_fraction":
            fraction_text(minimum_excess),
        "minimum_exact_excess_ler_float":
            float(minimum_excess),
        "maximum_exact_excess_ler_fraction":
            fraction_text(maximum_excess),
        "maximum_exact_excess_ler_float":
            float(maximum_excess),
        "individual_exact_excess_ler_fraction":
            [
                fraction_text(x)
                for x in excesses
            ],
    }

    payload = {
        "schema":
            "ising-bench.lambda0-gate1b-panel.v1",
        "validity": "ACCEPT",
        "status":
            "CHARACTERIZATION_COMPLETE",
        "runner_sha256":
            sha256_file(Path(__file__)),
        "seed_design": {
            "data_seeds": list(DATA_SEEDS),
            "training_seeds":
                list(TRAINING_SEEDS),
            "crossed_runs": 20,
        },
        "training_configuration": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        },
        "dataset": {
            "n": N,
            "train": N_TRAIN,
            "validation": N_VAL,
            "unused_sampled_test": (
                N - N_TRAIN - N_VAL
            ),
            "p_err": P_ERR,
            "q_x_fraction": "1/30",
        },
        "oracle": {
            "decision_table_sha256":
                EXPECTED_TABLE_SHA,
            "bayes_ler_fraction":
                fraction_text(bayes),
        },
        "validity_checks": checks,
        "runs": runs,
        "aggregate": aggregate,
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = OUT / "panel_result.json"

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 72)
    print("GATE-1B TRACK B CHARACTERIZATION")
    print("=" * 72)
    print(
        "exact-table matches:",
        f"{match_count}/20",
    )
    print(
        "median exact excess LER:",
        fraction_text(median_excess),
        f"({float(median_excess):.12f})",
    )
    print(
        "minimum exact excess LER:",
        fraction_text(minimum_excess),
        f"({float(minimum_excess):.12f})",
    )
    print(
        "maximum exact excess LER:",
        fraction_text(maximum_excess),
        f"({float(maximum_excess):.12f})",
    )
    print("PANEL STATUS: CHARACTERIZATION_COMPLETE")
    print("result:", out)
    print(
        "result SHA-256:",
        sha256_file(out),
    )


if __name__ == "__main__":
    main()
