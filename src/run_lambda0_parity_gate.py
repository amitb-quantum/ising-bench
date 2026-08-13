#!/usr/bin/env python3
"""
Prospective Lambda=0 learned-decoder parity gate.

Implements:
  prereg/LAMBDA0_PARITY_GATE_V1.md
  prereg/LAMBDA0_PARITY_GATE_V1_AMENDMENT_001.md

CPU only. Final test is evaluated exactly once after candidate selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ising_benchmark import (
    CNNDecoder,
    generate_surface_code_data,
    memoryless_baseline,
)

SEED = 1234
N = 100_000
DISTANCE = 3
P_ERR = 0.05

N_TRAIN = 60_000
N_VAL = 20_000
N_TEST = 20_000

NI_MARGIN = 0.005
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260812

DEVICE = torch.device("cpu")

CANDIDATES = {
    "A": {"batch_size": 256, "lr": 1e-3, "max_epochs": 200, "patience": 20},
    "B": {"batch_size": 512, "lr": 1e-3, "max_epochs": 200, "patience": 20},
    "C": {"batch_size": 256, "lr": 3e-4, "max_epochs": 400, "patience": 30},
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lambda0_parity_gate"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(DEVICE))
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(y).float().to(DEVICE)
        ).item()
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.float32)

    ler = float(np.mean(pred != y))
    return ler, float(loss), pred


def train_candidate(name, cfg, Xtr, ytr, Xval, yval):
    # Identical deterministic initialization for every candidate.
    torch.manual_seed(SEED)

    model = CNNDecoder(Xtr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(ytr).float(),
    )

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(SEED)

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
    )

    best_state = None
    best_val_ler = float("inf")
    best_val_loss = float("inf")
    best_epoch = None
    stale = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_ler, val_loss, _ = evaluate(model, Xval, yval)

        improved = (
            val_ler < best_val_ler
            or (val_ler == best_val_ler and val_loss < best_val_loss)
        )

        if improved:
            best_val_ler = val_ler
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"  {name} epoch={epoch:3d} "
                f"val_LER={val_ler:.6f} "
                f"val_BCE={val_loss:.6f} "
                f"best={best_val_ler:.6f}"
            )

        if stale >= cfg["patience"]:
            break

    if best_state is None:
        raise RuntimeError(f"candidate {name}: no checkpoint recorded")

    model.load_state_dict(best_state)

    return {
        "name": name,
        "model": model,
        "best_epoch": best_epoch,
        "val_ler": best_val_ler,
        "val_bce": best_val_loss,
        "epochs_executed": epoch,
        "config": cfg,
    }


def paired_bootstrap_upper95(err_learned, err_baseline):
    d = err_learned.astype(np.int8) - err_baseline.astype(np.int8)
    observed = float(np.mean(d))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_B, dtype=np.float64)

    # Chunked only for memory efficiency; each replicate remains a paired
    # resample of N_TEST final-test examples with replacement.
    chunk = 200
    pos = 0

    while pos < BOOTSTRAP_B:
        m = min(chunk, BOOTSTRAP_B - pos)
        idx = rng.integers(0, len(d), size=(m, len(d)))
        boot[pos:pos + m] = d[idx].mean(axis=1)
        pos += m

    u95 = float(np.quantile(boot, 0.95, method="linear"))
    return observed, u95


def main():
    print("=" * 72)
    print("LAMBDA=0 LEARNED-DECODER PARITY GATE")
    print("=" * 72)
    print("device: CPU")
    print("seed:", SEED)
    print("N:", N)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y, n_stab = generate_surface_code_data(DISTANCE, N, P_ERR)

    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xval, yval = X[N_TRAIN:N_TRAIN + N_VAL], y[N_TRAIN:N_TRAIN + N_VAL]
    Xte, yte = X[N_TRAIN + N_VAL:], y[N_TRAIN + N_VAL:]

    assert len(Xte) == N_TEST

    print(
        f"syndromes={X.shape} stabilizers={n_stab} "
        f"logical_flip_rate={y.mean():.6f}"
    )
    print(
        f"split: train={len(Xtr)} val={len(Xval)} test={len(Xte)}"
    )

    candidates = []

    for name in ("A", "B", "C"):
        print("\n" + "-" * 72)
        print("TRAIN CANDIDATE", name, CANDIDATES[name])
        result = train_candidate(
            name, CANDIDATES[name], Xtr, ytr, Xval, yval
        )
        candidates.append(result)
        print(
            f"candidate {name}: best_epoch={result['best_epoch']} "
            f"val_LER={result['val_ler']:.6f} "
            f"val_BCE={result['val_bce']:.6f}"
        )

    # Frozen selection rule:
    # lowest validation LER -> lowest validation BCE -> A/B/C order.
    order = {"A": 0, "B": 1, "C": 2}
    chosen = min(
        candidates,
        key=lambda r: (r["val_ler"], r["val_bce"], order[r["name"]]),
    )

    print("\n" + "=" * 72)
    print(
        f"SELECTED: {chosen['name']} "
        f"val_LER={chosen['val_ler']:.6f} "
        f"val_BCE={chosen['val_bce']:.6f}"
    )
    print("=" * 72)

    # FINAL TEST: evaluated only now, once candidate selection is complete.
    learned_ler, learned_bce, learned_pred = evaluate(
        chosen["model"], Xte, yte
    )

    base_pred = memoryless_baseline(Xtr, ytr, Xte)
    base_ler = float(np.mean(base_pred != yte))

    err_l = learned_pred != yte
    err_b = base_pred != yte

    delta, u95 = paired_bootstrap_upper95(err_l, err_b)
    verdict = "PASS" if u95 <= NI_MARGIN else "FAIL"

    print(f"final learned LER:  {learned_ler:.6f}")
    print(f"final logistic LER: {base_ler:.6f}")
    print(f"Delta_LER:          {delta:+.6f}")
    print(f"paired bootstrap U95:{u95:+.6f}")
    print(f"NI margin:           {NI_MARGIN:+.6f}")
    print(f"GATE VERDICT:        {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)

    prereg = ROOT / "prereg" / "LAMBDA0_PARITY_GATE_V1.md"
    amend = ROOT / "prereg" / "LAMBDA0_PARITY_GATE_V1_AMENDMENT_001.md"

    payload = {
        "schema": "ising-bench.lambda0-parity-gate.v1",
        "seed": SEED,
        "dataset": {
            "n": N,
            "distance": DISTANCE,
            "p_err": P_ERR,
            "train": N_TRAIN,
            "validation": N_VAL,
            "test": N_TEST,
        },
        "prereg_sha256": sha256_file(prereg),
        "amendment_001_sha256": sha256_file(amend),
        "candidate_results": [
            {
                "name": r["name"],
                "config": r["config"],
                "best_epoch": r["best_epoch"],
                "epochs_executed": r["epochs_executed"],
                "validation_ler": r["val_ler"],
                "validation_bce": r["val_bce"],
            }
            for r in candidates
        ],
        "selected_candidate": chosen["name"],
        "final": {
            "learned_ler": learned_ler,
            "learned_bce": learned_bce,
            "logistic_ler": base_ler,
            "delta_ler": delta,
            "bootstrap_upper95": u95,
            "noninferiority_margin": NI_MARGIN,
            "verdict": verdict,
        },
    }

    out = OUT / "gate_result.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("\nresult:", out)
    print("result SHA-256:", sha256_file(out))


if __name__ == "__main__":
    main()
