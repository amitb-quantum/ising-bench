#!/usr/bin/env python3
"""
Exact oracle for Lambda=0 Gate 1B.

Enumerates all 2^9 binary error configurations using exact Fraction
arithmetic. This module is the frozen population reference for the
binary parity-decoding task specified in:

  prereg/LAMBDA0_GATE1B_EXACT_ORACLE_V1.md

It does not train or inspect any learned model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import product
from pathlib import Path


N_BITS = 9

CHECK_SUPPORTS = (
    (0, 1, 3, 4),
    (1, 2, 4, 5),
    (3, 4, 6, 7),
    (4, 5, 7, 8),
    (0, 1),
    (6, 7),
    (1, 2),
    (7, 8),
)

LOGICAL_SUPPORT = (0, 3, 6)

Q_X = Fraction(1, 30)

EXPECTED_REPORTED_CHECKS = 8
EXPECTED_GF2_RANK = 6
EXPECTED_ATTAINABLE_SYNDROMES = 64
EXPECTED_KERNEL_DIM = 3
EXPECTED_ALGEBRAIC_DISTANCE = 3

EXPECTED_BAYES_LER = Fraction(
    373445699,
    38443359375,
)


def fraction_text(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def syndrome_of(error):
    return tuple(
        sum(error[i] for i in support) % 2
        for support in CHECK_SUPPORTS
    )


def logical_label_of(error):
    return sum(error[i] for i in LOGICAL_SUPPORT) % 2


def support_to_mask(support):
    mask = 0
    for i in support:
        mask ^= 1 << i
    return mask


def gf2_rank(support_rows):
    rows = [support_to_mask(row) for row in support_rows]
    rank = 0

    for col in range(N_BITS):
        pivot = None
        for r in range(rank, len(rows)):
            if (rows[r] >> col) & 1:
                pivot = r
                break

        if pivot is None:
            continue

        rows[rank], rows[pivot] = rows[pivot], rows[rank]

        for r in range(len(rows)):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]

        rank += 1

    return rank


def error_probability(error, q=Q_X):
    w = sum(error)
    return q**w * (1 - q) ** (N_BITS - w)


def enumerate_states(q=Q_X):
    states = []

    for error in product((0, 1), repeat=N_BITS):
        states.append(
            {
                "error": error,
                "weight": sum(error),
                "syndrome": syndrome_of(error),
                "label": logical_label_of(error),
                "probability": error_probability(error, q),
            }
        )

    return states


def posterior_masses(states):
    masses = {}

    for state in states:
        syndrome = state["syndrome"]
        label = state["label"]

        if syndrome not in masses:
            masses[syndrome] = [Fraction(0), Fraction(0)]

        masses[syndrome][label] += state["probability"]

    return masses


def bayes_table_from_masses(masses):
    table = {}
    ties = []

    for syndrome in sorted(masses):
        p0, p1 = masses[syndrome]

        if p0 == p1:
            ties.append(syndrome)
            continue

        table[syndrome] = 1 if p1 > p0 else 0

    return table, ties


def minimum_weight_table(states):
    best = {}

    for state in states:
        syndrome = state["syndrome"]
        label = state["label"]
        weight = state["weight"]

        if syndrome not in best:
            best[syndrome] = [None, None]

        old = best[syndrome][label]

        if old is None or weight < old:
            best[syndrome][label] = weight

    table = {}
    ties = []

    for syndrome in sorted(best):
        w0, w1 = best[syndrome]

        if w0 == w1:
            ties.append(syndrome)
            continue

        table[syndrome] = 0 if w0 < w1 else 1

    return table, ties


def exact_population_risk(decision_table, states):
    risk = Fraction(0)

    for state in states:
        decision = decision_table[state["syndrome"]]

        if decision != state["label"]:
            risk += state["probability"]

    return risk


def algebraic_distance(states):
    candidates = [
        state["weight"]
        for state in states
        if all(x == 0 for x in state["syndrome"])
        and state["label"] == 1
    ]

    if not candidates:
        return None

    return min(candidates)


def table_payload(table):
    rows = []

    for syndrome in sorted(table):
        rows.append(
            {
                "syndrome": "".join(str(x) for x in syndrome),
                "decision": int(table[syndrome]),
            }
        )

    return rows


def table_sha256(rows):
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(canonical).hexdigest()


def build_reference():
    rank_h = gf2_rank(CHECK_SUPPORTS)

    rank_h_plus_l = gf2_rank(
        CHECK_SUPPORTS + (LOGICAL_SUPPORT,)
    )

    logical_in_rowspace = rank_h_plus_l == rank_h

    states = enumerate_states(Q_X)

    total_probability = sum(
        (state["probability"] for state in states),
        Fraction(0),
    )

    masses = posterior_masses(states)

    bayes_table, posterior_ties = bayes_table_from_masses(masses)

    mw_table, mw_ties = minimum_weight_table(states)

    bayes_ler = exact_population_risk(
        bayes_table,
        states,
    )

    distance = algebraic_distance(states)

    assert len(CHECK_SUPPORTS) == EXPECTED_REPORTED_CHECKS
    assert rank_h == EXPECTED_GF2_RANK
    assert len(masses) == EXPECTED_ATTAINABLE_SYNDROMES
    assert N_BITS - rank_h == EXPECTED_KERNEL_DIM
    assert logical_in_rowspace is False
    assert distance == EXPECTED_ALGEBRAIC_DISTANCE

    assert total_probability == 1
    assert posterior_ties == []
    assert mw_ties == []

    assert bayes_ler == EXPECTED_BAYES_LER

    # At q_X = 1/30, exact Bayes and exact minimum-weight/coset
    # decisions must coincide for all attainable syndromes.
    assert bayes_table == mw_table

    rows = table_payload(bayes_table)

    reference = {
        "schema": "ising-bench.lambda0-exact-oracle.v1",
        "n_binary_error_variables": N_BITS,
        "reported_checks": len(CHECK_SUPPORTS),
        "check_supports": [
            list(x) for x in CHECK_SUPPORTS
        ],
        "gf2_rank": rank_h,
        "kernel_dimension": N_BITS - rank_h,
        "attainable_syndromes": len(masses),
        "logical_support": list(LOGICAL_SUPPORT),
        "logical_in_rowspace": logical_in_rowspace,
        "algebraic_distance": distance,
        "q_x_fraction": fraction_text(Q_X),
        "q_x_float": float(Q_X),
        "posterior_ties": len(posterior_ties),
        "minimum_weight_ties": len(mw_ties),
        "bayes_ler_fraction": fraction_text(bayes_ler),
        "bayes_ler_float": float(bayes_ler),
        "bayes_equals_minimum_weight_table": True,
        "decision_table_sha256": table_sha256(rows),
        "decision_table": rows,
    }

    return reference


def write_reference(path: Path):
    reference = build_reference()

    text = json.dumps(
        reference,
        indent=2,
        sort_keys=True,
    ) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

    return reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-reference",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    reference = build_reference()

    print("reported checks:", reference["reported_checks"])
    print("GF(2) rank:", reference["gf2_rank"])
    print(
        "attainable syndromes:",
        reference["attainable_syndromes"],
    )
    print(
        "logical in rowspace:",
        reference["logical_in_rowspace"],
    )
    print(
        "algebraic distance:",
        reference["algebraic_distance"],
    )
    print("q_X:", reference["q_x_fraction"])
    print(
        "posterior ties:",
        reference["posterior_ties"],
    )
    print(
        "Bayes LER:",
        reference["bayes_ler_fraction"],
    )
    print(
        "Bayes LER float:",
        f'{reference["bayes_ler_float"]:.18f}',
    )
    print(
        "Bayes == minimum-weight table:",
        reference["bayes_equals_minimum_weight_table"],
    )
    print(
        "decision-table SHA-256:",
        reference["decision_table_sha256"],
    )

    if args.write_reference is not None:
        write_reference(args.write_reference)
        print("reference written:", args.write_reference)


if __name__ == "__main__":
    main()
