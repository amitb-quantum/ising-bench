#!/usr/bin/env python3
"""
Corrected reanalysis of the Phase 3B reset/repreparation conditional dependence.

The original record reported single-test σ-values per delay (3.6σ at 30 µs,
2.4σ at 50 µs, 0.0σ and 1.0σ at 0/10 µs) WITHOUT a permutation null or
multiple-comparison correction. This script supplies both.

Protocol recap: measure YZ₀₁ → reset all qubits → fresh prep → measure YZ₀₁
again. The two measurements are stored concatenated in one classical register
'c'. We split per-shot into (1st, 2nd) outcomes, compute the conditional
dependence metric, and test it against a permutation null that SHUFFLES the
pairing between 1st and 2nd outcomes across shots — destroying any genuine
shot-to-shot conditional dependence while preserving both marginals exactly.

Metric (matches the record): D = |P(2nd=0|1st=0) - P(2nd=0|1st=1)|

Significance: permutation p-value per delay, then Bonferroni across the 4 delays.

IMPORTANT: this script first PRINTS the register width and the inferred bit
split, and asks you to sanity-check it before trusting the numbers. If the
split is wrong, the conditional-dependence values are meaningless, so verify
the printed marginals against the record's P(2nd=0|...) ~0.47-0.52 range.

Run:  python src/reanalyze_reprepare.py
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

REPREPARE_JSON = Path.home() / "ising-bench" / "_zenodo_stage" / "original_data" / \
                 "job-d62lmurc4tus73fdkbo0-result.json"
N_PERM = 20000          # permutations per delay (cheap, no QPU)
rng = np.random.default_rng(1234)

# Delay labels in pub order (from meta_temporal: 0,10,30,50 us)
DELAYS_US = [0.0, 10.0, 30.0, 50.0]


def load_bitarrays():
    """Deserialize the four pubs' BitArrays into per-shot 2D bit arrays.
    Returns list of (delay_us, ndarray[shots, num_bits])."""
    from qiskit.primitives.containers.bit_array import BitArray
    import qiskit.primitives.containers as _c  # noqa

    raw = json.load(open(REPREPARE_JSON))
    pubs = raw["__value__"]["pub_results"]
    out = []
    for i, pub in enumerate(pubs):
        data = pub["__value__"]["data"]["__value__"]
        c = data["fields"]["c"]  # the BitArray dict
        ba = _rebuild_bitarray(c)
        # ba is a BitArray; get an (shots, num_bits) uint8 array
        bits = _bitarray_to_bits(ba)
        out.append((DELAYS_US[i] if i < len(DELAYS_US) else float(i), bits))
    return out


def _rebuild_bitarray(c_dict):
    """Reconstruct a Qiskit BitArray from the type-tagged dict.
    num_bits is stored explicitly (=2 here: 1st and 2nd YZ01 measurement)."""
    from qiskit.primitives.containers.bit_array import BitArray
    val = c_dict["__value__"]
    num_bits = val["num_bits"]                      # explicit, no inference
    arr = _rebuild_ndarray(val["array"], num_bits)
    return BitArray(arr, num_bits)


def _rebuild_ndarray(nd_dict, num_bits):
    """Decode a base64 + zlib-compressed packed-byte array.
    Blob magic 0x789c = zlib. After decompress it's the packed uint8 buffer
    BitArray uses: shape (shots, ceil(num_bits/8))."""
    import base64, zlib
    b64 = nd_dict["__value__"]
    raw = base64.b64decode(b64)
    decompressed = zlib.decompress(raw)             # 0x789c zlib stream
    n_bytes = (num_bits + 7) // 8                    # packed bytes per shot
    arr = np.frombuffer(decompressed, dtype=np.uint8)
    if arr.size % n_bytes != 0:
        raise ValueError(f"buffer {arr.size} not divisible by {n_bytes} bytes/shot")
    return arr.reshape(-1, n_bytes)


def _bitarray_to_bits(ba):
    """BitArray -> (shots, num_bits) uint8, bit 0 = leftmost printed bit."""
    # BitArray.to_bool_array gives (shots, num_bits) in big-endian bit order
    try:
        return ba.to_bool_array().astype(np.uint8)
    except Exception:
        # manual unpack from packed bytes
        packed = ba.array  # (shots, n_bytes)
        bits = np.unpackbits(packed, axis=-1)
        return bits[:, -ba.num_bits:].astype(np.uint8)


def conditional_dependence(first, second):
    """D = |P(2nd=0|1st=0) - P(2nd=0|1st=1)|.  first/second are 0/1 arrays."""
    m0 = first == 0
    m1 = first == 1
    if m0.sum() == 0 or m1.sum() == 0:
        return np.nan, None, None
    p2_given0 = np.mean(second[m0] == 0)
    p2_given1 = np.mean(second[m1] == 0)
    return abs(p2_given0 - p2_given1), p2_given0, p2_given1


def permutation_pvalue(first, second, observed, n_perm):
    """Shuffle 'second' relative to 'first' to break pairing; null distribution
    of D. One-sided p = P(null >= observed)."""
    null = np.empty(n_perm)
    for k in range(n_perm):
        sh = rng.permutation(second)
        d, _, _ = conditional_dependence(first, sh)
        null[k] = d if not np.isnan(d) else 0.0
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return p, null


def main():
    if not REPREPARE_JSON.exists():
        print(f"file not found: {REPREPARE_JSON}")
        print("adjust REPREPARE_JSON path to where the file lives.")
        return

    print("Loading and deserializing BitArrays ...")
    data = load_bitarrays()

    # --- structural sanity check (DO NOT SKIP) ---
    _, bits0 = data[0]
    shots, num_bits = bits0.shape
    print(f"\nregister width = {num_bits} bits, {shots} shots per delay")
    print("EXPECTED: an even width if 1st+2nd YZ measurements are concatenated")
    print("(e.g. width 2 = [1st YZ01, 2nd YZ01], or wider with ancillas).")

    # num_bits is known = 2: bit 0 = 1st YZ01 measurement, bit 1 = 2nd.
    # No parity-of-halves needed; read the two bits directly.
    print(f"\nbit layout: bit 0 = 1st YZ01 measurement, bit 1 = 2nd (num_bits={num_bits})\n")
    print(f"{'delay':>7} {'D_obs':>9} {'P(2|1st=0)':>11} {'P(2|1st=1)':>11} "
          f"{'perm p':>10} {'reported σ':>11}")
    print("-" * 64)

    reported = {0.0: 0.0, 10.0: 1.0, 30.0: 3.6, 50.0: 2.4}
    results = []
    for delay, bits in data:
        # bits is (shots, num_bits); to_bool_array gives big-endian bit order.
        # first measurement = column 0, second = column 1.
        first = bits[:, 0].astype(np.uint8)
        second = bits[:, 1].astype(np.uint8) if bits.shape[1] > 1 else bits[:, 0]
        D, pg0, pg1 = conditional_dependence(first, second)
        if np.isnan(D):
            print(f"{delay:>7} {'n/a (degenerate marginal)':>40}")
            continue
        p, _ = permutation_pvalue(first, second, D, N_PERM)
        results.append((delay, D, p))
        print(f"{delay:>7} {D:>9.4f} {pg0:>11.4f} {pg1:>11.4f} {p:>10.4g} "
              f"{reported.get(delay,'?'):>11}")

    # Bonferroni across the delays actually tested
    n_tests = len(results)
    alpha_fw = 0.05
    alpha_bonf = alpha_fw / n_tests if n_tests else alpha_fw
    print("\n" + "=" * 64)
    print(f"Bonferroni across {n_tests} delays: per-test threshold "
          f"p < {alpha_bonf:.4g} for family-wise 0.05")
    survivors = [(d, D, p) for (d, D, p) in results if p < alpha_bonf]
    if survivors:
        print(f"SURVIVES correction: {[(d, round(p,5)) for d,_,p in survivors]}")
        print("The reprepare conditional dependence holds up under permutation")
        print("+ Bonferroni at these delays. 'When Reset Is Not Enough' stands.")
    else:
        print("NONE survive Bonferroni correction.")
        print("The reprepare conditional dependence does NOT hold up under a")
        print("permutation null with multiple-comparison correction. The 30 µs")
        print("3.6σ was likely inflated by single-test analysis across 4 delays.")
    print("\nNOTE: verify the printed P(2|1st=*) marginals match the record's")
    print("~0.47-0.52 values. If they don't, the bit-split convention is wrong")
    print("and these numbers need the layout corrected before trusting them.")


if __name__ == "__main__":
    main()
