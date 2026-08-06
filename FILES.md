# File Manifest — Zenodo v1.2 Bundle

This archive contains the original IBM Quantum job-result files from the February 2026 active-protocol campaign, plus follow-up May 2026 Z-basis idle-control experiments and analysis scripts.

The purpose of this version is to preserve the original experimental record while adding a stricter interpretation: the original active-protocol results are treated as suggestive, protocol-dependent evidence pending corrected reanalysis, while the follow-up idle-control study provides a clean controlled null for ambient Z-basis bit-flip correlation.

---

## Directory layout

```text
original_data/
  Original February 2026 IBM Quantum job results and metadata.

idle_control_data/
  May 2026 controlled Z-basis idle/identity experiment result files.

idle_control_results/
  Processed metrics from idle-correlation and permutation-null analyses.

analysis_scripts/
  Python scripts used to submit, fetch, analyze, and benchmark the idle-control data.
```

---

## original_data/

These files reproduce the original three-phase `ibm_fez` campaign.

| File                                      | Description                                                                                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `job-d61v0lao8gvs73f1gutg-result.json`    | Phase 1 spatial Y⊗Z parity-consistency experiment. Contains IBM job-result counts for the six-module spatial sweep.                      |
| `meta_d61v0lao8gvs73f1gutg.json`          | Metadata for the Phase 1 spatial experiment, including circuit/module information.                                                       |
| `job-d62h65ns6ggc73fgqee0-result.json`    | Phase 2 temporal sequential-measurement experiment. Contains counts for SIGNAL / ANCILLA-ONLY / ONE-SHOT style temporal-memory circuits. |
| `meta_temporal_d62h65ns6ggc73fgqee0.json` | Metadata for the Phase 2 temporal sequential-measurement experiment.                                                                     |
| `job-d62lmg3c4tus73fdkb9g-result.json`    | Phase 3A echo/control experiment. Included for completeness from the original campaign.                                                  |
| `meta_temporal_d62lmg3c4tus73fdkb9g.json` | Metadata for the Phase 3A echo/control experiment.                                                                                       |
| `job-d62lmurc4tus73fdkbo0-result.json`    | Phase 3B reset/repreparation experiment. Contains counts for the measure → reset → fresh prepare → measure protocol.                     |
| `meta_temporal_d62lmurc4tus73fdkbo0.json` | Metadata for the Phase 3B reset/repreparation experiment.                                                                                |

Important note: the original Phase 1–3 σ-values are single-test reported significances. They should be treated as provisional until reanalyzed with permutation/bootstrap nulls and multiple-comparison correction.

---

## idle_control_data/

These files contain the follow-up controlled Z-basis idle/identity experiments.

The protocol is:

```text
prepare |0...0⟩ → idle for τ → measure in Z basis
```

The ideal output is all zeros, so measured `1` bits are interpreted as residual bit-flip error events. Pairwise mutual information between error events is used to test for ambient bit-flip correlation.

### ibm_fez

| File                                                         | Description                                          |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| `idle_ibm_fez_Z_idle_q12_tau0us_d8dkekov14cs73digoc0.json`   | `ibm_fez`, Z-basis idle, 12-qubit patch, τ = 0 µs.   |
| `idle_ibm_fez_Z_idle_q12_tau30us_d8dkekq4gq0s73aqdsdg.json`  | `ibm_fez`, Z-basis idle, 12-qubit patch, τ = 30 µs.  |
| `idle_ibm_fez_Z_idle_q12_tau120us_d8dkel5mdsks73d4253g.json` | `ibm_fez`, Z-basis idle, 12-qubit patch, τ = 120 µs. |

### ibm_kingston

| File                                                              | Description                                               |
| ----------------------------------------------------------------- | --------------------------------------------------------- |
| `idle_ibm_kingston_Z_idle_q12_tau0us_d8dkek7d0j8c73f4ddt0.json`   | `ibm_kingston`, Z-basis idle, 12-qubit patch, τ = 0 µs.   |
| `idle_ibm_kingston_Z_idle_q12_tau30us_d8dkekdmdsks73d4250g.json`  | `ibm_kingston`, Z-basis idle, 12-qubit patch, τ = 30 µs.  |
| `idle_ibm_kingston_Z_idle_q12_tau120us_d8dkekgv14cs73digob0.json` | `ibm_kingston`, Z-basis idle, 12-qubit patch, τ = 120 µs. |

### ibm_marrakesh

| File                                                        | Description                                                                                            |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `idle_ibm_marrakesh_q12_tau0us_d8dipufd0j8c73f4b4a0.json`   | `ibm_marrakesh`, Z-basis idle, 12-qubit patch, τ = 0 µs.                                               |
| `idle_ibm_marrakesh_q12_tau10us_d8dipugv14cs73diej2g.json`  | `ibm_marrakesh`, Z-basis idle, 12-qubit patch, τ = 10 µs.                                              |
| `idle_ibm_marrakesh_q12_tau30us_d8dipui4gq0s73aqbkv0.json`  | `ibm_marrakesh`, Z-basis idle, 12-qubit patch, τ = 30 µs.                                              |
| `idle_ibm_marrakesh_q12_tau60us_d8diputmdsks73d3vsvg.json`  | `ibm_marrakesh`, Z-basis idle, 12-qubit patch, τ = 60 µs.                                              |
| `idle_ibm_marrakesh_q12_tau120us_d8dipv24gq0s73aqbl00.json` | `ibm_marrakesh`, Z-basis idle, 12-qubit patch, τ = 120 µs.                                             |
| `manifest_ibm_marrakesh_1780165882.json`                    | Submission manifest for the original `ibm_marrakesh` idle sweep, including selected patch and job IDs. |

Note: the `ibm_fez` and `ibm_kingston` idle sweeps used τ = 0, 30, and 120 µs. The earlier `ibm_marrakesh` sweep used τ = 0, 10, 30, 60, and 120 µs.

---

## idle_control_results/

| File                    | Description                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `idle_correlation.json` | Processed idle-correlation metrics, including error rates and mutual-information summaries.                                                                                                             |
| `permutation_test.json` | Final permutation-test output. Uses 500 permutations per qubit pair and Bonferroni correction over 66 pairs at 99% family-wise significance. Reports zero significant pairs across all tested backends. |

Main idle-control result:

```text
No qubit pair on ibm_fez, ibm_kingston, or ibm_marrakesh showed residual error mutual information surviving the 99% family-wise permutation null.
```

This supports the scoped statement:

```text
Controlled Z-basis idle circuits do not reveal detectable pairwise ambient bit-flip error correlation at the ~10^-4-bit mutual-information scale across the tested Heron-class systems.
```

---

## analysis_scripts/

| File                                 | Description                                                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `run_idle_correlation_experiment.py` | Submits controlled Z-basis idle/identity experiments to IBM Quantum backends. Selects a 12-qubit patch and sweeps idle duration τ. |
| `fetch_idle_results.py`              | Fetches completed IBM Runtime jobs and writes raw result JSON files into the idle-control data directory.                          |
| `analyze_idle_correlation.py`        | Computes residual bit-flip error rates and pairwise mutual-information summaries from the idle-control result files.               |
| `permutation_test_mi.py`             | Performs permutation-null testing for pairwise mutual information, with Bonferroni family-wise correction.                         |
| `mi_matrix.py`                       | Computes pairwise mutual-information matrices for raw count data. Used for exploratory/historical archive analysis.                |
| `ising_benchmark.py`                 | Two-arm learned-vs-memoryless benchmark scaffold. Includes simulated decoder benchmarking and real-count correlation probing.      |

---

## Interpretation boundary

The idle-control files do **not** show that quantum errors are never correlated.

They show that, under this specific controlled protocol:

```text
Z-basis idle identity circuits
12-qubit patches
0–120 µs idle durations
8192 shots
pairwise mutual information
99% family-wise permutation test
```

there is no detectable pairwise ambient bit-flip correlation above the current sensitivity scale.

The original active protocols may still indicate protocol-induced memory if they survive corrected reanalysis. In that case, the likely interpretation is not generic idle noise, but memory associated with measurement, reset, repreparation, stabilizer construction, resonator history, TLS defects, or other environmental/control degrees of freedom.

---

## Recommended citation

```bibtex
@dataset{brahmbhatt_protocol_dependent_error_memory_2026,
  author       = {{Amit Brahmbhatt}},
  title        = {{When Reset Is Not Enough: Evidence for Protocol-Dependent Error Memory in Heron-Class Quantum Processors}},
  year         = 2026,
  month        = may,
  publisher    = {Zenodo},
  version      = {1.2},
  doi          = {10.5281/zenodo.20467541},
  url          = {https://doi.org/10.5281/zenodo.20467541}
}
```
