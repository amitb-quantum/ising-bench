# ising-bench

Learned-vs-memoryless comparative benchmarking for quantum error structure,
built around NVIDIA's open **Ising** quantum-AI models (released 2026-04-14,
Apache-2.0) and the existing `quantum_central_db` measurement archive.

**Owner:** Quantum-Clarity LLC
**Thesis under test:** quantum errors are non-Markovian — spatially correlated,
environmentally mediated — contradicting the independent/memoryless assumption
of standard QEC. Goal here: detect and quantify that correlation in real data,
and benchmark learned decoders (which can exploit correlation) against
memoryless baselines (which cannot).

---

## Environment

- **Host:** WSL2 (Ubuntu) on laptop, NVIDIA **RTX A1000 6GB** (mobile).
- **Conda env:** `cudaq-ising`, Python **3.11.10** (pinned to avoid conda-forge
  cudaq segfaults).
- **GPU access:** via Windows-side NVIDIA driver; `nvidia-smi` works in WSL.
  Do **not** install a GPU driver inside WSL.
- **VRAM budget:** 6 GB fp32 → ~28 qubits max full statevector. Fine for
  decoder training and real-data work; caps large simulated code patches.

### Verified-working stack
```
cuda-quantum-cu12        # CUDA-Q, target 'nvidia' = cuStateVec fp32  ✓ sees GPU
torch (cu121)            # learned decoders                            ✓ sees CUDA
numpy scipy matplotlib pandas scikit-learn
qiskit qiskit-ibm-runtime
```

### Recreate env
```bash
conda create -y -n cudaq-ising python=3.11.10 pip
conda activate cudaq-ising
python -m pip install --upgrade pip setuptools wheel
pip install cuda-quantum-cu12
pip install numpy scipy matplotlib pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install qiskit qiskit-ibm-runtime
```

### Verify
```bash
python -c "import cudaq; cudaq.set_target('nvidia'); print(cudaq.get_target())"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Directory layout
```
~/ising-bench/
  src/                  scripts (below)
  data/raw/             untouched source data
  data/processed/       ML-ready tensors
  models/               decoder checkpoints
  results/figures/      plots
  results/logs/         run metrics (json)
  notebooks/            interactive exploration
```

---

## Scripts

### `src/ising_benchmark.py`
Two-arm comparative benchmark.
- **Simulated arm:** distance-3 surface code, Monte-Carlo Pauli-error sampling
  in the stabilizer frame, depolarizing noise. Learned CNN decoder vs.
  logistic-regression (memoryless) baseline, scored on logical error rate.
- **Real arm:** reads counts directly from JSON files on disk (NOT the DB —
  see Data Notes). Tests whether one qubit's bit is predictable from the others
  beyond its own marginal = a correlation probe.

Run: `python src/ising_benchmark.py [--skip-sim] [--skip-real]`

### `src/mi_matrix.py`
Per-width pairwise **mutual-information matrix** from the counts. Miller-Madow
bias-corrected. Emits heatmap per width + MI-vs-index-separation curve for
widths ≥5. No new dependencies.

Run: `python src/mi_matrix.py`

---

## Data Notes (IMPORTANT — these shaped every design choice)

Source: `~/quantum_central_db/` (~602 MB).

1. **`quantum_data.db` is NOT the right source.** Its `quantum_jobs` table has
   only 54 rows, `creation_date` is empty, and its `job_id`s **do not match**
   the JSON filenames on disk (different batch). Loaders read JSON files
   directly and sort on the in-file `completed_at`.

2. **JSON files are raw measurement counts, not QEC syndromes.** Schema:
   `backend_name, circuit_name, completed_at, depth, job_id, num_qubits,
   num_shots, results.counts{bitstring:int}, status, submitted_at`.
   No stabilizer/syndrome structure → cannot do real-hardware surface-code
   decoding from this data. (The simulated arm does real QEC; the real arm does
   correlation analysis. They are deliberately NOT the same task.)

3. **No qubit-layout metadata.** Only `num_qubits`. The transpiled
   `final_layout` (measurement-index → physical heavy-hex qubit) is absent.
   `03_circuit_definitions/` holds Cirq/Google calibration objects, not IBM
   layouts. Consequence: MI is by **measurement index, NOT physical distance**.
   Spatial-distance decay analysis is currently **blocked** on this.
   A dormant `final_layout_for()` hook in `mi_matrix.py` is ready if layouts
   are recovered.

4. **`circuit_name` is uninformative** — `remote_job` for all multi-qubit jobs,
   missing for width-2. So we do NOT know what state each circuit prepared.

5. **Two files deliberately excluded from all globs** (not result files,
   irrelevant to this work): `Phishing_exfiltration_script.json`,
   `Ransomware_detector_circuit.json`. Loaders skip any path containing
   `:sec`, `phishing`, or `exfiltration`.

---

## Results so far

### Simulated arm
Learned decoder LER 0.090 vs memoryless 0.022 — **baseline wins.** This is
correct/expected: noise is independent depolarizing, so there is no correlation
for the learned decoder to exploit, AND the net underfits (full-batch, 30 epochs,
8 stabilizers, near-linear task). Mirrors the known result that simple methods
beat learned ones on linearly-separable problems. The benchmark is an honest
instrument; the simulated task just isn't in the regime where learning helps.

### Real arm — MI matrix (mean off-diagonal MI, bits)
| width | n_jobs | mean off-diag MI | max MI |
|------:|-------:|-----------------:|-------:|
| 2     | 24     | 0.851            | 0.851  |
| 5     | 6      | 0.569            | 0.788  |
| 10    | 2      | 0.552            | 0.821  |
| 20    | 3      | 0.392            | 0.756  |
| 4     | 6      | 0.045            | 0.155  |
| 3     | 1      | 0.002            | 0.006  |

---

## ⚠️ Critical interpretation caveat (do not skip)

**The high MI values cannot currently be claimed as error correlation.**

Raw MI conflates (a) entanglement the circuit prepared *on purpose* (e.g. a
Bell/GHZ state has near-maximal MI by design) with (b) correlation from a shared
environment (the actual non-Markovian thesis). The data cannot separate these
because `circuit_name` is generic (`remote_job`) — we don't know the intended
state, so we can't subtract it.

The width-dependent pattern (high at 2/5/10/20, near-zero at 3/4) further
suggests the signal is **circuit-driven (designed entanglement), not hardware-
driven** — hardware correlation would not switch on/off with circuit width.

**To test the real thesis, MI must be computed on the RESIDUAL** = measured
output minus ideal (noiseless) circuit output. That requires knowing each
circuit. This data lacks it.

---

## Decision

Choose **Path B — purpose-built idle/identity experiment**.

The historical `quantum_central_db` data is useful for pipeline validation, but
not for a defensible hardware error-correlation claim. The counts show high
mutual information at some widths, but the source circuits and final layouts are
missing, so raw MI cannot distinguish prepared entanglement from correlated
hardware error.

The next phase generates controlled IBM Quantum jobs with:
- known idle/identity circuits,
- known ideal output distribution (all-zero),
- captured transpiled `final_layout`,
- a swept idle duration τ (around the ~30 µs characteristic timescale),
- contiguous heavy-hex patch placement (defined physical distances),
- residual-MI analysis on measured deviations from ideal,
- MI-vs-physical-distance analysis on the heavy-hex lattice.

Historical data remains useful as a negative-control / archive-forensics case,
but not as the main evidence for the non-Markovian thesis.

### Target result statement
"On controlled identity/idle circuits with known ideal all-zero output and
recorded physical layout, residual bit-flip events exhibit / do not exhibit
statistically significant mutual information as a function of physical qubit
distance, and as a function of idle duration τ."

### Implementation
`src/run_idle_correlation_experiment.py` — submits controlled circuits and
saves, per job: backend, job_id, num_qubits, shots, circuit_family, idle_us,
logical_qubits, physical_qubits, final_layout, depth, scheduled_duration,
submitted_at, completed_at, counts.

`src/analyze_idle_correlation.py` — residual-MI matrix + MI-vs-physical-distance
+ MI-vs-τ, using the recorded final_layout for true heavy-hex distances.

`src/permutation_test_mi.py` — permutation-test significance + family-wise
(Bonferroni) correction over all qubit pairs; reports the detection floor.

---

## Path B results — run 1 (ibm_marrakesh, 2026-05-30)

**Setup:** 12 qubits, contiguous heavy-hex patch [0,1,2,3,4,5,6,7,16,22,23,24],
idle τ ∈ {0, 10, 30, 60, 120} µs, 8192 shots, dynamical decoupling OFF.
Idle error rate ~0.5–0.6%, roughly flat in τ.

**Permutation test (500 perms/pair, Bonferroni over 66 pairs):**

| τ (µs) | err_rate | max raw MI | floor99 | sig pairs | verdict |
|-------:|---------:|-----------:|--------:|----------:|--------:|
| 0   | 0.0064 | 6.50e-04 | 3.55e-04 | 0 | null |
| 10  | 0.0063 | 2.87e-04 | 3.71e-04 | 0 | null |
| 30  | 0.0060 | 6.81e-04 | 3.77e-04 | 0 | null |
| 60  | 0.0054 | 4.58e-04 | 3.54e-04 | 0 | null |
| 120 | 0.0052 | 8.42e-04 | 3.41e-04 | 0 | null |

**Rigorous claim (use this wording, not "MI < X"):**
No qubit pair shows residual error mutual information exceeding the permutation
null at 99% family-wise significance after Bonferroni correction over 66 pairs,
at any tested τ. The experiment's **family-wise detection floor is ≈ 3.8×10⁻⁴
bits** — state this as a *sensitivity limit*, NOT as a maximum observed MI.
Several raw pairwise maxima exceed the per-run floor (largest 8.42×10⁻⁴ at
τ=120 µs), but none survive the corrected test — which is exactly what a sound
multiple-comparison test should do: reject isolated pairwise fluctuations that
are plausible under the null.

**What this does and does NOT say.** It does NOT say IBM hardware errors are
never correlated. It says this experiment did not detect statistically
significant pairwise residual bit-flip correlation above its sensitivity scale,
on this backend, this patch, under idle identity circuits, at these τ and shot
count. A weaker correlation may be physically present below current resolution.

**Connection to the archive.** The archive showed raw MI ~0.4–0.85 bits, but
those jobs lacked circuit identity and layout, so MI could not be read as error
correlation. The controlled experiment puts the residual idle-circuit signal
near the 10⁻⁴-bit scale and statistically null. The gap is the finding: the
historical high MI was almost certainly dominated by prepared circuit structure
(designed entanglement), not environmental error correlation.

**Next step (NOT just more shots).** Run the identical protocol on a second
backend with different noise characteristics. If it also nulls, the constraint
generalizes; if it shows signal, the effect is hardware-specific (a stronger,
more interesting result). Only after that comparison decide whether to drive the
single-backend bound lower. Empirically the sensitivity floor should improve
with more shots, roughly on the order of 1/N for this estimator/null setup, so
an order-of-magnitude tighter floor likely needs ~an order-of-magnitude more
shots — phrase this as an empirical expectation, not a theorem.

Caveat on protocol sensitivity: idle tomography is the *cleanest* protocol but
maybe not the most *sensitive*. Prior findings emphasized that simpler protocols
might miss correlations; the original effect may require active circuit
structure that bare idling does not generate.

---

## Path B results — run 2: cross-hardware sweep (2026-05-30)

**Setup:** identical Z-basis idle protocol on three 156-qubit Heron-class
systems. Marrakesh from run 1 (5 τ); Kingston + Fez added here (τ ∈ {0,30,120}
µs). 12-qubit contiguous patch, 8192 shots, DD off. Total QPU usage for the
6 new jobs: ~28 s.

**Per-backend permutation test (500 perms/pair, Bonferroni over 66 pairs):**

| backend | τ range | err_rate range | max raw MI | floor99 | sig pairs | verdict |
|--------|--------:|---------------:|-----------:|--------:|----------:|--------:|
| ibm_fez       | 0–120µs | 0.0070–0.0077 | 3.98e-04 | 4.16e-04 | 0 | null |
| ibm_kingston  | 0–120µs | 0.0089–0.0107 | 5.82e-04 | 4.19e-04 | 0 | null |
| ibm_marrakesh | 0–120µs | 0.0052–0.0064 | 8.42e-04 | 3.73e-04 | 0 | null |

### Result (canonical wording — use this)

Across three 156-qubit IBM Heron-class systems (`ibm_marrakesh`,
`ibm_kingston`, `ibm_fez`), on 12-qubit contiguous heavy-hex patches under
Z-basis idle identity circuits at idle durations spanning 0–120 µs and 8192
shots, no qubit pair on any device shows residual error mutual information
surviving the permutation null at 99% family-wise significance after Bonferroni
correction. Per-run detection floors cluster around 3.5–4.2×10⁻⁴ bits across all
backends and idle times. The null generalizes across the tested hardware.

What makes this more informative than three isolated nulls is that the devices
have different baseline error rates: Kingston is hottest (~0.89–1.07%),
Marrakesh coolest (~0.52–0.64%), Fez between (~0.70–0.77%). Yet all three show
the same structureless null at roughly the same detection scale. If the relevant
residual correlation scaled with the same mechanisms driving the baseline idle
error rate, Kingston would have been the most likely place to show signal. It
did not. The patch-selection rule also landed on the same nominal qubit-index
pattern across all three devices, reducing the chance that the result is merely
a placement artifact (note: same index pattern ≠ same physical/control
geometry — calibration, couplers, local defects differ by device).

### Boundary (do NOT over-read)

This constrains idle, Z-basis, single-shot residual **bit-flip** correlation on
Heron-class hardware up to 120 µs, at a sensitivity of ~4×10⁻⁴ bits. It does
**not** falsify the broader non-Markovian thesis. Specifically it does not rule
out: (1) correlation below the current floor; (2) correlation in protocols other
than idle; (3) operation-induced correlation during active gate layers;
(4) phase/dephasing-basis correlation invisible to Z-idle; (5) higher-order
multi-qubit correlation not captured by pairwise MI.

Sharp statement (not "errors are not correlated"):
> Controlled Z-basis idle circuits do not reveal detectable pairwise ambient
> bit-flip error correlation at the 10⁻⁴-bit MI scale across the tested
> Heron-class systems.

This narrows the hypothesis space: if the earlier non-Markovian signal is real,
it is more likely operation-induced, phase-type, protocol-dependent, or below
the current detection floor — not a large ambient Z-basis idle correlation.

### Next experiments (in priority order)
1. **X-basis idle sweep**, starting with Kingston (highest baseline error) —
   tests the dephasing-sensitive channel Z-idle deliberately does not probe.
   Use `--basis X_idle` (basis tag already wired into the manifest).
2. **Active-circuit protocol** with entangling layers — operation-induced
   crosstalk is now a more plausible location for the effect than bare idle.

This three-backend null is a complete technical result, worth writing up as a
standalone controlled-negative section (a strong section, not a whole paper).

---

## Background: NVIDIA Ising (context for the project)
Open-source quantum-AI model family (2026-04-14, Apache-2.0) targeting two
problems: (1) calibration — a vision-language model that tunes hardware params
in near-real-time; (2) decoding — 3D-CNN models over error syndromes, latency-
or accuracy-optimized. Hardware-agnostic, integrates with CUDA-Q + NVQLink.
Reported up to 2.5× faster / 3× more accurate real-time decoding. The thesis
overlap: learned decoders beat static (pyMatching-style) ones precisely when
errors are correlated — which is what this project tries to establish in data.

IP note: Apache-2.0 permits commercial fine-tuning, but keep QuantaCore
implementation (Basis Migration, Control Plane, Q-HAL) cleanly separated from
any Ising-derived code for provenance. (Not legal advice — confirm with IP
counsel re: US 63/952,786.)
