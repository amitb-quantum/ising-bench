# Lambda=0 Gate 1B — Exact-Oracle Calibration v1

Date frozen: 2026-08-12

## Purpose

Calibrate the learned-decoder training procedure against an exact,
enumerable population oracle under lambda=0.

The exact oracle performs the adjudication. The neural network is the
instrument under calibration.

This experiment does not establish scalable QEC decoder performance.

## Benchmark ceiling

The benchmark is a binary parity-decoding task with surface-code-shaped
supports. It is not represented as a faithful surface-code stabilizer
simulation.

The induced binary decoding task has algebraic distance 3.

No result from this gate constitutes a comparison with MWPM, PyMatching,
or a production quantum-error-correction decoder.

## Frozen binary task

Number of binary error variables: 9

Reported parity checks: 8

Check supports:

S0 = [0, 1, 3, 4]
S1 = [1, 2, 4, 5]
S2 = [3, 4, 6, 7]
S3 = [4, 5, 7, 8]
S4 = [0, 1]
S5 = [6, 7]
S6 = [1, 2]
S7 = [7, 8]

GF(2) rank of check matrix: 6

Number of attainable syndromes: 64

Logical-label support:

L = [0, 3, 6]

L is not in rowspace(H).

Kernel dimension: 3.

The minimum-weight zero-syndrome label-flipping error has weight 3.

## Noise parameter

The source parameter is:

p = 0.05

The actual independent binary X-error marginal is:

q_X = 2p/3 = 1/30

q_X, not p, is the binary marginal relevant to the exact population-risk
calculation.

## Exact iid reference

Enumerate all 2^9 = 512 binary error configurations.

For each error configuration e:

1. compute syndrome s = H e mod 2
2. compute logical label l = L e mod 2
3. assign exact iid probability

   P(e) = q_X^wt(e) * (1-q_X)^(9-wt(e))

For each attainable syndrome, sum exact probability mass separately for
logical labels 0 and 1.

The unique Bayes decision is the label with larger conditional mass.

There are no posterior ties among the 64 attainable syndromes.

Expected exact Bayes population LER at q_X = 1/30:

373445699 / 38443359375

approximately:

0.009714179641721282

The iid Bayes decision table is invariant for every:

0 < q_X < 1/2

Changing q_X within that interval changes population risk weights but not
the reference decision table.

The reference is described as the exact iid Bayes / minimum-weight coset
decision rule for this binary task.

## Exact learned-model adjudication

For a trained model, evaluate all 64 attainable syndromes exactly once and
extract its binary decision table.

Population LER is then computed exactly by summing probability mass over
all 512 error configurations.

For every model report:

- exact 64-bit table identity with the Bayes table
- number of differing attainable syndrome entries
- identities of differing syndrome entries
- exact population LER
- exact excess LER above Bayes

No sampled test set, bootstrap interval, or sampling-error margin is used
for Gate 1B oracle adjudication.

A table mismatch does not make the experiment invalid; it rejects exact
Bayes-table identity.

## Track A — Candidate-C replay identity

The original Gate-1 process did not persist Candidate C weights.

Attempt one deterministic reconstruction using the original Gate-1
dataset seed and training procedure.

Required execution environment is the frozen Gate-1 environment recorded
under:

provenance/LAMBDA0_GATE1_ENVIRONMENT.txt
provenance/LAMBDA0_GATE1_PIP_FREEZE.txt
provenance/LAMBDA0_GATE1_CONDA_EXPLICIT.txt

Do not retrospectively change PyTorch thread counts.

Expected discriminating replay anchors:

selected configuration: C
best epoch: 97
validation LER: 0.010600
validation BCE: 0.044843
sampled final LER: 0.009550

Replay identity verdict:

CONFIRM:
all recorded replay anchors reproduce under the frozen environment.

ABSTAIN:
one or more replay anchors fail to reproduce.

An ABSTAIN on Candidate-C identity does not alter the previously frozen
Gate-1 PASS and does not block Track B.

If replay CONFIRMs, persist and hash:

- model state_dict
- attainable-syndrome decision table
- exact population-risk result

## Track B — training-procedure capability

Training configuration is Candidate C:

batch_size = 256
learning_rate = 3e-4
max_epochs = 400
patience = 30
optimizer = Adam
loss = BCEWithLogitsLoss

Dataset size per run:

N = 100000

Split:

train = 60000
validation = 20000
final sampled test = not used for Gate-1B adjudication

The exact oracle evaluates population performance.

Use a full crossed seed panel:

data_seed in:
31001
31002
31003
31004

training_seed in:
41001
41002
41003
41004
41005

Total runs: 20.

data_seed controls synthetic training/validation-data generation.

training_seed controls:
- neural-network initialization
- minibatch shuffle order

No additional restart search or hyperparameter selection is permitted.

For each of the 20 runs record:

- data_seed
- training_seed
- best epoch
- validation LER
- validation BCE
- exact Bayes-table match
- number of differing table entries
- exact population LER
- exact excess population LER

Aggregate reporting must include:

- exact-table matches / 20
- median exact excess LER
- minimum exact excess LER
- maximum exact excess LER
- individual exact excess LER for all 20 runs

No pass/fail threshold on training reliability is introduced after seeing
these results. Gate 1B is a calibration/characterization experiment.

## Verdict separation

Structural/provenance validity uses:

ACCEPT / FLAG / ABSTAIN

Oracle-performance claims use:

CONFIRM / REJECT

Candidate-C replay identity may ABSTAIN if faithful reconstruction cannot
be established.

## Interpretation ceiling

Permitted external framing:

"We calibrated a learned-decoder training procedure against exact
population ground truth on an enumerable binary parity-decoding task of
algebraic distance 3."

Not permitted from this experiment alone:

- scalable decoder performance
- faithful surface-code decoding
- MWPM superiority
- PyMatching superiority
- quantum advantage
- production QEC performance
