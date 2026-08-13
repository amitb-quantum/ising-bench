# Lambda=0 Learned-Decoder Parity Gate v1
## Prospective Amendment 001 — Statistical and Determinism Specification

Date: 2026-08-12

This amendment was frozen before any Gate-1 candidate training was executed.

It supplements:

  prereg/LAMBDA0_PARITY_GATE_V1.md

Original preregistration SHA-256:

  5ba9c015a157f9d56bd148760b7a6eda76e581003c4da8a8c844c2925c677590

Historical reproduced baseline:

  learned LER  = 0.09025
  logistic LER = 0.02225
  Delta LER    = +0.06800

Baseline source commit:

  34fda3ff1bb38387cf1c869c91a01a763f96d4e5

Baseline source SHA-256:

  e61f028b7f3604f3f5025e65fd06f33debea76e6e5ff81117a12776d58b5835b

Baseline result SHA-256:

  f8c59588db8fcb550e77bf1e45511ce2ab35d069b8a990646bfb682ae8814dea


## 1. Primary estimand

For final-test example i define:

  e_L(i) = 1 if learned decoder is wrong, else 0
  e_B(i) = 1 if logistic baseline is wrong, else 0
  d(i)   = e_L(i) - e_B(i)

The paired logical-error-rate difference is:

  Delta_LER = mean_i d(i)

Positive Delta_LER means the learned decoder is worse.


## 2. Non-inferiority margin

The prospectively fixed absolute non-inferiority margin is:

  delta_NI = +0.005

The learned decoder is considered non-inferior only if the one-sided
95% upper confidence bound for Delta_LER is <= +0.005.


## 3. Paired confidence-bound procedure

Use a paired nonparametric bootstrap over final-test examples.

Final-test size:

  N_test = 20000

Bootstrap resamples:

  B = 10000

Bootstrap RNG:

  numpy.random.default_rng(20260812)

For each bootstrap replicate:

  - sample N_test indices with replacement
  - preserve learned/baseline pairing
  - compute mean d(i) on the resampled indices

Define:

  U95 = 95th percentile of the 10000 bootstrap Delta_LER estimates

using:

  numpy.quantile(bootstrap_deltas, 0.95, method="linear")


## 4. Gate verdict

PASS iff:

  U95 <= +0.005

Otherwise:

  FAIL

A FAIL after exhaustion of the frozen training-search budget terminates
the correlated-noise calibration program under this gate.

No lambda > 0 superiority result may be interpreted if Gate 1 FAILs.


## 5. Dataset generation and split

Generate exactly one synthetic lambda=0 dataset:

  N = 100000
  physical-error parameter p = 0.05
  data-generation seed = 1234

Use deterministic contiguous partitions after generation:

  samples 0:60000       train
  samples 60000:80000   validation
  samples 80000:100000  final test

The final-test partition must remain inaccessible to candidate selection.


## 6. Training determinism

Gate-1 execution is CPU-only.

Set:

  numpy seed = 1234
  torch seed = 1234

For minibatch shuffling use a dedicated PyTorch generator seeded with:

  1234

All candidates receive the identical train, validation, and final-test
partitions.

No additional random restart search is permitted.


## 7. Candidate selection

Evaluate only Candidates A, B, and C defined in the original
preregistration.

Candidate selection uses validation data only.

Primary selection criterion:

  lowest validation LER

If validation LER is exactly tied:

  choose lowest validation BCE loss

If still tied:

  choose in fixed order A, then B, then C

The final-test set is evaluated exactly once, after candidate selection.


## 8. Early stopping

Validation is evaluated once per epoch.

A checkpoint is considered improved when:

  validation LER decreases

or, if validation LER is exactly tied:

  validation BCE loss decreases.

Patience is counted in consecutive epochs without checkpoint improvement.

The best validation checkpoint, not the final epoch, is used for the
single final-test evaluation.


## 9. Interpretation ceiling

PASS permits only the statement:

  "The learned decoder reached prospective non-inferiority to a
  logistic-regression baseline under the frozen lambda=0 synthetic task."

FAIL permits only the statement:

  "The learned decoder did not reach prospective non-inferiority to the
  logistic-regression baseline under the frozen lambda=0 synthetic task.
  No correlated-noise superiority experiment was undertaken."

This gate does not compare against MWPM, PyMatching, or a production
surface-code decoder.

