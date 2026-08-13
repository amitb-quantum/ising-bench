# Lambda=0 Learned-Decoder Parity Gate v1

Date frozen: 2026-08-12
Purpose: establish learned-decoder competence under independent noise before
any correlated-noise calibration experiment.

## Scope

This is an instrument-calibration experiment on the existing synthetic
syndrome task in ising-bench. It is not an MWPM/PyMatching comparison and
does not establish production surface-code decoder performance.

## Null condition

Correlation parameter lambda = 0.
Independent per-data-qubit X-error sampling under the existing generator.

Physical-error parameter: p = 0.05
Seed: 1234
Dataset size: 100000

Split:
- train: 60%
- validation: 20%
- final test: 20%

The final test partition MUST NOT be used for candidate selection, training
decisions, hyperparameter selection, or early stopping.

## Comparator

sklearn LogisticRegression(max_iter=2000, C=1.0)

## Learned architecture

Keep the existing architecture unchanged:
input -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(1)

Loss: BCEWithLogitsLoss
Optimizer family: Adam

## Frozen training search

Candidate A:
- batch_size = 256
- learning_rate = 1e-3
- max_epochs = 200
- patience = 20

Candidate B:
- batch_size = 512
- learning_rate = 1e-3
- max_epochs = 200
- patience = 20

Candidate C:
- batch_size = 256
- learning_rate = 3e-4
- max_epochs = 400
- patience = 30

Select exactly one candidate using validation LER only.
No architecture expansion or additional hyperparameter search is permitted
under this gate.

## Primary endpoint

Delta_LER = LER_learned - LER_logistic

Both decoders are evaluated on the identical frozen final-test examples.

Use a paired 95% confidence interval for Delta_LER.

Non-inferiority margin:
delta_NI = +0.005 absolute LER.

## Verdict

PASS:
upper endpoint of paired 95% CI(Delta_LER) <= +0.005

FAIL:
upper endpoint > +0.005 after the frozen candidate search is exhausted

No correlated-noise (lambda > 0) sweep may be interpreted unless this gate
PASSes.

## Result-language ceiling

If PASS:
"The learned decoder reached prospective non-inferiority to a
logistic-regression baseline under the frozen lambda=0 synthetic task."

If FAIL:
"The learned decoder did not reach prospective non-inferiority to the
logistic-regression baseline under the frozen lambda=0 synthetic task.
No correlated-noise superiority experiment was undertaken."

Neither outcome constitutes a comparison against MWPM or PyMatching.
