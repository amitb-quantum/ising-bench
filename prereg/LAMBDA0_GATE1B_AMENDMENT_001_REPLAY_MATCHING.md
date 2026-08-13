# Lambda=0 Gate 1B
## Prospective Amendment 001 — Replay Matching Semantics

Date: 2026-08-12

This amendment was frozen before Candidate-C replay execution.

It supplements:

  prereg/LAMBDA0_GATE1B_EXACT_ORACLE_V1.md

## Purpose

Define unambiguous matching semantics for Track A Candidate-C replay.

The original Candidate-C model weights were not persisted. Therefore a
successful replay cannot establish byte-identical identity with the lost
state_dict. It can establish that a deterministic reconstruction conforms
to the prospectively recorded replay anchors under the frozen execution
environment.

## Required replay environment

Use the frozen Gate-1 environment without retrospectively changing
PyTorch thread counts.

Expected:

  Python 3.11.10
  torch 2.5.1+cu121
  numpy 2.4.6
  sklearn 1.8.0
  torch intra-op threads = 24
  torch inter-op threads = 24

Execution remains CPU-only.

## Source integrity

Before replay require:

  src/run_lambda0_parity_gate.py
  SHA-256 =
  231c51d3aa9e81dba927cb7f419c6718186e9822c20ddcf3b8133d870a2836b9

  src/ising_benchmark.py
  SHA-256 =
  e61f028b7f3604f3f5025e65fd06f33debea76e6e5ff81117a12776d58b5835b

Original raw Gate-1 result:

  adjudications/LAMBDA0_PARITY_GATE_V1_RAW_RESULT.json

Expected SHA-256:

  76c752f85ec8897a9ce152ab184bc4ed0d3ce171d2ce0782d3b51a3969d68d50

## Replay-anchor matching

The replay anchors were prospectively recorded in the Gate-1B
preregistration to six decimal places.

CONFIRM requires all of the following:

  selected configuration = C        exact
  best epoch             = 97       exact
  validation LER         = 0.010600 rounded to 6 decimal places
  validation BCE         = 0.044843 rounded to 6 decimal places
  sampled final LER      = 0.009550 rounded to 6 decimal places

Python round-to-six-decimal behavior is NOT used for adjudication.
Comparison is performed using formatted fixed-point decimal strings:

  format(value, ".6f")

against the literal expected strings above.

The following preserved raw values are diagnostics, not additional
identity gates:

  epochs_executed = 127
  validation BCE raw = 0.04484331235289574
  final learned BCE raw = 0.04164905846118927
  final logistic LER = 0.0234
  Delta LER = -0.01385
  bootstrap U95 = -0.0124

## Verdict semantics

CONFIRM means:

  "A deterministic Candidate-C reconstruction under the frozen
  environment reproduced all prospectively declared replay anchors."

CONFIRM does NOT mean:

  "The reconstructed state_dict is proven byte-identical to the lost
  original Candidate-C state_dict."

ABSTAIN means:

  one or more required replay anchors failed to reproduce.

An ABSTAIN does not alter the previously frozen Gate-1 PASS and does not
block Track B capability characterization.

If CONFIRM is obtained, persist and hash:

  - reconstructed state_dict
  - 64-entry attainable-syndrome decision table
  - exact population-risk adjudication
