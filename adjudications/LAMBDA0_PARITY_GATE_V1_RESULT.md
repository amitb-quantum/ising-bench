# Lambda=0 Learned-Decoder Parity Gate v1 — Result

Execution date: 2026-08-12

## Frozen inputs

Preregistration SHA-256:
5ba9c015a157f9d56bd148760b7a6eda76e581003c4da8a8c844c2925c677590

Amendment 001 SHA-256:
b5ed2fde55a61cacfe18e22ac3f5dcdf92a1cfe2f8fa898a3f23cc149771b180

Gate runner commit:
abe3167

Gate runner SHA-256:
231c51d3aa9e81dba927cb7f419c6718186e9822c20ddcf3b8133d870a2836b9

Raw result SHA-256:
76c752f85ec8897a9ce152ab184bc4ed0d3ce171d2ce0782d3b51a3969d68d50

## Candidate selection

Selected candidate: C

Validation LER:
0.010600

Validation BCE:
0.044843

Best epoch:
97

## Frozen final-test result

Learned decoder LER:
0.009550

Logistic-regression LER:
0.023400

Delta_LER:
-0.013850

Paired bootstrap one-sided U95:
-0.012400

Non-inferiority margin:
+0.005000

## Verdict

PASS

## Permitted interpretation

"The learned decoder reached prospective non-inferiority to a
logistic-regression baseline under the frozen lambda=0 synthetic task."

This result does not compare against MWPM, PyMatching, or a production
surface-code decoder.

No lambda > 0 result was evaluated as part of this gate.
