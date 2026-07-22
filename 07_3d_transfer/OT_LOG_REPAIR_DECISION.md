# Study 07 log-domain OT repair decision

Decision date: 2026-07-19  
Current status: exact blocker, one maximum-size block, and the fresh E18.5
calibration validated; workflow CUDA canaries pending

## Failure cause

The rejected translation-invariant candidate failed on E18.5 label `Septum`,
reference `GSM8323034_E18M_5`, with 363 source and 269 target spots. Its problem
SHA-256 is
`38b847acef034d2170764c57734a33e82815f3be2514a17ca044280574dba5e3`.
NumPy and CUDA costs were bit-identical and both methods failed at iteration
89, so changing the device or silently falling back to CPU could not repair
the problem.

## Exact solver evidence

- On the 363-by-269 failure block, FEAST `sinkhorn_log` and converged ordinary
  POT Sinkhorn both required 441 iterations with final residual
  approximately `9.921829e-6`.
- Log-domain versus ordinary relative plan L1 is `5.79e-15` on NumPy and
  `1.40e-15` on CUDA.
- CUDA log-domain runtime was approximately 0.270 seconds on the RTX A6000.
- On the E15.5 z-index 12 first 24,994,980-pair block, both methods converged
  in 473 iterations with final residual approximately `9.970742e-6`.
- Their relative plan L1 is `2.11e-13` on NumPy and `2.69e-15` on CUDA.
- CUDA log-domain runtime for that maximum-size block was approximately 3.253
  seconds. NumPy log-domain runtime was approximately 326.89 seconds, so
  publication generation remains explicitly CUDA-bound.

## Selected contract

Study 07 configuration v5 explicitly selects FEAST 1.1.0 candidate
`feast-ot-log-domain-cuda-repair-v2`, `sinkhorn_log`, float64, strict
nonconvergence handling, and CUDA generation. E18.5 assignment-randomness
calibration uses the same solver on the separately declared Torch/CPU backend.
There is no automatic method fallback and no silent CPU fallback.

Ordinary POT Sinkhorn passed a bounded nine-block Study 07 panel and was about
4.6 times faster than log-domain transport on the tested 25-million-pair CUDA
block. It is therefore a defensible sensitivity implementation, but it is not
selected for the 79,615-block publication workflow: the independent Study 06
block proves that a valid FEAST problem can contain a completely underflowed
ordinary kernel row, and the bounded Study 07 panel cannot exclude that failure
elsewhere. `sinkhorn_log` solves the same mathematical objective, matches
ordinary solutions where ordinary converges, removes that known numerical
failure class, and remains feasible on CUDA. This deliberately accepts the
recorded runtime cost in exchange for one fail-closed implementation throughout
the article run.

The expression-free E15.5/E18.5 blueprints retain their stable independent
blueprint contract. The exact-block evidence does not authorize the full
79,615-block workflow: bounded E15.5/E18.5 CUDA canaries must pass first.

## Calibration result

The fresh reference-only E18.5 calibration completed with assignment
randomness `0.5`, identical to the historical v2 calibration. The new
calibration SHA-256 is
`cb5f54537332fc15ec626cf5afc8c10664781bc3364543ec87c8d076d533c445`;
its solver/configuration manifest SHA-256 is
`92f35fd29f9f1496480e741487f69c07ef1c0f5590d7ed803708faa7a870d332`.
The manifest binds the five named reference hashes, public seed 2026, exact
wheel/source/provenance identity, full solver configuration, strict residual
gate, and calibration output hash. The absence of a retained internal
per-problem diagnostic table is recorded explicitly rather than hidden.
