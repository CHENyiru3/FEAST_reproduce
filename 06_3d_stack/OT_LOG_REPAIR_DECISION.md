# Study 06 log-domain OT repair decision

Decision date: 2026-07-20
Current status: package repair and full v5 preflight validated; CUDA generation
canaries pending

## Failure cause

The exact full-preflight blocker is the dense-gap-3 representative slice 076,
estimator seed 5027, label `13 CNU-HYa Glut`, with 116 source and 18 target
spots. Its transport problem SHA-256 under the rejected translation-invariant
contract is
`95b61bdf677e6e3ce68403d9d6e53cc4b567c7437e4840acad8d531cac36cc54`.

One source row has nearest `cost / epsilon = 1027.334`; its largest log-kernel
entry is `-1034.98`, so the entire exponential-kernel row underflows to zero.
This is not invalid input data and is not CUDA-specific.

## Exact solver evidence

- POT ordinary Sinkhorn: strict failure at iteration 0.
- POT translation-invariant Sinkhorn: strict failure at iteration 0.
- FEAST `sinkhorn_log`: converged in 454 iterations with final residual
  `9.9009e-6`, transport mass `0.876128`, and objective `1.24495915`.
- NumPy and Torch-CPU log-domain plans agree at relative L1
  `4.74e-15`.
- A tighter `1e-12` log-domain reference converged in 1,265 iterations. The
  declared-tolerance plan differs by relative L1 `4.897e-6`, with objective
  delta `1.06e-10`.
- KKT RMS for the declared solve is `4.92e-5`.

The comparison also sampled the six largest observed `max(cost)/epsilon`
problems. The exact blocker was the only ordinary-Sinkhorn failure; the
presence of a completely underflowed row/column, rather than the global ratio
alone, identifies the failure mode.

## Selected contract

Study 06 configuration v5 explicitly selects FEAST 1.1.0 candidate
`feast-ot-log-domain-cuda-repair-v2`, `sinkhorn_log`, float64, strict
nonconvergence handling, and CUDA generation. The assignment-randomness
preflight uses the identical solver on the explicitly declared Torch/CPU
backend. There is no automatic solver or device fallback.

The v4 full preflight completed all strict log-domain OT calls, so the original
numerical blocker is resolved for the reference-only gate. It did not authorize
the 93-output run: medium-gap assignment randomness changed from the frozen
`0.25` to `0.30`. The workflow stopped and preserved the complete plan.
Independent frozen-1.0.2 process repeats reproduced `0.30`, not the unpinned
historical `0.25`; this rules out the new solver/backend as the cause. Versioned
configuration v5 adopts `0.30`.

The fresh v5 preflight subsequently passed all three calibration gates and
froze 147 inputs plus 93 target assignments under
`outputs/preflight_otlog110_v5_20260720/`. Its plan SHA-256 is
`f4a852dafbadef60e6cdffb7616ab2523845cc78f7db89f57ed71f95b1fd0f33`
and its passed record SHA-256 is
`8b426b5b4924eab5db5ce8bc3ea7b6654b46b3950c53675b9ec9badd14fc57e9`.
The configuration file is byte-identical to the frozen copy. This closes the
Study 06 preflight blocker but does not by itself authorize article claims:
CUDA canaries, all 93 fresh outputs, independent validation, and new metrics
remain required.
