# Study 06 preflight blocker

Decision date: 2026-07-20
Completion update: 2026-07-31
Status: closed for declared v5 scope; 93/93 CUDA outputs validated and evaluated

## 2026-07-31 workflow completion

The repaired v5 contract passed its CUDA canaries and complete production
scope. Independent validation reopened all 93 outputs, confirmed exact target
identity, and found positive convergence evidence for every retained transport
record. The validation summary SHA-256 is
`c41bb104a06a6711070f1888533422ba11596d7e7bebf4fa044bc1aaeabf4a37`;
the evaluation provenance SHA-256 is
`2bb9d66f4fc3ad20c4e9683ba7fc5db4dedbb52c86b06ed515e1e9d5cea6c85b`.
The preflight blocker is closed for configuration
`study06-allen-abca1-conditional-stack-v5`. This does not generalize to other
solver configurations and does not itself authorize publication wording.

## 2026-07-20 repair update

FEAST candidate `feast-ot-log-domain-cuda-repair-v2` resolves the exact
iteration-zero defect. On the former 116-by-18 failure block, `sinkhorn_log`
converges in 454 iterations; NumPy and Torch-CPU plans agree at relative L1
`4.74e-15`, and the declared solution agrees with a tighter reference solve.

A fresh full v4 preflight then completed every strict OT call. It stopped at
the next scientific gate because medium-gap assignment randomness was `0.30`,
not the frozen historical `0.25`:

- dense gap 3: estimates `0.35, 0.35, 0.25`, median `0.35` (matched);
- medium gap 5: estimates `0.35, 0.30, 0.00`, median `0.30` (mismatch);
- sparse gap 10: estimates `0.35, 0.45, 0.20`, median `0.35` (matched).

The failed root is preserved under
`.work/archive/preflight_otlog110_torchcpu_ar_mismatch_20260720/` with plan
SHA-256
`ff70083a1115a5ad83af25560bdc2dfbf44259ed92197aa76a965eb285c80cce`.
No generation was launched. A bounded repeat/method/backend audit must classify
the `0.05` medium-density change before the expected value can be revised or
retained.

The change was then classified independently. Three fresh runs and two
additional process-level repeats under the still-installed frozen FEAST 1.0.2
environment returned `0.35, 0.30, 0.00`, exactly matching the log-domain v4
preflight. The saved historical record returned `0.35, 0.25, 0.10` but contains
no FEAST version, commit, wheel hash, or configuration hash and cannot be
reproduced under its declared seed/settings. The `0.30` value is therefore not
a log-solver or Torch-backend change; v5 declares the reproducible median and
required a completely fresh full preflight before generation.

That v5 preflight passed on 2026-07-20. It froze all 147 inputs and all 93
target assignments in
`outputs/preflight_otlog110_v5_20260720/`. The three density gates were:

- dense gap 3: observed `0.35000000000000003`, expected `0.35`;
- medium gap 5: observed `0.30000000000000004`, expected `0.30`;
- sparse gap 10: observed `0.35000000000000003`, expected `0.35`.

The independently repeated medium-gap audit found identical candidate scores
and generated-matrix hashes across two process-level log/Torch-CPU repeats.
Ordinary NumPy Sinkhorn, the current source tree, the candidate wheel, and the
installed frozen FEAST 1.0.2 wheel all reproduced the same `(0.35, 0.30,
0.00)` tuple. Its retained report is
`.work/audits/assignment_randomness_medium_20260720/REPORT.md`, SHA-256
`12663b050425122fcfd453582f454a00f4faf24e33fc3d7df5e98acf3df0d096`.

Fresh v5 evidence SHA-256 values:

- frozen configuration:
  `a554d9acff7790650edea45b0fd2bb78ad40cab87cb5f38cb14a06d5efc4c1de`;
- input manifest:
  `8d3ac86215cee6782e6d64a122bfa9b6aca7f04aff9412a5c3adcca42598036a`;
- plan:
  `f4a852dafbadef60e6cdffb7616ab2523845cc78f7db89f57ed71f95b1fd0f33`;
- passed preflight record:
  `8b426b5b4924eab5db5ce8bc3ea7b6654b46b3950c53675b9ec9badd14fc57e9`.

At that checkpoint no generation had been launched. The later 2026-07-31
completion record above closes the CUDA generation and independent-validation
gates.

## Historical numerical failure

The reference-only assignment-randomness gate fails inside the first dense
gap-3 estimate (slice 004, public seed 5026). POT emits
`divide by zero encountered in divide` and `Numerical errors at iteration 0`;
FEAST correctly converts that warning into `OptimalTransportError` under the
required fail-closed policy.

The initial full preflight and two independent one-thread full repeats all
failed before writing an assignment-randomness decision or `plan.json`. No
canary or production generation was launched.

## What was ruled out

- All 147 inputs and all 735 metadata hashes validate.
- Slice 004 contains finite nonnegative integer counts, no empty spots or
  genes, unique finite 2D coordinates, and exact `spatial_3d[:, :2]` agreement.
- `X` and `layers["counts"]` are exactly equal.
- The wrapper uses the declared counts layer and removes the unused 3D key for
  the within-slice 2D estimator.
- Four isolated exact estimator reproductions returned 0.35 with no warning.
  Instrumentation observed 99/99 converged solves with final residuals below
  `1e-5`.

The failure is therefore a preflight-context POT numerical-stability problem,
not an input-hash, coordinate, count-layer, or declared-AR mismatch. Very small
half-slice class supports (as low as 3 reference to 1 target spot) create
extreme unstabilized scaling and plausibly amplify the solver instability.

## Historical required repair and disposition

Keep `transport_nonconvergence="raise"`, epsilon, and tolerance unchanged.
Do not average attempts or silently discard rare labels. FEAST needs a
versioned, declared estimator repair with strict diagnostics. The pinned FEAST
1.1.0 candidate supplies the log-domain solver and the v5 preflight above
closes this requirement without changing epsilon, tolerance, rare-label
support, or strict nonconvergence handling.

Evidence:

- Tracked config SHA-256:
  `4211c061e0e286bf8ce7c22a6e72ae321e38cb91e6a2c0331ecb6a44d64c60d6`
- Full input-manifest SHA-256:
  `8d3ac86215cee6782e6d64a122bfa9b6aca7f04aff9412a5c3adcca42598036a`
- Initial failure record SHA-256:
  `ebcfcc7dbf5b0ffef80515f8b27caedd8a7b4486e2ffc6d7a944e2c035c6a9de`
- Preserved attempts: `.work/preflight_initial_20260719/`,
  `.work/preflight_repeat_a_20260719/`, and
  `.work/preflight_repeat_b_20260719/`.
