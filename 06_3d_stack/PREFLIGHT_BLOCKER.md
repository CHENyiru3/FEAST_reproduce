# Study 06 preflight blocker

Decision date: 2026-07-19  
Status: blocked before generation

## What failed

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

## Required repair

Keep `transport_nonconvergence="raise"`, epsilon, and tolerance unchanged.
Do not average attempts or silently discard rare labels. FEAST needs a
versioned, declared estimator repair—such as stabilized unbalanced Sinkhorn
and label/cost diagnostics, plus an explicit scientifically justified minimum
per-half label support contract. Rebuild and pin the wheel, then rerun the full
preflight into a new root. Gap-3, gap-5, and gap-10 AR decisions remain
unverified until that succeeds reproducibly.

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
