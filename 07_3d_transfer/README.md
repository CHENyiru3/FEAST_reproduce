# Study 07 — complete DevCCF 3D conditional transfer

This clean workflow replaces legacy Study 06 for publication reruns. It does
not import the historical experiment repository and never overwrites a result.
It transfers GSE269617 expression onto expression-free DevCCF blueprints for
both complete coordinate-system axes:

| Target age | References | z range | step | levels | blueprint spots |
|---|---:|---:|---:|---:|---:|
| E15.5 | 10 E14M/E14F slices | -4.12 to -0.98 | 0.02 | 158 | 2,330,927 |
| E18.5 | 5 E18M slices | -5.56 to -1.54 | 0.02 | 202 | 5,213,461 |

All references share the retained 550-gene panel. Reference fitting uses the
historical E15.5 thresholds `min_gene_spots=1`, `min_gene_mean=0.0`, and
`max_gene_zero_prop=1.0` for both ages. This preserves the identical ordered
550-gene panel while making E18.5 follow the E15.5 design. The blueprint
contains only exact DevCCF XY,
region, voxel identity, and z; target expression is never read or fabricated.
No cross-z smoothing is part of the publication candidate.

## Fixed scientific contract

- FEAST 1.0.2 wheel SHA-256
  `9dd912d883a03d51ed7105cecd914cf8f7f25f5355f57dfde1c77b6fef0b056b`
  from source commit `68816e5c1862a6fa2a49bc30609d617c7fa4b449`.
- Conditional public API: `FEAST.de_novo.fit_reference` followed by
  `FEAST.de_novo.simulate_from_reference`.
- Empirical-reference marginals, public seed `2026 + original sorted z index`.
- Sinkhorn iterations 1,000, tolerance `1e-5`, pair cap 25,000,000, and
  `transport_nonconvergence="raise"`.
- E15.5 assignment randomness is fixed at the author-declared value 0.4.
- Both ages use the historical E15.5 permissive reference filter and retain
  the same ordered 550-gene panel.
- E18.5 assignment randomness must first be estimated from the fitted five-
  reference cohort and frozen in a hash-bound calibration record. The target
  blueprint is not accessible to calibration.
- Every retained solver record must explicitly report convergence and a finite
  final error below its stop threshold.

Run with the clean wheel interpreter documented in `../environments/README.md`.
Do not put the mutable FEAST source checkout on `PYTHONPATH`.

## Preparation

From this directory:

```bash
python prepare.py --preflight-only
python prepare.py --age both
python calibrate.py
```

Preparation validates the 18 pinned input hashes, inspects all 15 H5AD
contracts, regenerates both NIfTI-derived blueprints, and writes age-specific
geometry/identity manifests below ignored `data/local/`.

## Canaries and shards

Run canaries before broad execution. These cover both endpoints, a central
slice, and the densest level.

```bash
python run.py --age E15.5 --shard-id canary --canary
python run.py --age E18.5 --shard-id canary --canary
```

The executable selection is derived from the prepared blueprint: endpoints,
the lower middle level, and the densest level. E15.5 additionally runs the
historical failure level `z=-3.88` (index 12) first. The currently prepared
canary sets are E15.5 `[12, 0, 78, 47, 157]` and E18.5
`[0, 100, 62, 201]`.

Only after each canary has positive convergence evidence should the remaining
indices be divided into non-overlapping shards. For example, the ranges below
avoid all age-specific canary indices and can run in parallel:

```bash
python run.py --age E15.5 --shard-id low --start-index 0 --stop-index 79 --exclude-indices 0 12 47 78
python run.py --age E15.5 --shard-id high --start-index 79 --stop-index 158 --exclude-indices 157
python run.py --age E18.5 --shard-id low --start-index 0 --stop-index 101 --exclude-indices 0 62 100
python run.py --age E18.5 --shard-id high --start-index 101 --stop-index 202 --exclude-indices 201
```

Shard directories are single-use. Each completed slice receives an atomic
record immediately, so if a later slice fails, leave the completed records
active and use a new shard ID only for missing indices. Archive any unrecorded
temporary/partial file before consolidation. Never rerun a recorded index into
a second active shard: consolidation rejects duplicates.

## Consolidation and decision

```bash
python consolidate.py --age E15.5
python consolidate.py --age E18.5
python validate.py --age both
```

Consolidation requires one and only one validated artifact for every original
z index and hardlinks it into `outputs/final/<age>/`; it does not duplicate the
large H5AD payload. Final validation reopens all 360 slices and checks exact
spot IDs, XY/XYZ geometry, regions, gene order, counts, matrix/file hashes,
seeds, configuration lineage, and positive convergence. Passing both volumes
writes a candidate-gate decision with `publication_canonical=false`; canonical
status remains pending article-level continuity metrics, visualization, and
author review.

The workflow deliberately contains no metrics or figures yet. Article-level
continuity metrics and visualization should be added only after this complete
identity/convergence gate succeeds.
