# Study 06: conditional 3D stack reconstruction

This directory cleanly regenerates the Allen Zhuang ABCA-1 semi-reference
stack benchmark. It produces 93 conditional FEAST slices: 49 for gap 3, 29
for gap 5, and 15 for gap 10. Historical outputs are never read or reused.

Current launch status: the exact POT underflow defect is repaired and the
historical medium-gap `0.25` value was classified as unpinned and
nonreproducible. Versioned configuration v5 declares the reproducible `0.30`
value. Its fresh full preflight passed, freezing 147 inputs and 93 target
assignments. CUDA canaries are the next gate, as recorded in
[`PREFLIGHT_BLOCKER.md`](PREFLIGHT_BLOCKER.md).

The target slice contributes only observed identity and geometry (`obs_names`,
`class`, `z`, `spatial`, `spatial_3d`, and gene names) during generation.
Target expression is reserved for the later `aggregate.py` evaluation stage.
Bracketing reference expression and explicitly declared reference-only label
donors are the sole expression inputs to FEAST.

The scientific contract is fail-closed:

- reference-only assignment-randomness preflight must reproduce 0.35, 0.30,
  and 0.35 for gaps 3, 5, and 10;
- unified OT uses 1,000 iterations, tolerance `1e-5`, a 25-million-pair block
  cap, and `transport_nonconvergence="raise"`;
- every saved transport record must contain positive convergence evidence;
- target spot order, coordinates, classes, z, and the 1,122-gene order must
  match exactly;
- no smoothing or z regularization is permitted;
- failed or interrupted work never becomes a final target artifact.

All FEAST stages must run from the clean installed wheel recorded by the
repository, not the mutable source checkout. Verify the interpreter first:

```bash
FEAST_PY=/path/to/clean-feast-environment/bin/python
REPRO_ROOT=..
CANDIDATE="$REPRO_ROOT/../FEAST/validation/package_builds/20260719_ot_log_repair_v2/provenance.json"
"$FEAST_PY" "$REPRO_ROOT/scripts/verify_feast_install.py" \
  --candidate-provenance "$CANDIDATE"
```

## 1. Freeze inputs and the target plan

Use a new output root. Preflight hashes all 147 source H5ADs, constructs the
93-target plan and donor support, runs the reference-only AR estimator, and
binds the result to the exact configuration and FEAST commit.

```bash
FEAST_PY=/path/to/clean-feast-environment/bin/python
DATA=/path/to/Allen_Zhuang_ABCA_1/h5ad
OUT=outputs/canary

$FEAST_PY preflight.py --data-dir "$DATA" --output-dir "$OUT"
```

If any AR differs from the declared value, preflight writes
`PREFLIGHT_FAILED.json` and stops. Do not override it; review the FEAST build,
input hashes, and estimator diagnostics.

## 2. Canaries and parallel shards

Run one declared target from each density first in the ignored canary root. A
target directory must not already exist; there is intentionally no reuse or
resume mode. The runner itself checks exact identity and positive convergence
before atomically exposing each canary.

```bash
$FEAST_PY run.py --output-dir "$OUT" --gap 3 --target-id 5
$FEAST_PY run.py --output-dir "$OUT" --gap 5 --target-id 6
$FEAST_PY run.py --output-dir "$OUT" --gap 10 --target-id 6
```

After reviewing the canary provenance, repeat preflight into the one publication
root, then shard production. Independent workers do not change original target
indices or seeds:

```bash
OUT=outputs/final
$FEAST_PY preflight.py --data-dir "$DATA" --output-dir "$OUT"
$FEAST_PY run.py --output-dir "$OUT" --gap 3 --shard-index 0 --shard-count 8
$FEAST_PY run.py --output-dir "$OUT" --gap 5 --shard-index 0 --shard-count 4
$FEAST_PY run.py --output-dir "$OUT" --gap 10 --shard-index 0 --shard-count 2
```

Launch the remaining shard indices as independent jobs. Shards must use the
same frozen preflight root. `.work/` holds interrupted or failed attempts;
only an atomically completed target directory is a candidate artifact.

## 3. Validate and evaluate

```bash
$FEAST_PY validate.py --output-dir "$OUT"
$FEAST_PY aggregate.py --output-dir "$OUT"
```

`validate.py` independently checks all 93 artifacts, exact target identity,
declared references/donors/weights/seeds, input and output hashes, count
validity, and every OT convergence record. `aggregate.py` is the first stage
allowed to read target expression. It atomically writes per-target and
per-density metrics, class–gene z trajectories, a deterministic within-slice
real-data split-half continuity baseline, score provenance, and an artifact
manifest under `evaluation/`. The 93 targets are ordered z levels, not
independent biological replicates; no across-target p-values are produced.

The historical comparison is a separate, hash-pinned review step. It does not
import historical code or accept historical H5ADs:

```bash
HISTORICAL=/path/to/pinned/cross_density_summary.csv
$FEAST_PY compare_historical.py \
  --output-dir "$OUT" \
  --historical-summary "$HISTORICAL"
```

This writes `evaluation/old_vs_new/` and declares the solver change from 200
to 1,000 iterations separately from comparable atomic metric changes. Every
comparison remains pending author review; it cannot update figures or
canonical decisions automatically.
