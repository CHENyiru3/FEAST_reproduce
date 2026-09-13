# Study 04B — real-slice incremental robustness

This directory implements the additional Study 04B protocol frozen in the
parent [`README.md`](../README.md). It is isolated from the validated Study 04A
outputs and refuses to overwrite a completed production root.

The primary matrix contains five conditions, three methods, and three method
seeds (45 GPU jobs). Raw `151675` is always the reference. Raw or
FEAST-perturbed `151676` is the query. Cross-section exact-spot retrieval is not
scored.

## Inputs

Create local hardlinks matching `data/input_checksums.csv`:

```text
data/local/151675.h5ad
data/local/151676.h5ad
```

The simulation stage must run with the clean installed FEAST wheel environment
declared in `config.yaml`.

## Execution

```bash
FEAST_PY=/path/to/envs/feast-prepublication-py311/bin/python3.11

$FEAST_PY run.py --dry-run
$FEAST_PY run.py --stage prepare
$FEAST_PY validate.py --stage prepared

# Production canaries: alpha 1.00, method seed 42.
$FEAST_PY run.py --stage methods --condition sim_1.00 --seed 42
$FEAST_PY validate.py --stage candidates --allow-incomplete

# Remaining matrix; verified canaries are skipped.
$FEAST_PY run.py --stage methods
$FEAST_PY score.py
$FEAST_PY validate.py --stage complete
```

Method workers run in their pinned external GPU environments. A candidate is
skipped only after its artifacts, provenance paths, input identity, fixed gene
panel, section order, CUDA execution, and zero-cross-section-edge contract have
validated. Invalid attempts are moved below the ignored production failure
directory; accepted candidates are never overwritten.

The score root contains atomic per-seed metrics, seed summaries, input-only
qualification diagnostics, and provenance. Composite scores and method winner
rankings are not produced.
