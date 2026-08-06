# FEAST publication reruns

Current execution counts and scientific dispositions are tracked in
[`RUN_STATUS.md`](RUN_STATUS.md).

This repository is the clean execution copy for the final FEAST article
reruns. The historical workspace at `FEAST_experiments` remains the audit
record; its repair attempts, archives, logs, and prior FEAST outputs are not
inputs to these workflows.

The original Git history is retained on `archive/legacy-experiments` and at
the annotated tag `legacy-experiments-20260630`. This clean line is a normal
descendant, so the cleanup is reviewable without rewriting the archive. See
[`ARCHIVE_POLICY.md`](ARCHIVE_POLICY.md) and the machine-readable
[`PUBLICATION_MANIFEST.json`](PUBLICATION_MANIFEST.json).
Current review figures and their exact numerical sources are listed in
[`FIGURE_SOURCE_MAP.md`](FIGURE_SOURCE_MAP.md).

## Active studies

| Study | Workflow | Fresh work required |
|---|---|---|
| 00 | Simulator benchmark | 12 reference-rank FEAST simulations and all metrics |
| 01 | Clustering | 81 simulations, 84 fixed-panel inputs, 252 method outputs |
| 02 | Alignment | 20 rotation inputs and 40 method outputs |
| 03 | Deconvolution | 6 simulations and 12 method outputs |
| 04 | Batch-effect removal | 12 batch inputs and 36 method outputs |
| 05 | 2D conditional transfer | 40 cross-slice plus 5 half-slice outputs |
| 06 | Conditional 3D stack | 93 held-out target-slice outputs |
| 07 | DevCCF 3D transfer | 158 E15.5 and 202 E18.5 z-level outputs |

Study 04 is the clean name for legacy Study 08. Publication figure builders
live in [`visualization/`](visualization/); only studies with complete,
validated clean outputs are added there.

## What may be reused

Only processed source datasets and the frozen external-simulator outputs used
by Study 00 may be materialized from the old workspace. The historical 12
FEAST OT-spatial outputs are removed from scope rather than regenerated or
relabelled. Each reused file must
match its declared SHA-256 before it is used. All FEAST-generated data,
downstream method outputs, metrics, and reports are regenerated under this
repository.

Local input hardlinks belong below each study's `data/local/` directory and
are ignored by Git. Publication cloud locations can replace those hardlinks
later without changing the scientific scripts.

## FEAST environment

Studies 00–05 use the wheel recorded in [`FEAST_BUILD.txt`](FEAST_BUILD.txt).
Studies 06 and 07 use the separately hash-pinned unreleased FEAST 1.1.0
log-domain repair candidate documented by their READMEs. Both clean FEAST
environments use supported Python 3.11.15 and NumPy 1.26.4. External methods
use the environments listed in
[`environments/README.md`](environments/README.md); those environments do not
expand FEAST's supported dependency range.

Before any full rerun:

```bash
python scripts/check_repository.py
python scripts/check_conditional_workflows.py
<candidate-python> scripts/test_conditional_workflows.py
<study-build-python> scripts/verify_feast_install.py
python -m pip check
python scripts/verify_rng.py \
  --input /path/to/one/real/article_input.h5ad \
  --input-artifact-id MERFISH_007 \
  --seed 2026 \
  --output validation/rng_article_gate_20260718.json
```

The RNG check launches three fresh processes with ambient NumPy seeds 1,
99991, and repeated 1. It requires identical output matrices from the fixed
public FEAST seed.

Studies 00–05 retain their recorded v1.0.2 provenance. Studies 06 and 07 were
freshly regenerated under the explicit repaired 1.1.0 candidate and retain
that separate lineage. No package release is unified or authorized by these
workflow results.

## Execution

There is no global experiment launcher. Follow each study README and run its
existing-style entry points in numerical order. Every command requires a new
output directory. Resume commands may skip only outputs that pass that study's
validation script.

Studies 00 and 01 use reference-rank spatial assignment and exact global gene
assignment. They must not use OT or the historical `PrefitSimulator` shortcut.

Studies 05–07 use the installed public conditional API (`fit_reference` and
`simulate_from_reference`) with explicit unified-OT settings and fail-closed
convergence. Study 05 pins FEAST 1.0.2; Studies 06/07 pin the repaired 1.1.0
candidate. Their complete scope, configuration changes, canary order, and
validation rules are recorded in
[`CONDITIONAL_RERUN_PLAN.md`](CONDITIONAL_RERUN_PLAN.md). Historical
conditional-OT H5ADs are audit evidence only and are never resumed or promoted.

If a fresh table materially changes a reported conclusion, preserve that
study's run and stop before integrating it into the publication results.
