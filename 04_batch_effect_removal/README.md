# Study 04 — batch-effect removal

This is the clean rerun of legacy Study 08. It regenerates every FEAST input,
all GraphST/STAMP/scVI outputs, and all atomic metrics. It does not read prior
Study 08 results, repair directories, audits, figures, or metric tables.

## Inputs

Place local hardlinks at:

```text
data/local/151673.h5ad   # clean reference and fixed-panel source
data/local/151508.h5ad   # query used only to characterize the deformation
```

Both files are ignored by Git. Edit only the path entries in `config.yaml` if
the data or external environments live elsewhere. The public seed is 42.

Study 04 contains 12 nonzero comparison conditions: two deformation modes by
six alpha values. The simulation stage also writes one alpha-zero baseline
for each mode (14 H5AD files total). Those two baseline count matrices must be
identical; keeping both paths preserves the verified per-mode layout used by
the method wrappers. Each method therefore runs exactly 12 jobs, not 14.

## Run

Set `FEAST_PY` to the Python interpreter containing the clean wheel recorded in
the repository-level `FEAST_BUILD.txt`, then prepare the inputs:

```bash
FEAST_PY=/path/to/clean-feast-wheel-env/bin/python
$FEAST_PY run.py --dry-run
$FEAST_PY run.py --stage simulate
$FEAST_PY run.py --stage panel
```

The simulation uses only the public `FEAST.characterize_batch`,
`FEAST.BatchDeformation`, and `FEAST.simulate_batch_effect` API. It passes the
public seed directly; run the repository-level ambient-RNG gate before the
production rerun.

Run the GPU methods one at a time so progress is easy to inspect:

```bash
$FEAST_PY run.py --stage methods --method GraphST
$FEAST_PY run.py --stage methods --method STAMP
$FEAST_PY run.py --stage methods --method scVI
```

The dispatcher invokes the configured external Python for each method and
requires both a clean `pip check` and actual CUDA execution; CPU fallback is
rejected. It skips only a fully hash-verified candidate. Failed or incomplete
attempts move below `outputs/final_rerun/failures/`; accepted outputs are never
silently replaced.

Finally build and validate the fresh report:

```bash
$FEAST_PY score.py
$FEAST_PY validate.py
```

`score.py` produces 60 rows: 36 all-spot primary rows, 12 GraphST PCA-10
representation-sensitivity rows, and 12 paired common-support sensitivity
rows. Alpha above 1 is kept as extrapolation. Composite scores, rankings, and
winner claims are prohibited.

To repeat the study, pass a nonexistent root to every command, for example
`--output-dir outputs/repeat_1` for `run.py`, then use the corresponding
`--run-dir` for `score.py` and `validate.py`. Never reuse a completed root.

## External-environment limitation

GraphST, STAMP, and scVI run in dedicated external environments. GraphST's
NumPy 1.23.x and scVI's NumPy 2.x are outside FEAST's supported `>=1.24,<2`
range; FEAST is not imported or executed in either worker, and every candidate
records that limitation explicitly. STAMP's isolated publication environment
uses NumPy 1.26.x and has passed both `pip check` and a real CUDA tensor
preflight; each method launch repeats the dependency gate. These external
stacks are not evidence that FEAST supports them. The validated result remains
a supplementary candidate until its fresh metrics receive scientific review.
