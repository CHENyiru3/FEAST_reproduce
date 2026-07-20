# Study 04 — batch-effect removal

This is the clean rerun of legacy Study 08. It regenerates every FEAST input,
all GraphST/STAMP/scVI outputs, and all atomic metrics. No historical artifact
is reused as a computational input. After fresh scoring, the report reads one
hash-pinned 60-row historical atomic table only to create a one-to-one
old-versus-new comparison; the invalid historical composite is recorded by
path and hash but is never compared numerically.

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

If a declared zero-library extrapolation run reaches a finite plateau and then
develops a non-finite Pyro variational parameter before the standard
20-epoch early-stop patience fires, the narrow unsupervised policy in
`stamp_numerical_retry_v1.yaml` permits a retry with patience 10. Data, graph,
model, seed, learning rate, and CUDA execution remain unchanged. The original
failure stays noncanonical, and the accepted candidate records the retry
configuration hash and stopping diagnostics. This policy is eligible only for
the two listed alpha-1.50 conditions and makes no hyperparameter-stability
claim. Scoring and validation reject patience 10 unless the candidate binds
the pinned policy, matching preserved failure and disposition hashes, the
single patience change, and the unchanged seed/data/graph/model/LR/CUDA
contract. Ordinary and eligible candidates completed with the standard
patience 20 remain valid without a retry declaration.

Finally build and validate the fresh report:

```bash
HISTORICAL_ATOMIC=/path/to/frozen/consolidated_atomic_metrics.csv
INVALID_COMPOSITE=/path/to/frozen/batch_correction_results.csv

$FEAST_PY score.py \
  --historical-atomic "$HISTORICAL_ATOMIC" \
  --invalid-composite "$INVALID_COMPOSITE"
$FEAST_PY validate.py \
  --historical-atomic "$HISTORICAL_ATOMIC" \
  --invalid-composite "$INVALID_COMPOSITE"
```

Both comparison paths are required explicitly so relocated checkouts do not
depend on a machine-specific repository layout. The historical atomic table
and invalid composite must still match their pinned SHA-256 values; the scorer
and independent validator record and cross-check the resolved paths and hashes.

`score.py` produces an exact 60-row key matrix: 48 all-spot rows (including 12
GraphST PCA-10 representation-sensitivity rows) and 12 paired common-support
sensitivity rows. The paired-support cells are fixed before scoring:
`diagonal_affine/1.25` removes one named query-zero spot and its reference
pair, `diagonal_affine/1.50` removes three, and `shift_only/1.50` removes two.
The reference fixed panel must have zero raw-zero rows. All-spots rows record
zero removals. A zero-variance named scVI expression pair is a validation
failure, not a silently dropped correlation.

The metric directory contains:

- `atomic_metrics.csv` — the strict 60-row atomic table;
- `primary_summary.csv` — 12 method/mode/alpha-stratum summaries;
- `common_support_sensitivity.csv` — 12 paired-support versus all-spot rows;
- `graphst_pca20_vs_pca10_sensitivity.csv` — the explicit 15-row GraphST
  representation sensitivity;
- `support_audit.csv` — exact query-zero IDs, pair counts, and artifact hashes;
- `old_vs_new_atomic_metrics.csv` — 60 one-to-one rows across all 10 atomic
  metrics, classified as RNG/API repair plus a full rerun;
- `old_vs_new_simulation_hashes.csv` — all 14 fresh simulation hashes against
  their pinned prior hashes, classified as cross-stage RNG/API repair plus a
  full rerun and explicitly not as repeat-drift evidence;
- `historical_composite_disposition.csv` — nonnumeric disposition only.

Validation independently reloads the simulation, NPZ, and result-H5AD
artifacts and recomputes all 60 atomic rows with validator-owned formulas. It
exact-compares that reconstruction to `atomic_metrics.csv` before using the
reported table to validate summaries, sensitivities, or old-versus-new files.

Alpha above 1 is kept as extrapolation. Composite scores, rankings, and winner
claims are prohibited. There is no active Study 04 manuscript claim, so fresh
numeric changes are author-review evidence and do not cause exit status 2.

To repeat the study, pass a nonexistent root to every command, for example
`--output-dir outputs/repeat_1` for `run.py`, then use the corresponding
`--run-dir` for `score.py` and `validate.py`. Never reuse a completed root.

## External-environment limitation

GraphST, STAMP, and scVI run in dedicated external environments. GraphST's
NumPy 1.23.x and scVI's NumPy 2.x are outside FEAST's supported `>=1.24,<2`
range; FEAST is not imported or executed in either worker, and every candidate
records that limitation explicitly. STAMP's isolated publication environment
uses NumPy 1.26.x. Every launch requires `pip check` and verified CUDA
execution. Validation additionally requires scVI's finite training history,
model state, CUDA device, and decoded expression under the common reference
batch. These external stacks are not evidence that FEAST supports them. The
validated result remains a supplementary candidate until its fresh metrics
receive scientific review.
