# Study 04 — batch-effect removal

This is the clean rerun of legacy Study 08. It regenerates every FEAST input,
all GraphST/STAMP/scVI outputs, and all atomic metrics. No historical artifact
is reused as a computational input. After fresh scoring, the report reads one
recorded 60-row historical atomic table only to create a one-to-one
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
claim. Scoring and validation reject patience 10 unless the candidate records
the declared policy, matching preserved failure and disposition evidence, the
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
and invalid composite are identified by their declared paths; the scorer and
independent validator record and cross-check the resolved paths and semantics.

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
- `support_audit.csv` — exact query-zero IDs, pair counts, and artifact paths;
- `old_vs_new_atomic_metrics.csv` — 60 one-to-one rows across all 10 atomic
  metrics, classified as RNG/API repair plus a full rerun;
- `old_vs_new_simulation_cells.csv` — all 14 fresh simulation cell counts against
  their prior records, classified as cross-stage RNG/API repair plus a
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

## Study 04B — real-slice incremental robustness

Study 04B is an additional, isolated experiment under
`real_slice_robustness/`. It does not replace or mutate the validated paired
pseudo-slice workflow above. Its scientific question is narrower than an
unqualified real-data batch-correction claim:

> When two real DLPFC sections already contain natural biological and
> technical differences, how stable are GraphST, STAMP, and scVI when a
> controlled additional marginal perturbation is applied to one section?

The fixed reference is raw section `151675`; the query is section `151676`.
They are different Visium sections/replicates from donor `Br8100`. They have no
biologically paired spots. Reused Visium barcode strings are therefore
prefixed with the section ID, and no cross-section spot-retrieval or spatial
edge is permitted.

### Frozen primary design

The primary deformation is the independently characterized Study 04A
diagonal-affine deformation from `151673` to `151508`. The same deformation is
applied only to raw `151676`. This avoids using the evaluation pair to both
define and assess the perturbation. The `151675` to `151676` deformation is
reserved for a separately declared sensitivity experiment and is not part of
the primary production matrix.

The five primary conditions are:

| Condition | Reference | Query | Role |
|---|---|---|---|
| `raw` | raw `151675` | raw `151676` | natural-data anchor |
| `sim_0.00` | raw `151675` | FEAST(`151676`, alpha 0.00) | resampling control |
| `sim_0.50` | raw `151675` | FEAST(`151676`, alpha 0.50) | interpolation |
| `sim_1.00` | raw `151675` | FEAST(`151676`, alpha 1.00) | primary endpoint |
| `sim_1.50` | raw `151675` | FEAST(`151676`, alpha 1.50) | extrapolation stress test |

The FEAST simulation seed is fixed at 42. Each condition is run with GraphST,
STAMP, and scVI using method seeds 42, 43, and 44, for 45 production jobs.
Method hyperparameters remain frozen from Study 04A. GraphST PCA-10 is primary
for dimensional parity with the native 10-dimensional STAMP and scVI outputs;
GraphST PCA-20 is a representation-sensitivity result.

The ordered 2,000-gene panel is selected from raw `151675` counts after
restricting to genes present in both sections. Panel selection, input QC,
simulation qualification, and all metric definitions are frozen before any
method result is scored. scVI receives raw counts with section as the only
batch covariate; the constant donor and evaluation layer labels are not model
covariates. Spatial graphs are constructed independently inside each section,
with a hard zero-cross-section-edge postcondition.

### Qualification and scoring contract

Before GPU production, every input must prove finite nonnegative integral
counts, exact query spot/gene/coordinate/label preservation across the
simulated ladder, fixed-panel support, and the requested deformation metadata.
Input-only diagnostics must distinguish the added `sim_0.00` to `sim_1.00`
perturbation from the raw-to-`sim_0.00` resampling effect. Failure of this gate
stops production rather than changing alpha or model settings after results
are observed.

Primary batch-removal metrics are computed within each shared DLPFC layer and
then macro-averaged equally across layers: batch ASW, local batch-mixing
entropy, and MMD. Primary biological-preservation metrics are layer ASW,
layer purity, cross-section layer-label-transfer macro-F1, cross-layer neighbor
contamination, and within-section neighborhood preservation. Query stability
between `sim_0.00` and each nonzero alpha uses k-nearest-neighbor overlap,
pairwise-distance correlation, and linear CKA. These are perturbation-stability
diagnostics, not cross-section spot alignment. Exact spot top-1 retrieval is
not a primary Study 04B metric.

Every condition also receives an uncorrected fixed-panel log-normalized PCA-10
baseline. Perturbation-response interpretation is relative to `sim_0.00`; the
raw arm separately measures simulator/resampling displacement. Scores are
reported per method seed and as seed mean/SD. Spots are not treated as
independent biological replicates. Composite scores, global rankings, and
winner claims remain prohibited.

### Conduct sequence and claim boundary

The workflow order is: establish semantic input contracts and the common panel; generate and
qualify the five inputs; compute uncorrected baselines; run and validate one
alpha-1.00 seed-42 canary for each method; run the remaining production matrix;
independently reconstruct all atomic metrics; and only then create plot-ready
tables and figures. No completed output root is overwritten.

If supported, the strongest intended claim is:

> In one within-donor pair of real DLPFC sections, GraphST, STAMP, and scVI
> showed distinct robustness and biological-preservation trade-offs as a
> controlled FEAST marginal perturbation was added to one section.

This design does not establish a batch-free ground truth, removal of the
natural section difference, cross-section spot correspondence, generalization
across donors or tissues, or a universally best method.

### Production completion and initial interpretation

Production root `real_slice_robustness/outputs/production_v1` completed on
2026-08-08. It contains all 45 method/condition/seed candidates, 65 atomic
metric rows (45 primary method rows, five uncorrected baselines, and 15 GraphST
PCA-20 sensitivity rows), and 195 seed-summary rows. Independent validation
reconstructed all 65 rows and 13 metric columns with maximum absolute error
`1.11e-16`. The combined Study 04A and Study 04B regression suite passed 13
tests.

The input qualification gate also passed. The added `sim_0.00` to `sim_1.00`
deformation exceeded raw-to-`sim_0.00` resampling displacement in every fitted
parameter dimension, while the query layer-pseudobulk correlation at
`sim_1.00` remained 0.9925. Thus the ladder is a controlled stress test rather
than arbitrary biological destruction.

The first-pass result supports a trade-off interpretation:

- scVI produced the strongest layer-conditioned batch mixing across the
  perturbation ladder. At `sim_1.00`, its seed-mean batch ASW/MMD were
  0.0233/0.0290 and entropy was 0.8167, compared with uncorrected
  0.0696/0.0805 and 0.5651. Its layer purity and layer-transfer F1 were not
  improved over the uncorrected baseline at that endpoint, and its query
  geometry was less stable relative to `sim_0.00`.
- GraphST retained the clearest spatial-neighborhood signal and relatively
  strong layer purity/F1, but its layer-conditioned batch separation increased
  with perturbation strength; the PCA-20 sensitivity changed magnitudes but
  not this qualitative conclusion.
- STAMP retained strong layer separation but had a non-monotone and
  seed-sensitive batch response. At `sim_1.00`, individual batch-ASW values
  ranged from 0.0473 to 0.1223; its improved mixing at `sim_1.50` should
  therefore be described as a stress-condition observation, not a smooth
  dose-response claim.
- All corrected representations substantially rewired input-PCA neighborhoods
  (primary within-section kNN preservation approximately 0.065 to 0.091).
  This metric compares different learned geometries and is best treated as a
  transformation-magnitude warning, not by itself as proof of biological loss.

These observations justify showing batch-removal, layer preservation, spatial
structure, and perturbation stability as separate panels. They do not justify
a composite score or a universal method ranking.
