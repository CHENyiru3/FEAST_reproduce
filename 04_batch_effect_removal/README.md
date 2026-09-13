# Study 04: batch-effect removal

Two complementary experiments evaluate GraphST, STAMP, and scVI. Study 04A
measures recovery from known perturbations of one slice; Study 04B measures
incremental robustness across real slices. Their outputs and methods remain
separate. Use the FEAST build and external interpreters declared by each config.

## Study 04A: controlled same-slice recovery

Run from `04_batch_effect_removal/`. Put DLPFC `151673.h5ad` and `151508.h5ad`
in `data/local/`; identities are in `data/input_checksums.csv`. Slice 151673
supplies the clean reference and fixed panel; 151508 characterizes deformation.

The design has two deformation modes and six nonzero alpha values: 12 comparison
conditions, plus two identical alpha-zero baseline matrices. Each external
method runs the 12 comparison jobs. Seed is 42; the primary endpoint is alpha 1.
Alpha above 1 is extrapolation. Change input/environment paths in `config.yaml`
for your installation and use a new output root for a repeat run.

```bash
FEAST_PY=/path/to/clean-feast-wheel-env/bin/python
"$FEAST_PY" run.py --dry-run
"$FEAST_PY" run.py --stage simulate
"$FEAST_PY" run.py --stage panel
"$FEAST_PY" run.py --stage methods --method GraphST
"$FEAST_PY" run.py --stage methods --method STAMP
"$FEAST_PY" run.py --stage methods --method scVI

HISTORICAL_ATOMIC=/path/to/frozen/consolidated_atomic_metrics.csv
INVALID_COMPOSITE=/path/to/frozen/batch_correction_results.csv
"$FEAST_PY" score.py --historical-atomic "$HISTORICAL_ATOMIC" --invalid-composite "$INVALID_COMPOSITE"
"$FEAST_PY" validate.py --historical-atomic "$HISTORICAL_ATOMIC" --invalid-composite "$INVALID_COMPOSITE"
```

The historical atomic table supports one-to-one comparisons. The invalid
composite is retained only for its disposition and is never compared numerically.
Scoring creates 60 atomic rows, support audits, and PCA/common-support sensitivity
tables. Validation independently recomputes the atomic metrics from artifacts.

Method launches require `pip check` and actual CUDA execution. GraphST's NumPy
1.23.x and scVI's NumPy 2.x are outside FEAST's supported range; FEAST does not
run in those workers. Completed candidates are reused only after validation;
failed attempts remain under the run's `failures/` directory.

The existing `stamp_numerical_retry_v1.yaml` permits patience 10 only for its two
specified alpha-1.50 numerical failures, retaining the failed attempt and all
other settings. Ordinary runs use patience 20. Do not generalize that exception.
For a repeat, pass `--output-dir` to the runner and the matching `--run-dir` to
scoring and validation; do not reuse a completed root.

## Study 04B: real-slice robustness

Run from `04_batch_effect_removal/real_slice_robustness/`. Put `151675.h5ad` and
`151676.h5ad` in its `data/local/` directory. Raw 151675 is the reference; raw or
FEAST-perturbed 151676 is the query. The config declares five conditions,
three methods, and three method seeds: 45 GPU jobs. Cross-section exact-spot
retrieval is not scored.

```bash
"$FEAST_PY" run.py --dry-run
"$FEAST_PY" run.py --stage prepare
"$FEAST_PY" validate.py --stage prepared
"$FEAST_PY" run.py --stage methods --condition sim_1.00 --seed 42
"$FEAST_PY" validate.py --stage candidates --allow-incomplete
"$FEAST_PY" run.py --stage methods
"$FEAST_PY" score.py
"$FEAST_PY" validate.py --stage complete
```

Candidates must satisfy their input identity, fixed panel, section order, CUDA,
and zero-cross-section-edge checks. Scoring retains atomic per-seed metrics,
seed summaries, and input-only qualification diagnostics. Composite scores and
method-winner rankings are not produced or authorized by either experiment.

## Figures

Run from the repository root. Inputs are Study 04A
`outputs/final_rerun_20260718_v2/metrics/` and Study 04B
`real_slice_robustness/outputs/production_v1/metrics/`.

```bash
python visualization/04_batch_effect_removal/plot.py
python visualization/04_batch_effect_removal/plot_additional.py
python visualization/04_batch_effect_removal/plot_real_slice_robustness.py
python visualization/04_batch_effect_removal/plot_combined_metrics.py
python visualization/04_batch_effect_removal/plot_same_slice_layer_transfer.py
python visualization/04_batch_effect_removal/plot_uncorrected_alpha_embeddings.py
python visualization/04_batch_effect_removal/plot_two_slice_context.py
python visualization/04_batch_effect_removal/plot_story.py
python visualization/04_batch_effect_removal/plot_overview.py
```

`plot_diagonal_combined.py` reads the embedding and layer-transfer CSVs produced
by the earlier plotters. `plot_task_table.py` draws the experiment summary.
These two scripts require `arial.ttf` and `arialbd.ttf` in the Python environment's
`fonts/` directory or a directory selected by `FEAST_FONT_DIR`.

Figure subdirectories group `same_slice/`, `two_slice/`, `overview/`, and
`supplementary/` outputs. Primary displays stop at alpha 1; extrapolation and
representation sensitivity remain explicit supplementary analyses. Cross-slice
robustness does not establish exact recovery of an unobserved biological truth.
