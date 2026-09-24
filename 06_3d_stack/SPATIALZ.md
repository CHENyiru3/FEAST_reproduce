# Native SpatialZ benchmark

`spatialz_generate.py` uses SpatialZ's own coordinates, cell count, inferred
class labels, and expression. `spatialz_evaluate.py` compares its slices and the
completed FEAST slices against the same held-out observations. No individual
generated cell is matched or moved to an observed target cell.

## Inputs and generation

Reuse `outputs/generative_five_reference_1.0.6_precision_v1/plan.json`:
49/30/17 retained slices, 95/114/127 targets, and the complete 1,122-gene panel.
The original 144 available slice positions and absent IDs remain unchanged.
SpatialZ reads only each target's nearest lower/upper retained reference H5ADs.
It consumes their `layers['counts']`, `obs['class']`, and `obsm['spatial']`.
Target z is taken from the plan; target expression, XY, labels, and cell count
are unavailable during generation. No generated slice becomes a reference.

For lower reference first, `alpha = (z_upper - z_target) / (z_upper - z_lower)`.
This is `1 - tau` in the existing FEAST plan. It accounts for irregular actual
z spacing instead of assuming evenly spaced slice IDs. `n_cell=None` lets
SpatialZ infer `int(alpha*n_lower + (1-alpha)*n_upper)` cells.

The researcher selected the [real-data evaluation tutorial](https://spatialz-tutorial.readthedocs.io/en/latest/Evaluation%20on%20a%20real%203D%20spatial%20transcriptomics%20data.html)
recipe: default synthesis, 1,000 iterations, 200 projections, learning rate
100,000, `k_sam=50`, `Beta=100`, `k_neighbors=1`, and `n_mag=1`. Settings are in
`config_spatialz.yaml`. The reference-derived count replaces the tutorial's
observed target count. There is no parameter search, fast-mode substitution,
extra donor supplementation, gene filtering, or expression renormalization
during generation. Seed is the existing per-target plan seed, set for both
SpatialZ's torch generator and its separate NumPy expression sampler.

The [Allen documentation](https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html)
identifies the source XY as measured coordinates after rotation/alignment to
the CCF, and z as section position. The adapter interprets their millimeter
scale using Allen's [CCF coordinate conventions](https://alleninstitute.github.io/abc_atlas_access/notebooks/merfish_ccf_registration_tutorial.html)
and converts XY by 1,000 to micrometers. This makes SpatialZ's hardcoded MENDER
radius 15 a 15-micrometer neighborhood. Direct use of the small input numbers
would make that radius cover a whole slice. Generated XY is converted back to
the input units on export. There is no target-informed alignment or rescaling.

## Environments and compute

Generation uses `/maiziezhou_lab2/yiru/envs/spatialz/bin/python`, Python 3.9.19,
PyTorch 1.13.0+cu117, and the existing SpatialZ checkout at commit
`e1b5ed01e933729149e08399febca08db16ced14`; see `../environments/README.md`.
The installed source is unchanged. A temporary replacement of MENDER's
`run_representation_mp` uses its same ordered map and concatenation with a
managed two-process pool instead of the hardcoded 200-process pool.
The runner limits numerical-library threads and restores the method afterward.
SpatialZ's full reference-by-query microenvironment similarity matrix and
gene-by-gene sampler are retained, so CPU memory and expression synthesis time
can still be substantial. CUDA accelerates the coordinate optimization.

Evaluation uses the existing Study 06 Python 3.11 precision environment. CUDA
is accessible on this host outside the restricted agent sandbox; an actual
PyTorch CUDA tensor operation passed on its RTX A6000 on September 14, 2026.

Run from the repository root, with a fresh output root:

```bash
tmux new-session -s feast-s06-spatialz
bash 06_3d_stack/run_spatialz.sh \
  06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  06_3d_stack/outputs/spatialz_native_evaluation_cuda_v1
```

Detach with Ctrl-b, then d. Reattach with `tmux attach -t feast-s06-spatialz`.
The script runs gap 3, 5, and 10 sequentially on CUDA 0 with two CPU threads
and two MENDER workers, then evaluates all 336 targets. Set `SPATIALZ_PYTHON`
and `EVALUATION_PYTHON` to change interpreter locations on another machine.
Follow `workflow.log` in the chosen output root.

For a single target, use the generator directly:

```bash
../envs/spatialz/bin/python -u 06_3d_stack/spatialz_generate.py \
  --feast-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  --output-root 06_3d_stack/outputs/spatialz_native_evaluation_cuda_pilot_20260914 \
  --gap 3 --target-id 5 --device cuda:0
```

Explicit `--target-id` or nonoverlapping `--shard-index`/`--shard-count` selections
can run remaining targets. Existing target directories are preserved, including
failed attempts; the runner does not silently overwrite or skip them.

Each target directory contains `generated.h5ad` and `run.json`, recording the
recipe, references, z, seed, source commit and modifications, environment,
device, elapsed time, status, and generated dimensions. H5AD stores native
expression in X/counts, native labels, generated XY in input units, and
`spatial_3d` with the requested z. Original FEAST outputs remain separate.

## Evaluation

```bash
../envs/feast-study06-1.0.6-precision/bin/python -u \
  06_3d_stack/spatialz_evaluate.py \
  --feast-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  --spatialz-root 06_3d_stack/outputs/spatialz_native_evaluation_cuda_pilot_20260914 \
  --output-dir 06_3d_stack/outputs/spatialz_native_evaluation_cuda_pilot_20260914/evaluation \
  --gap 3 --target-id 5
```

Omit selectors to require all 336 completed targets. Use a fresh evaluation
directory. FEAST's existing validation is retained; its exact-coordinate
requirements are not applied to SpatialZ. Both methods are evaluated by the
same functions here, with independent cells and graphs.

- Gene mean/variance correlations use the established Study 00 `log1p`
  statistic comparisons. Zero-fraction correlation/Wasserstein and relative
  gene-mean error use raw counts and the existing formulas.
- Expression Wasserstein is the median gene-wise empirical W1 after each
  cell is normalized to 10,000 counts and log1p transformed. Library-size and
  detected-gene-count W1 compare their raw-count distributions.
- Conditional expression W1 uses each method's own class memberships and
  target class-proportion weights, with exact empirical W1 for unequal cell
  counts. The formula matches the established metric when class sizes match.
  If any observed class is absent from generation, the whole conditional score
  is undefined (NaN). Per-class results and missing observed class mass expose
  that absence. No penalty or replacement expression is invented.
- Cell-count relative error and total variation of class proportions include
  all classes in either slice. Missing and extra class mass are reported.
- Moran and class-residual Moran profiles use raw counts, the target's top
  300 mean-expression genes, and a separate directed, row-normalized 6-NN graph
  for each slice. Residuals subtract each slice's own class means. The number
  of genes with defined paired Moran values is reported.

Outputs: `metrics.csv`, `comparison.csv`, `summary.csv`, `class_metrics.csv`,
per-target/method `gene_profiles/*.csv`, and `evaluation.json`. Summaries report
both selected target count and defined score count; NaN scores are not treated
as successes. No cellwise Pearson, paired zero-mask score, coordinate matching,
composite ranking, biological-replicate inference, or whole-stack continuity
claim is produced by this evaluator.

FEAST receives observed target coordinates and class labels and uses five
primary references plus declared supporting donors. SpatialZ infers target
geometry/composition from two references. This is a comparison of slice
reconstruction under those different information inputs, not an equal-input
test of expression prediction alone. SpatialZ cell-count/composition errors
therefore describe a task that FEAST is given directly.

## Complete three-method comparison

All 336 generated targets and their evaluation completed September 15, 2026
at 22:11 Chicago time. The three-method overview is rendered from these metric
tables and the existing ten-repeat resampling results:

```bash
../envs/feast-study06-1.0.6-precision/bin/python \
  visualization/06_3d_stack/plot_spatialz_comparison.py
```

Run from the repository root; use `--output-dir` with a fresh directory for an
additional rendering. The default output is
`visualization/06_3d_stack/figures/spatialz_full_comparison_v1`, containing
`REPORT.md`, two figures in PNG/PDF/editable SVG, plot data, and source paths.
The plotter checks agreement of the old and recomputed FEAST metrics on all
336 targets before adding resampling. It averages resampling repeats within
each target and shows medians and interquartile ranges across targets.
Conditional comparisons use the same supported target subsets for all three
methods within each gap (14/95, 11/114, and 18/127); the subsets differ across
gaps. No expression or spatial metric is recomputed by this plotter.

## Initial verification

The full 336-target CUDA run was launched September 14, 2026 at 07:44 Chicago
time in tmux session `feast-s06-spatialz`, writing to
`outputs/spatialz_native_evaluation_cuda_v1`. This records its launch, not
completion; inspect that root's `workflow.log` and target `run.json` files for
progress. The launcher evaluates both methods after all generation succeeds.

The September 14 CUDA pilot generated gap-3 slice 5 with all 1,122 genes,
9,629 native cells, and no all-zero expression rows. Generation took 198
seconds, including 24 seconds for coordinate optimization and 152 seconds for
expression synthesis. The observed slice has 12,097 cells; the generated
count was determined exclusively from reference slices 4 and 7.
Both SpatialZ and FEAST slice evaluation completed under
`outputs/spatialz_native_evaluation_cuda_pilot_20260914/evaluation`.
This is one pilot target, not a completed 336-target comparison.

Targeted checks passed for reference-only H5AD access, alpha direction,
counts-layer use, coordinate unit conversion, row-permutation invariance,
unequal cell counts, conditional metric parity with the existing evaluator,
and explicit missing-class reporting. The four existing Study 06 workflow
and resampling tests passed. The interrupted CPU pilot remains recorded
separately under `outputs/spatialz_native_evaluation_pilot_20260914`.
