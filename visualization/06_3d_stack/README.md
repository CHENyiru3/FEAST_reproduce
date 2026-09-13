# Study 06 visualization

> The figures and 93-target inputs described below are superseded for the
> intended reference-density question. Corrected figures must read
> `06_3d_stack/outputs/reference_density_rank_v1`, contain all 144 modeled
> positions in every arm, and distinguish retained real slices from simulated
> targets. Existing figures remain preserved and must not be overwritten.

## Current promoted comparison

`plot_resampling_comparison.py` builds the publication-facing one-row,
four-panel comparison
of FEAST with ten class-conditioned whole-spot resampling runs. It presents
conditional expression Wasserstein, conditional zero-fraction Wasserstein,
the existing Moran-I profile correlation, and label-residual Moran-I profile
correlation by reference gap. Typography, palette, box/point treatment, grids,
titles, and legend placement follow the Study 00 simulator-benchmark figure
style.

For each target, the ten conditional-resampling values are averaged before
display. The boxes therefore compare 93 FEAST targets with 93 matched baseline
means rather than visually weighting the baseline ten times more heavily.

```bash
python visualization/06_3d_stack/plot_resampling_comparison.py \
  --output-dir visualization/06_3d_stack/figures
```

The metric inputs are in
`06_3d_stack/outputs/final/baselines/conditional_metric_comparison/`. Ordered z
targets are displayed descriptively and are not treated as biological
replicates. Exact spot and per-spot profile Pearson are not displayed.

## Archived presentation history

The former atomic diagnostic, exact spatial Pearson, and per-spot 3D profile
figures and their derived plotting tables were moved to
`.archive/superseded-20260901/visualization/06_3d_stack/figures/` after the
replacement passed verification. Their validated evaluation inputs remain in
place.

The validated, unpromoted diagnostic is built by `plot.py` from the 93-target
final validation and evaluation only:

```bash
python visualization/06_3d_stack/plot.py
```

It emits `figures/conditional_stack_diagnostic.{pdf,svg,png}`, plotting CSVs,
and `figure_provenance.json`.

The expanded report set is rendered with:

```bash
python visualization/06_3d_stack/plot_local_spatial_fidelity.py
python visualization/06_3d_stack/plot_3d_profile_fidelity.py
python visualization/06_3d_stack/plot_3d_gene_distribution.py
```

`conditional_stack_local_spatial_fidelity` separates direct per-gene spatial
pattern recovery from the global distributional metrics. It reports mean and
median per-gene spotwise Pearson correlation plus Study 00-equivalent Moran-I
correlation, both by reference gap and across the ordered target z levels.

`conditional_stack_3d_profile_fidelity` shows every validated arm in matched
three-dimensional stacks (dense, medium, and sparse) from two fixed views.
Points encode each spot's Pearson correlation across the complete 1,122-gene
profile. The 3D point layer is deterministically sampled to at most 1,500
valid spots per target slice; the companion
`conditional_stack_profile_fidelity_by_z` figure uses all valid spots to show
the target-level median and q25–q75 interval over z. No gene is selected for
these profile figures.

`conditional_stack_3d_volume` shows all 147 real ground-truth slices followed
by the generated geometry for all three density arms, using matched oblique and
lateral views. `conditional_stack_3d_gene_distribution` shows `Slc17a7`,
`Gad2`, `Gfap`, and `Reln` from one matched oblique view. Its uniform
within-slice sample is not selected by expression. Real and generated raw counts
are displayed after `log1p`; each gene has one scale shared across ground truth
and all density arms and clipped at the positive-value 99th percentile for
display. Gray points are sampled zero-expression spatial context.

The compact 2×3 manuscript layout separates dense, medium, and sparse arms and
marks targets that use declared out-of-bracket label donors. A dedicated legend
band keeps those encodings and the continuity-baseline key outside the data
panels. Panel F shows generated–target and real split-half median z coherence;
the labels above each pair report their validated normalized ratio. The 93
points are ordered z levels, not independent biological replicates, and the
boxes are descriptive median/IQR summaries only.

Panels A–E use the same formulas and display names as the corresponding Study
00 simulator-benchmark metrics: `Mean Corr`, `Var Corr`, `Zero Frac WDist`,
`Real Error Mean`, and `Zero Jaccard`. These are shared fidelity concepts, not
cross-study effect sizes; Panel F remains the Study 06-specific 3D continuity
evaluation. The legacy Study 06 metrics remain in its evaluation tables for
the historical-comparison workflow.

PDF, SVG, and 600-DPI PNG exports use deterministic metadata. PDF fonts remain
editable TrueType and SVG labels remain text. Composite scores, winner rankings,
publication claims, and automatic figure promotion are prohibited.
