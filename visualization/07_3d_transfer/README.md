# Study 07 visualization

The figures directly in `figures/` were reproduced on 2026-09-09 from the
completed, validated two-reference-per-region **float32** run:
`07_3d_transfer/outputs/generative_transfer_two_reference_float32_v1`.
All 360 target levels, 7,544,388 positions and 550 genes remain in the evaluated
data. The original top-level figure files and this README were preserved under
`.archive/study07-figures-before-two-reference-20260909/` with their relative
layout. The separate `figures/two_reference_float32_20260909/` review set is
unchanged.

| Figure stem | Content |
| --- | --- |
| `full_axis_transfer_diagnostic` | Regional coverage and current FEAST continuity against the existing resampling median and 5–95% range |
| `full_axis_continuity_vs_conditional_resampling` | Matched adjacent-pair discontinuity distributions; control values are ten-run means |
| `full_axis_expression_summaries` | Unsmoothed counts per position, detected genes and zero fraction across z |
| `full_axis_3d_volume` | Both generated volumes from oblique and lateral views |
| `full_axis_3d_gene_distribution` | Slc17a7, Gad2, Gfap and Reln in 3D |
| `E15_5_full_axis_region_four_views` | E15.5 atlas regions in four views |
| `E18_5_full_axis_region_four_views` | E18.5 atlas regions in four views |
| `full_axis_marker_cross_sections` | Peak-support atlas and marker sections, linked to full-axis coverage |

Each figure has PDF, SVG and 600-dpi PNG exports, with associated plotting data
and provenance records. `study07_figures.pdf` combines all eight figures.

The existing resampling control uses complete reference spots pooled by age
and region, with replacement, for seeds 2026–2035. Its reference populations,
gene panel, target regional coverage and 358 adjacent-pair identities were
checked against the current data. No new control experiment was run. The
control retains its all-eligible-reference pool; it is not a matched
two-reference ablation. The 5–95% band describes variation across the ten
control runs and is not a confidence interval.

These are descriptive continuity and generated-expression displays. There is
no observed target expression, biological accuracy claim, or composite score.
Adjacent z pairs are not independent biological replicates. All metric values
are retained; logarithmic continuity displays use the existing floor at 1e-12
and axes spanning all plotted values. The 3D display sample remains uniform
within each slice (seed 2026; up to 700 positions per slice). Expression colors
use log1p counts and the existing positive-value 99th-percentile display cap.
Marker sections are selected independently by maximum support within each age,
not as homologous cross-age sections.

## Reproduction

The adapted plotting entry points used for this report are now in this
directory. From the repository root, use the study's plotting environment and
pass both directories explicitly:

```bash
FEAST_PY=/path/to/feast-environment/bin/python
INPUT=07_3d_transfer/outputs/generative_transfer_two_reference_float32_v1
FIGURES=visualization/07_3d_transfer/figures/reproduced
for script in plot_resampling_control plot_continuity_resampling_benchmark plot_expression_summaries plot_3d_gene_distribution plot_3d_region_four_views plot_axis_cross_sections; do
  "$FEAST_PY" "visualization/07_3d_transfer/$script.py" \
    --input-root "$INPUT" --output-dir "$FIGURES"
done
```

These commands require the completed validation/evaluation artifacts. The first
two also read the existing resampling control under
`07_3d_transfer/outputs/final/baselines/conditional_whole_spot_resampling/`;
that control must be supplied or reproduced separately. Atlas-based plots also
require the configured processed DevCCF volumes. Existing files are protected
by the entry points' overwrite checks; use a fresh figure directory.

The original run-local scripts and logs remain under `.work/` for provenance.
Rendered figures and plotting tables are excluded from the code-only repository.
