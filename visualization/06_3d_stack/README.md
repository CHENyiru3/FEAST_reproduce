# Study 06 visualizations

The active comparison includes the completed FEAST precision run, conditional
resampling and native SpatialZ on all 336 held-out targets. SpatialZ uses its
own generated cells, coordinates and class labels.

Start with the [publication figure index](figures/README.md). Benchmark panels
and legends consistently order methods as **FEAST, Resampling, SpatialZ**.
Older renders are retained in `figures/archived/`; they are not the active set.

Run from the repository root:

```bash
../envs/feast-study06-1.0.6-precision/bin/python visualization/06_3d_stack/plot_results.py
../envs/feast-study06-1.0.6-precision/bin/python visualization/06_3d_stack/plot_resampling_comparison.py --dpi 300
../envs/feast-study06-1.0.6-precision/bin/python visualization/06_3d_stack/plot_spatialz_comparison.py
../envs/feast-study06-1.0.6-precision/bin/python visualization/06_3d_stack/plot_spatialz_views.py
```

Use `--output-dir` with a fresh directory for another run. `--spatialz-root`
selects the SpatialZ run in the main, compact and native-view commands; `--input-root` selects FEAST in
the main and native-view commands. The compact comparison retains `--input-dir`
for the resampling metric directory. The full numerical report uses `--evaluation`
and `--baseline` to select its metric inputs. No generation or metric evaluation is rerun.

| Entry point | Rendered outputs | Contents |
|---|---|---|
| `plot_results.py` | [publication_with_spatialz_v1](figures/publication_with_spatialz_v1) | Three-method overview, conditional coverage, along-z profiles, reference coverage, native marker maps; explicitly FEAST-only z coherence |
| `plot_resampling_comparison.py` | [three_method_resampling_spatialz_v1](figures/three_method_resampling_spatialz_v1) | Four panels of target-level points and boxplots, including SpatialZ |
| `plot_spatialz_views.py` | [spatialz_native_views_v1](figures/spatialz_native_views_v1) | Sections 5/78/149, whole-volume expression, class-marker heatmaps and offline volume explorer |
| `plot_spatialz_comparison.py` | [spatialz_full_comparison_v1](figures/spatialz_full_comparison_v1) | Full numerical report, nine-metric overview, conditional and cell-recovery panels |

Marker maps retain Slc17a7, Gad2, Gfap and Reln and show every cell. The three
sections are the first, middle and last common held-out slice IDs, selected by
position. Axes and gene-wise color scales are shared across methods. Volume
plots and the offline HTML explorer use up to 1,200 cells per slice and method
across all 144 positions, including real anchors. The explorer can select a
single slice, hide anchors, choose a gene/gap, and synchronize rotations.
Heatmaps use all cells on held-out slices, with each method's own class labels;
absent classes are gray. Captions, provenance and CSV plot data accompany
PDF, editable SVG and PNG exports.

`plot_spatialz_views.py --render-only` reuses its saved volume points and class
summaries for layout changes. Section maps still read the three displayed
sections to show all cells.

Conditional metrics use identical supported targets across methods within
each gap: 14/95, 11/114 and 18/127. These subsets differ across gaps. Spatial
panels use all 336 targets. Resampling repeats are averaged within each slice;
slice variation is descriptive, not biological replication.

Moran-profile correlation does not establish correct anatomical placement.
FEAST receives target geometry and labels; SpatialZ generates them. No
SpatialZ-to-observed cell matching or cellwise difference is introduced.
The existing full-stack coherence panel remains explicitly FEAST-only because
the completed SpatialZ evaluator does not compute whole-stack continuity.

`plot_volume_differences.py` preserves the FEAST-only paired difference maps
and earlier matched-cell explorer. Use `plot_spatialz_views.py` for the updated
native-coordinate comparison. `plot.py`, `plot_3d_profile_fidelity.py`, and
`plot_local_spatial_fidelity.py` are historical diagnostics with older default
run paths and/or matched-cell assumptions. `plot_3d_gene_distribution.py`
provides the shared marker panel and sampling/display helpers. Earlier figure
directories are preserved.
