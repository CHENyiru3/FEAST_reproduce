# Study 05 visualization

## Combined cross-slice and half-held-out 4 × 4 panel

[Open the combined panel](figures/article_style/conditional_transfer_cross_and_half_4x4.pdf)
and its [caption](figures/article_style/conditional_transfer_cross_and_half_4x4_caption.md).
The four columns are DLPFC cross-slice (151676 → 151675, GFAP), DLPFC
half-held-out (151676, MBP), MERFISH cross-slice (006 → 007, Frzb), and
MERFISH half-held-out (007, Igfbp4). The rows are observed reference, held-out
truth, FEAST, and conditional resampling. All cases use AR = 0.3.

Half-held-out columns show the median-x split with a dashed line. In the three
target rows, observed left-half expression is muted and the evaluated right
half remains in full color. The reference row keeps the hidden half gray.
One shared, unclipped log1p-count colorbar covers all 16 maps. Data and scores
are unchanged; the old half-slice q99.5 display normalization is replaced by
the common raw-count display scale. The DLPFC half-held-out column keeps slice
151676 and uses MBP following visual comparison with SCGB2A2, TUBA1B, and MT-CO2.
Other columns retain their prior highest saved gene Pearson selections.
Selection is illustrative. All four resampling draws match their saved baseline metrics.
Arial PDF, editable SVG, and 600-DPI PNG exports are supplied.

```bash
python visualization/05_2d_conditional_transfer/plot_mixed_transfer_4x4.py
```

## Selected 4 × 4 cross-slice panel

[Open the selected panel](figures/article_style/cross_slice_selected_4x4.pdf)
and its [selection rationale and caption](figures/article_style/cross_slice_selected_4x4_caption.md).
Rows are source reference, held-out target, FEAST, and conditional resampling.
Columns show DLPFC within donor (151676 → 151675, GFAP), DLPFC cross donor
(151670 → 151675, SCGB2A2), MERFISH 006 → 007 (Frzb), and MERFISH 007 → 006
(Igfbp4). The selection uses strong existing marker examples at AR = 0.3 while
retaining both technologies, both DLPFC donor strata, and both MERFISH directions.
It is an illustrative selection based on held-out agreement, not an estimate of
typical-gene performance. The existing data are reused with one shared,
unclipped log1p-count expression scale and colorbar across all 16 panels.
Arial PDF text, editable SVG text, and 600-DPI PNG exports are supplied.

```bash
python visualization/05_2d_conditional_transfer/plot_selected_cross_slice.py
```

## Expanded article-style visualization PDF

[Open the 15-page article-style PDF](figures/article_style/study05_expanded_visualization.pdf)
and its [figure captions](figures/article_style/captions.md).
The layout follows the Study 00 spatial matrices and Study 01 compact panels:
method rows, direction/gene columns, bold headings, blue FEAST, gray resampling,
viridis expression, external legends, and tightly cropped exports.
It includes all eight cross-slice directions at AR = 0.3, with source/target
label maps and source, held-out truth, FEAST, and conditional-resampling
expression maps for two existing markers per dataset. Unsupported spots are
gray. Each gene uses one unclipped log1p-count scale across its four maps.
The marker set reuses SCGB2A2/Igfbp4 and the previously selected best-case
GFAP/Frzb; these illustrations are not a typical-gene accuracy estimate.

Additional pages show named direction-level metrics and baseline ranges,
marker expression distributions, target coverage, source-zero genes,
all five assignment-randomness settings, count-distribution diagnostics,
and the existing half-slice matrix with separate half-slice metrics.
The displayed resampling draw reproduces seed 2026, replicate 0 in the
original task sequence; its library-size Wasserstein distance was checked
against the saved baseline score for every cross-slice direction.

```bash
python visualization/05_2d_conditional_transfer/plot_expanded_report.py
```

This uses existing simulations and scores. Individual PDFs, editable SVGs,
600-DPI PNGs, plotting tables, captions, and provenance are in
`figures/article_style/`. Rebuilding replaces only this script's article-style
outputs. The earlier [17-page report](figures/expanded/study05_expanded_visualization.pdf)
and original figures remain available. The new figures retain vector text and
plots with dense spatial points rasterized at 600 DPI. The command uses the
existing analysis environment and `pdfunite`.

## Current promoted comparison

`plot_resampling_comparison.py` builds the publication-facing one-row,
four-panel comparison
of FEAST with ten label-conditioned whole-spot resampling runs. It presents
conditional expression Wasserstein, conditional zero-fraction Wasserstein,
the existing Moran-I profile correlation, and label-residual Moran-I profile
correlation. Exact spot Pearson and Spearman are not displayed. Typography,
palette, boxes/points, grids, titles, and legend placement follow the Study 00
simulator-benchmark figure style.

Each task is a paired FEAST/resampling observation joined by a thin line.
Tasks are grouped into DLPFC cross-slice, DLPFC half-slice, MERFISH
cross-slice, and MERFISH half-slice strata; orange ticks mark stratum medians.
No box or density estimate is used for these groups of only two to six tasks.

```bash
python visualization/05_2d_conditional_transfer/plot_resampling_comparison.py \
  --output-dir visualization/05_2d_conditional_transfer/figures
```

The metric inputs are in
`05_2d_conditional_transfer/outputs/baselines/conditional_metric_comparison/`.
Gray intervals are empirical 5–95% resampling ranges, not confidence
intervals. Target expression is used only for evaluation.

## Archived presentation history

The older aggregate-profile, exact-spot, and assignment-randomness figures and
their derived plotting tables were moved to
`.archive/superseded-20260901/visualization/05_2d_conditional_transfer/figures/`
after the replacement metrics passed verification. The sections below document
those historical plotting workflows; their validated score inputs remain in
place as scientific evidence.

`plot.py` builds the quantitative publication-review figure only from
validated fresh score, comparison, decision, and complete-validation
records. It distinguishes
DLPFC from MERFISH, cross-slice from half-slice generation, within-donor from
cross-donor DLPFC transfer, the predeclared primary assignment randomness
(`0.3`), and sensitivity settings. Target-support and globally
reference-observed-gene fractions are shown separately.

The compact four-panel manuscript layout uses a dedicated legend band. Panel A
contrasts aggregate mean, variance, Moran, and zero-profile fidelity; panel B
places the spotwise Pearson and Spearman evidence beside it. Open symbols are
individual directions or slices and filled symbols are stratum means. Panel C
shows assignment-randomness trajectories with the across-direction min–max
range, while panel D keeps each evaluation direction explicit and separates
the five declared strata with spacing rather than pooling them.

`plot_simulation_effect.py` adds two complementary graphics so the study is not
represented only by summary metrics:

- `conditional_transfer_simulation_design.pdf` explains exactly what FEAST
  receives in cross-slice and half-slice generation. Target coordinates and
  labels are supplied, but target expression remains hidden until scoring.
- `conditional_transfer_spatial_effect.pdf` is the compact main figure with
  DLPFC `151670` and MERFISH `006`; it shows known labels and coordinates plus
  held-out and generated log-library-size maps. In both expression columns,
  the observed half shows its real library size in muted viridis and the target
  half uses fully saturated color. A lightly shaded, outlined box encloses the
  masked right half in every spatial panel. The shorter library-size color
  scale is the only metric scale displayed.
- `conditional_transfer_spatial_effect_supplementary.pdf` contains the other
  three primary half-slice cases (DLPFC `151675`, DLPFC `151676`, and MERFISH
  `007`) in the same layout. Dataset-level condition-label legends retain the
  same label-to-color mapping. All five cases' spotwise all-gene profile values
  remain recorded in `figures/simulation_effect_plot_data.csv` and
  `figures/simulation_effect_summary.csv`.

`plot_ar_spatial_effect.py` adds the requested high-correlation best-case view.
It first selects the assignment-randomness value with the highest mean
cross-direction Moran-profile correlation within each dataset, then selects the
best direction at that AR. Within that case it chooses the highest-Pearson gene
among the top 5% of spatially structured target genes and displays the same
held-out target across every registered AR value. This selects DLPFC
`151676 → 151675`, AR `0.3`, `GFAP` (`r = 0.744`) and MERFISH
`007 → 006`, AR `0.2`, `Frzb` (`r = 0.611`). The blue title marks the best
aggregate AR. A companion trajectory shows that the aggregate-optimal AR does
not necessarily maximize the selected gene's coordinate-wise correlation. The
spatial maps display only Moran's I above each map; the Pearson and Moran's I
line trajectories are separated into
`conditional_transfer_ar_trajectory.pdf`.

`plot_expression_matrix.py` provides the compact spatial-expression matrix that
parallels Study 00. Its two rows compare held-out truth with FEAST simulation
across all five primary half-slice cases. The observed half shows its actual
ground-truth expression in a muted version of the shared viridis scale, while
the target half uses fully saturated color for held-out truth or FEAST. It uses
one common marker per dataset (`SCGB2A2` for DLPFC and
`Igfbp4` for MERFISH). Markers are selected only from held-out reference
structure: the highest median target Moran's I among genes present in every
dataset slice with 5–95% target zeros. FEAST agreement is not part of marker
selection. Each column shares its full-truth-plus-FEAST q99.5 log-expression
scale.

Run from the clean FEAST environment:

```bash
python visualization/05_2d_conditional_transfer/plot.py
python visualization/05_2d_conditional_transfer/plot_simulation_effect.py
python visualization/05_2d_conditional_transfer/plot_ar_spatial_effect.py
python visualization/05_2d_conditional_transfer/plot_expression_matrix.py
```

Each command stages a complete set of its own outputs, then replaces only those
script-owned files in `figures/`. Outputs owned by the other scripts are left
untouched.

The `figures/` root contains editable PDF/SVG and 600-DPI PNG exports. The
quantitative figure has three plotting-data tables and a provenance JSON
recording the validated row counts, upstream configuration, claim limits, and
output inventory. The simulation-design and general spatial graphics have an
all-spot plotting table, a five-slice summary table, and
`simulation_effect_provenance.json`, which records the five primary half-slice
cases and their upstream score configuration. The best-case AR maps and separate
trajectory have their own
`ar_spatial_effect_plot_data.csv` and `ar_spatial_effect_provenance.json`,
which record the ten displayed AR-specific cases and their selection rule. The
compact expression matrix has its own marker-selection table, per-slice scale
summary, and provenance JSON. PDF fonts use type 42 and SVG text is retained as
text for editing in Adobe
Illustrator. PDF, SVG, and PNG exports use deterministic metadata and stable SVG
element IDs, so unchanged code and evidence under the same software environment
reproduce identical figure bytes. Provenance JSON records the rerun time and is
therefore not byte-identical across runs.

The figure set is evidence for author review, not authorization for an accurate
spot-level imputation claim. It explicitly shows that high aggregate
distribution and Moran-profile fidelity coexist with weak DLPFC coordinate-wise
recovery. The general and supplementary spatial figures show library-size
context across all five half-slice targets, the compact expression matrix shows
reference-selected markers without selecting on FEAST performance, and the
separate best-case AR figure demonstrates that stronger local recovery is
possible for selected genes. These are deliberately distinct claims.
Historical 2D-transfer figures remain noncanonical and are not source material.
