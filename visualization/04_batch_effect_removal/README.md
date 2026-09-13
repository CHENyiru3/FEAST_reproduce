# Study 04: batch-effect-removal visualization

## Combined metric PDFs

Use `plot_combined_metrics.py` for the compact metric figure set:

```bash
python visualization/04_batch_effect_removal/plot_combined_metrics.py
```

- `figures/same_slice/same_slice_combined_metrics.pdf`: eight metric panels.
- `figures/two_slice/two_slice_combined_metrics.pdf`: six metric panels.

Panels A–F occupy the same positions in both PDFs: batch ASW, batch entropy,
distribution MMD², layer separation ASW, layer purity (k=30), and reference-to-query
layer transfer macro-F1. The same-slice
figure adds paired retrieval top-1 and paired retrieval median rank (log scale).
All eight metrics are available for GraphST, STAMP, and scVI. The scVI-only
paired expression correlation is retained in the original atomic tables and
excluded from this comparison. Unused grid space holds a shared legend.
Same-slice batch metrics use all spots, whereas
two-slice batch metrics retain their layer-conditioned definitions, so their
absolute values should not be treated as identical estimands.

The script reuses both existing input-validation routines and preserves the
registered α ≤ 1.0 display range, isolated two-slice raw anchor, individual
method seeds, seed means, and min–max bands. Each PDF has a companion plot-data
CSV and provenance JSON. Existing source tables, detailed diagnostics, and
GraphST representation-sensitivity figures are retained; the older plotting
commands below rebuild those detailed figures. No correction methods are retrained.

Layer transfer provides a biological-label measure alongside layer separation
and purity. The two-slice values come from its validated atomic table. For
same-slice, the script extends the approved spatial layer-transfer calculation
to α=0.25, 0.5, 0.75, and 1.0 in both modes using saved primary embeddings.
Only the 30 nearest reference embeddings and their labels determine each
prediction; query labels are used for evaluation. Spot order, explicit
majority votes, and an independent confusion-count macro-F1 are checked for
all 24 method/condition scores. The six α=1 method scores must also match the
existing same-slice spatial figure. Separate
`same_slice_layer_transfer_metric_ladder.csv` and provenance files record
this calculation; the original atomic tables remain unchanged.

## Same-slice cortical-layer transfer

```bash
python visualization/04_batch_effect_removal/plot_same_slice_layer_transfer.py
```

`figures/same_slice/same_slice_layer_transfer_spatial.pdf` has one page for
each perturbation mode at the registered α=1 endpoint. Each page shows query
ground-truth layers and query predictions from uncorrected input, GraphST,
STAMP, and scVI. All maps use the original slice-151673 query coordinates,
identical spatial limits and panel sizes, the same inverted-y orientation,
and the existing cortical-layer colors. Each prediction title reports macro-F1.

Reference α=0 and query α=1 inputs use the existing fixed gene panel. The
uncorrected representation reuses library normalization to 10,000, log1p,
PCA-20 (seed 42), and column standardization from `plot_additional.py`.
Saved GraphST outputs use the same-slice primary PCA-20 representation;
STAMP and scVI use their native 10-dimensional representations. All method
representations reuse the existing standardization across reference and query
spots. No correction method is retrained.

For each query spot, a uniform-vote classifier uses its 30 nearest reference
spots in the standardized embedding and their reference ground-truth labels.
Distance is Euclidean. Ties between class vote counts use the first label in
sorted order. Query labels enter only evaluation and the ground-truth display.
Spot IDs verify ordering and identify exported rows; spatial coordinates verify
geometry and place the dots. Neither supplies predictions, and known matching
spot pairs are not supplied to the classifier.

Macro-F1 is the equally weighted mean of the seven layer-specific F1 scores,
with F1 = 2TP / (2TP + FP + FN). The script checks predictions against explicit
neighbor-vote counts and checks sklearn macro-F1 against independent confusion
counts. It also verifies spot, batch, and label alignment, shared geometry
across modes, and identical rendered panel sizes. The prediction CSV contains
all 3,611 query spots for every method and mode; aggregate and per-layer scores,
two PNG previews, and brief provenance accompany the PDF. This is a new
diagnostic calculation from saved representations, not an existing same-slice
atomic metric or an alteration to the validated atomic tables.

**Layer transfer evaluates biological-label preservation.** A query spot can
receive the correct cortical layer from any nearby reference spots belonging
to that layer. **Exact-pair recovery evaluates individual spot identity:** the
specific corresponding reference spot must be recovered. These figures answer
different questions and are retained together. Comparing the uncorrected and
corrected α=1 maps shows whether correction improves layer transfer at that
endpoint. This figure alone does not measure a decline from α=0 performance or
establish that batch effects have been completely removed.

## Same-slice uncorrected input ladder

```bash
python visualization/04_batch_effect_removal/plot_uncorrected_alpha_embeddings.py
```

`figures/same_slice/uncorrected_embedding_alpha_ladder.pdf` contains one page
per mode (shift only and diagonal affine), displaying all saved α values from
0 to 1.5. Values above 1.0 are labeled as extrapolation stress tests; the
registered primary endpoint remains 1.0. Each column compares the fixed α=0
reference to that query input, with batch colors above and cortical-layer
colors in the middle row. A third row shows the matching gene-parameter cloud
at each α. The α=0 column contains identical copies of the reference input.

Preprocessing and UMAP settings are reused from `plot_additional.py` and its
representative embedding figure. Each column has an independent projection;
absolute UMAP geometry should not be compared across columns. Spot order and
layer labels are checked for every input. Separate PNG/SVG pages, a coordinate
CSV, and provenance accompany the PDF. No correction methods are rerun.
All panels use equal square viewports, with each embedding centered and scaled
uniformly to the same longest displayed extent. This preserves its proportions
while making visual size and padding consistent across α values.
The parameter row reuses the same 3,078 reference-selected genes and stored
parameter definitions as `plot_parameter_cloud.py`: log mean, log variance,
and zero proportion. These points represent genes, whereas UMAP points
represent spots. Every parameter panel uses identical axes and viewing angles;
the shared log-variance lower limit is extended to -3.9 to include α=1.5 without
clipping. Parameter positions retain their original values and scale. A
`uncorrected_embedding_alpha_ladder_parameter_cloud_plot_data.csv` table records
all displayed gene coordinates. The existing UMAP coordinates were reused for
this layout update; no projection or correction model was refitted.

## Two-slice embedding and spatial context

```bash
python visualization/04_batch_effect_removal/plot_two_slice_context.py
```

Two additional figures in `figures/two_slice/` show the registered α=1 endpoint
using method seed 42, chosen in advance rather than by performance:

- `two_slice_embedding_context.{pdf,png,svg}` compares uncorrected input,
  GraphST, STAMP, and scVI. The top row colors spots by section; the bottom row
  colors the identical UMAP coordinates by known cortical layer. Each method
  has its own UMAP, so absolute geometry is not comparable between columns.
- `two_slice_layer_transfer_spatial.{pdf,png,svg}` shows known reference and
  query layers alongside query-layer predictions from all four representations.
  The existing reference-to-query classifier uses 30 neighbors in the scorer's
  standardized primary representation. Each displayed macro-F1 is checked
  against the validated atomic score before rendering. These are transferred
  layer labels, not new unsupervised clusters or cross-section spot matches.

The script reuses saved embeddings without retraining correction methods.
GraphST uses its primary PCA-10 representation; STAMP and scVI use their native
10-dimensional representations. UMAP settings match the same-slice context
figure: 30 neighbors, minimum distance 0.3, Euclidean distance, random seed 42,
and one worker. Spot order is checked against each saved result. Plot-data
CSVs, source spatial annotations, and `two_slice_context_provenance.json`
accompany the figures. Colors match the same-slice layer and batch palette;
legends use the enlarged type from the compact metric figures.

## Detailed diagnostics

This directory renders the completed 60-row atomic-metric evidence selected by
`../../PUBLICATION_MANIFEST.json`. GraphST, STAMP, and scVI each have 12 validated
primary jobs, and the separate within-batch graph contract records zero
cross-batch edges. The primary figure shows six complementary metrics: batch
separation ASW, batch mixing entropy, distribution MMD², paired retrieval
top-1, domain separation ASW, and domain purity. It separates batch removal
from biological-signal preservation, with an explicit preferred direction in
every panel. The remaining four atomic metrics stay in the plot-ready CSVs; no
composite score or method ranking is used.

These figures are supplementary diagnostic candidates only. Study 04 has no
active manuscript claim; no composite score, method winner/ranking, or figure
promotion is authorized. The displayed figures and plot-ready tables end at the registered
`alpha = 1.0` endpoint. The eight displayed GraphST PCA-20/PCA-10 sensitivity
rows must accompany any future use. GraphST and scVI ran in external environments with NumPy versions outside
FEAST's supported range; FEAST was not imported in those workers. The declared
STAMP numerical-retry policy is retained in figure provenance.

Rebuild from the repository root with the supported reproduction environment:

```bash
python visualization/04_batch_effect_removal/plot.py
python visualization/04_batch_effect_removal/plot_additional.py
python visualization/04_batch_effect_removal/plot_story.py
python visualization/04_batch_effect_removal/plot_overview.py
```

Outputs are separated by scientific task:

- `figures/same_slice/` contains Study 04A controlled same-slice diagnostics,
  plot-data tables, and provenance.
- `figures/two_slice/` contains Study 04B real two-slice robustness diagnostics,
  plot-data tables, and provenance.
- `figures/supplementary/` contains the integrated six-panel synthesis combining
  selected evidence from both tasks.
- `figures/overview/` contains the standalone presentation-style schematic of
  the shared workflow and the two complementary scientific tasks.

## Core illustrative overview

`plot_overview.py` renders
`figures/overview/batch_effect_removal_design.{pdf,svg,png}` as a focused
two-component illustration rather than a quantitative result figure. It shows
the real slice setup and the query-expression perturbation only. The controlled
same-slice branch makes the known one-to-one spot correspondence visible and
contrasts the shift-only and diagonal-affine modes. The real two-slice branch
shows distinct sections, explicitly prohibits spot pairing, and visualizes the
registered `alpha = 0`, `0.5`, and `1.0` perturbation ladder. Correction methods
and evaluation metrics remain in the task-specific diagnostic figures.
All tissue thumbnails use the actual Visium spot coordinates from sections
151673, 151675, and 151676. The setup thumbnails use the real DLPFC layer
annotations; the perturbation thumbnails retain the same real slice geometry
but replace the layer colors with a conceptual orange expression-intensity
overlay. This emphasizes that counts change while spot coordinates, spot
identities, and layer labels remain fixed.

## Integrated supplementary story

`plot_story.py` combines only the already validated Study 04A and Study 04B
atomic tables into
`figures/supplementary/batch_removal_preservation_story.{pdf,svg,png}`.
Panels A–B place batch mixing against exact-pair recovery and local-domain
purity in the controlled same-slice experiment. Panels C–F retain the raw
natural-data anchor and the `alpha = 0`, `0.5`, and registered `1.0` real-slice
ladder. Batch entropy and MMD² summarize complementary neighborhood and
distributional batch-removal behavior; layer macro-F1 and spatial overlap
summarize biological preservation. Batch ASW is omitted as redundant with the
neighborhood-mixing view, and linear CKA remains in the full task-specific
diagnostics as a secondary representation-stability measure. Individual method
seeds and min–max bands remain visible. The deterministic uncorrected PCA
baseline is omitted from this integrated canvas but retained in the full
task-specific diagnostics.

The integrated figure uses no new experiment, composite score, or winner
ranking. Publication-facing panel and legend text uses scientific design names
rather than the internal 04A/04B task identifiers. Compact A–F indices support
manuscript captions without adding explanatory prose to the plotting canvas.
Its provenance is retained with the figure.
It remains a supplementary author-review candidate; the atomic grids,
representation sensitivity, and spatial recovery panels retain their
supplementary roles.

`graphst_pca_sensitivity.pdf` shows the direction-aware PCA-10 minus PCA-20
effect across the same six metrics and eight matched rows through `alpha = 1.0`.
Each metric is scaled
to its own observed PCA-10/PCA-20 span: positive values mean that PCA-10 has
the preferred metric direction, and negative values favor PCA-20. The raw
paired values and the normalized effects are both retained as plot-ready CSVs.
The deterministic PDF, SVG, and 600-dpi PNG candidates; plot-ready primary,
availability, and GraphST sensitivity tables; and `figure_provenance.json` are
written to `figures/same_slice/`. The script verifies every selected input path
and its semantic validation record before rendering.

`plot_additional.py` adds three complementary supplementary diagnostics without
rerunning GraphST, STAMP, or scVI:

- `batch_preservation_tradeoff` shows increasing-alpha trajectories for batch
  mixing versus paired-identity and local-domain preservation. It does not
  calculate a composite score or Pareto ranking.
- `paired_recovery_spatial` maps exact-pair recovery for both registered modes
  at `alpha = 1.0`, using the scorer's primary representations and the common
  DLPFC spot coordinates.
- `representative_embedding_context` shows the uncorrected input and all three
  methods at the diagonal-affine `alpha = 1.0` endpoint, selected by the maximum
  deterministic sampled batch ASW in the uncorrected standardized PCA-20
  representation among the displayed conditions. Each
  column uses an independent deterministic UMAP, so only within-column batch
  mixing and cortical-layer co-localization may be interpreted.

Plot-ready CSVs, the displayed-condition `embedding_condition_selection_data.csv` audit,
and `additional_figure_provenance.json` accompany this figure set.

## Study 04B: real-slice incremental robustness

`plot_real_slice_robustness.py` renders the independently validated Study 04B
production output. Raw 151675 is the fixed reference and raw or
FEAST-perturbed 151676 is the query. The raw pair is displayed as an isolated
natural-data anchor; connected trajectories begin at the `alpha = 0`
resampling control. `alpha = 1.0` is the primary and final displayed endpoint.

The Study 04B outputs are written to `figures/two_slice/`:

- `real_slice_batch_removal_metrics.{pdf,svg,png}`: layer-conditioned batch
  ASW, batch entropy, and MMD². Lower is preferred for ASW/MMD² and higher for
  entropy.
- `real_slice_biological_preservation.{pdf,svg,png}`: layer separation,
  layer purity, reference-to-query layer transfer, spatial-neighborhood
  overlap, within-section neighborhood preservation, and three query-stability
  diagnostics relative to the `alpha = 0` embedding.
- `real_slice_graphst_pca_sensitivity.{pdf,svg,png}`: the frozen GraphST
  PCA-10 primary representation versus its PCA-20 sensitivity representation.

Small points retain all three method seeds and shaded bands show their min–max
range. The uncorrected input is a single deterministic fixed-panel PCA
baseline, so it has no uncertainty band. Its within-section kNN preservation
equals one by construction and must not be interpreted as comparative method
performance.

The figures read the 65-row atomic table only after verifying the complete
independent validation contract: 45/45 method jobs,
65/65 reconstructed rows, 13 reconstructed metrics, and maximum absolute
reconstruction error `1.11e-16`. Plot-ready long-form values, seed summaries,
the complete GraphST sensitivity values, and
`real_slice_figure_provenance.json` accompany the figures.

Rebuild from the repository root with:

```bash
python visualization/04_batch_effect_removal/plot_real_slice_robustness.py
```

Study 04A and Study 04B answer complementary questions and remain separate in
the figure set. Study 04A is the controlled same-slice paired-recovery test;
Study 04B is a real-slice robustness test above natural section differences.
Exact cross-section spot retrieval is neither meaningful nor computed in Study
04B. No composite score, winner ranking, or figure promotion is authorized.

## Local fonts

`plot_task_table.py` and `plot_diagonal_combined.py` use Arial. Supply `arial.ttf`
and `arialbd.ttf` in the Python environment's `fonts/` directory, or set
`FEAST_FONT_DIR` to the directory containing those files. Font files are not
bundled with the source repository.
