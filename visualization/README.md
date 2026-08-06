# Visualization

Canonical figure-building layer for the clean FEAST publication reruns. Every
figure script reads only validated or checksum-frozen artifacts inside
`FEAST_reproduce`; the historical `FEAST_experiments` workspace is not an input.
Rendered figures stay within their subtask `figures/` directory. The
repository-level source and disposition index is
[`../FIGURE_SOURCE_MAP.md`](../FIGURE_SOURCE_MAP.md).

## Figure status

| Study | Figure | Status |
|---|---|---|
| 00 simulator benchmark | `simulator_benchmark_boxplots.pdf` | Rendering verified from the scCube-complete table; unpromoted because predecessor-table lineage is incomplete and no overall ranking is authorized. |
| 01 clustering | sensitivity, intervention, stability + spatial maps | **Verified.** 3×3 ground-truth sensitivity, realized-intervention diagnostics, within-method stability, and shared-scale/local-label spatial maps; registered slice 151676 is illustrative only. |
| 02 alignment | `alignment_sensitivity_diagnostic.pdf`, `alignment_spatial.pdf` | Diagnostic only; author metric/panel/scope selection required. Not promoted. |
| 03 deconvolution | `common_support_diagnostic.pdf`, `deconvolution_spatial.pdf` | Diagnostic only; scientific stop active. Not promoted. |
| 04 batch-effect removal | `atomic_metric_diagnostics.pdf`, `graphst_pca_sensitivity.pdf` | Supplementary diagnostics only; no winner ranking authorized. Not promoted. |
| 05 2D conditional transfer | `conditional_transfer_fidelity_and_limits.pdf` | Validated review candidate; claim-reframe required ("conditional-generation" not "accurate spot-level prediction"). |
| 06 conditional 3D stack | `conditional_stack_diagnostic.pdf` | Validated diagnostic from 93/93 fresh outputs; unpromoted pending author review. |
| 07 DevCCF 3D transfer | `full_axis_transfer_diagnostic.pdf` | Validated descriptive diagnostic from 158 + 202 full-axis outputs; no target-expression accuracy claim; unpromoted. |

**Verified rendering** means the current bytes, sources, and requested visual
design have been checked; it does not itself authorize a scientific claim or
article promotion. **Diagnostic** means a figure is rendered for review but not
authorized as a publication claim. **Blocked** means no generation or figure is
authorized.

## Unified style standard

All figure scripts must follow this contract when they are created or revised.
It combines the FEAST publication standard with the validated Study 00
benchmark and spatial-figure patterns. A figure may be scientifically
unpromoted, but its visual encoding and export must still be accurate.

### Reusable plotting contract

Call one `configure_matplotlib()` function before creating a figure. At minimum,
set the following `rcParams`; a study may use a more specific `svg.hashsalt`.

```python
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "feast-studyNN-figure-name",
})
```

### Typography

| Parameter | Value |
|---|---|
| Font family | DejaVu Sans |
| Base font size | 9 pt |
| Panel title | 10–11 pt, bold |
| Axis label | 9–10 pt |
| Tick label | 8 pt |
| Legend text | 8.5–9.5 pt |
| Annotation / footnote | 8–8.5 pt, `#777777`–`#888888` |

### Color

- Use colorblind-friendly palettes (Wong or equivalent).
- Avoid pure red/green pairings.
- Reserve blue `#0072B2` for FEAST or the method under discussion. Use
  reddish-purple `#CC79A7`, vermillion `#D55E00`, orange `#E69F00`, and
  neutral gray `#999999` for distinct comparators or support categories.
- Keep the color-to-method mapping fixed within a figure set; do not reuse a
  method color for a different method in another panel.
- Shaded bands / fill: alpha 0.15–0.20 for print visibility.
- Reference lines: alpha ≤ 0.5, dotted or dashed.
- Do not use color alone for a scientific distinction that must survive
  grayscale reproduction. Add a marker, line style, hatch, or direct label.

### Grid & spines

| Parameter | Value |
|---|---|
| Grid color | `#DDDDDD`–`#E8E8E8` |
| Grid linewidth | 0.5–0.6 pt |
| Grid alpha | 0.65–0.8 |
| Grid axis | y only (default); x only when needed |
| Top / right spines | Removed |
| Remaining spine color | `#555555` |
| Remaining spine linewidth | 0.8 pt |

### Layout

| Parameter | Value |
|---|---|
| Facecolor | White (figure, axes, savefig) |
| Legend frame | Off |
| Legend position | One shared upper/lower-center legend by default; use a panel legend only when it encodes a panel-specific quantity and does not cover data |
| Figure-level title | Omitted for manuscript assembly; a concise title and scope note are allowed for a standalone review candidate |
| Panel labels (A, B, …) | Omitted by default; use only when the caption needs unambiguous panel references |
| Subplot spacing | `hspace` 0.30–0.40, `wspace` 0.22–0.30 |

For a multi-metric benchmark, show the metric direction in every panel title,
keep the x-axis order readable, and state if the order is panel-specific. Use
log axes for orders-of-magnitude metrics rather than visually compressing them.
Show individual observations with box/summary marks when the number of source
datasets is small enough to support that reading.

### Spatial maps and continuous color

- Use a perceptually uniform sequential map (for example, `viridis`) for
  nonnegative expression or abundance. All sources compared within the same
  sample/gene panel must use the same scale.
- Use a diverging map centered at zero (for example, `RdBu_r`) only for signed
  quantities such as simulated-minus-reference expression.
- A colorbar must describe the exact normalization used by every panel it
  serves. Never show a single raw-value colorbar when panels use different
  limits.
- When rows require different symmetric limits, divide each row by its declared
  scale (for example, `q99(|delta|)` pooled across compared methods), label the
  colorbar as row-normalized, and print the raw `±` scale beside the row. This
  permits valid within-row comparison without implying between-row magnitude
  comparison.
- If a common row scale hides near-zero residual structure, a shared monotonic
  contrast transform (such as `AsinhNorm`) may expand the center and compress
  extremes. Apply it identically to every column in the row, retain the raw
  row scale, and name the transform in the colorbar; never replace it with
  method-specific column scales.
- On a white page, a light neutral-gray zero point can make near-matching
  spatial tissue visible while retaining a linear signed scale. State the
  blue/gray/red meaning in the legend or caption.
- When columns use separate expression scales, label the colorbar as
  column-normalized and state the percentile and pooling scope used to compute
  that scale.
- Keep spatial axes equal; hide coordinates only when their numeric values are
  not part of the scientific comparison. Retain a visible colorbar and clear
  row/column labels.
- Read only the required gene/vector from large H5AD inputs in backed mode.
  Deterministically downsample very dense point clouds, size points for the
  plotted density, and rasterize the point layer so PDF/SVG text remains
  editable and the output stays practical.
- For independent clustering results, categorical colours may be local to a
  panel. State this in the subtitle or caption and do not imply that the same
  colour denotes the same domain across methods. Do not remap labels merely to
  create cosmetic agreement. If a representative spatial slice is used,
  record its ID, artifact completeness, and selection rationale; retain the
  complete registered cohort in the quantitative panel.

### Output

| Parameter | Value |
|---|---|
| PDF fonttype | 42 (editable TrueType) |
| PS fonttype | 42 |
| SVG fonttype | `none` (text retained as text) |
| SVG hashsalt | `feast-study{NN}-publication` |
| PNG DPI | 600 |
| Formats | PDF + SVG + PNG for every figure |
| Provenance | `figure_provenance.json` adjacent to every figure set (or a uniquely named companion JSON when multiple scripts share one `figures/` directory) |
| Plotting data | CSV tables alongside figures (plot-ready long-form) |

For a dense spatial figure, vector text and annotations must remain editable;
the point cloud itself may be rasterized. Before publication or handoff, check
that `pdffonts` reports embedded TrueType text and that the SVG contains text
nodes rather than converting all labels to paths.

### Scientific annotations

- No composite score, winner claim, or aggregate ranking unless explicitly
  authorized in the publication decision record.
- Every figure provenance record must state `figure_promotion_authorized: false`
  until the corresponding author decision is recorded.
- Footnote text is omitted by default; if included, use 8 pt `#777777`–`#888888`
  centered below the figure.
- A status caveat must be factual and visually subordinate to the plotted
  evidence; it cannot be used as a result headline.

## Per-study details

### 00 — Simulator benchmark

- **Script:** `00_simulator_benchmark/plot.py`
- **Input:** `../00_simulator_benchmark/outputs/final_metrics/simulator_quality_metrics.csv`
- **Output:** `figures/simulator_benchmark_boxplots.{pdf,svg,png}` (2×4 consolidated) + 8 individual `panel_*.{pdf,svg,png}` + `simulator_delta_spatial.{pdf,svg,png}` + `slice_panel_spatial_comparison.{pdf,svg,png}`
- **Data:** `plot_data_long.csv`, `metric_medians.csv`
- **Design:** 8 metrics displayed; 5 simulators ordered per-panel by median (best→worst); unavailable methods last; log scale on 4 Wasserstein/divergence metrics. No overall ranking.
- **Verified style:** Okabe–Ito method palette; uniform filled-circle dots;
  5-method centered frameless legend; explicit DejaVu Sans rcParams; minimalist
  spines; no near-identity footnote. The delta map uses a row-normalized
  zero-centered colorbar with each raw scale printed at the row label. The
  expression map uses a shared within-column normalized scale. Cos Div is in
  panel G. Scientific promotion remains blocked by the source-lineage
  disposition.

### 01 — Clustering sensitivity

- **Scripts:** `01_clustering/plot.py`, `plot_intervention_profile.py`, `plot_clustering_stability.py`, `plot_spatial_expression.py`, `plot_clustering_spatial.py`
- **Input:** `../01_clustering/outputs/final_rerun_20260718_report/fixed_panel_benchmark_metrics.csv`
- **Output:** `figures/alteration_sensitivity.{pdf,svg,png}` (3×3 ground-truth sensitivity), `intervention_profile.{pdf,svg,png}`, `clustering_stability_vs_baseline.{pdf,svg,png}`, `alteration_spatial_expression.{pdf,svg,png}`, and `alteration_clustering_spatial_{mean,variance,sparsity}.{pdf,svg,png}`
- **Data:** `baseline_reference.csv`, `alteration_plot_data.csv`, `intervention_profile_data.csv`, `clustering_stability_data.csv`
- **Design:** ARI / NMI / AMI rows × Mean / Variance / Sparsity columns quantify agreement with ground truth. `intervention_profile` reads FEAST-recorded realized relative mean, variance, and zero-fraction diagnostics; `clustering_stability_vs_baseline` displays the more direct `1 − ARI` partition-change quantity against the same-method baseline (0 = identical partition; higher = stronger reorganization), while retaining ARI/NMI in its data CSV. Both retain thin individual-slice traces and solid cohort means. Sparsity merges negative and positive logit shifts on a signed x-axis (−1.0 … 0 … +1.0), with a dashed vertical neutral setting. The illustrative spatial maps use slice 151676; expression uses one pooled log1p colour scale. Each clustering-map figure is a 3 × 8 dose sequence: reference domains plus all seven registered levels of one alteration factor, with the neutral FEAST baseline centred. Predicted cluster colours are panel-local (not cross-method domain correspondences).
- **Verified style:** Shared y-axis per metric row; larger labels; upper-right frameless method legend; PDF/SVG/600-DPI PNG exports; spatial maps use rasterized point layers with editable vector text.

### 02 — Alignment sensitivity

- **Scripts:** `02_alignment/plot.py`, `plot_spatial.py`
- **Input:** `../02_alignment/outputs/final_rerun_20260718_v2/scores/alignment_metrics.csv`
- **Output:** `figures/alignment_sensitivity_diagnostic.{pdf,svg,png}` and
  `figures/alignment_spatial.{pdf,svg,png}`
- **Data:** `plot_data.csv`, `alignment_spatial_plot_data.csv`
- **Design:** The quantitative diagnostic is a 2 × 2 layout: PASTE/Spateo
  rows × direct alignment-error columns. Every line preserves all five raw
  rotation results; simulation condition is distinguished by a fixed
  colour/marker mapping. Mean spatial error shares one log scale across
  methods. GE correlation remains in the data table but is not treated as an
  alignment result because it is invariant to method and input rotation. The
  spatial diagnostic uses all four registered conditions at 45°; its rotated,
  Spateo, and PASTE panels share coordinate limits and a reference-x `viridis`
  scale, with gray target-reference points behind each result.
- **Verified style:** DejaVu Sans with embedded TrueType PDF text and editable
  SVG text; y-only grids; shared, frame-free condition legend; 600-DPI PNG;
  rasterized dense point layers with vector text; separate provenance records
  for quantitative and spatial outputs.

### 03 — Deconvolution diagnostic

- **Scripts:** `03_deconvolution/plot.py`, `plot_spatial.py`
- **Input:** `../03_deconvolution/outputs/final_rerun_20260718_v2/scores_common_support_20260719_v1/deconvolution_scores.csv`
- **Output:** `figures/common_support_diagnostic.{pdf,svg,png}` and
  `figures/deconvolution_spatial.{pdf,svg,png}`
- **Data:** common-support score/audit tables plus
  `deconvolution_spatial_plot_data.csv` and `deconvolution_spatial_palette.csv`
- **Design:** The numerical diagnostic remains a common-support, no-ranking
  panel. The spatial composition diagnostic is a 2 × 3 layout for the
  manifest-verified MERFISH slice Zhuang-ABCA-1.100: single cells;
  ground-truth composition at ×0.25 and ×0.10; then matching Cell2location and
  RCTD compositions at the same resolutions. The shared legend contains the
  30 reference classes plus the two explicitly retained slice-100 classes.
  Fine labels are summed by name before plotting. Ground truth, Cell2location,
  and RCTD pie glyphs have matched centres, diameters, colours, ordering,
  background, and shared coordinates at each resolution. One deterministic
  ×0.10 ROI is rendered in three separate enlarged axes, without covering any
  tissue map.
- **Verified style:** White background; 10–11 pt bold panel titles; 3-column
  legend; no map axes; rasterized points/wedges with vector text and ROI boxes;
  PDF/SVG/600-DPI PNG plus spatial source provenance. The scientific
  stop and the prohibition on aggregate rankings remain in force.

### 04 — Batch-effect removal

- **Script:** `04_batch_effect_removal/plot.py`
- **Input:** `../04_batch_effect_removal/outputs/final_rerun_20260718_v2/metrics/atomic_metrics.csv`
- **Output:** `figures/atomic_metric_diagnostics.{pdf,svg,png}`, `figures/graphst_pca_sensitivity.{pdf,svg,png}`
- **Style gaps vs standard:**
  - The current script already sets explicit DejaVu Sans typography,
    `#DDDDDD` y-only grids, and centered/shared legends where appropriate.
  - Its PNG export is 360 DPI and must be raised to 600 DPI before promotion.
  - Extrapolation shading is study-specific and acceptable when the boundary
    at `alpha = 1` remains explicit.

### 05 — 2D conditional transfer

- **Script:** `05_2d_conditional_transfer/plot.py`
- **Input:** `../05_2d_conditional_transfer/outputs/scores/summary.csv`
- **Output:** `figures/conditional_transfer_fidelity_and_limits.{pdf,svg,png}`
- **Style gaps vs standard:**
  - Legend per-axis (standard: shared centered legend)
  - Grid color `#E0DFDF` (close to standard)
  - Has per-axis text annotations (study-specific; acceptable)
  - Figure dimensions 12.4×9.2" (acceptable; 2×2 layout)
  - Default PNG export is 300 DPI and must be raised to 600 DPI before
    promotion.

### 06 — Conditional 3D stack

- **Script:** `06_3d_stack/plot.py`
- **Input:** `../06_3d_stack/outputs/final/evaluation/target_metrics.csv`, `continuity_summary.csv`, validated plan and provenance
- **Output:** `figures/conditional_stack_diagnostic.{pdf,svg,png}`
- **Data:** `target_metric_plot_data.csv`, `continuity_plot_data.csv`
- **Design:** Atomic fidelity by density plus generated-versus-target continuity normalized by the deterministic real split-half baseline. Out-of-bracket label-donor targets are marked. No composite score or winner ranking.

### 07 — Full-axis DevCCF transfer

- **Script:** `07_3d_transfer/plot.py`
- **Input:** `../07_3d_transfer/outputs/final/evaluation/region_coverage.csv`, `adjacent_z_continuity.csv`, both final manifests and validation record
- **Output:** `figures/full_axis_transfer_diagnostic.{pdf,svg,png}`
- **Data:** `region_coverage_plot.csv`, `adjacent_z_plot.csv`
- **Design:** Complete E15.5/E18.5 regional support over the declared z axes plus adjacent-z `1 - Pearson correlation` on logarithmic axes. Descriptive continuity only; no target-expression ground truth or accuracy claim.

## Style and export preflight

Apply this checklist before replacing a canonical figure or requesting author
review:

1. Read only validated/frozen inputs, and preserve every scientific disposition
   and claim restriction in the rendered figure and provenance.
2. Apply the reusable `rcParams` contract, white backgrounds, y-only light
   grid, and removed top/right spines.
3. Use a shared frameless legend when encodings are common across panels; check
   that it does not cover data.
4. Audit every continuous colorbar against its normalization scope. For spatial
   maps, verify common scales within each comparison group and disclose any
   row/column normalization.
5. Export PDF, SVG, and 600-DPI PNG; use `pdf.fonttype: 42`,
   `ps.fonttype: 42`, and `svg.fonttype: "none"`.
6. Inspect the full-size PNG and PDF, then check PDF fonts and SVG text nodes.
7. Emit figure provenance JSON with SHA-256 hashes for all inputs and outputs,
   plus plot-ready CSV tables alongside the figure set.
8. Render first to a temporary `--output-dir` when practical; replace canonical
   files only after visual and export checks pass, then update any recorded
   figure hash.

## Rebuilding

All scripts are self-contained and run from the repository root with the
supported reproduction environment (Python 3.11, NumPy 1.26). Example:

```bash
python visualization/00_simulator_benchmark/plot.py
python visualization/01_clustering/plot.py
python visualization/01_clustering/plot_intervention_profile.py
python visualization/01_clustering/plot_clustering_stability.py
python visualization/01_clustering/plot_spatial_expression.py
python visualization/01_clustering/plot_clustering_spatial.py
python visualization/06_3d_stack/plot.py
python visualization/07_3d_transfer/plot.py
```

Scripts verify their declared input SHA-256 values against the study's frozen
validation/evaluation provenance (and, where applicable, the repository
publication manifest) before reading data. Use `--output-dir` to redirect
output when validating a new candidate run.
