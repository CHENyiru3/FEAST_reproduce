# Figure style

## Unified style standard

All figure scripts must follow this contract when they are created or revised.
It combines the FEAST publication standard with the validated Study 00
benchmark and spatial-figure patterns. A figure may be scientifically
unpromoted, but its visual encoding and export must still be accurate.

### Reusable plotting contract

Call one `configure_matplotlib()` function before creating a figure. At minimum,
set the following `rcParams`.

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
