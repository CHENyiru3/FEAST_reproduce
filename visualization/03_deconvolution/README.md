# Study 03: deconvolution visualization

The current figures use the independently validated `cell_class` rerun in
`../../03_deconvolution/outputs/cell_class_rerun_20260810_v1`. The previous
fine-`cell_type` diagnostic is retained only in the `.work/` backup and the
historical output roots; it is not a source for these figures.

Both RCTD and Cell2location receive the same fresh simulation for every
slice-resolution pair, are scored on the exact same positive-library spots,
and use the same slice-specific classes with at least 50 reference cells. The
retained supports are 7 classes for S007, 14 for S050, and 15 for S100. Truth
mass from rarer classes is represented as `Other` for both methods.

## Numerical comparison

Rebuild with:

```bash
python visualization/03_deconvolution/plot.py
```

`plot.py` delegates to `plot_cell_class.py`, which requires the scorer
provenance and `validation_cell_class.csv` before rendering. It writes
`common_support_diagnostic.{pdf,svg,png}`, the plot-ready score/support tables,
and `figure_provenance.json` to `figures/`.

Arial is embedded from the active Conda environment's `mscorefonts`
installation.

The figure connects ×0.10 and ×0.25 only within the same slice and method; no
line crosses slices. A wider horizontal method dodge, taller panels, and tighter
data-derived y-ranges separate near-equal points without changing their y-values.
It reports mean Jensen–Shannon divergence, mean Pearson correlation, and RMSE.
No panel letters or aggregate winner are rendered. Across the six registered
pairs, RCTD has a slightly lower mean JSD (0.0638 versus 0.0701), Cell2location has a lower mean
RMSE (0.0576 versus 0.0614), and mean Pearson correlation is nearly identical
(0.8438 versus 0.8441). This is a mixed descriptive result, not a method
ranking.

The figure and its provenance are retained in the figures directory.

## Spatial composition comparison

Rebuild with:

```bash
python visualization/03_deconvolution/plot_spatial.py
```

The spatial figure uses the validated S100 `cell_class` run at resolutions
×0.25 and ×0.10. Its rows show ground truth, Cell2location, and RCTD; the final
column shows the same fixed ROI for all three rows. Every panel within a
resolution uses identical centers, pie radii, class order, colors, background,
and coordinate limits. Empty locations are hidden consistently, and the
Cell2location output records them as reinserted zero rows.

The S100 reference contains 26 observed classes, while both prediction methods
fit the same 15 classes that meet the ≥50-cell threshold. For a directly
comparable display, all panels use those same 15 named classes plus `Other`;
the latter aggregates truth mass from the 11 rarer classes. The legend therefore
contains exactly 16 entries and no absent S100 classes. Panel letters are omitted
for manuscript-stage lettering. Eight formerly confusable palette entries were
shifted across indigo, pink, forest green, dark mustard, teal, violet, brown, and
magenta while preserving one fixed class-to-color mapping across all panels.

The PDF and SVG rasterize single-cell points and pie wedges at 600 DPI, which
keeps the dense panels practical to edit in Illustrator. Text, legend swatches,
and ROI outlines remain vector and editable. Arial is embedded from the active
Conda environment's `mscorefonts` installation. The PNG is also exported at
600 DPI. The script writes the long-form
plot table, palette table, and `deconvolution_spatial_provenance.json`.

The spatial figure and its provenance are retained in the figures directory.
