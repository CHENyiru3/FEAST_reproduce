# Study 03: deconvolution visualization

This directory renders the completed, corrected common-support evidence selected
by `../../PUBLICATION_MANIFEST.json`. The input consists of all six fresh
same-expression comparisons for RCTD and Cell2location, scored on the exact
positive-spot intersection and the common named cell-type support plus `Other`.

The result is a diagnostic candidate, not an authorized publication figure.
Pearson correlation and summed RMSE changed their historical comparative
directions. The scientific stop therefore remains active: no publication claim,
aggregate winner, or method ranking is authorized.

Rebuild from the repository root with the supported reproduction environment:

```bash
python visualization/03_deconvolution/plot.py
```

The deterministic PDF, SVG, and PNG candidate; plot-ready score, support, and
direction tables; and `figure_provenance.json` are written to `figures/`.
The script verifies every selected input against its manifest SHA-256 before
rendering.

## Spatial deconvolution composition diagnostic

`plot_spatial.py` replaces the earlier dominant-cell-type map with a
composition-preserving comparison of both inference methods:

```text
single-cell MERFISH | Ground truth ×0.25 | Ground truth ×0.10 | enlarged ROI
32-class legend     | Cell2location ×0.25 | Cell2location ×0.10 | enlarged ROI
32-class legend     | RCTD ×0.25         | RCTD ×0.10         | enlarged ROI
```

It uses the manifest-verified `Zhuang-ABCA-1.100` simulation, ground truth,
Cell2location, and RCTD outputs as one internally consistent experiment. Its
30-class reference display ontology is supplemented by the two biologically
observed, explicitly labelled classes `16 HY MM Glut` and `25 Pineal Glut`; no
class is removed or merged for visual convenience. Fine cell labels are summed
by name before rendering. At each resolution, ground truth, Cell2location, and
RCTD use exactly the same spot centres, pie-glyph radii, fixed wedge order,
palette, white background, coordinate limits, and fixed ROI. The three ×0.10
ROI enlargements are separate axes rather than overlays, so no tissue data are
covered. Bins with zero source cells are hidden from all three composition maps:
Cell2location's corresponding normalized model outputs remain in the exported
long-form table but are not treated as an observable spatial composition.

Rebuild with:

```bash
python visualization/03_deconvolution/plot_spatial.py
```

The script writes `deconvolution_spatial.{pdf,svg,png}`, its long-form spatial
plot table, palette table, and `deconvolution_spatial_provenance.json`. It
checks the displayed reference, simulations, truth proportions, and
Cell2location and RCTD outputs against the simulation and method manifests.
This remains a descriptive visualization; it does **not** override the Study 03
scientific stop or authorize an aggregate method ranking.

For an alternate registered slice, request an explicit preview. For example,
slice 050 contains `15 HY Gnrh1 Glut`, which is outside the 30-class primary
ontology and is therefore retained as a labeled 31st category rather than
silently merged:

```bash
python visualization/03_deconvolution/plot_spatial.py \
  --source-case 050 --include-additional-classes \
  --output-dir /tmp/study03_case050_preview
```
