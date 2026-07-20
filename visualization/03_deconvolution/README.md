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
