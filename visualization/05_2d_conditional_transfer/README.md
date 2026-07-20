# Study 05 visualization

`plot.py` builds the publication-review figure only from the validated fresh
score, comparison, and complete-validation records. It distinguishes DLPFC
from MERFISH, cross-slice from half-slice generation, within-donor from
cross-donor DLPFC transfer, the predeclared primary assignment randomness
(`0.3`), and sensitivity settings. Target-support and globally
reference-observed-gene fractions are shown separately.

Run from the clean FEAST environment:

```bash
python visualization/05_2d_conditional_transfer/plot.py
```

The fresh `figures/` root contains editable PDF/SVG, a PNG preview, three
plotting-data tables, and a provenance JSON binding all 45 validated H5AD
hashes. PDF fonts use type 42 and SVG text is retained as text for editing in
Adobe Illustrator.

The figure is evidence for author review, not authorization for an accurate
spot-level imputation claim. It explicitly shows that high aggregate
distribution and Moran-profile fidelity coexist with near-zero DLPFC median
spotwise gene correlations. Historical 2D-transfer figures remain
noncanonical and are not source material.
