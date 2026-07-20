# Study 02: alignment visualization

Visualization code and rendered figures for Study 02 belong here. Build them
only from complete, validated outputs under `../../02_alignment/outputs/`;
historical figures are not inputs.

The clean 20-row Spateo and 20-row PASTE matrix is complete and validated. The
registered source is
`../../02_alignment/outputs/final_rerun_20260718_v2/scores/alignment_metrics.csv`
(SHA-256
`d405d2b8bea4d834cee729013803d2cb9c3ba2193f25d5c9df08dc1446723e92`).

The six prespecified sparsity-condition relations have zero reversals versus
the prior corrected candidate. Historical grid-snapped aggregates are
noncanonical and not directly comparable. Performance varies by alteration,
so no aggregate method ranking, winner claim, or publication figure is
authorized until the author selects the panel, metric, alteration strata, and
claim. Use `comparison_v2/publication_decision.json` for this disposition.

## Unpromoted diagnostic candidate

`plot.py` reads the Study 02 paths and SHA-256 values directly from
`../../PUBLICATION_MANIFEST.json`, requires complete positive validation for
all 40 method rows, and refuses to run if the manifest authorizes a figure or
method-ranking claim. Rebuild from the repository root with:

```bash
python visualization/02_alignment/plot.py
```

The fresh `figures/` candidate contains the PNG/PDF/SVG diagnostic,
`plot_data.csv`, and `figure_provenance.json`. It shows raw rows separately for
each displayed metric and alteration; it makes no cross-alteration aggregate,
method ranking, or winner claim. This candidate is **unpromoted** and remains
**author-selection-required**. It must not be used as a publication figure
until the author declares the panel, metric, alteration strata, and claim.
