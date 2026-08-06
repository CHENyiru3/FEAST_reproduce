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

## Unpromoted diagnostic candidates

`plot.py` reads the Study 02 paths and SHA-256 values directly from
`../../PUBLICATION_MANIFEST.json`, requires complete positive validation for
all 40 method rows, and refuses to run if the manifest authorizes a figure or
method-ranking claim. Rebuild from the repository root with:

```bash
python visualization/02_alignment/plot.py
```

`plot.py` produces `alignment_sensitivity_diagnostic.{pdf,svg,png}` together
with `plot_data.csv` and `figure_provenance.json`. The compact 2 × 2 layout
uses rows for the alignment method and columns for the two direct alignment
errors. Lines retain every validated rotation job; colour and marker encode the
four simulated conditions. Mean spatial error uses a shared log scale across
methods. Rotation-recovery panels retain raw linear units with a separate
within-method range so zero-valued solutions remain legible. GE correlation is
retained in `plot_data.csv`, but is not a plotted alignment outcome because it
is invariant across methods and input rotations in this design.

`plot_spatial.py` produces `alignment_spatial.{pdf,svg,png}` with the
plot-ready `alignment_spatial_plot_data.csv` and its own provenance record. It
shows the four registered conditions at the prespecified 45° input rotation.
Every panel shares the same coordinate limits and reference-x colour scale;
the light-gray background is the target reference geometry. The script checks
every displayed H5AD against the rotation and method manifests before reading
coordinates.

Both are **unpromoted** descriptive diagnostics. They preserve raw condition
and method results but make no cross-alteration aggregate, method ranking, or
winner claim. They must not be used as publication figures until the author
declares the panel, metric, alteration strata, and claim.
