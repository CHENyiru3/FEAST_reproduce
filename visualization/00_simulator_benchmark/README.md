# Study 00: simulator benchmark visualization

> **Author-review stop:** this rendering uses the fresh scCube-complete metric
> root, but it is not a canonical article figure. The predecessor `metrics_v2`
> file was overwritten outside the fresh-only workflow, so its original bytes
> cannot be verified. Do not promote a figure, ranking, or winner claim until
> that source-lineage break and the changed atomic metrics are dispositioned.

The figure-candidate source is the validated 60-row atomic metric table:

```text
../../00_simulator_benchmark/outputs/final_metrics/
    simulator_quality_metrics.csv
```

It contains the fresh default FEAST arm (stored under the traceable source ID
`FEAST_reference_rank`) and four frozen external simulators. The historical
FEAST OT-spatial arm is outside the clean rerun and is not plotted. Eight atomic
metrics are displayed. `gene_variance_wasserstein` is replaced by
`cosine_divergence` per author request; `gene_mean_wasserstein` is omitted as a
mean-fidelity redundancy while mean correlation and relative mean error are
retained. The selection is independent of which method ranks first. The
underlying 60-row table still preserves `gene_variance_wasserstein`, including
its worsening for FEAST. This source does not authorize a composite score, an
overall simulator winner, or a final publication claim without author review.

Within each subpanel, simulators are ordered by the median atomic-metric value
from best to worst, following that metric's higher- or lower-is-better
direction. Unavailable methods appear last. These orders are metric-specific
and do not define an overall simulator ranking.

Study 00 uses a 12-pt base font (rather than the repository-wide 9-pt default)
so labels remain readable in the eight-panel benchmark and the dense spatial
companion figures. This is a presentation-only setting: it does not alter data,
normalization, ordering, or scientific status.

## Metric coverage and overlap

The panel order is mean abundance (`Mean Corr`, then `Real Error Mean`), gene
variance, spatial autocorrelation (Moran I), zero/sparsity, paired spot-level
expression, and total-count scale. `Real Error Mean` is the display name for
the stored `relative_error_mean`: mean absolute gene-mean error normalized by
the reference gene mean. This wording change does not alter its calculation.

No two displayed metrics are identical. The closest pair is `Zero Frac WDist`
(the distribution of each gene's zero fraction) and `Zero Jaccard` (agreement
of the paired spot-by-gene zero mask). They are kept adjacent because they
jointly test sparsity at different granularities; in this 60-row table they are
strongly direction-aligned (Spearman rho about 0.95), so they should not be
read as two independent pieces of evidence. Mean correlation and relative mean
error are likewise complementary: correlation tests cross-gene shape whereas
relative error tests quantitative calibration.

Rebuild from the repository root:

```bash
python visualization/00_simulator_benchmark/plot.py
```

Figures, plot-ready tables, and `figure_provenance.json` are written to the
local `figures/` directory. The renderer also verifies and hash-binds the
Study 00 publication decision, uses deterministic PDF/SVG/PNG metadata, and
records that figure promotion, a composite score, and a winner claim remain
unauthorized. Use `--metrics-csv`, `--decision-json`, and `--output-dir` only
when validating a new candidate run.

## Spatial companion figures

The following visual checks use the same curated six-platform panel and export
PDF, SVG, and 600-DPI PNG files to `figures/`:

```bash
python visualization/00_simulator_benchmark/plot_delta_spatial.py
python visualization/00_simulator_benchmark/plot_slice_panel.py
```

`simulator_delta_spatial` normalizes each platform/gene row by its shared
99th percentile of absolute log1p delta across FEAST, SRTsim, and scCube. Its
linear colorbar is therefore comparable within a row, not between rows. Zero
delta is a light neutral gray rather than white, so near-matching tissue remains
visible against the white page. The row label reports the raw q99 delta scale.
`slice_panel_spatial_comparison` uses a
shared 99.5th-percentile log1p scale across its four sources within each
dataset/gene column; its colorbar reports the resulting column-normalized
expression.
