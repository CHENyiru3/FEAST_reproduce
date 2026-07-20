# Study 01: clustering visualization

The publication figure candidate is built only from the Study 01 metrics and decision
registered in `../../PUBLICATION_MANIFEST.json`. Historical tables and figures
are not inputs. The script verifies both registered SHA-256 values before it
reads the data.

The registered metric source is
`../../01_clustering/outputs/final_rerun_20260718_report/fixed_panel_benchmark_metrics.csv`
(SHA-256
`10abbadbcf8c26d2b85e92a3357a8380fe4f4bc46882e6d42d2388a3d1882fab`).
It contains complete fixed-panel GraphST, STAGATE+mclust, and label-free Leiden
results. From the supported Python 3.11 / NumPy 1.26 environment, rebuild from
the repository root with:

```bash
python visualization/01_clustering/plot.py
```

The fresh `figures/` directory contains the PDF, PNG, and SVG figure, three
plot-ready CSV files, and `figure_provenance.json` with input and output
SHA-256 values.

Panel A shows FEAST-baseline and raw-slice scores averaged across the same
three slices. Panel B exposes each slice-level difference and the registered
three-slice method mean. The supported interpretation is limited to close
method means across these three slices: the figure does not claim uniform
slice-level equivalence, method superiority, or a benchmark winner, and it
remains unpromoted pending author review.
