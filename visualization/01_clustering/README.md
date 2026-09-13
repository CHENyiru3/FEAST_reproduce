# Study 01: clustering visualization

The primary sensitivity figure combines the Study 01 metrics and decision
registered in `../../PUBLICATION_MANIFEST.json` with the validated additive
mean/variance extreme-level extension. The script verifies the declared source
paths and the extension validation recorded in its `validation.json` before
reading either table. Historical tables and figures are not inputs.

The registered metric source is
`../../01_clustering/outputs/final_rerun_20260718_report/fixed_panel_benchmark_metrics.csv`.
The additive source is
`../../01_clustering/outputs/expanded_mean_variance_fc_0_20_5_20260806/report/expanded_benchmark_metrics.csv`;
its complete 12-simulation/36-method-result validation is read
from the adjacent `validation.json`.
Rebuild from the repository root with:

```bash
python visualization/01_clustering/plot.py
python visualization/01_clustering/plot_spatial_expression.py
python visualization/01_clustering/plot_clustering_spatial.py
python visualization/01_clustering/plot_intervention_profile.py
python visualization/01_clustering/plot_clustering_stability.py
python visualization/01_clustering/plot_main_figure.py
```

`alteration_sensitivity.{pdf,svg,png}` is a 3 × 3 summary (ARI/NMI/AMI ×
mean/variance/sparsity).  Thin traces show the three registered DLPFC slices;
the solid traces and markers show their method-specific mean.  A dashed
vertical line marks the neutral simulation level. Mean and variance include
the added requested FC 0.2 and FC 5 levels and use a log-scaled x-axis so the
extremes and original levels remain legible. These x values are requested
interventions; realized fold changes can differ and are documented in
`realized_alteration_diagnostics.csv`. The figure describes alteration
sensitivity only: it does not claim method superiority, uniform equivalence,
an exact dose-response, or a benchmark winner.

The spatial maps use the registered illustrative slice **151676**.  It has
complete reference, simulation, and method-result artifacts and seven
reference domains; the sensitivity figure continues to include all three
registered slices.  `alteration_spatial_expression.{pdf,svg,png}` has a single
shared log1p(MBP) scale across every panel.  The three
Each `alteration_clustering_spatial_*` figure displays all seven registered
levels of one factor in columns, with the neutral FEAST baseline centred and
the three clustering methods in rows. Colours denote independently produced,
panel-local cluster IDs and do not imply a cross-method domain correspondence.
The adjacent spatial provenance JSON files record exact input paths, source
scripts, display slice, and outputs.

`intervention_profile.{pdf,svg,png}` is a cohort-level simulator diagnostic.
It reads FEAST's recorded realized relative mean, variance, and zero-fraction
changes from every registered simulation and shows the three individual slices
plus their mean. It verifies that a named control produces a measurable
statistical intervention; it is not a clustering score.

`clustering_stability_vs_baseline.{pdf,svg,png}` compares every perturbed
partition with the FEAST-baseline partition from the same method and slice.
It displays the intuitive partition-change quantity **1 − ARI**: zero means an
identical partition and a higher value means a stronger reorganization. ARI
and NMI remain in the adjacent CSV, and both are invariant to a permutation of
cluster labels, so no predicted-to-reference label matching is used. Every
input `clusters.csv` is checked for the recorded method-run schema and spot order. This
is a descriptive response profile, not a cross-method winner ranking.

`clustering_intervention_response.{pdf,svg,png}` is the compact main-figure
candidate. Its left panel shows the pre-specified lower and higher simulation
settings relative to FEAST baseline for each control in representative slice
151676; its right panel shows the corresponding descriptive partition response
across all registered slices. It uses the existing shared spatial-expression
scale and baseline-relative 1 − ARI calculation, and it does not rank methods.
