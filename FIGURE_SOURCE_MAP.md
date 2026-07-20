# Publication figure-source map

Checkpoint: 2026-07-19

This map identifies the current clean-rerun figure candidates and their direct
scientific sources. A rendered figure is not automatically authorized for the
article. Every current figure remains unpromoted until the corresponding author
decision is recorded.

| Study | Review figure | PDF SHA-256 | Direct numerical source | Decision |
|---|---|---|---|---|
| 00 simulator benchmark | `visualization/00_simulator_benchmark/figures/simulator_benchmark_boxplots.pdf` | `93144b0ff3e5be9f639b3d77a9f95ed117d056ee7724ea8bacd3f41c56e14f99` | `00_simulator_benchmark/outputs/final_rerun_20260718_metrics_v2/simulator_quality_metrics.csv` | Unpromoted; author review of the worsened variance-Wasserstein metric required |
| 01 clustering | `visualization/01_clustering/figures/baseline_clustering_comparison.pdf` | `7c2ef87b7b8d536f17619e99de061bb2447a00e2f9d3b52c87b88ec8587283f5` | `01_clustering/outputs/final_rerun_20260718_report/fixed_panel_benchmark_metrics.csv` | Unpromoted; only three-slice method-mean closeness is supported |
| 02 alignment | `visualization/02_alignment/figures/alignment_sensitivity_diagnostic.pdf` | `a41f3258a43c305e1c869aa4246e71860ead9476bd5399c354ca00dbc4d7016c` | `02_alignment/outputs/final_rerun_20260718_v2/scores/alignment_metrics.csv` | Diagnostic only; author must select metric, panel, and alteration scope |
| 03 deconvolution | `visualization/03_deconvolution/figures/common_support_diagnostic.pdf` | `fcc4f60eafb2547dbd4497cac4ca4ca6dea19562da92c0bcce2fc40ef9248e5d` | `03_deconvolution/outputs/final_rerun_20260718_v2/scores_common_support_20260719_v1/deconvolution_scores.csv` | Scientific stop; headline directions changed and no ranking is authorized |
| 04 batch-effect removal | `visualization/04_batch_effect_removal/figures/atomic_metric_diagnostics.pdf` | `8b430773ee6f52975fa811cd3403c8d27826a9352f7119ba6e05aef86b524013` | `04_batch_effect_removal/outputs/final_rerun_20260718_v2/metrics/atomic_metrics.csv` | Supplementary diagnostic only; no winner ranking is authorized |
| 04 batch-effect removal | `visualization/04_batch_effect_removal/figures/graphst_pca_sensitivity.pdf` | `208672181a0d1f6379617e2bed7d2a7e998645ca63ae677027231380b74484ea` | `04_batch_effect_removal/outputs/final_rerun_20260718_v2/metrics/atomic_metrics.csv` plus the declared PCA sensitivity rerun | Supplementary sensitivity diagnostic only |
| 05 2D conditional transfer | `visualization/05_2d_conditional_transfer/figures/conditional_transfer_fidelity_and_limits.pdf` | `03f8fa7c74d9f715e8a8977d518ed6def622f87171b8b0cad65e938574271985` | `05_2d_conditional_transfer/outputs/scores/summary.csv`, support audit, and old-versus-new table | Validated review candidate; use conditional-generation wording, not accurate spot-level prediction |
| 06 conditional 3D stack | Not rendered | — | `06_3d_stack/PREFLIGHT_BLOCKER.md`; no generated output | Blocked by strict reference-only AR preflight failure |
| 07 DevCCF 3D transfer | Not rendered | — | Corrected 550-gene blueprints and E18.5 reference-only AR=0.5; canaries incomplete | Blocked until 158 E15.5 and 202 E18.5 outputs validate |

The machine-readable hash and input lineage for each rendered candidate is in
its adjacent `figures/figure_provenance.json`. SVG counterparts retain text as
text, and PDF files use editable TrueType text where recorded by that
provenance. Historical figures from `FEAST_experiments` are not figure sources.
