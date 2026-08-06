# Publication figure-source map

Checkpoint: 2026-08-06

This map identifies the current clean-rerun figure candidates and their direct
scientific sources. A rendered figure is not automatically authorized for the
article. Every current figure remains unpromoted until the corresponding author
decision is recorded.

| Study | Review figure | PDF SHA-256 | Direct numerical source | Decision |
|---|---|---|---|---|
| 00 simulator benchmark | `visualization/00_simulator_benchmark/figures/simulator_benchmark_boxplots.pdf` | `86bb513e782bead9e4d8c6708796edc216c5163c350ac47766e3bcb5f1835ae3` | `00_simulator_benchmark/outputs/final_metrics/simulator_quality_metrics.csv` | Fresh scCube-complete rendering validated but unpromoted; the overwritten predecessor-table lineage and worsened variance-Wasserstein metric require author disposition, and no ranking is authorized |
| 01 clustering | `visualization/01_clustering/figures/alteration_sensitivity.pdf` | `5a4ee0bd3132f3bc7e8e304aa37316091cfc385395e4c77a8b757200c0679bb9` | `01_clustering/outputs/final_rerun_20260718_report/fixed_panel_benchmark_metrics.csv` | Rendering verified; the scientific statement remains limited to three-slice method-mean closeness |
| 02 alignment | `visualization/02_alignment/figures/alignment_sensitivity_diagnostic.pdf` | `de61d0424ab1959cfbb0d4846442c15c239e27e76ed426e31cac52b4492e1f49` | `02_alignment/outputs/final_rerun_20260718_v2/scores/alignment_metrics.csv` | Diagnostic only; author must select metric, panel, and alteration scope |
| 03 deconvolution | `visualization/03_deconvolution/figures/common_support_diagnostic.pdf` | `f090f8967fba63c0296ffdc1db91f25ec64d40a8288cf6bd84cf4153b221c3cc` | `03_deconvolution/outputs/final_rerun_20260718_v2/scores_common_support_20260719_v1/deconvolution_scores.csv` | Scientific stop; headline directions changed and no ranking is authorized |
| 03 deconvolution | `visualization/03_deconvolution/figures/deconvolution_spatial.pdf` | `330ab6eda1107c0e98d8493d283ba4d9a4eef55b39385b6705b77922b94aad3b` | Manifest-verified Zhuang-ABCA-1.007 reference, simulations, truth proportions, and Cell2location proportions at resolutions 0.25 / 0.10 | Descriptive Cell2location-versus-ground-truth composition diagnostic only; it does not lift the scientific stop or authorize a method ranking |
| 04 batch-effect removal | `visualization/04_batch_effect_removal/figures/atomic_metric_diagnostics.pdf` | `8b430773ee6f52975fa811cd3403c8d27826a9352f7119ba6e05aef86b524013` | `04_batch_effect_removal/outputs/final_rerun_20260718_v2/metrics/atomic_metrics.csv` | Supplementary diagnostic only; no winner ranking is authorized |
| 04 batch-effect removal | `visualization/04_batch_effect_removal/figures/graphst_pca_sensitivity.pdf` | `208672181a0d1f6379617e2bed7d2a7e998645ca63ae677027231380b74484ea` | `04_batch_effect_removal/outputs/final_rerun_20260718_v2/metrics/atomic_metrics.csv` plus the declared PCA sensitivity rerun | Supplementary sensitivity diagnostic only |
| 05 2D conditional transfer | `visualization/05_2d_conditional_transfer/figures/conditional_transfer_fidelity_and_limits.pdf` | `03f8fa7c74d9f715e8a8977d518ed6def622f87171b8b0cad65e938574271985` | `05_2d_conditional_transfer/outputs/scores/summary.csv`, support audit, and old-versus-new table | Validated review candidate; use conditional-generation wording, not accurate spot-level prediction |
| 06 conditional 3D stack | `visualization/06_3d_stack/figures/conditional_stack_diagnostic.pdf` | `7f018ad5af52557240fd455b4546436204ced627a8f6cb0aa55d3f884bc3bb80` | `06_3d_stack/outputs/final/evaluation/target_metrics.csv`, `continuity_summary.csv`, validation summary, and final plan | Validated 93-output diagnostic; unpromoted, no composite score or winner ranking |
| 07 DevCCF 3D transfer | `visualization/07_3d_transfer/figures/full_axis_transfer_diagnostic.pdf` | `62eb4085709d39e2abfea521e8ee33df8cba9b3131ce21eb8a64a831c6ddd098` | `07_3d_transfer/outputs/final/evaluation/region_coverage.csv`, `adjacent_z_continuity.csv`, final validation, and both age manifests | Validated descriptive full-axis diagnostic; unpromoted and no target-expression accuracy claim |

The machine-readable hash and input lineage for each rendered candidate is in
its adjacent `figures/figure_provenance.json`. SVG counterparts retain text as
text, and PDF files use editable TrueType text where recorded by that
provenance. Historical figures from `FEAST_experiments` are not figure sources.
