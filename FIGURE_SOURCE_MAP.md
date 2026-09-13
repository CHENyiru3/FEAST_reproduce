# Figure code and analysis sources

Use this index with the per-study visualization README. Paths below are relative
to the repository root. Analysis inputs and rendered outputs are local and are
not included in the code-only GitHub repository. This index does not select
final manuscript figures or authorize scientific claims.

| Study | Figure code and commands | Main local analysis sources |
|---|---|---|
| 00 | [Simulator benchmark](visualization/00_simulator_benchmark/README.md) | `00_simulator_benchmark/outputs/final_metrics/`, fresh simulations and external simulator inputs |
| 01 | [Clustering](visualization/01_clustering/README.md) | `01_clustering/outputs/final_rerun_20260718_report/` and registered perturbation outputs |
| 02 | [Alignment](visualization/02_alignment/README.md) | `02_alignment/outputs/paste2_rerun_20260807_v1/` and `fixed_plate_rerun_20260806_v1/` |
| 03 | [Deconvolution](visualization/03_deconvolution/README.md) | `03_deconvolution/outputs/cell_class_rerun_20260810_v1/` |
| 04 | [Batch-effect removal](visualization/04_batch_effect_removal/README.md) | `04_batch_effect_removal/outputs/final_rerun_20260718_v2/` and `real_slice_robustness/outputs/production_v1/` |
| 05 | [2D conditional transfer](visualization/05_2d_conditional_transfer/README.md) | `05_2d_conditional_transfer/outputs/final/`, `scores/`, and `baselines/` |
| 06 | [3D reconstruction](visualization/06_3d_stack/README.md) | Select an explicitly validated Study 06 output root; latest recorded run: `generative_five_reference_1.0.6_precision_v1` |
| 07 | [DevCCF transfer](visualization/07_3d_transfer/README.md) | `07_3d_transfer/outputs/generative_transfer_two_reference_float32_v1/`; descriptive resampling control under `outputs/final/baselines/` |
| 08 | [Workflow illustrations](visualization/08_workflow/README.md) | Processed transcriptomic sections, generated sections, and DevCCF atlas volumes |

Some composition scripts read tables produced by earlier plotting scripts;
follow the study's command order. Older figure sets and their original source
map are retained locally under `.archive/`. Existing numerical provenance
records stay beside the local outputs.
