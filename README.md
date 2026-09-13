# FEAST methodology and reproduction

Experiment, preprocessing, evaluation, and figure code for FEAST. The FEAST
method implementation is maintained separately. Datasets, generated results,
figures, logs, and archives stay local and are excluded from the source upload.

## Reproduce

1. Follow [preprocessing/README.md](preprocessing/README.md) to download and
   prepare inputs. Original annotation tables may need to be supplied separately.
2. Use the [environment notes](environments/README.md) and the selected study's
   configuration. Different studies require different FEAST builds.
3. Follow the study README for input paths, generation, methods, scoring,
   validation, and plotting. Use fresh output roots and documented resume options.

| Study | Instructions |
|---|---|
| 00 — Simulator benchmark | [README](00_simulator_benchmark/README.md) |
| 01 — Clustering | [README](01_clustering/README.md) |
| 02 — Alignment | [README](02_alignment/README.md) |
| 03 — Deconvolution | [README](03_deconvolution/README.md) |
| 04 — Batch-effect removal | [README](04_batch_effect_removal/README.md) |
| 05 — 2D conditional transfer | [README](05_2d_conditional_transfer/README.md) |
| 06 — Local 3D reconstruction | [README](06_3d_stack/README.md) |
| 07 — DevCCF expression transfer | [README](07_3d_transfer/README.md) |

Plotting code and workflow illustrations are in [visualization/](visualization/README.md).
Each study README includes its figure instructions. `preprocessing/downloads/`
contains one folder per dataset; `environments/` contains the recorded inventories.
`scripts/` and `publication/` retain validation tools and provenance records.

Study 06's latest recorded precision-build run has not been established complete
by this cleanup. Study 07's current workflow is `two_reference_float32/` within
its study folder; it requires the recorded local FEAST source. A clone alone
does not contain all inputs needed to reproduce existing figures.

The [publication guide](publication/README.md) retains artifact-preservation and
release requirements. Its existing mixed-build manifest check currently fails
at Study 02; this code repository is not a declaration of a final scientific release.
