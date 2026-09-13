# FEAST methodology and reproduction

Code for the FEAST simulation studies, downstream benchmarks, and manuscript
visualizations. The FEAST method implementation is maintained separately; this
repository contains the experiment configurations, runners, evaluation code,
and figure builders.

This is a **code-only repository**. Datasets, generated H5ADs, fitted models,
metrics, rendered figures, logs, and local archives are excluded from Git.
Use the [preprocessing scripts](preprocessing/README.md) to prepare upstream
inputs, then run the relevant analysis before building its figures.

## Studies

| Study | Scope | Instructions |
|---|---|---|
| 00 | Simulator quality benchmarks | [Simulator benchmark](00_simulator_benchmark/README.md) |
| 01 | Clustering under expression perturbations | [Clustering](01_clustering/README.md) |
| 02 | Fixed-plate alignment with PASTE2 and Spateo | [Alignment](02_alignment/README.md) |
| 03 | Cell-class deconvolution with RCTD and Cell2location | [Deconvolution](03_deconvolution/README.md) |
| 04 | Controlled batch removal and real-slice robustness | [Batch-effect removal](04_batch_effect_removal/README.md) |
| 05 | Cross-slice transfer and half-slice conditional generation | [2D transfer](05_2d_conditional_transfer/README.md) |
| 06 | Local 3D reconstruction across reference densities | [3D reconstruction](06_3d_stack/README.md) |
| 07 | Expression transfer to DevCCF atlas coordinates | [3D transfer](07_3d_transfer/README.md) |

## Reproduce a study

1. Select a study and read its README and YAML configuration. Working-directory
   conventions differ; use the directory shown in that study's commands.
2. Install the study's recorded FEAST build and the required external-method
   environments. See [environment notes](environments/README.md) and
   [the original FEAST build record](FEAST_BUILD.txt). Studies use different
   builds; one installation does not reproduce every study.
3. Supply the processed datasets at the configured paths, usually under
   `<study>/data/local/`. The small `data/input_checksums.csv` files record input
   identities. [Download helpers](preprocessing/downloads/README.md) and
   [preprocessing instructions](preprocessing/README.md) are included; some
   annotation tables must be supplied separately.
4. Run preparation/generation, downstream methods, scoring, and validation in
   the order documented by the study. Use a fresh output directory for a new
   run and only the documented resume options for interrupted work.
5. Build figures with the scripts in [visualization/](visualization/README.md).
   The [source map](FIGURE_SOURCE_MAP.md) connects plotting code to analysis
   inputs. Generated figures and plot-data tables stay local.

Study 07's completed two-reference float32 workflow is in
[`07_3d_transfer/two_reference_float32/`](07_3d_transfer/two_reference_float32/README.md).
Study 06's latest recorded configuration is `config_1.0.6_precision.yaml`;
its full production completion was not established by this cleanup.

## Repository layout

- `preprocessing/`: upstream converters, annotation code, and download helpers.
- `00_*`–`07_*`: study code, configurations, input identifiers, and tests.
- `visualization/`: plotting scripts and workflow illustrations.
- `environments/`: recorded dependency inventories and setup notes.
- `scripts/`, `publication/`: existing validation and publication-provenance tools.
- `.archive/`, `.work/`, `outputs/`, `figures/`: local artifacts, excluded from Git.

The [repository review](docs/REPOSITORY_REVIEW.md) records remaining portability
and reproducibility issues. Historical execution notes are archived locally;
[ARCHIVE_POLICY.md](ARCHIVE_POLICY.md) describes preservation and release rules.
The publication manifest retains its existing release-authorization status;
code preparation does not declare a final scientific release.
