# Figures

Each study README contains its plotting commands, required inputs, and scientific
limits. Run analysis and validation first; some composite figures read tables
from earlier plotting steps. PDF/SVG/PNG, plot-data tables, and provenance stay
in ignored local `figures/` directories.

| Study | Analysis and figure instructions |
|---|---|
| 00 simulator benchmark | [README](../00_simulator_benchmark/README.md#figures) |
| 01 clustering | [README](../01_clustering/README.md#figures) |
| 02 alignment | [README](../02_alignment/README.md#figures) |
| 03 deconvolution | [README](../03_deconvolution/README.md#figures) |
| 04 batch-effect removal | [README](../04_batch_effect_removal/README.md#figures) |
| 05 conditional transfer | [README](../05_2d_conditional_transfer/README.md#figures) |
| 06 local 3D reconstruction | [README](../06_3d_stack/README.md#figures) |
| 07 DevCCF transfer | [README](../07_3d_transfer/README.md#figures) |

## Workflow illustrations

`08_workflow/` builds data-derived components and anatomical figures for manual
assembly. From the repository root:

```bash
python visualization/08_workflow/build_components.py
python visualization/08_workflow/build_anatomical_workflow.py
python visualization/08_workflow/build_anatomical_workflow.py --components-only
```

These require the processed transcriptomic sections, generated examples, and
DevCCF volumes, plus the recorded `scikit-image`/`trimesh` environment. The default
Study 06 illustration uses 082 → 083 → 084; Study 07 uses the E15.5 atlas and
its generated section. Use `--output-dir` for a new rendering directory.
Expression-free blueprint components do not read held-out target expression.
The figures do not validate either study or establish target-expression accuracy.

## Figure conventions

Retain each script's established design and recorded selection/normalization.
Defaults are DejaVu Sans, white backgrounds, frameless legends, colorblind-safe
colors, light y-grids, and hidden top/right spines. Keep FEAST blue (`#0072B2`).
Study 00 uses 12-point base text; other existing scripts may have documented
font overrides. New or revised figures should retain readable editable text.

Export PDF/SVG and 600-DPI PNG; use PDF/PS fonttype 42 and SVG fonttype `none`.
Rasterize dense spatial marks while keeping text/vector annotations editable.
Keep plot-ready CSVs and a uniquely named provenance JSON beside each figure
set. Check embedded PDF fonts and SVG text before manuscript assembly.

Use equal spatial aspect ratios and explicitly labeled shared color scales.
Use sequential maps for nonnegative quantities and zero-centered diverging maps
for signed differences. Record percentile caps and transforms; never imply
cross-panel comparability when normalization differs. Cluster colors are
panel-local unless an actual correspondence is established. Record illustrative
slice/gene selection and keep the registered cohort in quantitative comparisons.

No composite, aggregate ranking, winner claim, or figure promotion is authorized
by rendering. Provenance retains `figure_promotion_authorized: false` until the
corresponding author decision is recorded. Existing claim limitations in the
study and publication records remain binding.
