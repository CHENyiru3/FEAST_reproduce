# Study 06: local generative reconstruction

Active configuration: `config_1.0.6_precision.yaml`. Historical workflow files
are in `archive/pre_generative_20260905`; historical outputs remain unchanged and
are legacy. The active output is `outputs/generative_five_reference_1.0.6_precision_v1`.

The latest recorded run uses the separate FEAST 1.0.6 precision build declared
in the configuration. It completed on September 13 at 08:25 Chicago time:
all 336 targets passed independent validation with positive transport convergence,
and evaluation and diagnostic figures completed successfully. Earlier runs
remain local and separate. This source repository does not include their outputs.

The three retained reference pools contain 49, 30, and 17 slices. Generate 95,
114, and 127 targets, respectively, covering 144 positions per reconstruction.
Slices 4–150, absent IDs 75/94/112, boundary anchors 4/150, gap-10 anchor 89, and
all 1,122 genes are preserved. Target expression is evaluation-only.

`FEAST.simulate_local_references` selects five primary references, including the
actual-z bracket. Its exponential bandwidth is the median adjacent actual-z
spacing of the retained pool. Local groups with positive reference populations
below 50 merge through within-slice 6-NN contacts. Original labels are retained.
Explicit supporting donors remain separate references and contribute only to
unsupported groups. There is no cross-z smoothing or global statistical model.

Reference gene-statistic tables use generative parameter clouds, hybrid global
SciPy Hungarian assignment, threefold candidates, and interpolated PPFs. FEAST
1.0.6 uses direct Student-t inversion at quantiles >= 0.995 and numerical scaling
to prevent spline overflow. Candidates exceeding float64 use extended precision
until the existing bounded assignment features are calculated; selected parameters
must be representable in float64 before count generation. Tables
are cached by reference identity, merged members, genes and parameter seed.
Each target fuses tables in log/logit space, applies one shared theta-space
batch draw (SD 0.005), then uses full count conversion at the target group size
and the spatial-intensity decoder with boundary multiplier 1.1.

Run `preflight.py --config config_1.0.6_precision.yaml --data-dir ... --output-dir <fresh-run>`
with the installed local research build. It performs three whole-reference
holdouts per density and the AR grid 0–0.5 by 0.05, fitting all genes and scoring
20 genes selected only from training references. Generation uses strict CUDA
float64 OT, epsilon 0.05, 1,000 iterations, tolerance 1e-5, 25-million-pair cap,
and raises on nonconvergence. Use two workers on GPU 0 with two CPU threads each.
Run `run.py --output-dir <fresh-run> --gap 3 --shard-index 0 --shard-count 2` (and the
other nonoverlapping shards), then `validate.py` and `aggregate.py`.

`diagnose_generative.py` is a bounded two-reference empirical, two-reference
core, and five-reference core diagnostic. Its smaller reference scope is
explicitly recorded and is not production calibration.


Pass the selected configuration to `preflight.py`. Generation, validation, and
aggregation read `frozen_config.yaml` from that output root; they do not accept
`--config`. Inspect `--help` for each stage. The older
`config_reference_density.yaml` and `config_1.0.6.yaml` are retained for prior
runs and existing validation tools. Do not infer the intended run from defaults.
Historical launch commands and session logs are local archive material.

## Five-reference resampling baseline

The current baseline completed all 336 targets × 10 repeats on September 13.
Matched comparison metrics, full-stack continuity summaries and comparison
figures are available in the directories below.

`conditional_resampling_baseline.py` samples complete count vectors with
replacement within original cell classes, using the completed plan's five
reference weights. Weights are renormalized over references supporting the
class; unsupported classes use their declared donor. This baseline retains
original classes rather than applying FEAST's modeling-group merges.

Run the combined evaluator from the repository root:

```bash
python 06_3d_stack/evaluate_resampling_metrics.py \
  --output-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1
```

It evaluates all 336 targets with ten repeats (seeds 2026–2035), saving metrics
and summaries under `baselines/five_reference_conditional_resampling/` within
the selected output root. It reuses the established conditional Wasserstein and
Moran-profile metrics, also computes the standard atomic metrics, and evaluates
continuity across the same 144-position stacks including real anchors. Target
expression and XY coordinates are evaluation-only. No bootstrap H5ADs are saved.

After evaluation, render the existing comparison:

```bash
python visualization/06_3d_stack/plot_resampling_comparison.py \
  --input-dir 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1/baselines/five_reference_conditional_resampling \
  --output-dir visualization/06_3d_stack/figures/five_reference_resampling_v1
```

Use a fresh output directory for an additional run. Repeat variation describes
Monte Carlo variability, not biological replication. The archived two-bracket
baseline scripts retain their historical method and are not the current entry points.

## Native SpatialZ benchmark

SpatialZ generation and slice-level comparison are implemented in
`spatialz_generate.py` and `spatialz_evaluate.py`, using
`config_spatialz.yaml`. SpatialZ generates its own coordinates, cell count,
class labels, and expression from two bracketing retained references.
It shares the existing 336 held-out target positions and 1,122-gene panel.
See [SPATIALZ.md](SPATIALZ.md) for the selected tutorial recipe, coordinate
units, metrics for unmatched cells, environment, and CUDA/tmux commands.
`run_spatialz.sh` runs the three density arms and their matched evaluation.

## Figures

Run from the repository root after validation.

The publication overview and along-z profiles now include FEAST, resampling
and SpatialZ. Native marker maps use each method's coordinates and the
existing four-gene panel:

```bash
python visualization/06_3d_stack/plot_results.py \
  --input-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  --spatialz-root 06_3d_stack/outputs/spatialz_native_evaluation_cuda_v1 \
  --output-dir visualization/06_3d_stack/figures/publication_with_spatialz_v1
```

Exports include PDF, editable SVG, PNG, plot-data tables and captions.
The marker maps show Slc17a7, Gad2, Gfap and Reln on slice 5, the first slice held
out in all three density arms. Every cell is shown on its method's own
coordinates, using shared axes and gene-wise color scales. The three-method
overview uses completed metrics that support unmatched cells; it does not
compute cellwise cosine distances or zero-mask overlap for SpatialZ.
Reference coverage remains separate at 2.9 × 1.5 inches. Full-stack z coherence
is exported separately and explicitly labeled FEAST-only.

Native-coordinate volume views, marker maps for sections 5/78/149, class-marker
heatmaps and an offline Observed/FEAST/SpatialZ explorer are generated with:

```bash
python visualization/06_3d_stack/plot_spatialz_views.py
```

These are saved under `visualization/06_3d_stack/figures/spatialz_native_views_v1`.
See [the visualization guide](../visualization/06_3d_stack/README.md) for all
updated entry points and output locations.

The earlier FEAST-only matched blue/red difference maps and explorer remain
available with:

```bash
python visualization/06_3d_stack/plot_volume_differences.py \
  --input-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  --output-dir visualization/06_3d_stack/figures/whole_volume_1.0.6
```

Use `--render-only` to reuse its saved plot data for layout updates. The volume
views display the same sampled cells across methods. Differences are
`log1p(simulated counts) - log1p(observed counts)`, with shared gene-wise color
limits based on all generated cells. Retained anchors are excluded from error
statistics. Sections 5, 78 and 149 are selected by position, not performance.

The original compact diagnostic remains available:

```bash
python visualization/06_3d_stack/plot.py \
  --input-root 06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1 \
  --output-dir visualization/06_3d_stack/figures/precision
```

This requires that run's completed validation and evaluation artifacts. Other plotting scripts retain older
93-target defaults and must not be presented as results for the corrected
144-position stacks. Ordered z levels are not independent biological replicates;
no composite, winner claim, or automatic figure promotion is authorized.

## Class-marker and GO preservation

The requested class-versus-rest analysis is implemented in
`pathway_analysis/run.py` and `pathway_analysis/summarize.py`. It uses the
provided mouse GO BP 2025.1 GMT and the existing Study 05 pathway environment.
See `pathway_analysis/README.md` for commands, explicit parameters, progress
locations and interpretation. This is descriptive recovery given supplied
class labels; serial slices and cells are not independent donor replicates.
