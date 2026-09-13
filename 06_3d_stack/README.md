# Study 06: local generative reconstruction

Active configuration: `config_1.0.6_precision.yaml`. Historical workflow files
are in `archive/pre_generative_20260905`; historical outputs remain unchanged and
are legacy. The active output is `outputs/generative_five_reference_1.0.6_precision_v1`.

The latest recorded run uses the separate FEAST 1.0.6 precision build declared
in the configuration. Its September 13 notes report production in progress;
full completion was not established during repository cleanup. Earlier runs
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
