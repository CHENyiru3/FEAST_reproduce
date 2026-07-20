# Study 00: simulator benchmark visualization

> **Author-review stop:** this rendered figure still uses the pre-repair metric
> root. A coordinate-support repair recovered pairwise metrics for 20 rows and
> left only `scCube/Slideseq_001` unresolved. Do not promote or publish this
> figure until the new 11-dataset complete-case pairwise results are reviewed
> and an updated rendering is authorized.

The figure-candidate source is the validated 60-row atomic metric table:

```text
../../00_simulator_benchmark/outputs/final_rerun_20260718_metrics_v2/
    simulator_quality_metrics.csv
```

It contains the fresh default FEAST arm (stored under the traceable source ID
`FEAST_reference_rank`) and four frozen external simulators. The historical
FEAST OT-spatial arm is outside the clean rerun and is not plotted. Eight atomic
metrics are displayed. `cosine_divergence` is omitted because it is a redundant
profile-similarity summary with incomplete method support;
`gene_mean_wasserstein` is omitted as a mean-fidelity redundancy while mean
correlation and relative mean error are retained. The selection is independent
of which method ranks first. `gene_variance_wasserstein` worsened and remains
visible. This source does not authorize a composite score, an overall simulator
winner, or a final publication claim without author review.

Within each subpanel, simulators are ordered by the median atomic-metric value
from best to worst, following that metric's higher- or lower-is-better
direction. Unavailable methods appear last. These orders are metric-specific
and do not define an overall simulator ranking.

Rebuild from the repository root:

```bash
python visualization/00_simulator_benchmark/plot.py
```

Figures, plot-ready tables, and `figure_provenance.json` are written to the
local `figures/` directory. The renderer also verifies and hash-binds the
Study 00 publication decision, uses deterministic PDF/SVG/PNG metadata, and
records that figure promotion, a composite score, and a winner claim remain
unauthorized. Use `--metrics-csv`, `--decision-json`, and `--output-dir` only
when validating a new candidate run.
