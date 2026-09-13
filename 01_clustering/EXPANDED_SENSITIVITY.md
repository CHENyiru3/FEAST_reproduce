# Study 01 additive mean/variance extremes

This extension adds mean FC 0.2 and 5.0 and variance FC 0.2 and 5.0 to the
three frozen Study 01 slices. It does not overwrite or alter
`outputs/final_rerun_20260718` or its report.

The extension uses the frozen raw inputs, fixed 3,000-gene panels, public seed,
FEAST assignment contract, preprocessing, and clustering settings. Its output
root is `outputs/expanded_mean_variance_fc_0_20_5_20260806`.

Run from this directory with the pinned environments:

```bash
FEAST_PY=/path/to/envs/feast-reproduce-py311-68816e5/bin/python
OUTPUT=outputs/expanded_mean_variance_fc_0_20_5_20260806

$FEAST_PY run_expanded_sensitivity.py --config expanded_config.yaml \
  --output-root "$OUTPUT" simulate --raw-dir data/local

$FEAST_PY run_expanded_sensitivity.py --config expanded_config.yaml \
  --output-root "$OUTPUT" panels

$FEAST_PY run_expanded_sensitivity.py \
  --config expanded_config.yaml --output-root "$OUTPUT" method \
  --method GraphST --python /path/to/envs/GraphST/bin/python

$FEAST_PY run_expanded_sensitivity.py --config expanded_config.yaml \
  --output-root "$OUTPUT" method --method Leiden_unsupervised --python "$FEAST_PY"
```

STAGATE must inherit the CUDA library path recorded in the frozen Study 01
STAGATE provenance before running its method stage. Finally, run the `score`
stage with the FEAST environment. Each simulation and method stage supports
`--resume`; only hash-verified completed candidates are skipped.

Validate the completed extension and export the realized-alteration table:

```bash
$FEAST_PY validate_expanded_sensitivity.py --config expanded_config.yaml \
  --output-root "$OUTPUT"
```

The completed report contains:

- `expanded_benchmark_metrics.csv`: 36 rows (12 panels x 3 methods).
- `combined_mean_variance_metrics.csv`: frozen and new mean/variance results.
- `combined_mean_variance_summary.csv`: per-level, per-method summaries.
- `realized_alteration_diagnostics.csv`: requested, target-stage, and
  realized count-stage fold changes for mean, per-gene variance, and zero
  proportion.
- `validation.json`: checksums and validation counts for all 12 simulations,
  12 panels, 36 method jobs, and 36 metric rows.

The FC in a condition name is the requested intervention, not a claim that
count decoding realizes it exactly. Interpret the clustering curves together
with `realized_alteration_diagnostics.csv`. In particular, zero-proportion FC
is exactly 1.0 for every added condition, while the realized mean and variance
FCs can differ substantially from their nominal levels.
