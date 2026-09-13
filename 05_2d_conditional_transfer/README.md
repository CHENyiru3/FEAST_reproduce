# Study 05: 2D conditional transfer

The design has 45 FEAST candidates: 40 cross-slice outputs over five assignment-
randomness values, plus five half-slice outputs at the primary value 0.3.
DLPFC slices are 151670, 151675, and 151676; MERFISH slices are 006 and 007.
The fixed panels contain 17,391 and 1,122 genes respectively.

DLPFC 151670 is from donor Br5595; 151675/151676 are from Br8100. Keep
within-donor, cross-donor, and within-slice results separate. MERFISH donor
identity is not declared and must not be inferred from adjacent slice IDs.

## Inputs and method

Run from `05_2d_conditional_transfer/` with the FEAST 1.0.2 environment recorded
in [FEAST_BUILD.txt](../FEAST_BUILD.txt). Supply the five inputs under
`data/local/dlpfc/` and `data/local/merfish/`, matching `data/input_checksums.csv`.

A target label is supported only when the source has at least 20 spots with
that label. All excluded labels/spots are recorded. The half-slice task uses
`x <= median(x)` as visible reference and `x > median(x)` as the masked target.
Generation receives target coordinates and labels; target expression is loaded
only for evaluation. This is conditional generation given known labels.

Zero-evidence reference genes remain in the fixed panel (`min_gene_spots=0`).
Scores retain both the full panel and a reference-observed-gene sensitivity
subset. The conditional API uses explicit unified OT, 1,000 Sinkhorn iterations,
and `transport_nonconvergence="raise"`.

```bash
FEAST_PY=/path/to/clean-feast-environment/bin/python
"$FEAST_PY" ../scripts/verify_feast_install.py
"$FEAST_PY" -m pip check
"$FEAST_PY" preflight.py
"$FEAST_PY" run.py --mode cross_slice --dataset dlpfc \
  --source 151675 --target 151676 --assignment-randomness 0.3
"$FEAST_PY" run.py --mode mask_half --dataset dlpfc --slice 151675
```

Run one strict cross-slice and half-slice canary per dataset before completing
the remaining candidates declared in `config.yaml`. Each candidate contains
`generated.h5ad`, provenance, and transport diagnostics below
`outputs/final/<mode>/<dataset>/<direction>/ar_<value>/`.
Use the documented resume behavior; never replace a completed candidate with
another configuration.

## Validation and controls

```bash
"$FEAST_PY" validate.py --all --report outputs/study05_validation.json
"$FEAST_PY" score.py --generation-dir outputs/final --output-dir outputs/scores
"$FEAST_PY" validate.py --all --scores-dir outputs/scores \
  --report outputs/study05_complete_validation.json
"$FEAST_PY" conditional_resampling_baseline.py
"$FEAST_PY" evaluate_resampling_metrics.py
"$FEAST_PY" linear_baseline.py
```

The primary empirical control uses ten whole-spot draws conditioned on source
labels. `evaluate_resampling_metrics.py` compares conditional expression and
zero-fraction Wasserstein distances, Moran-I profiles, and label-residual Moran-I
profiles. Results are in `outputs/baselines/conditional_metric_comparison/`.
The label-mean linear control is a separate lower-bound diagnostic.

`compare_historical.py` accepts `--historical-root`, `--fresh-score-dir`, and
`--output`; only the original ten 151675↔151676 cross-slice rows have valid
historical counterparts. Historical masked-half results are leakage-tainted.
The [publication decision](PUBLICATION_DECISION.md) is retained because the
plotting and publication records consume it. High distributional similarity
does not establish accurate coordinate-wise prediction or authorize a winner.

## Figures

Run from the repository root after the required analysis/control stages:

```bash
python visualization/05_2d_conditional_transfer/plot_resampling_comparison.py
python visualization/05_2d_conditional_transfer/plot_expanded_report.py
python visualization/05_2d_conditional_transfer/plot_selected_cross_slice.py
python visualization/05_2d_conditional_transfer/plot_mixed_transfer_4x4.py
```

The expanded report requires `pdfunite` and supplies intermediate article-style
plot data used by the selected panels. Resampling ranges are empirical 5–95%
ranges over ten runs, not confidence intervals. Selected marker panels include
examples chosen using held-out agreement and are illustrative, not typical-gene
accuracy estimates. The 4×4 panels share an unclipped log1p-count scale.

Additional diagnostics are `plot.py`, `plot_simulation_effect.py`,
`plot_ar_spatial_effect.py`, and `plot_expression_matrix.py` in the same folder.
The AR view deliberately selects best-case agreement. The expression matrix
uses reference-selected spatial markers and its recorded per-column scale.
These displays have distinct selection rules and cannot be treated as the same
accuracy claim. Output tables, captions, and figures remain local.
