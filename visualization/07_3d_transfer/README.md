# Study 07 visualization

The validated, unpromoted diagnostic is built by `plot.py` from the final
descriptive evaluation and both independently validated manifests only:

```bash
python visualization/07_3d_transfer/plot.py
```

It emits `figures/full_axis_transfer_diagnostic.{pdf,svg,png}`, plotting CSVs,
and `figure_provenance.json`. The PDF SHA-256 is
`62eb4085709d39e2abfea521e8ee33df8cba9b3131ce21eb8a64a831c6ddd098`;
the figure-provenance SHA-256 is
`ef53d938ae35558da635cb60013279a0a7ee48911e99f11757342b16694f81fc`.

E15.5 (158 levels, -4.12 through -0.98) and E18.5 (202 levels, -5.56 through
-1.54) remain distinct and are rendered side by side over their complete z
extent. Adjacent-z panels display `1 - Pearson correlation` on a logarithmic
axis; lower values indicate smoother descriptive continuity. There is no
target-expression ground truth, accuracy claim, composite score, or automatic
figure promotion. Partial historical stacks, historical smoothing, and failed
conditional-OT slices are prohibited inputs.
