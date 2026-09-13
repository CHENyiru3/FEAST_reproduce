# Study 07: expression transfer to DevCCF atlas coordinates

The latest reported workflow uses two references per modeling region and
float32 transport. Its generation, consolidation, validation, and evaluation
code is in [two_reference_float32/](two_reference_float32/README.md). These are
the scripts used for the completed 360-level run and the September 9 report,
previously stored only in ignored `.work/` directories.

E15.5 has 158 levels and 2,330,927 positions; E18.5 has 202 levels and 5,213,461
positions. Both use 550 genes. Target blueprints provide coordinates and region
labels without expression. Evaluation is descriptive; it cannot establish
accuracy against unobserved target expression.

See the [visualization instructions](../visualization/07_3d_transfer/README.md)
for plotting the current results and comparing continuity with the existing
whole-spot resampling control.

The Python files and `config.yaml` directly in this directory retain the older
full-reference, verified-wheel workflow. Its calibration helpers, blueprint
preparation code, and tests remain available, but its defaults do not reproduce
the latest two-reference results. Earlier workflows under `archive/` and local
execution notes under `.work/` are excluded from the code upload. The previous
README is preserved in `.archive/code-only-cleanup-20260913/07_3d_transfer/` at
the repository root.

`conditional_resampling_baseline.py` retains the all-eligible-reference,
within-region whole-spot resampling algorithm used as the descriptive control.
It requires a validated run and its matching configuration (`--config`), and
accepts a fresh `--output-dir`. Its evaluation helpers are unchanged from the
archived implementation. This control pools all eligible references; it is not
a two-reference ablation.
