# Study 06 visualization

The validated, unpromoted diagnostic is built by `plot.py` from the 93-target
final validation and evaluation only:

```bash
python visualization/06_3d_stack/plot.py
```

It emits `figures/conditional_stack_diagnostic.{pdf,svg,png}`, plotting CSVs,
and `figure_provenance.json`. The PDF SHA-256 is
`7f018ad5af52557240fd455b4546436204ced627a8f6cb0aa55d3f884bc3bb80`;
the figure-provenance SHA-256 is
`b117a1d68d2e1786210774eda6ab5f1cc0367822e969f139530be640c01ce85a`.

The figure separates dense, medium, and sparse arms and marks targets that use
declared out-of-bracket label donors. Between-z summaries are normalized by
the declared real-data split-half baseline. The 93 targets are ordered z
levels, not independent biological replicates. Composite scores, winner
rankings, and automatic figure promotion are prohibited.
