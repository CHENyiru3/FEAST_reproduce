# Study 04: batch-effect-removal visualization

This directory renders the completed 60-row atomic-metric evidence selected by
`../../PUBLICATION_MANIFEST.json`. GraphST, STAMP, and scVI each have 12 validated
primary jobs, and the separate within-batch graph contract records zero
cross-batch edges.

These figures are supplementary diagnostic candidates only. Study 04 has no
active manuscript claim; no composite score, method winner/ranking, or figure
promotion is authorized. Values above `alpha = 1` are extrapolation evidence.
The 15 paired GraphST PCA-20/PCA-10 sensitivity rows must accompany any future
use. GraphST and scVI ran in external environments with NumPy versions outside
FEAST's supported range; FEAST was not imported in those workers. The declared
STAMP numerical-retry policy is retained in figure provenance.

Rebuild from the repository root with the supported reproduction environment:

```bash
python visualization/04_batch_effect_removal/plot.py
```

The deterministic PDF, SVG, and PNG candidates; plot-ready primary,
availability, and GraphST sensitivity tables; and `figure_provenance.json` are
written to `figures/`. The script verifies every selected input against its
manifest SHA-256 before rendering.
