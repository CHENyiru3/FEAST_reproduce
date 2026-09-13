# Study 02: alignment visualization

Active Study 02 figures compare **PASTE2** and **Spateo** on the fixed-plate
simulation.  All current visualizations use the independently rerun PASTE2
outputs in `../../02_alignment/outputs/paste2_rerun_20260807_v1/` and the
corresponding validated Spateo outputs in
`../../02_alignment/outputs/fixed_plate_rerun_20260806_v1/`.

`plot_fixed_plate_results.py` renders the two primary coordinate-based metrics
across the seven rotation angles: normalized spatial error and
rotation-recovery error. Both use the rigid-alignment layer applied to every
retained moving spot and are directly comparable between PASTE2 and Spateo.
Coordinate-NN spot accuracy is retained in the plot-data CSV as a secondary,
ceiling-prone metric; region accuracy, label-transfer ARI, expression
correlation, and partial-coupling `transport_*` diagnostics remain auxiliary.
`plot_spatial.py` renders the 45-degree four-condition spatial comparison.
Both write PDF, SVG, PNG, plot-data, and provenance files under `figures/`.
Rebuild from the repository root with:

```bash
python visualization/02_alignment/plot_fixed_plate_results.py
python visualization/02_alignment/plot_spatial.py
```

`plot_fixed_plate_rotation.py` creates the method-neutral illustration of the
fixed-plate rotation setup. `plot_spatial.py` is the corresponding result
figure: every row shows one registered expression condition (baseline, mean
×0.5, variance ×2.0, sparsity ×0.5), and its columns show the Spateo and
PASTE2 outputs. Every cell uses the same two-plane view: the full gray target
slice lies below the aligned moving slice, with sampled exact barcode-pair
connectors. Result spots use shared exact barcode-pair residual bins: blue
≤0.005, orange 0.005–0.01, and red >0.01 reference spot spacings. These bins
describe coordinate residuals, not failed transport matches.

## Archived superseded material

The superseded fixed-plate results and their prior figures are retained only
for traceability under the repository-root
`.archive/code-only-cleanup-20260913/visualization/02_alignment/archive/` and
`../../02_alignment/outputs/archive/paste_fixed_plate_20260806_v1/`.  They are
not inputs to active scripts and must not appear in current Study 02 figures.
