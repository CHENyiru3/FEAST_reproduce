# Visualization

This is the canonical figure-building layer for the clean FEAST publication
reruns. The repository-level source and disposition index is
[`../FIGURE_SOURCE_MAP.md`](../FIGURE_SOURCE_MAP.md). Each reproduction subtask
has its own visualization directory:

| Study | Visualization directory | Status |
|---|---|---|
| 00 simulator benchmark | [`00_simulator_benchmark/`](00_simulator_benchmark/) | Deterministically rebuilt; author claim review required |
| 01 clustering | [`01_clustering/`](01_clustering/) | Deterministically rebuilt; three-slice mean scope only; unpromoted |
| 02 alignment | [`02_alignment/`](02_alignment/) | Diagnostic rebuilt; author panel/metric scope required; unpromoted |
| 03 deconvolution | [`03_deconvolution/`](03_deconvolution/) | Diagnostic rebuilt; scientific stop remains active; unpromoted |
| 04 batch-effect removal | [`04_batch_effect_removal/`](04_batch_effect_removal/) | Supplementary diagnostics rebuilt; author review required; unpromoted |
| 05 2D conditional transfer | [`05_2d_conditional_transfer/`](05_2d_conditional_transfer/) | Deterministically rebuilt from 45/45 validated outputs; claim reframe and author review required; unpromoted |
| 06 conditional 3D stack | [`06_3d_stack/`](06_3d_stack/) | Blocked at reference-only AR preflight; no generation or figure authorized |
| 07 DevCCF 3D transfer | [`07_3d_transfer/`](07_3d_transfer/) | Corrected 550-gene calibration complete; canaries in progress; no figure yet |

Figure scripts read only regenerated or checksum-frozen artifacts inside
`FEAST_reproduce`; the historical `FEAST_experiments` workspace is not an
input. Rendered figures stay within the corresponding subtask directory. A
rendered diagnostic is not a promoted publication claim: each figure
provenance record preserves its current scientific and authorization status.
