# Visualization

This is the canonical figure-building layer for the clean FEAST publication
reruns. Each reproduction subtask has its own visualization directory:

| Study | Visualization directory | Status |
|---|---|---|
| 00 simulator benchmark | [`00_simulator_benchmark/`](00_simulator_benchmark/) | Rebuilt |
| 01 clustering | [`01_clustering/`](01_clustering/) | Awaiting complete validated outputs |
| 02 alignment | [`02_alignment/`](02_alignment/) | Awaiting complete validated outputs |
| 03 deconvolution | [`03_deconvolution/`](03_deconvolution/) | Awaiting complete validated outputs |
| 04 batch-effect removal | [`04_batch_effect_removal/`](04_batch_effect_removal/) | Awaiting complete validated outputs |

Figure scripts read only regenerated or checksum-frozen artifacts inside
`FEAST_reproduce`; the historical `FEAST_experiments` workspace is not an
input. Rendered figures stay within the corresponding subtask directory.
