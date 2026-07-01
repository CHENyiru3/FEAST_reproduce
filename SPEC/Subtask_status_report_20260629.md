# FEAST Experiment Subtask Status Report

**Generated:** 2026-06-29 09:54 CDT  
**Scope:** `/maiziezhou_lab2/yiru/FEAST_experiments`  
**Definition of reportable:** result can be used in manuscript/report tables if the stated caveat/source is preserved.  

## Executive Summary

| Subtask | Current status | Reportability | Recommended reporting source |
|---|---|---|---|
| 00 Simulator benchmark | Complete archives exist; current optimized rerun is incomplete/worse for FEAST | **Reportable from archive only** | `00_simulator_benchmark/outputs/archived_20260623_110615/simulator_quality_metrics.csv` or `reported_previous_results/latest_pre_optimization/00_simulator_benchmark/` |
| 01 Clustering | Rerun fixed; 243/243 rows ok | **Reportable** | `01_clustering/results/clustering_benchmark_results.csv` |
| 02 Alignment | Completed with Spateo + PASTE; PASTE is not equivalent to old SPACEL | **Conditionally reportable** | `02_alignment/results/`, with explicit method caveat |
| 03 Deconvolution | Completed; 12/12 rows ok | **Reportable with method caveat** | `03_deconvolution/results/` |
| 04 2D conditional transfer | Completed; AR sweep available | **Reportable** | `04_2d_conditional_transfer/results/ar_*/summary.csv` |
| 05 3D stack | Completed; all density summaries available | **Reportable** | `05_3d_stack/outputs/cross_density_summary.csv` |
| 06 3D transfer | Failed during `.h5ad` write | **Not reportable yet** | Needs rerun after metadata serialization fix |
| 07 Visualization | Partial figures exist; not all panels are current | **Conditionally reportable** | Use only figures tied to locked result sources |
| 08 Batch effect removal | Current rerun not done; archived outputs exist | **Archive-only / conditional** | `08_batch_effect_removal/*/archive/archived_20260626_210556/` |
| 09 Resource scaling | Timing runs complete; summary script failed | **Reportable after manual summary** | `09_helper_tests/results/scaling_20260627_112455_timing.csv` |

No active FEAST experiment workload was running at inspection time. Existing tmux sessions were idle bash panes.

## Cross-Cutting Issues

| Issue | Impact | Action |
|---|---|---|
| Current optimized 00 benchmark does not reproduce archived FEAST quality | Do not use current optimized 00 table for headline simulator quality | Report archived 2026-06-23 results; keep current run as diagnostic only |
| Current 00 optimized FEAST results are missing OpenST samples | Bug to fix later | Track as OpenST optimized-rerun issue |
| `StudentT_mixture_model.py` has uncommitted PPF clamp fix | Prevents float/infinity issues but not committed | Commit after review/tests |
| 06 transfer fails writing `uns/de_novo/quantile_field/rank_scope_metadata/fallbacks` | Blocks 06 and downstream 08 rerun queue | Convert fallback metadata to JSON-safe strings before writing `.h5ad` |
| 09 scaling summary step failed by executing CSV as Python | Raw timing is valid; summary artifact missing | Build summary from CSV manually or fix summary invocation |
| Full-Xenium `FEAST_OT_Spatial` peaks at ~354.6 GB | Too high for a practical recommended configuration | Treat as a stress-test upper bound; report `FEAST_Rank` for full Xenium unless OT is downsampled or memory-capped |
| Alignment changed from SPACEL to PASTE | Benchmark meaning changed | Report as Spateo + PASTE, not as direct reproduction of SPACEL panel |

## 00 Simulator Benchmark

**Status:** Complete archived result exists; optimized/current rerun should not be used as the main quality table.

### Available Result Sets

| Source | Rows | FEAST samples per mode | Main note |
|---|---:|---:|---|
| `outputs/archived_20260623_110615/simulator_quality_metrics.csv` | 64 | 11 | Best FEAST quality; preferred for reporting |
| `reported_previous_results/latest_pre_optimization/00_simulator_benchmark/simulator_quality_metrics.csv` | 64 | 11 | Same metrics as archive; convenient stable copy |
| `outputs/feast_legacy_quality_20260627_111554/benchmarks/simulator_quality_metrics.csv` | 64 | 11 | Completed rerun, but FEAST quality below archive |
| `outputs/benchmarks/simulator_quality_metrics.csv` | 68 | 10 | Current optimized table; OpenST missing for FEAST and quality lower |

### Preferred Archived Metrics

| Simulator | Mean corr | Variance corr | Zero Jaccard | Moran corr | Composite |
|---|---:|---:|---:|---:|---:|
| FEAST_OT_Spatial | 0.9934 | 0.9450 | 0.9570 | 0.9575 | 0.6342 |
| FEAST_Rank | 0.9934 | 0.9452 | 0.9903 | 0.9853 | 0.6392 |
| SRTsim | 0.9999 | 0.9933 | 0.9986 | 0.9923 | 0.6488 |
| Splatter | 0.0092 | 0.0031 | 0.8852 | 0.9345 | 0.5204 |
| Splatter_Simple | -0.0041 | -0.0043 | 0.4224 | 0.9297 | 0.1465 |
| scCube | 0.9963 | 0.9971 | 0.9066 | 0.9800 | 0.6078 |

**Reportability:** Use the archived/pre-optimization result for manuscript quality. Do not use the current optimized result for headline claims until the OpenST omission and FEAST quality regression are resolved.

## 01 Clustering Benchmark

**Status:** Fixed rerun complete.

| Metric | Value |
|---|---:|
| Rows | 243 |
| Status | 243 ok, 0 failed |
| Slices | 3 |
| Simulation IDs | 27 |
| Methods | GraphST, STAGATE_mclust, Leiden |

### Method Means

| Method | ARI | NMI | CHAOS | PAS |
|---|---:|---:|---:|---:|
| GraphST | 0.3685 | 0.4591 | 0.0594 | 0.0137 |
| STAGATE_mclust | 0.3399 | 0.4645 | 0.0614 | 0.1220 |
| Leiden | 0.0804 | 0.1188 | 0.1001 | 0.8253 |

### Mean ARI by Slice

| Slice | GraphST | STAGATE_mclust | Leiden |
|---|---:|---:|---:|
| 151508 | 0.2576 | 0.3283 | 0.0761 |
| 151670 | 0.5296 | 0.3647 | 0.0598 |
| 151676 | 0.3242 | 0.3267 | 0.1052 |

**Reportability:** Reportable. The previous 36-missing-row issue was fixed by rerunning failed/stale rows. Use the current `01_clustering/results/` output, not the stale root-level `outputs/03_clustering` table.

## 02 Alignment Benchmark

**Status:** Completed, but method set changed. Current results contain `spateo` and `paste`; old SPACEL results were archived and should not be mixed into a current PASTE table without explanation.

### Current Method Means

| Method | NN accuracy | NN region accuracy | Mean spatial error | GE correlation | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| spateo | 0.7341 | 0.8378 | 2881.0035 | 0.8187 | 10.6590 |
| paste | 0.1977 | 0.5325 | 1865.9052 | 0.7113 | 22.5135 |

### Current Summary Rows

| Alteration | Method | NN accuracy | Region accuracy | GE correlation | Success rate | Angles |
|---|---|---:|---:|---:|---:|---:|
| baseline | paste | 0.2001 | 0.5324 | 0.7588 | 1.0 | 5 |
| mean_0.5 | paste | 0.1982 | 0.5331 | 0.7238 | 1.0 | 5 |
| sparsity_0.5 | paste | 0.1931 | 0.5319 | 0.7747 | 1.0 | 5 |
| variance_2.0 | paste | 0.1993 | 0.5328 | 0.5881 | 1.0 | 5 |
| baseline | spateo | 0.8662 | 0.9717 | 0.9005 | 1.0 | 5 |
| mean_0.5 | spateo | 0.6870 | 0.7935 | 0.8126 | 1.0 | 5 |
| sparsity_0.5 | spateo | 0.6738 | 0.7914 | 0.8497 | 1.0 | 5 |
| variance_2.0 | spateo | 0.7094 | 0.7946 | 0.7121 | 1.0 | 5 |

**Reportability:** Conditional. It is valid to report as a Spateo + PASTE benchmark, but not as a direct reproduction of the previous SPACEL benchmark. The PASTE numbers are weak under the current metric design, so avoid framing PASTE as a drop-in replacement that resolves the old SPACEL annotation issue.

## 03 Deconvolution Benchmark

**Status:** Completed.

| Metric | Value |
|---|---:|
| Rows | 12 |
| Status | 12 ok, 0 failed |
| Methods | cell2location, rctd |
| Resolutions | 0.10, 0.25 |

### Method Means

| Method | JSD mean | Pearson mean | Summed RMSE |
|---|---:|---:|---:|
| cell2location | 0.8966 | 0.1501 | 0.0281 |
| rctd | 0.8464 | 0.0234 | 0.0961 |

**Reportability:** Reportable with caveat. Cell2location is the stronger method in this run. RCTD is much weaker and should be presented as method-specific underperformance rather than a FEAST simulation failure.

## 04 2D Conditional Transfer

**Status:** Completed. The AR sweep is available in current results.

| Assignment randomness | Rows | Mean corr | Variance corr | Moran corr | Zero KS | Median gene Pearson |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2 | 0.9988 | 0.9976 | 0.6950 | 0.0207 | 0.0172 |
| 0.1 | 2 | 0.9988 | 0.9976 | 0.7620 | 0.0207 | 0.0152 |
| 0.2 | 2 | 0.9988 | 0.9976 | 0.8699 | 0.0207 | 0.0091 |
| 0.3 | 2 | 0.9988 | 0.9976 | 0.9182 | 0.0207 | 0.0053 |
| 0.5 | 2 | 0.9988 | 0.9976 | 0.8812 | 0.0207 | 0.0038 |

**Reportability:** Reportable. `assignment_randomness=0.3` is the best balance in current summaries: mean/variance fidelity remains near 1.0 and Moran correlation peaks around 0.9182.

## 05 3D Stack Reconstruction

**Status:** Completed. All density outputs and summaries exist.

### Cross-Density Summary

| Density | Targets | Mean corr | Variance corr | Moran corr | Zero KS | Z coherence | Real split-half Z coherence |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_gap3 | 49 | 0.9391 | 0.8539 | 0.4464 | 0.2900 | 0.0965 | 0.4489 |
| medium_gap5 | 29 | 0.9312 | 0.8590 | 0.4772 | 0.2752 | 0.1122 | 0.4646 |
| sparse_gap10 | 15 | 0.9309 | 0.8744 | 0.4404 | 0.3307 | 0.1900 | 0.4894 |

**Reportability:** Reportable. Main strength is mean/variance preservation. Spatial Moran and z-coherence are moderate, so claims should emphasize approximate 3D reconstruction rather than near-perfect z-profile recovery.

## 06 3D Transfer to DevCCF

**Status:** Failed during current rerun.

### Evidence

| Artifact | Status |
|---|---|
| `06_3d_transfer/outputs/E15.5_blueprints_full.json` | Exists |
| `06_3d_transfer/outputs/E18.5_blueprints_full.json` | Exists |
| `06_3d_transfer/outputs/E15.5_test/` | Partial generated files exist |
| `06_3d_transfer/logs/transfer_3d_20260627_191129.log` | Failure log exists |

Failure:

```text
TypeError: Can't implicitly convert non-string objects to strings
Error raised while writing key '/uns/de_novo/quantile_field/rank_scope_metadata/fallbacks'
```

**Reportability:** Not reportable yet. The failure is an AnnData/HDF5 serialization problem, likely from non-string objects stored in `.uns`. Fix by sanitizing `rank_scope_metadata["fallbacks"]` before `write_h5ad()`, then rerun 06.

## 07 Visualization

**Status:** Partial/current figures exist.

| Figure group | Status |
|---|---|
| `figures/simulator_benchmark/` | Exists |
| `figures/slice_panel_compare/` | Exists |
| `figures/clustering/` | Exists |
| Alignment / deconvolution / 2D transfer / 3D transfer / stack figure scripts | Scripts exist, but not all final figures are present/current |
| Total figure files under `07_Visualization/figures` | 37 |

**Reportability:** Conditional. Use figures only when their source data is locked to a reportable result source. Simulator figures should be regenerated or explicitly tied to the archived 00 table. 3D transfer figures should wait until 06 is fixed.

## 08 Batch Effect Removal

**Status:** Current rerun was not completed. Archived outputs exist.

### Archived Available Outputs

| Component | Archived result |
|---|---|
| effect_simulation | `effect_simulation/archive/archived_20260626_210556/outputs_previous/20260624_013147/` and `20260624_015506/` |
| effect_verification | `effect_verification/archive/archived_20260626_210556/outputs_previous/20260624_003618/` |

### Archived Simulation Manifest

Both archived simulation runs contain 14 rows: 2 deformation modes x 7 alpha levels. Example baseline size: 3611 spots x 19140 genes.

### Archived Verification Affine Fits

| Slice | d_mu | d_omega | d_pi0 | b_mu | b_omega | b_pi0 | r2_mu | r2_omega | r2_pi0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 151508 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 151670 | 1.0029 | 1.1391 | 1.0363 | 0.4519 | 0.0134 | -0.5699 | 0.9665 | 0.8154 | 0.9260 |
| 151676 | 0.9678 | 1.2799 | 1.0374 | 0.5733 | 0.0158 | -0.8063 | 0.9554 | 0.8060 | 0.9311 |

**Reportability:** Archive-only and conditional. These results can support method demonstration if clearly labeled as archived/pre-current-rerun. Do not claim current rerun completion until 08 is run after 06 is unblocked, if the queue dependency still applies.

## 09 Resource Scaling / Helper Tests

**Status:** Raw timing complete; summary step failed.

| Metric | Value |
|---|---:|
| Rows | 26 |
| Status | 26 OK |
| Average FEAST_Rank time | 149.30 s |
| Average FEAST_Rank memory | 9.72 GB |
| Average FEAST_OT_Spatial time | 523.87 s |
| Average FEAST_OT_Spatial memory | 80.03 GB |

### Representative Scaling Rows

| Config | Mode | Spots | Genes | Wall seconds | Max RSS MB |
|---|---|---:|---:|---:|---:|
| dlpfc_151675_full | FEAST_Rank | 3565 | 18641 | 216.48 | 3284.8 |
| dlpfc_151675_full | FEAST_OT_Spatial | 3565 | 18641 | 219.65 | 3694.4 |
| openst_g28943_s47217 | FEAST_Rank | 47217 | 28943 | 214.15 | 43843.0 |
| openst_g28943_s47217 | FEAST_OT_Spatial | 47217 | 28943 | 1006.61 | 69020.3 |
| xenium_s377957_g0523 | FEAST_Rank | 377957 | 523 | 59.42 | 7517.6 |
| xenium_s377957_g0523 | FEAST_OT_Spatial | 377957 | 523 | 1623.64 | 354602.7 |

**Reportability:** Reportable after making a clean summary table from the raw CSV, but split the interpretation by mode. `FEAST_Rank` is reportable as the practical full-Xenium configuration. Full-Xenium `FEAST_OT_Spatial` should be reported only as a stress-test/upper-bound result because its ~354.6 GB peak RSS is not a reasonable production requirement. For OT spatial, report capped or downsampled spot counts unless the implementation is changed to enforce a memory budget.

## Recommended Next Actions

1. Commit or explicitly shelve the FEAST `StudentT_mixture_model.py` PPF clamp fix after validation.
2. Fix 06 metadata serialization by converting `fallbacks` under `.uns` to JSON-safe strings/lists/scalars.
3. Rerun 06, then rerun 08 if current-version batch-effect results are required.
4. For 00, report the archived 2026-06-23/pre-optimization table and separately mark current optimized OpenST omission as a bug.
5. Generate a locked `09_helper_tests` summary CSV from `scaling_20260627_112455_timing.csv`.
6. Regenerate 07 figures only from the chosen reportable source tables to avoid mixing archived and current result sets.
