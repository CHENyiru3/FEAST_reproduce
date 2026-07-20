# Study 05 publication decision

Decision date: 2026-07-19  
Status: validated evidence; publication claim reframe and author approval required

## Evidence completed

- All 45 fresh FEAST candidates completed under configuration
  `study05-two-dataset-conditional-half-slice-v2`.
- The exact matrix contains 40 cross-slice and 5 deterministic half-slice
  candidates. All candidates have positive label-level transport convergence
  evidence and passed identity, support, panel, input, output, and provenance
  validation.
- The score root contains 45 summary rows, 587,367 ordered per-gene rows, and
  45 support rows. The validator independently reconstructs all full-panel and
  globally reference-observed summary values from the per-gene table.
- Target expression was unavailable to generation and was loaded only by the
  scoring stage.
- The five historical summaries and ten historical per-gene tables were read
  from the frozen restored snapshot. Historical 17,924-gene metrics were also
  recomputed on the exact fresh 17,391-gene panel, separating the 533-gene
  panel effect from the fresh unified-OT method delta.

The final generation matrix contains 45 H5AD files totaling 1,396,118,620
bytes. The complete validation report SHA-256 is
`63d0aada03bb65e0896a08426e55a8207a4bd05ac6141741325c9c3213c427ae`.

## Primary results

The assignment-randomness value 0.3 was declared primary before scoring.
Values below are means within the stated stratum.

| Dataset/task | Rows | Mean corr. | Var. corr. | Moran corr. | Zero KS | Median gene Pearson | Retained target |
|---|---:|---:|---:|---:|---:|---:|---:|
| DLPFC cross-slice, within donor | 2 | 0.998751 | 0.997643 | 0.919014 | 0.021333 | 0.007237 | 1.000000 |
| DLPFC cross-slice, cross donor | 4 | 0.991394 | 0.981540 | 0.832458 | 0.047021 | -0.001957 | 0.918218 |
| DLPFC half-slice | 3 | 0.992611 | 0.997407 | 0.772375 | 0.092883 | -0.001864 | 1.000000 |
| MERFISH cross-slice | 2 | 0.985216 | 0.967052 | 0.909426 | 0.081996 | 0.044327 | 0.996718 |
| MERFISH half-slice | 2 | 0.994889 | 0.982465 | 0.869146 | 0.028520 | 0.043152 | 0.996367 |

On the original `151675↔151676` scope, the primary fresh Moran correlations
are 0.913926 and 0.924102. After restricting historical per-gene tables to the
fresh panel, the method-only deltas are +0.000909 and -0.000666. Mean, variance,
and zero-fraction metrics are unchanged by the method on the common panel to
the precision recorded in the comparison table.

## Publication interpretation

The evidence supports a bounded claim that FEAST performs label-aware
conditional empirical-rank generation from reference expression plus observed
target XY coordinates and labels, with strong agreement in gene-level mean,
variance, and Moran-statistic profiles on these slices.

The evidence does **not** support accurate coordinate-wise expression
recovery. Median per-gene spotwise correlations are near zero for DLPFC and
approximately 0.04 for MERFISH. Therefore manuscript text and figures must not
call the half-slice task accurate spot-level imputation, reconstruction, or
prediction. Use `conditional half-slice generation` and report the spotwise
metrics beside aggregate fidelity metrics.

Additional required caveats:

- Target class/layer labels and target geometry are observed covariates.
- Directions from `151670` exclude unsupported Layers 1–2 and retain only
  83.11% or 84.17% of target spots; they are not full-target evaluations.
- MERFISH donor identity is not declared by these inputs.
- `reference-observed` means nonzero anywhere in the retained reference, not
  nonzero within every conditional label.
- This five-slice result does not establish performance for other tissues,
  platforms, donors, or label-free settings.

## Disposition

The generated H5ADs, scores, support audit, and old-versus-new table are
accepted as validated publication candidates. They are not promoted to an
unqualified canonical article claim until the author accepts the wording above
and reviews the editable figure. The global scientific disposition therefore
remains blocked; no package release, tag, or PyPI action is authorized.

Key evidence hashes:

- `config.yaml`: `b9584840f7875c0cbdfe08286be728b57b66c32f689452eb770f4f45d2162683`
- `outputs/scores/summary.csv`: `7e6b7a8979d80f49e69646c392e56ac0f9a28adb6b8a44a6223d9e17c94d9b40`
- `outputs/scores/per_gene_metrics.csv`: `3a521275df3815726059d3639089670e952e811415595c56f617786e8acc108f`
- `outputs/scores/support_audit.csv`: `be3fc4200188acfcaed218742a55f0f48bd6348b6930a35be0b64921f85a1a61`
- `outputs/scores/provenance.json`: `0db3dfe062fd997d11b8426759e11d5a5e5ed354d103c35932c3854b3a64f8c1`
- `outputs/old_vs_new_metrics.csv`: `86dcab623cda2a7d3eaf4f3843cf5a75a2d0e6365f3d371f6ba7b8b415f9eae3`
- `outputs/old_vs_new_metrics_provenance.json`: `3f9817afaf1cdebbe479aafea27414882940d727a40c7eef15637a9d03719531`

The score/comparison hashes above are checked again before integration because
tracked documentation or validator changes do not alter generation artifacts
but a regenerated score root would necessarily produce new provenance hashes.
