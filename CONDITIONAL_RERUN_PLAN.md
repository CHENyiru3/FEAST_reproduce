# Conditional-transfer publication rerun plan

This plan adds three fresh FEAST workflows after clean Studies 00–04. The
legacy experiment numbers are mapped to new sequential clean-study numbers:

| Clean study | Legacy study | Publication scope |
|---|---|---|
| `05_2d_conditional_transfer` | `04_2d_conditional_transfer` | DLPFC and MERFISH cross-slice transfer plus deterministic half-slice generation |
| `06_3d_stack` | `05_3d_stack` | 93 held-out Allen Zhuang slices at three reference densities |
| `07_3d_transfer` | `06_3d_transfer` | complete E15.5 and E18.5 DevCCF coordinate-system volumes |

Historical FEAST outputs remain audit evidence only. No historical H5AD,
metric, manifest, or figure is copied into a fresh candidate root.

Implementation checkpoint (2026-07-31): all three generation, validation, and
evaluation workflows are implemented. Their isolated workflow gates report 16,
16, and 16 passing tests. Study 05 completed and independently validated all
45 fresh candidates, scores, the historical common-panel comparison, and the
editable review figure. Study 06 completed 93/93 CUDA generations, strict
validation, atomic evaluation, and its editable diagnostic. Study 07 completed
158/158 E15.5 plus 202/202 E18.5 CUDA generations, consolidation, strict
validation, descriptive continuity evaluation, and its editable diagnostic.
All three remain subject to the scientific wording and author-review limits
declared below.

## Shared execution contract

Study 05 imports the installed 1.0.2 wheel recorded in `FEAST_BUILD.txt`.
Studies 06 and 07 import the repaired 1.1.0 candidate recorded in
`../FEAST/validation/package_builds/20260719_ot_log_repair_v2/provenance.json`.
No workflow imports the mutable FEAST checkout or a historical experiment
directory. Before a canary or full run, the selected interpreter must pass:

```bash
python scripts/verify_feast_install.py [--candidate-provenance <path>]
python -m pip check
```

The conditional transport settings are explicit rather than inherited from
package defaults:

```yaml
epsilon: 0.05
sinkhorn_iter: 1000
sinkhorn_tol: 1.0e-5
unbalanced_transport: true
reg_m: 5.0
geometry_weight: 1.0
boundary_weight: 0.25
max_transport_pairs: 25000000
transport_nonconvergence: raise
```

The iteration limit changes the historical contract value of 200 to the
declared unified-OT value of 1000. It is a declared numerical configuration
change and must appear in the workflow's repair decision or score provenance;
the tracked old-versus-new CSVs separately report the calibrated scientific
parameter changes. A finite plan is not proof of convergence. Every saved
transport record must report `converged=true`, a
finite final error below its recorded tolerance, finite positive transported
mass, and policy `raise`. A solver warning or missing positive evidence makes
the candidate fail.

Every output must bind its configuration ID, FEAST version and commit, wheel
SHA-256, public seed, input artifact IDs and SHA-256 values, solver diagnostics,
runner SHA-256, and output SHA-256. H5AD validation requires named observation
identity, exact ordered genes, exact blueprint coordinates and labels, finite
nonnegative counts, an identical `counts` layer, and successful reread after
closing the writer.

Only `outputs/final/` is publication-facing. Shards and canaries use ignored
`.work/`; superseded or failed material uses ignored `.archive/`. Full runs do
not overwrite or reuse prior candidates. Resume may skip only a file that the
study validator accepts and whose manifest hash still matches.

## Study 05: 2D conditional transfer and half-slice generation

- DLPFC inputs: `151670` (`Br5595`) plus `151675` and `151676`
  (`Br8100`); all six ordered directions.
- MERFISH inputs: `Zhuang-ABCA-1.006` and `.007`; both ordered directions.
- Cross-slice assignment randomness: `0.0`, `0.1`, `0.2`, `0.3`, and `0.5`.
- Half-slice task: visible `x <= median(x)` conditions generation for held-out `x > median(x)`
  at the primary setting `0.3`, once for each of the five slices.
- Genes: fixed ordered 17,391-gene DLPFC and 1,122-gene MERFISH panels.
- Source-label support: at least 20 reference spots; excluded target support is
  retained in the audit rather than silently pooled.
- Seed: 2026 for every declared configuration.
- Expected outputs: 40 cross-slice plus 5 half-slice H5ADs = 45.

Target expression is evaluation-only and is never loaded by generation. A
half-slice runner loads expression only for visible rows; its target contract
contains exact masked IDs, XY, and known conditional labels. The 97/146/129
DLPFC genes with zero visible-half evidence remain in the fixed panel and are
declared explicitly; scoring reports both complete-panel and
reference-observed-gene sensitivity metrics. Run one cross-slice and one
half-slice canary per technology before dispatching independent jobs. The
historical numerical comparison is restricted to the original ten
`151675`↔`151676` cross-slice rows; historical masking is leakage-tainted and
noncanonical.

Completion record: all 45 fresh outputs passed strict identity, panel, support,
hash, and positive-convergence validation. Aggregate distribution and
Moran-profile fidelity is strong on these bounded slices, but median per-gene
spotwise correlations are near zero for DLPFC and approximately 0.04 for
MERFISH. Publication wording must therefore use `conditional half-slice
generation`, not accurate spot-level imputation, reconstruction, or prediction.
The validated evidence remains unpromoted pending author review.

## Study 06: conditional 3D stack imputation

- Inputs: 147 available Allen Zhuang ABCA-1 slices with 1,122 genes.
- Dense gap 3: 49 targets.
- Medium gap 5: 29 targets.
- Sparse gap 10: 15 targets.
- Expected outputs: 93 H5ADs.
- Seed: `2026 + gap * 10000 + target_index`.
- Z regularization and spot smoothing: disabled.

Assignment randomness is estimated once per density using reference data only.
The reproducible v5 values are dense `0.35`, medium `0.30`, and sparse `0.35`. The
preflight writes the estimates and stops if they differ; accepted values are
then frozen for every target in that arm.

Completion gate: the initial ordinary-POT preflight failure was repaired by the
versioned log-domain candidate without relaxing the fail-closed policy. The
fresh v5 preflight, CUDA canaries, all 93 generations, independent validation,
and atomic evaluation passed. The repair classification and exact hashes are
recorded in `06_3d_stack/PREFLIGHT_BLOCKER.md` and
`06_3d_stack/OT_LOG_REPAIR_DECISION.md`.

Each target uses exact observed XY, class labels, and z. Primary reference
weights are linear between the bracketing slices. Any out-of-bracket label
donor is selected by the declared nearest-z rule and written to the target
manifest. Run one canary per density, then shard independent target jobs.
Aggregate only after all 93 target validators pass. Between-z statistics must
be compared with the declared real-data split-half baseline and must not treat
the 93 targets as independent biological replicates.

## Study 07: full two-age DevCCF transfer

### E15.5

- References: ten E14 spatial-transcriptomics slices.
- Full z range: -4.12 through -0.98 at 0.02 spacing.
- Expected levels: 158.
- Blueprint spots: 2,330,927.
- Assignment randomness: 0.4, frozen from the reference-only estimate.
- Seed: `2026 + original_sorted_z_index` (2026–2183).

### E18.5

- References: five E18M spatial-transcriptomics slices.
- Full z range: -5.56 through -1.54 at 0.02 spacing.
- Expected levels: 202.
- Blueprint spots: 5,213,461.
- Assignment randomness: estimate from E18M references only, record it, then
  freeze it before generating any publication slice. The corrected v2
  reference-only calibration selected 0.5.
- Seed: `2026 + original_sorted_z_index` (2026–2227).

Both blueprint sets are regenerated from their age-specific NIfTI volume and
the common region schema. Atlas observation IDs are deterministic functions of
age, original z index, and voxel/row identity. The generation code receives
only XY, z, and region labels; no target expression exists.

Both ages use the historical E15.5 reference-filter contract
`min_gene_spots=1`, `min_gene_mean=0.0`, and `max_gene_zero_prop=1.0`. This
retains the same ordered 550-gene input panel for E15.5 and E18.5; stricter
package defaults would retain only 169 and 181 genes, respectively, and are
not part of this publication design.

Completion record: consolidation retained exactly 158 E15.5 and 202 E18.5
unique levels. Independent validation confirmed exact blueprint identity and
positive convergence for all 12,290 transport records. Descriptive full-axis
coverage and adjacent-z continuity metrics and an editable diagnostic are
complete. Because no target expression exists, these outputs do not support an
expression-accuracy claim.

Canaries cover both endpoints, a middle level, and the densest level for each
age. E15.5 additionally reruns historical failure level z=-3.88 first. After
canaries pass, age-specific z-index shards write to `.work/`. Consolidation
requires exactly 158 and 202 unique levels, the original 0.02 grid, no missing
or duplicate IDs, the expected 550-gene order, and one valid manifest row per
H5AD. The two final age roots remain separate and are visualized side by side.

## Execution order and stop rules

1. Build input checksum manifests and verify the installed FEAST wheel.
2. Run the representative fixed-seed RNG and strict-convergence canaries.
3. Freeze reference-only assignment-randomness decisions.
4. Study 06 and Study 07 completed their CUDA canaries before production.
5. Study 06 completed 93 targets; Study 07 completed its measured E15.5/E18.5
   z-index shards and exact consolidation.
6. Independent validation and declared evaluation are complete. Study 06 has
   no historical comparison because no separately approved hash-pinned summary
   was supplied; Study 07 has no target-expression ground truth.
7. Do not update canonical decisions or the scientific disposition until
   author review of the supported wording and diagnostic figures.
8. After all accepted candidates and author decisions are recorded, rebuild
   the terminal artifact freeze and clean-environment engineering gate.

Study 05 uses the declared NumPy/POT CPU path. Studies 06 and 07 use the pinned
Torch/CUDA log-domain implementation and completed on the NVIDIA RTX A6000
without silent CPU or solver fallback. Parallel independent target/z-index
workers did not change seeds, configuration IDs, or output identity.
