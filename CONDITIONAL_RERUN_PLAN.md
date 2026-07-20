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

Implementation checkpoint (2026-07-19): all three generation, validation, and
comparison workflows are implemented. Their isolated test gate reports 16, 16,
and 11 passing tests. Study 05 completed and independently validated all 45
fresh candidates, scores, the historical common-panel comparison, and the
editable review figure. Study 06 has a verified real metadata plan, and both
complete Study 07 expression-free blueprints are prepared. Study 06 is blocked
before generation after three strict full-preflight POT failures. Study 07 was
corrected to the historical 550-gene E15.5 reference-fit contract for both
ages; E18.5 reference-only calibration selected 0.5 and age canaries are the
only generation currently authorized.

## Shared execution contract

All three workflows must import the installed wheel recorded in
`FEAST_BUILD.txt`, not the mutable FEAST checkout or a historical experiment
directory. Before a canary or full run, the selected interpreter must pass:

```bash
python scripts/verify_feast_install.py
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
unified FEAST v1.0.2 value of 1000. It is a declared numerical configuration
change and must appear in old-versus-new records. A finite plan is not proof of
convergence. Every saved transport record must report `converged=true`, a
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
The expected values are dense `0.35`, medium `0.25`, and sparse `0.35`. The
preflight writes the estimates and stops if they differ; accepted values are
then frozen for every target in that arm.

Current gate: the initial full preflight and two independent one-thread repeats
all failed during the first dense-gap-3 estimator call with POT numerical
errors at iteration 0. Isolated calls can pass, so no declared AR is accepted
and no canary may run. The blocker and required versioned solver/estimator
repair are recorded in `06_3d_stack/PREFLIGHT_BLOCKER.md`.

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
4. Keep Study 06 stopped until its versioned estimator/solver repair passes
   reproducible full preflights; Study 05 is complete.
5. Run Study 07 in measured z-index shards, beginning with E15.5 and E18.5
   canaries before allocating full workers.
6. Independently validate outputs, recompute metrics, and write old-versus-new
   tables.
7. Stop any workflow whose headline conclusion changes. Do not update figures,
   canonical decisions, or the scientific disposition until author review.
8. After all accepted candidates pass, regenerate remaining visualizations,
   update the figure-source map, revalidate all method contracts, and rebuild
   the final artifact freeze and clean-environment engineering gate. Study 05's
   review figure and source-map entry are already complete.

The current implementation is NumPy/POT CPU code. CUDA does not accelerate
these transports. Parallelism is therefore across independent direction,
target, or z-index jobs with explicit BLAS thread limits and measured memory
use; it must not change seeds, configuration IDs, or output identity.
