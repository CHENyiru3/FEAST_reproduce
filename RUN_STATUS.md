# FEAST publication rerun status

Checkpoint: 2026-07-19. The originally requested clean Study 00–04
computations are complete. The author has expanded the fresh-rerun scope to
clean Studies 05–07. Study 05 is now fully generated, scored, and validated;
Study 06 is blocked at its fail-closed reference-only preflight; Study 07 has a
validated 550-gene calibration contract and age-specific canaries in progress.

FEAST remains provisional version `1.0.2` at commit
`68816e5c1862a6fa2a49bc30609d617c7fa4b449`. The scientific disposition is
blocked and unapproved. No release commit, tag, or PyPI publication has been
created.

## Completion matrix

| Study | Fresh completion | Retained status |
|---|---|---|
| 00 simulator benchmark | 12/12 FEAST simulations; 48 retained external outputs; 60/60 atomic metric rows | Validated evidence; author claim decision required |
| 01 clustering | 81/81 simulations; 84/84 panels; 84/84 GraphST, STAGATE+mclust, and label-free Leiden; 252/252 metrics | Validated for three-slice method-mean closeness only |
| 02 alignment | 20/20 rotations; 20/20 PASTE; 20/20 Spateo; 40/40 metrics | Validated evidence; author comparison scope required |
| 03 deconvolution | 6/6 simulations; 6/6 RCTD; 6/6 Cell2location; 12/12 common-support scores | Evidence-complete scientific stop after headline directions changed |
| 04 batch-effect removal (legacy 08) | 12/12 GraphST; 12/12 STAMP; 12/12 scVI; 60/60 atomic metrics | Validated supplementary candidate only |
| 05 2D conditional transfer (legacy 04) | 45/45 generations; 45/45 summary/support rows; 587,367 per-gene rows; 16/16 tests; figure rebuilt | Validated evidence; accurate spot-level prediction claim rejected; bounded conditional-generation wording needs author approval |
| 06 conditional 3D stack (legacy 05) | 0/93 generations; 16/16 tests; 147 inputs verified; 3/3 full AR preflights failed at strict OT | Blocked before generation; versioned estimator/solver repair required |
| 07 DevCCF 3D transfer (legacy 06) | 0/360 accepted generations; 13/13 tests; both v2 blueprints prepared; E18.5 reference-only AR=0.5 | Age canaries in progress; no production launch yet |

GraphST, STAGATE, Spateo, STAMP, scVI, and Cell2location jobs selected for
Studies 00–04 completed on the NVIDIA RTX A6000. Studies 05–07 use the current
NumPy/POT conditional transport and are CPU workflows; Study 05 completed via
parallel independent jobs, not CUDA. External method environments remain
isolated and may use NumPy versions unsupported by FEAST; FEAST was not
imported in those workers.

The Study 07 expression-free blueprint hashes are
`15a8ecf002c2bfa03d0ebe55a1822e51097293f11cf8ca3d476067c092c1254a`
for E15.5 and
`59524de6d47bd93e14eb5c73e80cf4b785874b6f1dd0c0cf6feb6dac7d04dd17`
for E18.5. Their exact scopes are 158/202 z levels and
2,330,927/5,213,461 spots. These are prepared geometry inputs, not FEAST
simulation results or canonical article outputs. Configuration v2 applies the
historical E15.5 `1/0.0/1.0` reference filter to both ages, retaining the same
ordered 550-gene panel; E18.5 reference-only calibration selected 0.5.

## Scoped cross-study gates

- RNG is resolved only for the selected fresh Study 00/01 configurations. The
  public-seed-2026 three-process gate is hash-identical under ambient NumPy
  seeds 1, 99991, and 1. This is not an unseeded or all-mode guarantee.
- Blocked assignment is avoided, not generally fixed. Every selected fresh
  Study 00/01 simulation uses SciPy exact-global assignment with
  `assignment_blocks=false`. Core v4 retains the adversarial relative fidelity
  gap `376.256785922`, so no blocked/global equivalence or production-scale
  stability claim is allowed.
- The 156 historical conditional-OT outputs remain noncanonical: 31 explicit
  failure-saved, 26 failure-tainted, 99 without positive convergence evidence,
  and 0 verifiably converged. The fresh Study 05 candidate consumes none of
  them and records positive convergence evidence for all 45 new outputs.
- All 84 selected STAGATE outputs come from the clean sequential run. Historical
  concurrency-tainted rows remain noncanonical, and no bitwise-repeatability
  claim is made.

The independently validated v4 decision layer contains nine decisions, 40
figure-source rows, and six old-versus-new rows. It predates the fresh Study 05
run and is preserved as a point-in-time record because it pins the preceding
disposition bytes. A fresh terminal record must bind the new Study 05 evidence,
the updated disposition, status files, contracts, core v4, and final
engineering gate.

## Study-specific decisions

### Study 00

The simulation manifest SHA-256 is
`a7919681f5e376f1f8a7d1435892d0531d143389cf60254e6f9e87e9a658f7eb`;
the corrected 60-row atomic table SHA-256 is
`b80e01a740ac084867997189ac80c148e4f52ceee79c4f721985d9d98df75879`.
The external simulator outputs are retained, while all FEAST outputs are fresh.
No composite or conditional-OT arm is retained. Mean
`gene_variance_wasserstein` worsens from `1.3374325851` to `1.4484000857`;
the author must approve, narrow, rewrite, or remove the simulator claim.

### Study 01

The fresh 252-row metric table SHA-256 is
`10abbadbcf8c26d2b85e92a3357a8380fe4f4bc46882e6d42d2388a3d1882fab`.
The maximum absolute FEAST-versus-real method-mean baseline gap is
`0.0078601533`, while the maximum slice-level gap is `0.0774812245`.
The supported statement is three-slice method-mean closeness, not uniform slice
equivalence. Leiden selection is label-free; the historical truth-selected
table remains sensitivity evidence only.

### Study 02

All 40 method candidates and 40 score rows validate. The method manifest,
validation table, and metric hashes are
`881a221125e804529650e83d96d6e918f97aa069fb80c233fe6029c1215fb220`,
`65266a8c2752e21bd0c3ac74d4e6d5f1c26499095042dad3137c2da97cc67901`,
and `d405d2b8bea4d834cee729013803d2cb9c3ba2193f25d5c9df08dc1446723e92`.
The author must select metric, panel, alteration strata, and claim before any
table, figure, ranking, or winner statement is promoted.

### Study 03

RCTD and Cell2location use the same six expression simulations by path and
SHA-256. Exact positive spot support, common named types plus `__other__`, and
symmetric zero-library exclusion validate independently. Fresh mean
JSD/Pearson/summed-RMSE values are `0.320446/0.789990/0.054500` for RCTD and
`0.521656/0.605519/0.085278` for Cell2location.

Historical winners RCTD/Cell2location/Cell2location become RCTD/RCTD/RCTD;
Pearson and summed-RMSE change direction. This is an evidence-complete
scientific stop. The author must remove or narrowly rewrite the comparison; no
superiority, ranking, winner, table, or figure is authorized.

### Study 04 / legacy Study 08

The exact 36-row method manifest, 60-row atomic table, and independent
validation hashes are
`a12c9133caadb2a3692085a1cc25c0932c23bce7cb8af44f5ab1d8777a1410fe`,
`a6c78a602d4b0025628917d2c8a015cb7de19c962dfe14aedac8b333c18a4b2a`,
and `f48b6c4d8284c6710d26ca352a715daee504178d9b19b12e3fd2b7a50066ce1d`.
Zero cross-batch edges are asserted. This is a supplementary candidate only;
later scope expansion must preserve the STAMP retry, unsupported external
NumPy, extrapolation, and GraphST PCA-sensitivity limitations.

### Study 05 / legacy Study 04

All 45 fresh candidates validate: 40 cross-slice directions/randomness jobs and
five deterministic low-x-observed to high-x-held-out half-slice jobs. The score
root contains 45 summary rows, 45 support rows, and 587,367 per-gene rows. The
complete validation and summary hashes are
`63d0aada03bb65e0896a08426e55a8207a4bd05ac6141741325c9c3213c427ae`
and
`7e6b7a8979d80f49e69646c392e56ac0f9a28adb6b8a44a6223d9e17c94d9b40`.
Target expression was not loaded during generation.

At the predeclared assignment-randomness value 0.3, aggregate gene mean,
variance, and Moran-profile agreement is strong on the bounded slices. That
does not translate into coordinate-wise expression recovery: median per-gene
spotwise correlations are near zero for DLPFC and approximately 0.04 for
MERFISH. The supported interpretation is label-aware conditional generation
given observed target XY and labels, not accurate spot-level imputation,
reconstruction, or prediction. The editable review figure is complete but is
not promoted pending author acceptance of that wording.

### Study 06 / legacy Study 05

All 147 inputs and 735 metadata hashes validate, but the initial full preflight
and two independent one-thread repeats all stop during the first dense-gap-3
reference-only AR estimate. POT reports divide-by-zero and numerical errors at
iteration 0, which FEAST correctly rejects under `transport_nonconvergence=
"raise"`. Isolated slice-004 diagnostics pass, so the current classification is
preflight-context numerical instability amplified by tiny class supports, not
corrupt input or an AR mismatch. Gap-3, gap-5, and gap-10 AR values remain
unverified. No canary or production generation is allowed until a versioned
solver/estimator repair passes fresh reproducible preflights.

### Study 07 / legacy Study 06

The first E18.5 calibration attempt correctly exposed an internal contract
mismatch: strict package-default filters retained 181 rather than the claimed
550 genes. Historical E15.5 actually used permissive `1/0.0/1.0` thresholds and
saved all 550 genes. Configuration v2 now applies that exact contract to both
ages. Both expression-free blueprints were regenerated with fresh provenance,
13 tests pass, and the corrected E18.5 reference-only calibration selected 0.5.
Age-specific strict-convergence canaries are the only generation currently in
progress.

## Package, core, and contracts

The post-disposition engineering gate is
`publication/outputs/final_gate_20260719_v2/`. It binds the updated scientific
disposition and release checklist. Source and installed-wheel suites each report
108 passed with one expected alignment xfail. Strict Sphinx, supported and wheel
`pip check`, wheel/sdist builds, twine, snapshot rehash, and all 28 independent
gate checks pass. The gate summary, wheel, and sdist hashes are
`0c73788b1ea02a3fd7da3e1c7e8c0f9a3a7b245339c648cc6e564e0078bcd658`,
`10ab398c68b34bd8ed3bd1fe5831cebc982eb76c6736a68a722664955b764a2d`,
and `8ba0a53b16ae9afbc9550e287306fb40cd402464527e5f0819773ae427dfbac2`.

Core v4 completed without execution errors but retains one scientific failure:
the adversarial blocked-assignment case. Fresh exact-global candidates do not
consume it. The result and run-manifest hashes are
`06d50c29a41d6fb9a3c0df4a78e5126f2a581920610ff69ccdc0e7ac7b256aa5`
and `a7e0b1923e45811e55fa68d9e40cefaa75fcfbfd23834834e2fc71f2a5f7df76`.

All eight v2 method contracts validate. Studies 00, 01, 02, and clean Study
04/legacy 08 are `validated_candidate`; Study 03 is evidence-bound at
`decision_required`. The fresh Study 05 evidence validates separately but its
article claim remains unapproved. Study 06 is preflight-blocked; Study 07
remains `validation_required` while its canaries and full volume are incomplete.

## Remaining integration actions

1. Generate and independently validate the fresh v5 terminal decision record.
2. Build and independently rehash the final candidate artifact freeze.
3. Revalidate all eight method contracts and repository policy gates against
   those terminal records.
4. Repair and re-pin the Study 06 estimator/solver before any generation; in
   parallel, complete and validate the Study 07 canaries and then its authorized
   fresh scope. Do not consume the 156 historical conditional-OT H5ADs.
5. Obtain author decisions for Studies 00–03 and Study 05, and accept or
   otherwise dispose the two historical base-freeze divergences.
6. Keep release approval blocked. Do not create a tag or publish to PyPI.
