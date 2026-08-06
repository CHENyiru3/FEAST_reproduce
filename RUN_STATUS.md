# FEAST publication rerun status

Checkpoint: 2026-07-31. The selected fresh executions for Studies 00–05 are
complete, but Study 00 remains scientifically unpromoted because an ignored
predecessor metric table was overwritten and its original bytes are
unavailable. Study 06 has completed preflight, CUDA canaries, 93/93 fresh
generations, strict validation, atomic evaluation, and its unpromoted
diagnostic. Study 07 has completed both age-specific CUDA canaries, 360/360
fresh full-axis generations, consolidation, strict validation, descriptive
continuity evaluation, and its unpromoted diagnostic.

The formal FEAST release remains provisional and unapproved. Studies 06 and 07
pin the repaired FEAST 1.1.0 candidate wheel (SHA-256
`3ad31888faf367a91aec9d46902a5e89759837b5c7ea0e45ca93276674e68883`)
to source commit `68816e5c1862a6fa2a49bc30609d617c7fa4b449` plus the recorded repair
snapshot. The scientific disposition remains blocked and unapproved. No
release commit, tag, or PyPI publication has been created.

## Completion matrix

| Study | Fresh completion | Retained status |
|---|---|---|
| 00 simulator benchmark | 12/12 FEAST simulations; 48 retained external outputs; fresh scCube Slide-seq completion; 60/60 atomic metric rows | Pairwise evidence complete, but predecessor-table lineage is incomplete; author disposition required and no ranking/figure promotion authorized |
| 01 clustering | 81/81 simulations; 84/84 panels; 84/84 GraphST, STAGATE+mclust, and label-free Leiden; 252/252 metrics | Validated for three-slice method-mean closeness only |
| 02 alignment | 20/20 rotations; 20/20 PASTE; 20/20 Spateo; 40/40 metrics | Validated evidence; author comparison scope required |
| 03 deconvolution | 6/6 simulations; 6/6 RCTD; 6/6 Cell2location; 12/12 common-support scores | Evidence-complete scientific stop after headline directions changed |
| 04 batch-effect removal (legacy 08) | 12/12 GraphST; 12/12 STAMP; 12/12 scVI; 60/60 atomic metrics | Validated supplementary candidate only |
| 05 2D conditional transfer (legacy 04) | 45/45 generations; 45/45 summary/support rows; 587,367 per-gene rows; 16/16 tests; figure rebuilt | Validated evidence; accurate spot-level prediction claim rejected; bounded conditional-generation wording needs author approval |
| 06 conditional 3D stack (legacy 05) | 93/93 CUDA generations; 93/93 independently validated; atomic evaluation and editable diagnostic complete; 16/16 workflow tests | Validated candidate for declared v5 scope; figure/claim remains unpromoted |
| 07 DevCCF 3D transfer (legacy 06) | 158/158 E15.5 plus 202/202 E18.5 CUDA generations; all 360 independently validated; descriptive evaluation and editable diagnostic complete; 16/16 workflow/evaluator tests | Validated descriptive candidate; no target-expression accuracy claim and no canonical promotion |

GraphST, STAGATE, Spateo, STAMP, scVI, and Cell2location jobs selected for
Studies 00–04 completed on the NVIDIA RTX A6000. Study 05 completed via
parallel independent CPU jobs. Studies 06 and 07 used the pinned Torch/CUDA
log-domain transport on the NVIDIA RTX A6000 with no silent CPU or method
fallback. External method environments remain isolated and may use NumPy
versions unsupported by FEAST; FEAST was not imported in those workers.

The Study 07 expression-free blueprint hashes are
`b03b48ee68929f0cfb8571ef1d7b7251bef9f95a3d253303f95701ca471bc1ff`
for E15.5 and
`b05fefd347163ae42ee74ef4852f9fe3bae045888b6248913981f4b39b7fce9b`
for E18.5. Their exact scopes are 158/202 z levels and
2,330,927/5,213,461 spots. These are prepared geometry inputs, not FEAST
simulation results or canonical article outputs. Configuration v5 applies the
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
`a7919681f5e376f1f8a7d1435892d0531d143389cf60254e6f9e87e9a658f7eb`.
The fresh scCube-complete 60-row atomic table is
`00_simulator_benchmark/outputs/final_metrics/simulator_quality_metrics.csv`,
SHA-256
`a315f525ac11fcc0666d94bab5b0840e058744bd4e9fc610eeb7942fddc94fb6`.
All pairwise cosine-divergence values are present, including
`scCube/Slideseq_001`. The external simulator outputs are retained, while all
FEAST outputs are fresh. No composite or conditional-OT arm is retained. Mean
FEAST `gene_variance_wasserstein` is `1.4484000857` and remains explicitly
visible in the complete table.

This evidence is not canonical because the ignored predecessor `metrics_v2`
file expected at SHA-256
`b80e01a740ac084867997189ac80c148e4f52ceee79c4f721985d9d98df75879`
was overwritten in place; its current SHA-256 is
`f497e386d0d6d7acf32f672140a5653af14b971bbe7b6e37f9da564d866044d3`
and the expected original bytes were not recovered. The final provenance and
v3 contract preserve this source-lineage break. The author must disposition
the lineage and atomic changes before any simulator ranking, winner claim, or
figure promotion.

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

The versioned log-domain repair closed the iteration-zero POT underflow defect
without relaxing strict nonconvergence handling. Fresh configuration v5
reproduced reference-only assignment-randomness values 0.35, 0.30, and 0.35
for gaps 3, 5, and 10, respectively. The CUDA canaries and all 93 production
targets then completed. Independent validation reports 93/93 outputs, exact
target identity, and positive convergence evidence throughout. Validation
summary SHA-256:
`c41bb104a06a6711070f1888533422ba11596d7e7bebf4fa044bc1aaeabf4a37`.

Atomic evaluation is complete with score-provenance SHA-256
`2bb9d66f4fc3ad20c4e9683ba7fc5db4dedbb52c86b06ed515e1e9d5cea6c85b`.
The target slices are ordered z levels, not independent biological replicates;
no across-target inferential test, composite score, or winner ranking is
allowed. The editable diagnostic is validated but unpromoted.

### Study 07 / legacy Study 06

The first E18.5 calibration attempt correctly exposed an internal contract
mismatch: strict package-default filters retained 181 rather than the claimed
550 genes. Configuration v5 applies the historical E15.5 permissive
`1/0.0/1.0` contract to both ages and retains the same ordered 550 genes. The
corrected E18.5 reference-only calibration selected 0.5. Both age-specific
CUDA canary sets and the complete scope then finished.

Final consolidation contains exactly 158 E15.5 and 202 E18.5 unique levels.
Independent validation reopened all 360 H5ADs, confirmed exact blueprint
identity, and found all 12,290 transport records converged. Validation SHA-256:
`2e9a1607b8e21e1e035c2b8353a0be78bcf96782a58ba7266d0138183b17aa1f`.
Descriptive full-axis coverage and adjacent-z continuity evaluation is complete
(provenance SHA-256
`6e41d14bdff42dba10ee473f3b2477efd63bfd7fcfe2b648168ea738a906453c`).
Because the blueprints have no target expression, no accuracy claim is
authorized. The editable diagnostic remains unpromoted.

## Package, core, and contracts

The preserved FEAST 1.0.2 post-disposition engineering gate is
`publication/outputs/final_gate_20260719_v2/`. It binds the updated scientific
disposition and release checklist. Source and installed-wheel suites each report
108 passed with one expected alignment xfail. Strict Sphinx, supported and wheel
`pip check`, wheel/sdist builds, twine, snapshot rehash, and all 28 independent
gate checks pass. The gate summary, wheel, and sdist hashes are
`0c73788b1ea02a3fd7da3e1c7e8c0f9a3a7b245339c648cc6e564e0078bcd658`,
`10ab398c68b34bd8ed3bd1fe5831cebc982eb76c6736a68a722664955b764a2d`,
and `8ba0a53b16ae9afbc9550e287306fb40cd402464527e5f0819773ae427dfbac2`.

The additional 2026-07-31 candidate-source gate reports 92/92 passing tests and
`pip check` success in the supported Python 3.11.15 / NumPy 1.26.4
environment. The conditional workflow gate reports 48/48 passing tests (16
each for Studies 05, 06, and 07), repository and publication-manifest checks
pass, and the installed provisional 1.1.0 wheel verifies. Its warning that the
live source differs from the wheel snapshot is intentional: the wheel is a
local, unreleased repair candidate, not a final package release.

Core v4 completed without execution errors but retains one scientific failure:
the adversarial blocked-assignment case. Fresh exact-global candidates do not
consume it. The result and run-manifest hashes are
`06d50c29a41d6fb9a3c0df4a78e5126f2a581920610ff69ccdc0e7ac7b256aa5`
and `a7e0b1923e45811e55fa68d9e40cefaa75fcfbfd23834834e2fc71f2a5f7df76`.

The frozen v2 method contracts are preserved as point-in-time evidence, but
they no longer validate the current disk because the ignored Study 00
`metrics_v2` file was overwritten after that gate. This failure is not hidden
or repaired in place. The additive `publication/method_contract_v3/` layer
pins all eight v2 predecessors, binds the current direct evidence, records the
Study 00 lineage break, and validates 8/8. Every v3 publication claim and
release authorization remains false. Studies 06 and 07 pass their independent
output gates for the declared v5 configurations; their figures and scientific
wording remain unpromoted pending author review.

## Remaining integration actions

1. Obtain author decisions for Studies 00–03 and Study 05, including an
   explicit disposition of the Study 00 predecessor-table lineage break and
   the historical base-freeze divergences.
2. Review the Study 06 diagnostic/wording and the descriptive-only Study 07
   diagnostic; explicitly promote, revise, or omit each from the article.
3. Run the optional Study 06 historical comparison only if an author-approved,
   hash-pinned historical summary is supplied; do not infer or substitute it.
4. After those scientific decisions, generate the fresh terminal decision
   record, rebuild the candidate artifact freeze, and rerun all eight v3
   contracts and repository gates against that terminal state.
5. Keep release approval blocked. Do not create a tag or publish to PyPI.
