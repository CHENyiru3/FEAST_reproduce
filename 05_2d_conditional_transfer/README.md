# Study 05: 2D conditional transfer and half-slice generation

This clean workflow evaluates FEAST on two distinct conditional-generation
tasks and two technologies. It declares exactly 45 fresh FEAST candidates:

| Dataset | Slices | Cross-slice jobs | Half-slice jobs |
|---|---|---:|---:|
| DLPFC Visium | `151670`, `151675`, `151676` | 6 directions × 5 assignment-randomness settings = 30 | 3 |
| MERFISH Zhuang-ABCA-1 | `006`, `007` | 2 directions × 5 settings = 10 | 2 |

The DLPFC selection contains two donors: `151670` is from `Br5595`, while
`151675` and `151676` are from `Br8100`. Results must therefore retain the
within-donor versus cross-donor direction stratum. The fixed ordered panels are
17,391 DLPFC genes and 1,122 MERFISH genes.

## Source-only support contract

FEAST can condition only on labels represented in its reference. A target
label is retained only when the source contains at least 20 spots with that
label. This rule is applied before generation using labels, never target
expression, and every excluded target label and spot is recorded.

In particular, `151670` does not contain Layers 1–2. Its outgoing targets
retain 2,963/3,565 spots for `151675` and 2,888/3,431 for `151676`. MERFISH
cross-slice directions retain 9,660/9,693 and 10,409/10,442 target spots. These
support differences are reported per direction and must not be hidden by a
pooled score.

## Deterministic half-slice experiment

Each of the five slices is split at the median of `obsm['spatial'][:, 0]`:

- visible reference: `x <= median(x)`;
- masked target: `x > median(x)`;
- assignment randomness: the primary value 0.3 only.

The split is disjoint, exhaustive, and frozen before generation. The runner
loads expression only for retained visible-reference rows. It creates an
expression-free target contract containing the masked spot IDs, coordinates,
and known conditional labels. Masked expression is first loaded by `score.py`,
after all 45 generation candidates have independently validated.

All three DLPFC masked halves retain 100% of geometrically masked spots after
the source-label gate. MERFISH `006` and `007` retain 5,210/5,221 and
4,821/4,846 spots. The visible DLPFC halves contain 97, 146, and 129 genes with
zero reference counts. FEAST retains the fixed 17,391-gene panel with
`min_gene_spots=0`; those zero-evidence genes are explicitly identified, and
scores are reported both for the complete panel and for the
reference-observed-gene sensitivity subset.

This is conditional half-slice expression generation given known labels. It
is not label-free spatial imputation.

## Inputs and preflight

The five ignored local inputs are hardlinks below `data/local/dlpfc/` and
`data/local/merfish/`. Exact IDs, shapes, and SHA-256 values are frozen in
`data/input_checksums.csv`.

Use the clean wheel interpreter recorded in `../FEAST_BUILD.txt`:

```bash
FEAST_PY=/path/to/clean-feast-environment/bin/python
REPRO_ROOT="$(git rev-parse --show-toplevel)"

"$FEAST_PY" "$REPRO_ROOT/scripts/verify_feast_install.py"
"$FEAST_PY" -m pip check
cd "$REPRO_ROOT/05_2d_conditional_transfer"
"$FEAST_PY" preflight.py
```

## Run one candidate

Cross-slice example:

```bash
"$FEAST_PY" run.py \
  --mode cross_slice --dataset dlpfc \
  --source 151675 --target 151676 \
  --assignment-randomness 0.3
```

Half-slice example:

```bash
"$FEAST_PY" run.py \
  --mode mask_half --dataset dlpfc --slice 151675
```

Run one strict cross-slice canary and one strict half-slice canary per dataset
before dispatching the remaining independent CPU jobs. A candidate is exposed
atomically as:

```text
outputs/final/<mode>/<dataset>/<direction>/ar_<value>/
├── generated.h5ad
├── provenance.json
└── transport_diagnostics.csv
```

The runner refuses replacement. Interrupted or failed work remains under
ignored `.work/`. Every candidate binds the exact input/config/runner/wheel
hashes, source-only support, target identity and coordinates, public seed, and
positive convergence diagnostics. Target IDs are verified from the blueprint
column before FEAST's internal row index is rebound to public `obs_names`.

The public FEAST conditional implementation is CPU/NumPy based; it has no CUDA
backend selector. Before fitting, the runner caches the immutable reference
`counts` layer once in the same dense float32 representation FEAST requests
internally. This is a value-preserving wrapper optimization that avoids
repeated sparse densification in the empirical decoder. Its strategy, original
storage type, dtype, and shape are recorded in every candidate's provenance.

Study 05 is explicitly two-dimensional. Some MERFISH inputs also carry a
`spatial_3d` alias, which the public FEAST coordinate resolver would otherwise
prefer. The wrapper removes that alias only from the in-memory reference copy
and passes the frozen two-column `obsm['spatial']` coordinates on both sides;
the removal and dimensionality are recorded in the support provenance.

## Validate, score, and compare

```bash
"$FEAST_PY" validate.py --all \
  --report outputs/study05_validation.json

"$FEAST_PY" score.py \
  --generation-dir outputs/final \
  --output-dir outputs/scores

"$FEAST_PY" validate.py --all \
  --scores-dir outputs/scores \
  --report outputs/study05_complete_validation.json
```

The score root contains a 45-row summary, the complete per-gene table, a
45-row support audit, and provenance. Dataset, mode, donor stratum, direction,
assignment randomness, and support fraction remain explicit; no across-mode
or across-technology composite is created.

For DLPFC, `donor_stratum` is `within_donor` for `151675↔151676`,
`cross_donor` for directions involving `151670`, and `within_slice` for the
mask experiment. MERFISH donor identity is not declared by these inputs and is
recorded as `donor_not_declared`, not inferred from adjacent slice names.

Historical numerical comparison is restricted to the original ten
`151675`↔`151676` cross-slice rows:

```bash
"$FEAST_PY" compare_historical.py \
  --historical-root <historical-cross-slice-root> \
  --fresh-score-dir outputs/scores \
  --output <old-versus-new.csv>
```

The other 35 candidates have no valid historical numerical counterpart. The
legacy masked-half implementation passed the held-out target AnnData into
generation, so its masking scores are leakage-tainted and noncanonical.

Fresh candidates use the public unified API `fit_reference` plus
`simulate_from_reference`, strict nonconvergence policy `raise`, and 1,000
Sinkhorn iterations rather than the historical 200. Passing validation creates
evidence only; publication-canonical status still requires metric review and
author approval.
