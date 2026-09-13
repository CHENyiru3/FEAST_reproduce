# FEAST publication method contract v2 candidate

This directory is a non-destructive evidence-binding candidate for the eight
article workflow contracts. It does not replace or modify the frozen v1
contracts in `FEAST_experiments/SPEC/before_publication/method_contract`.
Each v2 record identifies its v1 predecessor by path and then records the clean
rerun evidence that is currently available.

This is an integration record, not release authorization. Every record has
`release.authorized=false`, and no record authorizes a composite score, method
ranking, winner, or unresolved manuscript claim.

## Evidence roots

- `REPRO`: this repository root, discovered from `validate_contract.py`
- `EXP`: the sibling `FEAST_experiments` repository
- `PKG`: the sibling `FEAST` repository

The validator resolves the small files listed in each `evidence` array. H5AD
artifacts are bound through their declared manifests; the
validator deliberately does not rehash large scientific outputs.

## Current disposition

| Legacy contract | Clean rerun | v2 candidate status | Bound evidence |
|---|---|---|---|
| 00 simulator | clean Study 00 | `validated_candidate` | 12 fresh FEAST simulations, 48 retained external rows, corrected atomic metrics, old-versus-new decision |
| 01 clustering | clean Study 01 | `validated_candidate` | 81 fresh simulations, 84 fixed panels, 84 rows for each of GraphST/STAGATE/label-free Leiden, 252 metrics |
| 02 alignment | clean Study 02 | `validated_candidate` | 20 rotations, 40 method outputs, 40 metrics, validation and decision records |
| 03 deconvolution | clean Study 03 | `decision_required` | six simulations, twelve method outputs, common-support scores, support audit, old-versus-new evidence, independent validation, and scientific-stop decision |
| 04 2D conditional | none | `validation_required` | frozen v1 contract plus the aggregate conditional-OT audit |
| 05 3D imputation | none | `validation_required` | frozen v1 contract plus the aggregate conditional-OT audit |
| 06 DevCCF transfer | none | `validation_required` | frozen v1 contract plus the aggregate conditional-OT audit |
| 08 batch removal | clean Study 04 | `validated_candidate` | 14 simulations, 2,000-gene panel, 36 method candidates, 60 atomic metric rows, old-versus-new evidence, and independent validation |

The three conditional workflows remain noncanonical as a group: 156 active
H5ADs, 31 explicit failure-saved outputs, 26 failure-tainted outputs, 99 with
no positive tolerance evidence, and zero verifiably converged outputs. A finite
transport plan is not convergence. Any retained article-scope rerun must use
`transport_nonconvergence="raise"` and a new output root.

## Scientific and author decisions still pending

Clean Study 03 is evidence-complete but stopped because the headline direction
changed. Historical JSD/Pearson/RMSE winners of RCTD/Cell2location/Cell2location
became RCTD/RCTD/RCTD. The fresh means and exact common-support audit are bound,
but they do not authorize a superiority claim, ranking, winner, table, or
figure. Authors must decide whether to remove or narrowly restate the comparison
while disclosing `__other__`, symmetric zero-library exclusions, and the
external Cell2location environment.

Clean Study 04 is now bound as a validated supplementary candidate. Its
independent validation does not authorize an active claim, table, figure,
composite, ranking, or winner. Author review remains required for all numeric
deltas, the STAMP patience-only retry, external method environments, alpha
extrapolation, and the GraphST PCA-20/PCA-10 sensitivity.

## Validation

Run from this directory:

```bash
python validate_contract.py
```

The standard-library validator checks JSON structure, the exact eight legacy
mappings, v1 predecessor paths, evidence-file paths, status/evidence
consistency, conditional-OT counts, and the universal no-release/no-ranking
gates. It does not make a scientific decision.
