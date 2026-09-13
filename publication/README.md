# Publication records and release requirements

These tools retain the recorded scientific provenance and release requirements.
They are not a current experiment launcher or a declaration that every study
has finished. The study READMEs identify the active workflows.

## Retained tools

- `build_final_freeze.py` and `validate_final_freeze.py` build/check the existing
  artifact inventory. Their generated reports stay local.
- `method_contract_v2/` contains the July 19 historical method records and
  validator. Evidence resolves against this repository and the sibling
  `FEAST_experiments` and `FEAST` repositories.
- `method_contract_v3/` is the addendum referencing the v2 JSON records and newer
  evidence. The predecessor records and path relationships remain intact.

```bash
python scripts/check_repository.py
python scripts/check_release_manifest.py
python publication/method_contract_v2/validate_contract.py
python publication/method_contract_v3/validate_contract.py
```

Run from the repository root; contract validation requires its referenced local
evidence. The release-manifest check still assumes the original FEAST commit
for every study and fails at Study 02's different pin. It also predates the
removed Study 06 `config.yaml`. Simplifying documentation does not reconcile
those scientific/build records or bypass their checks.

## Preservation and authorization

The original history is retained at commit
`a582b9b0e42681fcd5f6c52918b4123322f930a2`, branch
`archive/legacy-experiments`, and tag `legacy-experiments-20260630`.
Never force-update those archival references. Local superseded artifacts remain
under `.archive/`, preserving their original relative layout. Do not rewrite,
relabel, or silently promote them.

A publication candidate requires its study validation and recorded input,
configuration, runner, environment, and output provenance. The existing freeze's
76 failure-evidence and eight noncanonical-evidence files remain at their bound
paths unless a replacement freeze is built and independently validated.

Keep results tied to the build that generated them. A changed FEAST build
requires a fresh output root and an explicit impact decision. The recorded
1.0.2 wheel is verified against its build record; distributions may be local or
immutable release assets and must not be rebuilt under the same version after
source changes.

`PUBLICATION_MANIFEST.json` remains `in_progress` with `release_authorized=false`
until all studies are validated and cross-study integration is complete.
No final article tag may be created while authorization is false. A rendered
figure or code push does not authorize a publication claim, composite score,
method ranking, or release. Retained method/claim decisions remain binding.
