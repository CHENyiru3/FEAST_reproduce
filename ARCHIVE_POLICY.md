# Repository and artifact lineage

The original experiment repository is preserved at commit
`a582b9b0e42681fcd5f6c52918b4123322f930a2`. Its archival references are:

- branch `archive/legacy-experiments`;
- annotated tag `legacy-experiments-20260630`.

The clean rerun line is a normal descendant of that commit. Its cleanup commit
removes historical repair scripts from the active tree while retaining them in
the archive. The archive branch and tag must never be force-updated.

## Scientific artifacts

Materialized inputs, run roots, logs, and method outputs remain ignored. A file
is a publication candidate only when its study validator accepts it and its
manifest records the exact input, configuration, runner, environment, and
output hashes. Ignored output directories are not evidence by themselves.

Superseded local run roots are moved out of publication-facing `outputs/`
directories and retained below `.archive/`, which is ignored by Git. Their
original repository-relative layout is preserved beneath the dated archive
root so provenance can be recovered without confusing an old run with the
current candidate. The current candidate roots are exactly those declared in
`PUBLICATION_MANIFEST.json`; cleanup must not rename, rewrite, or silently
promote archived artifacts.

The terminal freeze deliberately retains 76 failure-evidence files and eight
noncanonical-evidence files inside selected candidate roots. They are
path-bound audit evidence, not competing candidate versions, and must remain
in place unless a new freeze is built and independently validated.

The FEAST v1.0.2 wheel and source distribution are identified by SHA-256 in
`FEAST_BUILD.txt` and `PUBLICATION_MANIFEST.json`. They may be kept locally or
attached as immutable release assets, but they are not ordinary source files
and must not be rebuilt under the same version after a source change.

## FEAST release boundary

Completed article studies remain tied to the recorded v1.0.2 build. The current
FEAST release removes compatibility arguments and repairs RNG, convergence,
and aggregation behavior. It therefore requires a fresh output root and a
targeted impact decision; existing artifacts must never be relabelled as
current-release results.

## Release authorization

`PUBLICATION_MANIFEST.json` remains `in_progress` and
`release_authorized=false` until every study is validated and cross-study
integration is complete. No final article tag may be created while that gate is
false.
