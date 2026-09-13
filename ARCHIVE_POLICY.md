# Repository and artifact lineage

The original experiment repository is preserved at commit
`a582b9b0e42681fcd5f6c52918b4123322f930a2`. Its archival references are:

- branch `archive/legacy-experiments`;
- annotated tag `legacy-experiments-20260630`.

The clean rerun line is a normal descendant of that commit. Its cleanup commit
removes historical repair scripts from the active tree while retaining them in
the archive. The archive branch and tag must never be force-updated.

## Code-only distribution

The GitHub source tree excludes scientific output directories, rendered
`visualization/**/figures/` trees, validation reports, dated publication-freeze
reports, logs, caches, and local archives. Removing an artifact from Git tracking
does not delete the local artifact or alter its provenance. Existing Git history
is retained. The cleanup record is in [docs/REPOSITORY_REVIEW.md](docs/REPOSITORY_REVIEW.md).

## Scientific artifacts

Materialized inputs, run roots, logs, and method outputs remain ignored. A file
is a publication candidate only when its study validator accepts it and its
manifest records the input paths, configuration, runner, environment, and
output paths. Ignored output directories are not evidence by themselves.

Superseded local run roots are moved out of publication-facing `outputs/`
directories and retained below `.archive/`, which is ignored by Git. Their
original repository-relative layout is preserved beneath the dated archive
root so provenance can be recovered without confusing an old run with the
current candidate. The manifest records earlier publication candidates; newer study runs are
documented in their study READMEs. Cleanup must not rename, rewrite, or silently
promote archived artifacts. The manifest is not a complete index of the newest runs.

The terminal freeze deliberately retains 76 failure-evidence files and eight
noncanonical-evidence files inside selected candidate roots. They are
path-bound audit evidence, not competing candidate versions, and must remain
in place unless a new freeze is built and independently validated.

The FEAST v1.0.2 wheel is verified at installation against the build-record
digest. The wheel and source distribution may be kept locally or attached as
immutable release assets, but they are not ordinary source files and must not
be rebuilt under the same version after a source change.

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
