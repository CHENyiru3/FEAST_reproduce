# Code-only repository review — 2026-09-13

## Cleanup

The public scope is source code, configurations, input identifiers, environment
inventories, and useful reproduction documentation. Scientific outputs remain
local at their existing paths because figure builders can read intermediate
plot-data tables and resampling controls there.

Generated figure trees, validation reports, and dated publication-freeze reports
were removed from Git tracking without deleting local files. New ignore rules
also exclude historical `archive/` folders and local tool caches. Existing Git
history is unchanged, so older commits still contain previously tracked figures.

Superseded root run notes, the conditional rerun plan, the old alignment figure
archive, and root pytest/Ruff caches moved to
`.archive/code-only-cleanup-20260913/`, preserving their relative layout.
Original copies of rewritten documentation and replaced Study 07 plotting
scripts are retained there. Existing study archives stay at their recorded
paths and are excluded from the code upload.

The Study 07 workflow used for the completed two-reference float32 run was
recovered from `.work/study07-two-reference-float32-20260906/` into
`07_3d_transfer/two_reference_float32/`. The plotting scripts used for the
September 9 report were recovered from
`.work/study07-root-figures-20260909/plotting/` into the study's visualization
folder. Only location-dependent paths were adjusted in these recovered files.
The old verified-wheel workflow and its tests remain available separately.
The still-used Study 07 conditional-resampling baseline was also recovered from
the historical archive; its evaluation helpers match the archived implementation.

## Findings that affect reproduction

- **Different FEAST builds:** the root 1.0.2 build record does not cover every
  current workflow. Study 06 references a separate 1.0.6 precision build;
  Study 07's completed float32 run imported local FEAST source. The latter
  records runtime details but does not verify an immutable source build.
- **Study 06 remains in progress in its latest recorded notes.** This review
  did not run generation or determine scientific completion. Pass the selected
  configuration and output root explicitly; older script defaults still exist.
- **External inputs are required:** processed datasets, some historical
  comparison tables, blueprints, and the Study 07 resampling control are local.
  A GitHub clone alone cannot render every figure. Upstream conversion and acquisition code is now included in
  `preprocessing/`; dataset distribution, original label tables, and a complete
  environment specification remain separate work.
- **Existing CI fails:** before cleanup, `check_repository.py` reported local
  paths and generated provenance files. `check_release_manifest.py` failed at
  Study 02 because it assumes every study uses the original FEAST commit; it
  also expects the removed Study 06 `config.yaml`. The manifest and conditional
  checks describe older study configurations. Release authorization and the
  existing checks have not been bypassed or relaxed to claim a successful release.
- **Older publication records are not a current result index:** dated freezes
  and method contracts remain historical provenance. Use the study READMEs and
  plotting commands to identify the current code paths.
- **No repository license or citation file was present.** A license and the
  final manuscript citation require the authors' chosen terms and reference.

## Verification results

- The source-repository check passes after checking Git-visible code instead
  of ignored outputs and replacing machine-specific documentation paths.
- Two Study 04 plotting scripts now locate their existing Arial font files via
  `FEAST_FONT_DIR` or the Python environment's `fonts/` directory.
- At the initial cleanup, all 167 Python sources passed syntax parsing and the
  Git-visible working tree was approximately 2.2 MiB, compared with 1,508 MiB
  before cleanup. The preprocessing addition is verified separately below.
- Revised guide links resolve, and the recovered Study 07 configuration loads
  with unchanged scientific settings and identical resolved local paths.
- All 11 checked entry points load with `--help`: the recovered generation,
  consolidation, validation, evaluation, six plotting commands, and resampling
  baseline. Generation-related imports require the documented local-source
  `PYTHONPATH`; the installed environment alone lacks the needed FEAST API.
- The release-manifest check still fails for the pre-existing mixed-build
  mismatch described above. Its scientific requirements remain unchanged.

## Verification scope

Cleanup verification covers Python syntax, documentation links, Git exclusions,
configuration path resolution, and command-line loading of relocated code where
its existing environment is available. No scientific outputs were regenerated,
and no methods or scientific thresholds were changed during the cleanup or
preprocessing-code migration. The GitHub code update includes the existing
working-tree changes alongside this cleanup; scientific release
authorization remains unchanged.

## Preprocessing code added

`preprocessing/` contains the 12 original Python scripts from
`Reproduce/Preprocess_helper` and eight raw-data download scripts with unchanged logic and
their existing URL tables. Only machine-specific Python path defaults were
adjusted. The guide maps datasets to studies, documents QC defaults, required
external labels, and the GSE269617/DevCCF annotation sequence. No raw or processed
data, per-cell labels, or run logs were copied or regenerated.

The complete source tree contains 179 Python files, all of which pass syntax
parsing. All 11 preprocessing entry points load with `--help`, all eight Bash
download scripts pass `bash -n`, and the download URL tables match their sources.
The Python processing logic matches the original files after substituting path
defaults. Three download scripts received only trailing-blank-line cleanup.
