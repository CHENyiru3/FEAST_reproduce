# Execution environments

Studies 00–05 use the clean FEAST 1.0.2 wheel and supported Python 3.11.15 /
NumPy 1.26.4 environment recorded in `../FEAST_BUILD.txt`. Studies 06 and 07
instead pin the unreleased FEAST 1.1.0 log-domain repair candidate recorded in
`../../FEAST/validation/package_builds/20260719_ot_log_repair_v2/provenance.json`
and run its declared Torch/CUDA path. At the current site these are installed
in separate clean environments; those paths are local infrastructure, not part
of the scientific contract.

The observed package inventory for that exact execution environment is stored
in [`feast_py311_68816e5/pip-list.txt`](feast_py311_68816e5/pip-list.txt).
It is an audit snapshot rather than a promise that every platform can solve the
same transitive environment.

External methods remain isolated in their method-specific environments:

- GraphST
- STAGATE and mclust
- Spateo
- PASTE
- RCTD / spacexr
- Cell2location
- STAMP
- scVI
- scCube

Each study README identifies the command or interpreter needed for its method.
Before publication, export the exact package list for every retained method
environment into this directory. Unsupported external NumPy versions must be
reported as method-environment limitations and are not FEAST support claims.

Current external stacks observed before execution:

| Method | Python | NumPy | Disposition |
|---|---:|---:|---|
| GraphST | 3.8.20 | 1.23.4 | Unsupported by FEAST; FEAST is not run there |
| STAGATE | 3.8.20 | 1.23.5 | Unsupported by FEAST; FEAST is not run there |
| STAMP | 3.10.20 | 1.26.4 | Isolated publication env; `pip check` and CUDA tensor preflight pass |
| Cell2location / scVI | 3.10.20 | 2.2.6 | Unsupported by FEAST; FEAST is not run there |
| Spateo CUDA candidate | 3.11.15 | 2.4.6 | Unsupported by FEAST; FEAST is not run there |
| scCube Slide-seq rerun | 3.8.20 | 1.23.5 | Unsupported external method environment; FEAST is not imported there |

The Spateo candidate also contains an unrelated older FEAST distribution, so
its environment-level `pip check` is not clean. Spateo method workers do not
import FEAST; FEAST simulation and validation stages stay in their explicitly
pinned clean 1.0.2 or repaired-candidate environments. This limitation must
remain visible in any retained Study 02 provenance.

The original STAMP environment remains unchanged and conflicted. Study 04 uses
an isolated full-copy publication environment whose exact inventory and repair
record are in [`stamp_publication_20260718/`](stamp_publication_20260718/).
Its `pip check`, import smoke test, and real CUDA tensor preflight pass; these
checks do not turn the external method environment into a FEAST support claim.
