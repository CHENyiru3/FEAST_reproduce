# Environments

Use the FEAST build declared by the selected study and separate environments
for external methods. The inventories below record observed installations;
they are not one installable environment or a cross-platform lock.

| Stack | Inventory | Key requirements |
|---|---|---|
| Original FEAST 1.0.2 | [pip list](feast_py311_68816e5-pip-list.txt) | Python 3.11.15, NumPy 1.26.4; build in [FEAST_BUILD.txt](../FEAST_BUILD.txt) |
| Study 04 STAMP | [pip list](stamp_publication_20260718-pip-list.txt), [conda packages](stamp_publication_20260718-conda-explicit.txt) | scTM 0.1.3, NumPy 1.26.4, PyTorch 2.6.0+cu124, Squidpy 1.2.2 |
| Study 06 SpatialZ | [pip list](spatialz_e1b5ed0-pip-list.txt) | Python 3.9.19, NumPy 1.22.2, PyTorch 1.13.0+cu117 |

Study 06's `config_1.0.6_precision.yaml` records a separate FEAST precision
build. Study 07's current two-reference float32 workflow imports the recorded
local FEAST source through `PYTHONPATH`; its dependency environment alone lacks
the required API. Follow the study READMEs instead of substituting one build
for another.

External stacks include GraphST, STAGATE/mclust, PASTE2, Spateo, RCTD/spacexr,
Cell2location, STAMP, scVI, scCube, and SpatialZ. Their versions are independent
of FEAST support. GraphST/STAGATE/scCube use NumPy 1.23.x; Cell2location/scVI and
the Spateo candidate use NumPy 2.x. FEAST must run in its own supported environment.
The Spateo candidate's `pip check` reports an unrelated older FEAST conflict;
that limitation remains part of Study 02 provenance.

The isolated STAMP environment uses leidenalg 0.12.0, igraph 1.0.0, and texttable
1.7.0; its unused conflicting SpatialData branch was removed during its original
setup. The source environment was preserved. The package inventories retain
that installation without asking other environments to adopt its dependencies.

SpatialZ uses source commit `e1b5ed01e933729149e08399febca08db16ced14`
from `senlin-lin/SpatialZ`. Because that source has no package metadata,
`SpatialZ.py` and `Synthesize.py` are linked from the pinned checkout into the
method environment's `site-packages`. FEAST is not run there.

For preprocessing, use Python 3.11 with AnnData, NumPy, pandas, SciPy, Scanpy,
and h5py; optional atlas plots require Matplotlib. The two Study 04 Arial-based
plotters also require local font files; see the study's figure instructions.
