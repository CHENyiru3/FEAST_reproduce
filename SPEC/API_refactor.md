# FEAST API Refactor

**Branch:** `api-refactor`
**Date:** 2026-06-27
**Status:** In progress

## Motivation

The old API was organized by *task* (alignment, deconvolution, de_novo) rather than by
*function* (simulate, generate, fit).  This forced users to understand internal package
structure and led to three different import paths for the same operation.

## Design — Function-faced API

```
FEAST
├── simulate()          # ST reference → synthetic ST slice
├── generate()          # blueprint + param_cloud → virtual ST slice
├── generate_from()     # reference + blueprint → conditional virtual ST
├── fit()               # learn parameter-cloud model from data
├── decode()            # param_cloud + rank_scores → count matrix
├── Alteration          # expression alteration config
├── spatial_transform   # .rotate(), .warp()
│
├── [backward compat]
│   ├── AlterationConfig    (→ Alteration)
│   ├── simulator           (subpackage, unchanged)
│   ├── alignment           (subpackage, unchanged)
│   ├── deconvolution       (subpackage, unchanged)
│   └── de_novo             (subpackage, unchanged)
```

## Before / After

| Task                     | Old import                                                        | New import                                   |
|--------------------------|-------------------------------------------------------------------|----------------------------------------------|
| Single-slice simulation  | `from FEAST.FEAST_core.simulator import simulate_single_slice`   | `from FEAST import simulate`                 |
| Expression alteration    | `from FEAST.modeling.marginal_alteration import AlterationConfig` | `from FEAST import Alteration`               |
| Fit parameter cloud      | `from FEAST.FEAST_core.parameter_cloud import GeneParameterSimulator` | `from FEAST import fit` (or `GeneParameterSimulator`) |
| De novo generation       | `from FEAST.de_novo.builder import simulate_from_design`         | `from FEAST import generate`                 |
| Conditional generation   | `from FEAST.de_novo.conditional import simulate_from_reference`  | `from FEAST import generate_from`            |
| Count decoding           | `from FEAST.FEAST_core.count_decoding import decode_counts_by_rank` | `from FEAST import decode`                |
| Spatial rotation         | `RotationTransformer(adata).transform_sequencing(angle=30)`      | `from FEAST.spatial_transform import rotate` |
| Spatial warp             | `WarpTransformer(adata)....`                                     | `from FEAST.spatial_transform import warp`   |

## What was removed

- **`FEAST` class** — thin stateful wrapper that no experiment used directly (removed
  from public API; underlying functions still exist).
- **`FEAST_core` from public surface** — still importable for backward compat, but
  no longer documented.

## Files changed

### New
- `src/FEAST/spatial_transform.py` — `rotate()`, `warp()` pure functions

### Modified
- `src/FEAST/__init__.py` — rewrote public surface with `simulate`, `generate`,
  `generate_from`, `fit`, `decode`, `Alteration`; kept backward-compat aliases

### Unchanged (implementation)
- `src/FEAST/FEAST_core/simulator.py`
- `src/FEAST/FEAST_core/parameter_cloud.py`
- `src/FEAST/FEAST_core/count_decoding.py`
- `src/FEAST/modeling/marginal_alteration.py`
- `src/FEAST/de_novo/*.py`

## Experiment script migration

Each experiment directory below must be updated to use the new import paths.

- [x] API surface implemented (`__init__.py`, `spatial_transform.py`)
- [x] 00_simulator_benchmark — 2 files
- [x] 01_clustering — 3 files  
- [x] 02_alignment — 2 files
- [x] 03_deconvolution — 0 files (already uses backward-compat imports)
- [x] 04_2d_conditional_transfer — 1 file
- [x] 05_3d_stack — 1 file
- [x] 06_3d_transfer — 3 files
- [x] 07_Visualization — skip (no FEAST imports)
- [x] 08_batch_effect_removal — 2 files
- [x] **Total: 14 files migrated across 8 experiment directories**

## Verification checklist

- [ ] `from FEAST import simulate, generate, generate_from, fit, decode, Alteration` succeeds
- [ ] `from FEAST.spatial_transform import rotate, warp` succeeds
- [ ] Old imports still work (backward compat)
- [ ] All experiment scripts run with new imports
- [ ] Results identical to pre-refactor
