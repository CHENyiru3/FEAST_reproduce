# FEAST Package Publishing Preparation

## Audit Summary

Full audit of 28 source files in `FEAST/src/FEAST/` and 82 experiment files across 8 subtask directories in `FEAST_experiments/`.

### Issues Found

| Category | Count | Severity |
|----------|-------|----------|
| Bare `except:` blocks | 2 | P0 |
| Global `warnings.filterwarnings('ignore')` | 2 | P0 |
| `except Exception` swallowing errors | 4 | P1 |
| Dead methods (unreachable) | ~460 lines | P1 |
| API kwargs TypeError bug | 1 | P1 |
| TPS fallback returns random noise | 1 | P1 |
| Unused imports | 9 | P2 |
| Debug print statements | 3 | P2 |
| Duplicate function pairs | 7 | P2 |
| Dead dict/var | 2 | P2 |
| Agent-narration comments | ~6 files | P3 |
| Trivial restatement comments | ~60 instances | P3 |
| Ornamental dividers | ~20 files | P3 |
| Redundant type coercions | ~50 | P2 |
| Redundant `.copy()` | 1 | P2 |
| Dead public method | 1 | P2 |

### Module Usage by Experiment

| FEAST Module | Used by | Status |
|-------------|---------|--------|
| `simulator.py` | 00, 01, 02, 04, 05, 08 | Core (6/8) |
| `parameter_cloud.py` | 01, 08 | Active |
| `marginal_alteration.py` | 01, 02 | Active |
| `z_regularize.py` | 05, 06 | Active |
| `z_spot_smooth.py` | 05, 06 | Active |
| `theta_transform.py` | 08 | Active |
| `deconvolution_simulator.py` | 03 | Active |
| `spatial_align_alter.py` | 02 | Active |
| `APIs.py` | 02 | Active |
| `conditional.py` | 06 | Active |
| `core.py` | 06 | Active |
| `count_decoding.py` | 01 | Active |
| `_ot_transport.py` | 00 (tests only) | Low usage |
| `quantile_field.py` | 00 (tests only) | Low usage |
| `builder.py` | none | Public API, no experiment coverage |
| `pattern.py` | none | Public API, no experiment coverage |
| `stack.py` | none | Public API, no experiment coverage |
| `alignment_simulator.py` | via APIs.py only | Thin |
| `generate_deconvolution.py` | via APIs.py (unreached path) | No coverage |
| `Beta_mixture_model.py` | internal (parameter_cloud) | Indirect |
| `StudentT_mixture_model.py` | internal (parameter_cloud) | Indirect |
| `_metadata.py` | internal (conditional) | Indirect |

---

## Subtask 1: P0 Safety Fixes

**Files:** `StudentT_mixture_model.py`, `Beta_mixture_model.py`

- Replace 2 bare `except:` with `except Exception:`
- Replace 2 global `warnings.filterwarnings('ignore')` with targeted `with warnings.catch_warnings():` at call sites

**Verification:** Import both modules, confirm no blanket warning suppression.

---

## Subtask 2: P1 Critical Fixes

**Files:** `simulator.py`, `APIs.py`, `spatial_align_alter.py`, `builder.py`, `alignment_simulator.py`

- Remove ~460 lines dead `_apply_deterministic_rank_assignment` call tree (10 methods)
- Remove `_SIMULATION_TO_PARAMETER` dead dict
- Fix `APIs.py:189` — `simulate_alignment_benchmark()` kwargs mismatch
- Fix TPS fallback in `spatial_align_alter.py:410` — re-raise instead of returning random noise
- Replace 4 `except Exception` with specific exceptions

**Verification:** Import all core modules, run `01_clustering/smoke_test.py`.

---

## Subtask 3: P2 Cleanup

**Files:** 9 files across FEAST/src and FEAST_experiments

- Remove 9 unused imports
- Remove 3 `[DEBUG]` print statements
- Deduplicate 7 function pairs -> consolidate into shared internal helpers
- Remove `_minimal_internal_space` dead method

**Verification:** All public API symbols still accessible via `from FEAST.de_novo import *`.

---

## Subtask 4: Comment & Style Cleanup

**Files:** ~25 files

- Strip agent-narration comments (NEW/UPDATED headers, "We adjust...", "CRITICAL FIX", numbered Q-Q checklist)
- Strip trivial restatements (most in `simulator.py`, `generate_deconvolution.py`, `Beta_mixture_model.py`)
- Clean ornamental dividers in experiment scripts
- Remove `# placeholder` in `z_regularize.py:357`

**Verification:** `git diff --stat` shows substantial comment reduction, zero functional changes.

---

## Subtask 5: Over-Defensive Cleanup

**Files:** `builder.py`, `conditional.py`, `parameter_cloud.py`

- Remove ~50 redundant type casts on typed dataclass fields
- Remove redundant `.copy()` in `parameter_cloud.py:93`
- Replace global RNG save/restore with local `default_rng()` in `conditional.py`

**Verification:** Fixed-seed simulation produces identical output.

---

## Subtask 6: Experiment Import Cleanup

**Files:** `cloud_extraction.py`, `z_regularize.py`, `run_simulation.py`

- Remove `theta_to_stats` import (production code never calls it)
- Remove `calibrate_counts_to_regularized_means` import (manual loop used instead)
- Remove duplicate `AlterationConfig` import

**Verification:** Each file imports without error.

---

## Subtask 7: Full Verification

1. **Import smoke test** — import every public symbol from FEAST
2. **Quick simulation test** — `01_clustering/smoke_test.py`
3. **Experiment import check** — verify all experiment scripts parse
4. **Dead code grep** — confirm zero references to removed symbols

---

## Subtask 8: Commit & Merge

1. Commit all changes on `publish-prep` with structured message
2. Push to remote
3. Merge to `main`
4. Tag `v1.0.1`
