# Subtask: Simulator Quality Benchmark

Compares FEAST (Rank decode, OT decode) against external simulators (Splatter, SRTsim)
on single-slice simulation quality. Evaluates across 8 datasets using three metric
categories: distribution fidelity, spatial structure preservation, and structured novelty.

## Simulators compared

| Simulator | Decode method | Description |
|-----------|--------------|-------------|
| FEAST_Rank | rank + reference_rank calibration | Rank-based decode with reference calibration |
| FEAST_OT | quantile (hybrid_ot) | Default OT-based transport + quantile decode |
| Splatter | — | Bioconductor Splatter (Poisson-Gamma model) |
| SRTsim | — | SRTsim spatial simulator |

## Datasets (8)

DLPFC Visium, MERFISH (006, 007), OpenST, Stereo-seq, Slide-seq, Xenium

## Pipeline

```
prepare_inputs → run_feast → run_external → build_inventory → build_metrics → plot
```

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# 1. Prepare simulator inputs
python scripts/prepare_simulator_inputs.py \
  --data-root $FEAST_DATA_ROOT \
  --output-dir outputs/sim_inputs

# 2. Run FEAST simulators
bash scripts/run_feast_simulator_benchmark.sh

# 3. Run external simulators
bash scripts/external_simulators/run_external_simulators.sh

# 4. Build inventory
python scripts/build_simulator_inventory.py \
  --sim-dir outputs/simulations \
  --exper-data $FEAST_DATA_ROOT \
  --output outputs/benchmarks/simulator_inventory.csv

# 5. Build quality metrics
python scripts/build_simulator_quality_metrics.py \
  --inventory outputs/benchmarks/simulator_inventory_available.csv \
  --exper-data $FEAST_DATA_ROOT \
  --metrics outputs/benchmarks/simulator_quality_metrics.csv \
  --summary outputs/benchmarks/simulator_quality_summary.csv

# Or run everything at once
bash run.sh
```

## Results

| Simulator | Composite Score | Zero Pres. | Struct. Score | Novelty | Identity Flags |
|-----------|----------------|------------|---------------|---------|----------------|
| FEAST_Rank | **0.816** | 0.941 | 0.709 | **0.893** | 0 |
| Splatter | 0.640 | 0.858 | 0.426 | 0.501 | 0 |
| FEAST_OT | 0.556 | 0.729 | 0.580 | 0.068 | 0 |
| SRTsim | 0.539 | **0.984** | **0.965** | 0.000 | 9/10 |

**Key finding**: FEAST_Rank leads on composite score. SRTsim excels at preservation
metrics but 9/10 outputs are flagged as near-identity (copies input too faithfully).

## Tests

```bash
pytest tests/ -v
```

Covers: metric identity, spatial structure, distribution fidelity, output contract,
zero structure.

## Environments

| Stage | Conda Env |
|-------|-----------|
| FEAST simulation, metrics, plotting | `feast-py311-conda` |
| Splatter | R with Splatter Bioconductor package |
| SRTsim | R with SRTsim package |
| scCube | `scCube` (Python) |

## Known Issue

squidpy ≥1.4 breaks on `anndata.io` import with anndata ≥0.11. The `build_simulator_quality_metrics.py`
script imports squidpy for spatial autocorrelation (Moran's I). Fix: pin `squidpy==1.3.0` or
patch the import to use scanpy's Moran's I directly.
