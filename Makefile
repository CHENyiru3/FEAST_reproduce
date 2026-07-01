# FEAST Reproduction Makefile
# ============================================================================
# Each subtask is a self-contained numbered directory.
# Usage:
#   make all                     # Run everything
#   make subtask_01              # 2D conditional transfer
#   make subtask_02              # 3D stack reconstruction
#   make subtask_03              # Clustering benchmark
#   make subtask_04              # Alignment benchmark
#   make subtask_05              # Deconvolution benchmark
#   make clean                   # Remove simulation outputs
#   make distclean               # Remove everything except scripts + results
# ============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

FEAST_DATA_ROOT ?= $$(pwd)/data
FEAST_ENV := /maiziezhou_lab2/yiru/envs/feast-py311-conda
STAGATE_ENV := /maiziezhou_lab2/yiru/envs/STAGATE
GRAPHST_ENV := /maiziezhou_lab2/yiru/envs/GraphST
SPACEL_ENV := /maiziezhou_lab2/yiru/envs/SPACEL
CELL2LOC_ENV := /maiziezhou_lab2/yiru/envs/cell2loc_env

PYTHONPATH := $$(pwd)/../FEAST/src:$$PYTHONPATH
export PYTHONPATH

SEED ?= 2026
MPLCONFIGDIR ?= /tmp/matplotlib
export MPLCONFIGDIR

# ============================================================================
# Top-level
# ============================================================================

.PHONY: all
all: subtask_01 subtask_02 subtask_03 subtask_04 subtask_05 subtask_06

# ============================================================================
# Subtask 01 — 2D Conditional Transfer
# ============================================================================

.PHONY: subtask_01
subtask_01: subtask_01_dlpfc subtask_01_merfish

.PHONY: subtask_01_dlpfc
subtask_01_dlpfc:
	conda run -p $(FEAST_ENV) python 01_2d_conditional_transfer/run.py \
		--mode cross_slice \
		--data-dir $(FEAST_DATA_ROOT)/spatialLIBD_DLPFC_Visium/h5ad \
		--annotation-key ground_truth \
		--directions 151675:151676 151676:151675 \
		--n-top-genes 3000 \
		--seed $(SEED) \
		--output-dir outputs/01_2d_conditional_transfer/dlpfc/cross_3000 \
		--overwrite
	conda run -p $(FEAST_ENV) python 01_2d_conditional_transfer/run.py \
		--mode mask_complete \
		--data-dir $(FEAST_DATA_ROOT)/spatialLIBD_DLPFC_Visium/h5ad \
		--annotation-key ground_truth \
		--directions 151675 151676 \
		--n-top-genes 3000 \
		--seed $(SEED) \
		--output-dir outputs/01_2d_conditional_transfer/dlpfc/mask_3000 \
		--overwrite

.PHONY: subtask_01_merfish
subtask_01_merfish:
	conda run -p $(FEAST_ENV) python 01_2d_conditional_transfer/run.py \
		--mode cross_slice \
		--data-dir $(FEAST_DATA_ROOT)/Allen_Zhuang_ABCA_1/h5ad \
		--annotation-key class \
		--directions Zhuang-ABCA-1.006:Zhuang-ABCA-1.007 Zhuang-ABCA-1.007:Zhuang-ABCA-1.006 \
		--seed $(SEED) \
		--output-dir outputs/01_2d_conditional_transfer/merfish/cross_full \
		--overwrite
	conda run -p $(FEAST_ENV) python 01_2d_conditional_transfer/run.py \
		--mode mask_complete \
		--data-dir $(FEAST_DATA_ROOT)/Allen_Zhuang_ABCA_1/h5ad \
		--annotation-key class \
		--directions Zhuang-ABCA-1.006 Zhuang-ABCA-1.007 \
		--seed $(SEED) \
		--output-dir outputs/01_2d_conditional_transfer/merfish/mask_full \
		--overwrite

# ============================================================================
# Subtask 02 — 3D Stack
# ============================================================================

.PHONY: subtask_02
subtask_02:
	conda run -p $(FEAST_ENV) python 02_3d_stack/run.py \
		--data-dir $(FEAST_DATA_ROOT)/Allen_Zhuang_ABCA_1/h5ad \
		--densities 3 5 10 \
		--label-key class \
		--seed $(SEED) \
		--output-dir outputs/02_3d_stack

# ============================================================================
# Subtask 03 — Clustering Benchmark
# ============================================================================

.PHONY: subtask_03
subtask_03: subtask_03_sim subtask_03_hvg
	bash 03_clustering/run_pipeline.sh
	conda run -p $(FEAST_ENV) python 03_clustering/benchmark.py \
		--input-dir outputs/03_clustering/methods \
		--simulation-manifest outputs/03_clustering/simulations/simulation_manifest.csv \
		--output outputs/03_clustering/clustering_benchmark_results.csv

.PHONY: subtask_03_sim
subtask_03_sim:
	conda run -p $(FEAST_ENV) python 03_clustering/run_simulation.py \
		--input-dir $(FEAST_DATA_ROOT)/spatialLIBD_DLPFC_Visium/h5ad \
		--slices 151508,151670,151676 \
		--output-dir outputs/03_clustering/simulations \
		--parameter-mode reference_stats \
		--seed $(SEED)

.PHONY: subtask_03_hvg
subtask_03_hvg: subtask_03_sim
	conda run -p $(FEAST_ENV) python 03_clustering/run_hvg.py \
		--simulation-manifest outputs/03_clustering/simulations/simulation_manifest.csv \
		--output-dir outputs/03_clustering/hvg_inputs \
		--n-top-genes 3000

# ============================================================================
# Subtask 03.1 — Clustering Benchmark (hungarian x ot_spatial)
# ============================================================================
# Simulation only target. The full pipeline (hvg + methods + benchmark)
# can be run by adjusting the output-dir in run_pipeline.sh and benchmark.py
# to point at outputs/03_clustering_ot/.

.PHONY: subtask_03_1_sim
subtask_03_1_sim:
	conda run -p $(FEAST_ENV) python 03_clustering/run_simulation_ot.py \
		--input-dir $(FEAST_DATA_ROOT)/spatialLIBD_DLPFC_Visium/h5ad \
		--slices 151508,151670,151676 \
		--output-dir outputs/03_clustering_ot/simulations \
		--parameter-mode hungarian \
		--spatial-mode ot_spatial \
		--seed $(SEED)

# ============================================================================
# Subtask 04 — Alignment Benchmark
# ============================================================================

.PHONY: subtask_04
subtask_04: subtask_04_sim
	bash 04_alignment/run_methods.sh
	conda run -p $(FEAST_ENV) python 04_alignment/benchmark.py \
		--simulation-manifest outputs/04_alignment/simulation_manifest.csv \
		--methods-dir outputs/04_alignment/methods \
		--reference outputs/04_alignment/reference.h5ad \
		--output outputs/04_alignment/alignment_benchmark_results.csv

.PHONY: subtask_04_sim
subtask_04_sim:
	conda run -p $(FEAST_ENV) python 04_alignment/run_simulation.py \
		--input $(FEAST_DATA_ROOT)/spatialLIBD_DLPFC_Visium/h5ad/151675.h5ad \
		--seed $(SEED) \
		--output-dir outputs/04_alignment

# ============================================================================
# Subtask 05 — Deconvolution Benchmark
# ============================================================================

.PHONY: subtask_05
subtask_05: subtask_05_sim
	bash 05_deconvolution/run_methods.sh
	conda run -p $(FEAST_ENV) python 05_deconvolution/benchmark.py \
		--truth-dir outputs/05_deconvolution/ground_truth \
		--prediction-dirs outputs/05_deconvolution/cell2location \
		--slices 007 050 100 \
		--resolutions 0.1 0.25 \
		--output outputs/05_deconvolution/deconvolution_benchmark_results.csv

.PHONY: subtask_05_sim
subtask_05_sim:
	conda run -p $(FEAST_ENV) python 05_deconvolution/run_deconvolution.py \
		--input-dir $(FEAST_DATA_ROOT)/Allen_Zhuang_ABCA_1/h5ad \
		--seed $(SEED) \
		--output-dir outputs/05_deconvolution

# ============================================================================
# Subtask 06 — Simulator Quality Benchmark
# ============================================================================

.PHONY: subtask_06
subtask_06:
	bash 06_simulator_benchmark/run.sh

# ============================================================================
# Utilities
# ============================================================================

.PHONY: clean
clean:
	rm -rf outputs/

.PHONY: distclean
distclean:
	rm -rf outputs/

.PHONY: list
list:
	@echo "Available targets:"
	@echo "  all           — Run everything"
	@echo "  subtask_01    — 2D conditional transfer (DLPFC + MERFISH)"
	@echo "  subtask_02    — 3D semi-reference stack reconstruction"
	@echo "  subtask_03    — Clustering benchmark"
	@echo "  subtask_04    — Alignment benchmark"
	@echo "  subtask_05    — Deconvolution benchmark"
	@echo "  subtask_06    — Simulator quality benchmark"
	@echo "  clean         — Remove simulation outputs"
