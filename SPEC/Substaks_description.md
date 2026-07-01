# FEAST Subtask Descriptions

Comprehensive descriptions of all 9 experiment subtasks (00–08) comprising the FEAST publication benchmark suite. Each section covers: motivation (why), method and API (how), theoretical justification (why it works), key parameters, and expected outputs.

---

## Subtask 00: Simulator Benchmark

### Why

Before claiming FEAST as a simulation platform, we must demonstrate that its generative engine actually produces synthetic data that is competitive with — and ideally surpasses — established external simulators. This subtask answers the question: **does `FEAST.FEAST_core.simulator.simulate_single_slice` match or outperform SRTsim, Splatter (Poisson-Gamma), and scCube on distributional fidelity, spatial structure preservation, and realistic novelty across diverse real-world spatial transcriptomics datasets?** It serves as a validation gate for the core simulation pipeline used by all downstream tasks (01–08).

### Method & API

The pipeline is driven by the single-entry-point function `simulate_single_slice`:

```python
from FEAST.FEAST_core.simulator import simulate_single_slice
```

Internally, `simulate_single_slice` orchestrates three stages:

1. **Parameter fitting** via `run_parameter_cloud_fitting` (defined in `FEAST.FEAST_core.simulator`):
   - Instantiates `GeneParameterSimulator` (from `FEAST.FEAST_core.parameter_cloud`).
   - For `parameter_mode="hungarian"` (generative): calls `GeneParameterSimulator.fit()` which fits per-gene marginal distributions (StudentT and Beta mixtures via `FEAST.modeling.StudentT_mixture_model.StudentTMixtureMarginalModeler` and `FEAST.modeling.Beta_mixture_model.BetaMixtureMarginalModeler`) and a vine copula over the uniform scores (`pyvinecopulib`). Then `build_gene_parameter_table()` performs Hungarian-assignment or copula-rank OT matching of each simulated gene profile to the closest real-gene target statistics (`mean`, `variance`, `zero_prop`).
   - For `parameter_mode="reference_stats"` (empirical): calls `fit_statistics_only()` which extracts mean/variance/zero_prop directly from the input data without copula modeling.
   - The fitted parameter table is converted to count-decoder parameters via `convert_params_for_new_simulator`.

2. **Spatial assignment** — two modes:
   - `spatial_mode="reference_rank"`: Uses the reference counts directly as the quantile input for decoding. Preserves within-gene rank ordering from the reference slice.
   - `spatial_mode="ot_spatial"`: Converts reference counts to per-gene rank quantiles via `midpoint_rank_normalize` (from `FEAST.de_novo.quantile_field`), then transports those quantiles from source to target spatial coordinates using `sinkhorn_transport` (entropic-regularized OT from `FEAST.de_novo._ot_transport`, via the `POT` library). Datasets exceeding 50,000 spots are handled by `_block_ot_transport` which partitions space into grid tiles with overlap to avoid O(n²) memory blowup.

3. **Count decoding** via `decode_counts_by_rank` (from `FEAST.FEAST_core.count_decoding`): given model parameters (NB/ZINB/Poisson theta per gene) and a rank-quantile field, reconstructs integer count matrices by mapping quantiles through the inverse CDF of the fitted count distribution, then clipping to `boundary_multiplier * reference_max`.

The benchmark evaluates two FEAST configurations (FEAST_Rank: `hungarian + reference_rank + hybrid`; FEAST_OT_Spatial: `hungarian + ot_spatial + hybrid`) against SRTsim, Splatter (two variants: full and simple), and scCube across 11 real reference datasets spanning Visium, MERFISH, OpenST, Stereo-seq, Slide-seq, and Xenium platforms. Quality is measured by 10 atomic metrics covering three categories: distribution fidelity (`log1p` mean/variance Pearson correlation, zero-fraction Jaccard, library-size Wasserstein), spatial structure preservation (spatial autocorrelation via Moran's I), and identity detection (flags for near-identical outputs).

### Why This Works

The theoretical justification rests on decomposing the simulation problem into three independent sub-problems:

- **Marginal modeling**: Each gene's expression distribution is modeled as a flexible mixture (StudentT for positive counts, Beta for zero-inflation), capturing arbitrary skew, multimodality, and zero-inflation without distributional assumptions. The StudentT is heavy-tailed, naturally accommodating outlier expression values common in UMI-based data.

- **Dependency modeling**: The vine copula captures the full rank-correlation structure across genes without assuming joint normality. This means correlated gene programs and co-expression patterns are reconstructed faithfully.

- **Rank-based decoding**: `decode_counts_by_rank` reconstructs counts by inverting the fitted count distribution CDF at positions given by a within-gene rank-quantile field. Because the rank field comes from the reference data (or is transported via OT), the simulated data inherits the spatial organization of the reference without needing to model spatial autocorrelation explicitly. This is a non-parametric approach to spatial pattern preservation.

The external simulators serve as baselines: SRTsim uses a simpler Poisson-Gamma framework, Splatter is designed for scRNA-seq (not spatial) and lacks explicit spatial modeling, and scCube uses a VAE-based generative model. The comparison tests whether FEAST's copula + marginal mixture + rank-decoding architecture yields superior fidelity.

### Key Parameters

| Parameter | Value | Justification |
|---|---|---|
| `parameter_mode` | `hungarian` | Uses the full copula + Hungarian assignment pipeline for the most realistic generative simulation |
| `spatial_mode` | `reference_rank` or `ot_spatial` | Tests both rank-preserving (fidelity-maximal) and OT-transported (novel-spatial-context) scenarios |
| `assignment_method` | `hybrid` | Hybrid OT cost (80% raw, 20% log-space) for gene-profile matching balances high- and low-expression gene fidelity |
| `clip_overshoot_factor` | `0.0` | No clipping; the `boundary_multiplier` of 1.1 alone ensures counts stay within realistic ranges |
| `boundary_multiplier` | `1.1` | Allows simulated counts up to 110% of reference max per gene |
| `use_heuristic_search` | `False` | Generative mode uses copula-rank OT, not heuristic screening |
| `random_seed` | `2026` | Fixed seed for reproducibility across all runs |

### Expected Output

1. **Simulated `.h5ad` files** (`outputs/simulation_saved/{FEAST_Rank, FEAST_OT_Spatial, srtsim_updated, splat, splatSimple, sccube}/*.h5ad`): One per dataset per simulator, containing the synthetic count matrix with matching spatial coordinates and metadata.
2. **`simulator_inventory.csv`** (`outputs/benchmarks/`): Manifest of all simulated outputs with simulator name, dataset, hourglass ID, and success/failure status.
3. **`simulator_quality_metrics.csv`** (`outputs/benchmarks/`): 10 per-sample metrics including log1p mean/variance correlation, zero-fraction Jaccard, library-size Wasserstein distance, spatial autocorrelation preservation, and identity flags.

---

## Subtask 01: Clustering Benchmark

### Why

Spatial clustering is the most common analytical workflow applied to spatial transcriptomics data, yet its robustness to systematic data-quality perturbations is poorly characterized. This subtask answers the question: **can three widely-used spatial clustering methods (GraphST, STAGATE+mclust, Leiden) recover known spatial domains from FEAST-simulated data when the underlying gene-expression marginals are systematically altered in mean, variance, and sparsity?** By using FEAST to generate controlled perturbations on real DLPFC slices with verified ground-truth cortical layers, this experiment establishes a causal sensitivity baseline: how much can expression statistics drift before clustering accuracy degrades, and which perturbation axes are most disruptive.

### Method & API

The simulation pipeline uses a **copula-preserving marginal alteration** strategy that guarantees the gene-gene dependency structure (vine copula) is held constant while per-gene marginal distributions are perturbed. This cleanly isolates the effect of marginal changes on downstream clustering.

**Core classes and functions used:**

```python
from FEAST.FEAST_core.parameter_cloud import GeneParameterSimulator, convert_params_for_new_simulator
from FEAST.FEAST_core.count_decoding import decode_counts_by_rank
from FEAST.modeling.marginal_alteration import AlterationConfig
from FEAST.FEAST_core.simulator import simulate_single_slice
```

**Pipeline flow (pre-fit fast path using `PrefitSimulator` class defined in `run_simulation.py`):**

1. **Fit once per slice**: `GeneParameterSimulator.fit(adata)` fits per-gene StudentT and Beta mixture marginals plus a vine copula to the raw counts. The `hybrid_alpha` of 0.2 weights the OT cost function 20% log-space and 80% raw-space during gene-profile assignment.

2. **Clone and alter per condition**: For each alteration condition (7 types, multiple levels each, totaling ~23 per slice), the fitted simulator is deep-copied and `build_gene_parameter_table(alteration_config=..., use_distributional_alteration=True)` is called. The key insight: **vine copulas are invariant under strictly monotonic marginal transformations**, so the copula fitted once is valid for all marginal perturbations. This avoids re-fitting the copula 27 times per slice, reducing runtime from hours to minutes.

3. **Marginal alteration (distribution-level, not post-hoc scalar)**:
   - `AlterationConfig` modifies the fitted mixture model parameters (theta) directly BEFORE sampling:
     - **Mean**: `μ' = α_μ · μ` via pure log10-location shift of StudentT component means. With `variance_coupling="fano"`, variance is also shifted by `α_μ`, maintaining the Fano-factor relationship observed in real UMI data.
     - **Variance**: `σ²' = α_v · σ²` (level shift) with `ρ_v` controlling component dispersion heterogeneity (`s_k' = ρ_v · s_k`). `ρ_v=1.0` preserves the relative spread of component means.
     - **Sparsity**: `logit(z') = logit(z) + δ_z`, a direct additive shift in logit space applied to Beta mixture component means. Component concentrations `S_k = α_k + β_k` are preserved, so the shift only changes detection probability, not the shape of the zero-inflation curve.
     - **Combined**: Simultaneous multi-axis perturbations for realistic scenarios (e.g., "noisy_tissue" with elevated variance and sparsity).

4. **Convert and decode**: `convert_params_for_new_simulator(table, n_spots=adata.n_obs)` maps the altered parameter table to count-model parameters (NB/ZINB/Poisson theta per gene). `decode_counts_by_rank(reference_matrix, model_params, ...)` reconstructs the count matrix by applying the inverse CDF of each gene's count distribution to the reference rank-quantile field, then clipping at `boundary_multiplier=1.1`.

5. **Clustering methods** applied to the simulated outputs:
   - **GraphST**: Self-supervised graph contrastive learning on spatial neighborhood graphs, followed by mclust clustering.
   - **STAGATE**: Graph attention auto-encoder with a spatial regularization term, followed by mclust on the latent representation.
   - **Leiden**: Standard PCA (50 PCs) + k-NN graph (k=15) + Leiden clustering at resolutions [0.2, 0.4, 0.6, 0.8, 1.0], with the best resolution selected by maximum NMI against ground truth.

6. **Evaluation metrics**: ARI, NMI, AMI, Homogeneity, Completeness, V-measure, CHAOS, PAS, computed per (slice, alteration, method) combination against the known DLPFC cortical layer labels.

### Why This Works

The theoretical justification has three layers:

- **Copula-marginal decoupling**: By Sklar's theorem, any multivariate distribution decomposes into marginals and a copula. The vine copula captures all rank-correlation structure (gene programs, co-expression networks). Monotonic marginal transformations leave the copula invariant. Therefore, altering marginal model parameters (theta) before sampling changes only the marginal distributions while perfectly preserving the gene-gene dependency architecture. This guarantees that observed differences in clustering performance are caused by marginal changes, not by confounded changes in gene co-expression patterns.

- **Distribution-level intervention**: Unlike naive post-hoc scalar multiplication (`X' ← α * X`), which distorts both the mean and the shape of the distribution (e.g., scaling a ZINB variable by a constant does not produce another ZINB variable), the `use_distributional_alteration=True` path modifies the fitted mixture model parameters directly. The altered parameters define a valid ZINB/NB/Poisson distribution whose mean (or variance, or zero proportion) equals the target value. The resulting count data is therefore a legitimate draw from a properly-defined count distribution, not an ad-hoc transformation.

- **Ground-truth spatial domains**: The DLPFC slices come with expert-annotated cortical layer labels (ground_truth), providing a gold-standard benchmark. Because FEAST preserves the spatial rank field from the reference, the simulated data retains the same spatial domain organization. If clustering accuracy degrades under a particular perturbation (e.g., severe sparsity), we can directly attribute the degradation to the perturbation rather than to simulation artifacts.

### Key Parameters

| Parameter | Value | Justification |
|---|---|---|
| `parameter_mode` | `hungarian` | Full generative pipeline with copula + Hungarian assignment |
| `spatial_mode` | `reference_rank` | Preserves within-gene spatial rank ordering from reference (no OT transport) |
| `use_distributional_alteration` | `True` | Alters mixture model parameters (theta) before sampling, not post-hoc scalar multiply |
| `assignment_method` | `hybrid` | Hybrid OT cost (20% log-space, 80% raw-space) for gene-profile matching |
| `hybrid_alpha` | `0.2` | Weight for log-space distance in hybrid OT cost |
| `boundary_multiplier` | `1.1` | Max count per gene capped at 110% of reference |
| `clip_overshoot_factor` | `0.0` | No additional clipping |
| `random_seed` | `2026` | Fixed seed for reproducibility across all 3 × ~23 = 69 simulations |
| Mean `variance_coupling` | `"fano"` | When mean is altered, variance auto-scales by the same fold-change |
| Variance `heterogeneity_scale` (`ρ_v`) | `1.0` | Preserves relative dispersion of StudentT component means |
| Alteration levels (mean) | `[0.5, 0.67, 0.8, 1.25, 1.5, 2.0]` | Covers expression suppression to strong overexpression |
| Alteration levels (variance) | `[0.5, 0.67, 0.8, 1.25, 1.5, 2.0]` | Covers underdispersion to overdispersion |
| Alteration levels (sparsity δ_z) | `[-1.0, -0.5, -0.25, +0.25, +0.5, +1.0]` | Covers reduced sparsity to increased dropout |
| DLPFC slices | `151508, 151670, 151676` | Three Visium slices with expert-annotated cortical layers |

### Expected Output

1. **Simulated `.h5ad` files** (`outputs/simulations/{slice_id}/{simulation_id}.h5ad`): 69 files (3 slices × 23 alterations including baseline).
2. **`simulation_manifest.csv`**: Registry of all 69 simulation jobs.
3. **Clustering outputs**: Predicted cluster labels from GraphST, STAGATE+mclust, and Leiden for every (slice, alteration) combination.
4. **`clustering_benchmark_results.csv`**: All 8 clustering metrics per (slice, alteration, method) triplet.

---

## Subtask 02: Alignment Benchmark

### Why

Spatial transcriptomics slides are often misaligned between adjacent tissue sections, rotated during imaging, or warped during sample handling. Before any multi-slice integration or 3D reconstruction can proceed, a spatial alignment step must recover the relative rotation and translation between slices. But alignment methods are typically validated on real data where the ground-truth transformation is unknown, so evaluation reduces to visual inspection or indirect biological plausibility checks.

This subtask answers: **Can FEAST produce realistic simulated spatial data with an exactly known rotation angle, so that alignment methods can be benchmarked against a gold-standard Procrustes recovery?** It tests whether popular alignment tools (Spateo's `morpho_align` and SPACEL) can recover the correct rotation from expression-altered simulated slices, and at what angle magnitudes they break down.

### Method & API

Two complementary pipelines exist, differing in abstraction level.

**Pipeline A — Unified high-level API** (`run_simulation_api.py`): Uses `FEAST(adata).simulate_alignment()` from `FEAST.FEAST_core.APIs`, which internally chains:
1. `FEAST_core.simulator.simulate_single_slice()` — fits a marginal-distribution model (Student-T mixture) to each gene of the input Visium data, then generates a realistic simulated AnnData with the same spatial topology, gene expression distributions, and count properties.
2. `alignment.alignment_simulator.simulate_alignment_rotation()` — applies a `RotationTransformer` (`FEAST.alignment.spatial_align_alter`) at the requested angle. Uses `data_type="sequencing"` with `filter_edge_spots=True` and `edge_margin_ratio=0.03` to crop edge artifacts that would leak the rotation angle through spatial bounds alone.
3. Returns a `(original_adata, transformed_adata)` tuple, preserving per-spot barcode identity so ground truth is exact.

**Pipeline B — Manual assembly** (`run_simulation.py`): Bypasses `FEAST()` and directly invokes:
- `FEAST_core.simulator.simulate_single_slice(adata, parameter_mode="hungarian", spatial_mode="reference_rank", alteration_config=alt_config, random_seed=seed)`
- `FEAST.alignment.spatial_align_alter.RotationTransformer(simulated).transform_imaging()` or `.transform_sequencing()`
This pipeline additionally tests expression **alterations** (baseline, mean-shift, variance-shift, sparsity-change) via `FEAST.modeling.marginal_alteration.AlterationConfig`, injected upstream of rotation.

**Evaluation** (`benchmark.py`): For each (alteration, method, angle) combination:
1. Loads `aligned_coordinates.csv` produced by the method and computes **coordinate-level metrics**: Procrustes-recovered rotation angle, mean/median/RMSE spatial error, Pearson-x and Pearson-y coordinate correlation.
2. Computes **unified nearest-neighbor metrics** (method-agnostic): NN spot-match accuracy, NN region accuracy, mean NN spatial distance, and per-spot gene expression correlation between matched reference-moving spot pairs.
3. For Spateo only: loads the `mapping_matrix.npy` probability matrix and computes morpho accuracy, precision, recall, F1, and bidirectional consistency.

Primary input: Visium slice 151675 (human DLPFC). Rotation angles tested: 1, 5, 10, 30, 45, 60 degrees.

### Why This Works

FEAST's simulation engine models each gene's expression as a Student-T mixture fit to the real data's marginal distribution, then samples new counts that are spatially arranged via optimal-transport matching to the observed spatial reference. When a rotation is applied after simulation, the per-spot barcode identities rotate with the coordinates, so the ground-truth correspondence between the reference and the rotated slice is exact — every spot `i` in the reference maps to spot `i` in the rotated slice.

This creates a **doubly-challenging** benchmark: alignment methods must recover the rotation despite (a) expression differences between the real reference and the simulated rotated slice, and (b) the fact that the simulation introduces gene-specific random variation on top of the rotation. The Procrustes rotation recovered from the aligned coordinates can be compared directly to the injected angle, giving a single-number accuracy score that is impossible to obtain from real data alone.

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `parameter_mode` | `"hungarian"` | Hungarian algorithm for one-to-one parameter-to-spot assignment |
| `spatial_mode` | `"reference_rank"` | Ranks parameters by spatial smoothness before assignment |
| `data_type` | `"sequencing"` (API) or `"imaging"` (manual) | sequencing mode discretizes onto a grid first |
| `filter_edge_spots` | `True` (API) | Removes outermost 3% of spots, preventing edge-boundary cues |
| `random_seed` | 2026 | Fixed for reproducibility |

### Key Findings

1. **Spateo dominates small angles (1–30°).** Across the 1–30° range, Spateo's `morpho_align` achieves ~96% nearest-neighbor spot accuracy. The Procrustes-recovered rotation is within 1° of ground truth.

2. **Spateo fails catastrophically at 45°.** At 45°, `morpho_align` converges to a wrong local minimum, recovering -166° instead of +45°. The optimization landscape has degenerate valleys at near-symmetric configurations.

3. **Spateo crashes at 60°.** At 60°, the rotated coordinates become collinear (only 34 spots survive, all sharing the same x-coordinate). Spateo's coordinate checker drops the zero-variance dimension, leaving 1D data that fails validation. Even SPACEL gets 0% accuracy — no 2D alignment method can solve rotation from collinear data.

4. **SPACEL is more robust but lower accuracy.** SPACEL completes all angles without crashing but its NN accuracy ceiling is lower (~70–80%). It trades precision for reliability.

5. **Both failures are reproducible.** Re-running Spateo at 45° produced byte-for-byte identical results (-166.2939° recovery, 0.0 NN accuracy), confirming method limitations rather than flukes.

### Expected Output

- **`simulation_manifest.csv`**: Maps every generated `.h5ad` file to its alteration ID, type, fold change, angle, and dimensions.
- **Per-angle `.h5ad` files**: `{alteration_id}_rotated_{angle}.h5ad` — one per alteration-angle combination.
- **`alignment_benchmark_results.csv`**: Per-job results including recovered rotation angle, NN accuracy, NN region accuracy, spatial error, Pearson coordinate correlation, gene expression correlation, and morpho metrics (Spateo-only).
- **`alignment_benchmark_summary.csv`**: Aggregated summary grouped by method and alteration type.

---

## Subtask 03: Deconvolution Benchmark

### Why

Spatial transcriptomics technologies measure expression at spots or pixels that contain multiple cells. Cell-type deconvolution methods estimate the proportion of each cell type at every spatial location, which is essential for understanding tissue organization. However, validating deconvolution accuracy is notoriously difficult because real spatial data lack ground-truth per-spot cell-type proportions — single-cell resolution is not available for the same tissue.

This subtask answers: **Can FEAST generate realistic pseudo-bulk spatial data from single-cell-resolution inputs, where the true per-spot cell-type proportions are known exactly, enabling rigorous benchmarking of deconvolution tools (Cell2location, RCTD)?** It tests whether these methods can recover the known proportions from lower-resolution aggregated data, and at what downsampling factors performance degrades.

### Method & API

The pipeline uses a single convenience function with a two-stage internal architecture.

**Entry point**: `FEAST.deconvolution.simulate_deconvolution_from_single_cells()` (from `FEAST.deconvolution.deconvolution_simulator`).

**Stage 1 — Single-cell simulation**: Internally calls `FEAST.FEAST_core.simulator.simulate_single_slice()` with the same marginal-distribution modeling engine used in Subtask 02. Takes a single-cell-resolution MERFISH AnnData with cell-type labels and generates a realistic simulated AnnData.

**Stage 2 — Spatial aggregation**: The simulated single-cell spots are downsampled into pseudo-bulk spots via hexagonal grid aggregation. Internal sub-steps (from `FEAST.deconvolution.generate_deconvolution`):
1. `create_low_resolution_grid()` — generates hexagonal grid points at the target resolution.
2. `filter_grid_to_tissue_shape()` — removes grid points outside the tissue boundary using alpha-shape with `alpha=0.01`.
3. `assign_original_spots_to_grid()` — assigns each single-cell spot to its nearest hexagonal grid point.
4. `aggregate_gene_expression()` — sums expression counts across all spots assigned to each grid point.
5. `calculate_cell_type_proportions_for_lowres()` — computes the exact proportion of each cell type among the assigned spots.

The result is a low-resolution AnnData whose `.obsm["cell_type_proportions"]` contains the **exact ground-truth proportions**, while `.X` contains summed pseudo-bulk counts that deconvolution methods must decompose.

**Benchmarked methods**: Cell2location (Bayesian hierarchical model) and RCTD (Robust Cell Type Decomposition via maximum likelihood).

**Evaluation** (`benchmark.py`): Jensen-Shannon Divergence, Pearson correlation (per-cell-type), Distance correlation, summed RMSE, per-cell-type RMSE, per (method, slice, resolution) combination.

**Input data**: MERFISH Zhuang-ABCA-1 mouse brain slices (IDs: 007, 050, 100), chosen for well-characterized cortical cell types and single-cell resolution.

**Downsampling factors**: 0.1 (10% of original resolution) and 0.25 (25% of original resolution).

### Why This Works

The two-stage design solves a fundamental tension in deconvolution benchmarking. If we simply aggregated real MERFISH spots into larger pseudo-bulk spots, we would know the exact proportions (by counting the single-cell labels per pseudo-spot), but the expression data would be trivially deterministic — no method is needed because the proportions are simply the observed cell counts divided by the total.

FEAST resolves this by first generating simulated expression through its mixture-model simulator (Stage 1), which re-samples counts while preserving the per-spot cell-type identity, then aggregating these simulated spots into pseudo-bulk measurements (Stage 2). The ground-truth proportions are computed from the known cell-type labels of the spots assigned to each pseudo-bulk location, while the expression is the sum of stochastically-simulated counts. This means **the expression is non-trivial to decompose** (it contains simulation noise and gene-gene covariation), yet **the true proportions are known with certainty**, creating a rigorous benchmark.

The choice of hexagonal grids (rather than square grids or k-means clusters) ensures isotropic spatial aggregation that does not introduce directional bias. The alpha-shape filtering (`alpha=0.01`) ensures pseudo-bulk spots respect tissue boundaries.

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `downsampling_factor` | 0.1, 0.25 | Tests resolution limits of each method |
| `grid_type` | `"hexagonal"` | Isotropic spatial coverage; avoids axis-aligned bias |
| `alpha` | 0.01 | Concave-hull tightness; avoids empty "spots" |
| `cell_type_key` | `"cell_type"` | Per-cell type labels for ground-truth generation |
| `random_seed` | 2026 | Fixed seed for reproducibility |
| `parameter_mode` | `"hungarian"` | Best one-to-one parameter-spot assignment quality |
| `spatial_mode` | `"reference_rank"` | Preserves spatial expression gradients |

### Expected Output

- **Per-slice, per-resolution `.h5ad` files**: Integer pseudo-bulk counts in `.X`, spatial coordinates in `.obsm["spatial"]`, and ground-truth proportions in `.obsm["cell_type_proportions"]`.
- **Per-slice, per-resolution truth CSVs**: Spot-by-cell-type matrix of exact proportions.
- **Deconvolution method output CSVs**: Per-method proportion estimates at each (slice, resolution) combination.
- **`deconvolution_benchmark_results.csv`**: Per-method, per-slice, per-resolution results with JSD, Pearson, distance correlation, RMSE.
- **`deconvolution_benchmark_summary.csv`**: Aggregated summary with best-performing method per metric.

---

## Subtask 04: 2D Conditional Transfer

### Why

Tests whether FEAST can reconstruct a target slice's full transcriptome from only its spatial geometry and cell-type labels, without access to the target's actual gene counts. This measures how much spatial information is recoverable through conditional generation: a reference model fitted on a source slice (expression + coordinates + labels) generates expression for a target slice using only a `SliceBlueprint` (coordinates + domain labels). The experiment quantifies the spatial fidelity gap between source-driven reconstruction and the held-out ground truth via mean correlation, variance correlation, and Moran's I correlation.

### Method & API

Uses the FEAST `de_novo` conditional generation pipeline rather than the monolithic `simulate_single_slice` to gain access to `assignment_randomness`, which is essential for breaking OT-driven spatial over-smoothing. The pipeline is:

1. `FEAST.de_novo.conditional.fit_reference(source_adata, label_key=..., config=ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0))` — fits per-class log-mean expression and per-gene rank-normalized quantile profiles on the source slice. Minimal QC filtering preserves all common genes.

2. `FEAST.de_novo.core.SliceBlueprint(coordinates=target_coords, domain_map=target_labels)` — encapsulates the target's spatial coordinates and cell-type domain map as a geometry-only specification.

3. `FEAST.de_novo.conditional.simulate_from_reference(model, blueprint, config=SimulationConfig(assignment_randomness=...), random_seed=...)` — generates target counts by (a) Sinkhorn-regularized OT transporting rank-normalized expression quantiles from source to target spots based on spatial proximity; (b) blending in random source samples at weight `assignment_randomness` to restore spatial structure; (c) decoding final counts via rank-matching to per-class reference distributions.

The experiment also supports a `mask_complete` mode: split a single slice spatially (e.g., at the x-axis median), fit on one half, and reconstruct the masked half from geometry alone, testing intra-slice generalization.

### Why This Works

Optimal transport (Sinkhorn) computes a coupling that maps each target spot to a weighted combination of source spots based on normalized spatial distances and label agreement. Transporting rank-normalized expression through this coupling averages expression profiles across nearby source spots, producing a spatially over-smoothed output — Moran's I correlation drops sharply because the transport plan acts as a spatial low-pass filter. The `assignment_randomness` parameter in `_transport_reference_latent_scores()` counteracts this: it blends random source-sample latent scores into the transported scores at weight `alpha` (0 to 1), computed as `(1 - alpha) * transported + alpha * random_source_sample`. Without it (AR=0.0), the generated expression loses spatial structure and `moran_corr` crashes to ~0.09. With AR=0.3, the injected stochastic variation restores tissue-level spatial patterning, recovering `moran_corr` to ~0.95 per the SPEC. The hybrid assignment also preserves per-class mean expression accuracy (mean_corr near 1.0) because the class-conditional quantile decoder remains intact.

### Key Parameters

| Parameter | Values / Details |
|---|---|
| `assignment_randomness` sweep | 0.0, 0.1, 0.2, 0.3, 0.5 |
| `ReferenceFitConfig` | min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0 |
| Number of common genes | 17,924 (full intersection of DLPFC slices 151675 and 151676) |
| Direction pairs | 151675 → 151676, 151676 → 151675 (bidirectional cross-validation) |
| Dataset | spatialLIBD DLPFC Visium (10x Genomics, human dorsolateral prefrontal cortex) |
| Annotation key | `ground_truth` (layer-level cortical labels) |
| OT parameters | epsilon=0.05, sinkhorn_iter=200, unbalanced=True, reg_m=5.0 |
| Modes | `cross_slice` (source→target) and `mask_complete` (split-half intra-slice) |

### Expected Output

Per direction directory (e.g., `151675_to_151676/`):
- `generated.h5ad` — full generated AnnData with counts, spatial coords, and metadata
- `per_gene_metrics.csv` — per-gene Pearson r, Spearman rho, generated/target mean, variance, zero proportion, and Moran's I
- `panel_summary.json` — aggregate `mean_corr`, `var_corr`, `moran_corr`, `zero_ks`, `median_gene_pearson`, `median_gene_spearman`
- `layer_mean_matrix_generated.csv` / `layer_mean_matrix_target.csv` — per-label mean expression matrices
- `layer_zero_matrix_generated.csv` / `layer_zero_matrix_target.csv` — per-label zero-proportion matrices
- `moran_metrics.csv` — per-gene generated vs. target Moran's I comparison

Top-level outputs:
- `summary.csv` — one row per direction with all aggregate metrics
- `metadata.json` — full experimental configuration, git commit, timestamps, input files
- `gene_panel.json` — gene selection metadata and full gene list

---

## Subtask 05: 3D Stack Reconstruction

### Why

Tests FEAST's ability to reconstruct held-out tissue slices in a 3D volumetric stack from only flanking reference slices, validating the full pipeline: (1) optimal transport spatial transfer from two flanking references, (2) bilateral XY+Z spot-level smoothing, and (3) z-regularization via analytically-solved quadratic penalties. This experiment answers whether FEAST's conditional generation can faithfully interpolate intermediate z-levels in dense 3D tissues, a critical capability for volumetric spatial transcriptomics where not all sections are experimentally assayed. Three density levels (hold out every 3rd, 5th, or 10th slice) test robustness as the reference-target z-gap widens.

### Method & API

Operates on the Zhuang-ABCA-1 MERFISH dataset (147 coronal slices, ~1122 genes, ~10k spots each). The pipeline proceeds in five stages:

**Stage 1 — Target assignment.** For each density gap G in {3, 5, 10}, every G-th slice starting from index `radius+1` is designated a target, with its nearest lower and upper available slices as flanking references. Labels missing from both flanking references are supplemented from the nearest available support slice by z-distance.

**Stage 2 — Generation via `simulate_single_slice`.** Both flanking references are concatenated into a single reference AnnData. `FEAST.FEAST_core.simulator.simulate_single_slice(adata=reference_combined, annotation_key=label_key, parameter_mode='hungarian', spatial_mode='ot_spatial', target_adata=target, assignment_method='hybrid', random_seed=...)` generates the target slice.

**Stage 3 — Spot-level cross-z bilateral smoothing (optional).** `FEAST.de_novo.z_spot_smooth.smooth_cross_z_spots(z_slices, k_xy=10, k_z=5, blend_lambda=0.3)` applies a bilateral filter across XY neighbors and adjacent z-slices.

**Stage 4 — Z-regularization.** For each (class, gene) pair, a quadratic-penalty system is assembled across all z-nodes:

- `FEAST.de_novo.z_regularize.z_penalty_matrix(z_values, lambda_1, lambda_2)` — constructs the (N_nodes × N_nodes) penalty matrix encoding first-derivative and second-derivative quadratic costs on adjacent z-steps.
- `FEAST.de_novo.z_regularize.class_anchor_weight(n_spots, multiplier)` — computes per-node diagonal weights proportional to spot counts. Small classes are excluded to avoid noisy anchors.
- `FEAST.de_novo.z_regularize.regularize_mean_profiles(y_matrix, node_weights, z_values, lambda_1, lambda_2, ridge)` — solves the linear system `(P + diag(weights + ridge)) * solved_log = weights * y` via Cholesky decomposition.
- `FEAST.de_novo.z_regularize.calibrate_counts_to_regularized_means(counts, labels, gene_names, target_regularized)` — adjusts spot-level counts to match regularized class means.

**Stage 5 — Evaluation.** `FEAST.de_novo.z_regularize.compute_z_coherence(records)` computes per-(class, gene) correlation between expression profiles at adjacent z-levels. A real-data split-half baseline provides the gold-standard upper bound.

### Why This Works

OT transport from two flanking slices provides an initial estimate weighted by z-distance, but each target is generated independently, leading to inter-slice discontinuities. The z-regularization analytically solves a quadratic optimization that enforces two constraints simultaneously: (1) first-derivative penalty prevents abrupt jumps between adjacent z-levels; (2) second-derivative penalty prevents oscillatory profiles. The combined system `(P + W) * x = W * y` has a unique closed-form solution because P is positive semi-definite and W adds positive definiteness. Class-anchor weights proportional to spot counts prevent rare cell types from over-constraining the solution.

### Key Parameters

| Parameter | Values / Details |
|---|---|
| Density gaps | 3 (dense, 49 targets), 5 (medium, 29 targets), 10 (sparse, 15 targets) |
| Z-regularization lambdas | lambda_1=0.01 (first-derivative), lambda_2=1e-5 (second-derivative), ridge=1e-8 |
| Anchor weights | reference_beta=1.0, target_weight=1.0, min_class_spots=2 |
| Spot smoothing | k_xy=10, k_z=5, blend_lambda=0.3 |
| Dataset | Zhuang-ABCA-1 MERFISH, 147 coronal slices, 1122-gene panel |

### Expected Output

- **cross_density_summary.csv** — one row per density with all aggregate metrics
- Per density: `summary.csv`, `z_coherence_metrics.csv`, `real_z_coherence_split_half.csv` (gold standard)
- Per target: `generated.h5ad`, `per_gene_metrics.csv`, `per_class_metrics.csv`, `moran_metrics.csv`, `panel_summary.json`
- Z-regularized variants when enabled

---

## Subtask 06: 3D Transfer to DevCCF

### Why

This subtask transfers 2D MERFISH reference gene expression profiles (from the GSE269617 mouse brain dataset, 9 broad anatomical regions) onto the 3D DevCCF (Developmental Common Coordinate Framework) anatomical atlas. The DevCCF is a canonical 3D reference atlas of the developing mouse brain, but it contains only anatomical region labels — no gene expression. By generating synthetic MERFISH-like expression at every coronal cross-section of the DevCCF volume, this subtask produces a full 3D spatial transcriptomics resource. The result is a 3D stack of expression profiles that faithfully reproduces the per-region expression signatures observed in the real MERFISH data, projected onto the standardized DevCCF coordinate system. This enables 3D visualization, cross-age comparison, and serves as a template for integrating other spatial transcriptomics modalities.

### Method & API

The pipeline has three stages:

**Stage 1 — Blueprint extraction** (`extract_devccf_blueprints.py`): Reads DevCCF broad-region annotation NIfTI volumes, parses NIfTI-1 headers, and extracts 2D coronal cross-sections at each voxel level containing non-background tissue. Voxel indices are converted to world coordinates (mm) via the sform affine matrix. A region schema TSV maps integer region IDs to string labels. Each z-level blueprint is serialized as JSON with x, y world coordinates and region label arrays.

**Stage 2 — Conditional generation** (`transfer_to_devccf.py`):
- `FEAST.de_novo.conditional.fit_reference()` — fits a `SimulationReference` model from all loaded GSE269617 reference slices.
- `FEAST.de_novo.core.SliceBlueprint()` — constructs geometry+labels specification per z-level from DevCCF blueprints.
- `FEAST.de_novo.conditional.simulate_from_reference()` — generates synthetic gene expression per z-level by sampling from the reference-conditioned model given the blueprint geometry.
- `FEAST.de_novo.core.assign_generated_coordinates()` — attaches 3D spatial coordinates to each generated AnnData (`coordinate_system="devccf_world"`).

**Stage 3 — Post-hoc z-regularization** (`z_regularize.py`):
- `FEAST.de_novo.z_regularize.regularize_mean_profiles()` — solves quadratic-penalty optimization for smooth z-profiles.
- `FEAST.de_novo.z_regularize.class_anchor_weight()` — weights per-z nodes by spot count.
- `FEAST.de_novo.z_regularize.compute_z_coherence()` — evaluates adjacent-z expression correlation.

### Why This Works

FEAST's reference-conditioned generation decomposes expression patterns into two components: a per-region marginal distribution (the parameter cloud in theta-space) and a spatial quantile field (Q) that encodes intra-region spatial structure. When transferring to DevCCF, the blueprint supplies only the target geometry and region labels. `simulate_from_reference()` uses the learned reference model to synthesize expression values that respect both the marginal per-region statistics and the spatial correlation structure of the reference data. Since DevCCF regions correspond to the same anatomical structures as the reference, the same biological expression programs apply. The z-regularization step enforces smoothness across the third dimension, producing a 3D-coherent volume.

### Key Parameters

| Parameter | Description |
|---|---|
| `ReferenceFitConfig(min_gene_spots, min_gene_mean, max_gene_zero_prop)` | Gene filtering during reference fitting |
| `SimulationConfig(coordinate_scale, verbose)` | Simulation behavior |
| `--sigma` / `--lambda1` / `--lambda2` | z-regularization penalty strengths |
| `--max-reference-spots` | Subsample reference slices for memory control |
| `--min-spots` | Minimum spots per label per z-level |

### Expected Output

- Blueprint JSON file: per-z entries with x, y, region arrays, z_world, voxel_j, n_spots
- Per-z-level generated h5ad files: AnnData with generated counts, 3D spatial coordinates, `obs["z"]`
- `manifest.csv`: mapping z_world, filename, n_spots, z_idx
- Post-z-regularization: `regularized_z*.h5ad` files
- `z_coherence_metrics.csv` and `z_coherence_summary.csv`

---

## Subtask 07: Visualization

### Why

This subtask generates all publication-quality figures for the FEAST manuscript. It is a pure visualization layer with zero FEAST Python imports: it reads pre-computed h5ad and CSV artifacts produced by all other subtasks (01 clustering, 02 alignment, 04 2D conditional transfer, 06 simulator benchmark, 06 3D transfer) and renders them as polished matplotlib figures. This clean separation ensures figure reproducibility independent of the FEAST computation pipeline and makes it straightforward to regenerate all panels after parameter adjustments without re-running expensive simulations.

### Method

Zero FEAST imports. All scripts use `matplotlib` (Agg backend for headless rendering), `seaborn`, `scanpy` (only for reading `.h5ad` spatial data), `pandas`, and `numpy`. Each script is self-contained with hardcoded sample/gene selections and file paths.

| Script | Figure Description |
|---|---|
| `visualization_single_simulator.py` | 2×4 multi-panel boxplot comparing 6 simulation methods across 7 quality metrics |
| `visualization_reference_vs_feast.py` | Side-by-side spatial gene expression overlays for 6 platforms (Visium, MERFISH, OpenST, Slide-seq, Stereo-seq, Xenium) |
| `visualization_clustering_robustness_curve.py` | 4×4 grid: robustness curves (ARI, NMI, PAS, CHAOS) for STAGATE and GraphST across mean/variance/sparsity alterations |
| `visualization_clustering_spatial_panel.py` | Spatial domain assignment panels at varying alteration fold changes |
| `visualization_alignment.py` | 2×3 multi-metric panel: alignment performance vs. rotation angle for Spateo and Spacel |
| `prepare_alignment_metrics.py` | Helper: transforms benchmark CSV into long-format metrics CSV for plotting |

### Expected Output

All figures written to `/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/` organized by analysis type: `simulator_benchmark/`, `slice_panel_compare/`, `clustering/`, `alignment/`, plus directories for 2d_transfer, 3d_transfer, stack, and deconvolution.

---

## Subtask 08: Batch Effect Removal

### Why

This subtask validates FEAST's ability to simulate and assess batch effects in spatial transcriptomics data. Batch effects — systematic technical variation between experimental batches — are pervasive in spatial transcriptomics and can confound biological interpretation. FEAST models expression in a parameter space (theta: log_mu, log_omega, logit_pi0) where batch effects manifest as structured deformations (diagonal affine transformations). This subtask has two halves: (1) `effect_simulation` demonstrates that FEAST can introduce controllable, interpretable batch effects by deforming the theta parameter cloud while preserving spatial structure, and (2) `effect_verification` demonstrates that FEAST can detect, characterize, and quantify batch effects between real biological slices by fitting affine deformation models and computing distribution distance metrics.

### Method & API

**effect_simulation/** — Controllable batch effect generation:

- `batch_simulator.py`: Defines two empirical batch deformation signatures derived from DLPFC slices 151673 (clean) and 151508 (noisy):
  - `DLPFC_SHIFT_ONLY`: Pure translation in theta-space (D = [1, 1, 1], b = [-0.819, -0.037, +0.851]).
  - `DLPFC_DIAGONAL_AFFINE`: Full diagonal affine (D = [0.951, 0.507, 0.922], b = [-0.967, +0.011, +1.082]).
  - `simulate_batch_ladder()`: Calls `FEAST.FEAST_core.simulator.simulate_batch_effect()` for each alpha in [0.0, 0.25, ..., 1.5]. The alpha parameter interpolates between no deformation (alpha=0, identity) and full empirical deformation (alpha=1), with extrapolation beyond.
  - The `simulate_batch_effect()` API applies: `theta_batch = theta_ref * ((1-alpha) + alpha*D) + alpha*b`, then decodes counts from the deformed theta values while preserving the spatial quantile field Q.
- `evaluate_simulation.py`: Validates the batch ladder by checking monotonicity, theta formula fidelity, centroid shift proportionality, spatial preservation, and Moran's I stability.

**effect_verification/** — Batch effect detection and characterization:

- `cloud_extraction.py`: Extracts theta parameter clouds from AnnData slices using `FEAST.FEAST_core.theta_transform.stats_to_theta()`.
- `metrics.py`: `compute_centroids()` (per-slice means and batch shifts), `compute_covariances()` (3×3 cov matrices), `pairwise_distances()` (MMD, Wasserstein, Riemannian distance), `slice_qc_metrics()`.
- `affine_model.py`: Fits the diagonal affine deformation model `theta_s[:, k] = d_k * theta_ref[:, k] + b_k` via OLS. Reports D, b, R², residual variance.
- `domain_analysis.py`: Separates biological from technical variation by splitting slices by spatial domain.
- `run_assessment.py`: 8-step pipeline orchestrating all modules. Outputs JSON, CSV, and NPZ files.

### Why This Works

A diagonal affine deformation in theta-space (D diag + b) is the natural parameterization of a batch effect in FEAST's generative model. The three theta dimensions have clear interpretations: log_mu controls overall expression level (batch effects often alter detection sensitivity), log_omega controls overdispersion relative to Poisson (batch effects alter technical noise), and logit_pi0 controls zero-inflation (batch effects alter dropout rates). A diagonal affine transformation independently scales and shifts each of these dimensions, which is both interpretable and flexible enough to capture realistic batch effects. By decoupling P (marginal distribution deformation) from Q (spatial structure), the simulation preserves the spatial organization of the tissue while cleanly altering expression statistics.

### Key Parameters

| Parameter | Description |
|---|---|
| `DLPFC_SHIFT_ONLY` | D = [1,1,1], b = [-0.819, -0.037, +0.851]; pure translation |
| `DLPFC_DIAGONAL_AFFINE` | D = [0.951, 0.507, 0.922], b = [-0.967, +0.011, +1.082]; full diagonal affine |
| `alpha` sweep | 0.0 to 1.5 in 0.25 increments |
| `distance_metrics` | centroid_euclidean, covariance_frobenius, mmd_rbf, wasserstein, riemannian |

### Expected Output

**effect_simulation:**
- Per-alpha h5ad files with deformed counts, `uns["batch_alpha"]`, `uns["batch_deformation"]`
- `manifest.csv`: per-slice quality metrics
- `evaluation_report.json`: pass/fail status for all validation checks

**effect_verification:**
- `qc_summary.csv`, `centroids.json`, `covariances.npz`, pairwise distance matrices
- `affine_params.csv`: fitted D and b, R², residual variance
- `domain_shifts.csv`, `domain_variance_decomposition.csv`

**Test suite:** 89 tests covering cloud extraction, metrics computation, affine model fitting, domain analysis, FEAST core additions, and full integration — all passing.
