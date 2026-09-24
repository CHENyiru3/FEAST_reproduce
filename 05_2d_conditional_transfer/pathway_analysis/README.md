# Study 05 layer/class pathway recovery

This analysis compares real held-out expression, FEAST, and ten existing
label-conditioned whole-spot resampling draws. It covers all 13 primary tasks
at assignment randomness 0.3, retaining dataset, transfer direction, mode,
donor stratum, and layer/class identity.

## Run

Use the existing environment with the project-local GSEApy 1.3.1 installation.
The shared environment is unchanged. To recreate the local dependency if needed:

```bash
/maiziezhou_lab2/yiru/miniconda3/envs/sctm-pub-20260718/bin/python -m pip install --no-deps \
  --target 05_2d_conditional_transfer/pathway_analysis/.packages/gseapy_1_3_1 gseapy==1.3.1
```

Then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
  /maiziezhou_lab2/yiru/miniconda3/envs/sctm-pub-20260718/bin/python \
  05_2d_conditional_transfer/pathway_analysis/run.py

/maiziezhou_lab2/yiru/miniconda3/envs/sctm-pub-20260718/bin/python \
  05_2d_conditional_transfer/pathway_analysis/summarize.py
```

Run these from the FEAST_reproduce root. `run.py` continues the same analysis
using completed task rankings and enrichment tables. The rankings remain in
`results/`; the researcher-approved adaptive multilevel enrichment and final
comparison are in `results_multilevel_gseapy131/`. The initial classical-permutation
outputs in `results/` remain separate and are not mixed into the final report.
The provisional GSEApy 1.3.0 adaptive results are preserved under `results_multilevel/`
and must not be used for final claims; see their `VALIDATION_NOTE.md`. Version 1.3.1
fixes an upstream p-value bug and correctly marks insufficient same-sign null
support as unavailable. Do not reuse an output directory for changed scientific
settings or inputs.

## Methods

- Targets are exactly the generated spot IDs, coordinates, and conditional
  labels. Half-held-out comparisons include only the right/held-out half.
  The existing source-label support rule is retained; unsupported labels are
  absent from the evaluation, not filled in.
- Gene ranking is each supported label versus all other supported labels,
  computed separately for each expression source. Counts are normalized to
  10,000 per spot and log1p transformed. Scanpy Wilcoxon signed z scores use
  tie correction, `use_raw=False`, and every gene in the fixed panel.
- No additional spot filtering, HVG selection, regression, reclustering, or
  DEG p-value filtering is performed. The 17,391-gene DLPFC panel and
  1,122-gene MERFISH panel define the separate ranked universes.
- Exact species-specific gene symbols are matched to MSigDB GO Biological
  Process 2025.1 human C5 and mouse M5. Terms are restricted to 15–500 genes
  matched to the corresponding panel. `gene_sets/sources.json` records the
  public download URLs; `results_multilevel_gseapy131/*_gene_set_coverage.csv` records full versus
  matched term sizes and eligibility.
- GSEA settings are explicit in `run.py` and recorded with software versions
  in `results_multilevel_gseapy131/parameters.json`: weighted adaptive multilevel GSEA
  (weight 1, sample size 101, eps 1e-50), 1,000 simple permutations for NES
  normalization, seed 2026, and no automatic plotting. Gene-score ties are
  ordered by gene symbol without artificial jitter. Their frequency is
  recorded in `ranking_inventory.csv`.
- The ten resampling draws use seeds 2026–2035 and the original primary-task
  RNG sequence. Every reconstructed draw is checked against its existing
  library-size Wasserstein result before enrichment.

## Interpretation and outputs

`summarize.py` exports enrichment-profile Spearman correlation, NES mean
absolute error, direction agreement, and recall/precision/Jaccard of positive
real-data terms (NES > 0 and GSEA FDR q < 0.05). Empty denominators are NA.
Baseline intervals are empirical 5–95% Monte Carlo ranges across ten draws.
Each task/layer uses the same jointly testable terms across all 12 profiles.
Undefined NES/significance values remain NA; excluded counts are recorded in
`testability_audit.csv`. Heatmaps show unavailable values in gray, and a baseline
mean is unavailable if any of its ten draws is unavailable.

The full per-term results are retained. Figure terms are chosen from real
positive enrichment only (up to two terms per layer/class). Heatmaps compare
Real, FEAST, and mean resampling NES. Their dots indicate real/FEAST q < 0.05;
the baseline mean is not assigned a significance call.

This is descriptive functional-profile recovery given supplied labels.
Spots, task directions sharing slices, overlapping GO terms, and resampling
draws are not independent biological replicates. GSEA q-values are computed
within a ranking, not across the entire study. MERFISH panel coverage must be
reported separately from DLPFC.

- `results_multilevel_gseapy131/REPORT.md`: numerical results and interpretation limits.
- `results_multilevel_gseapy131/recovery_metrics.csv`: each task/layer/method/replicate.
- `results_multilevel_gseapy131/term_level_comparison.csv.gz`: real, FEAST, and baseline enrichment.
- `figures_gseapy131/study05_go_pathway_analysis.pdf`: Arial summary and task heatmaps.
- Per-figure PDF, SVG, and 600-DPI PNG exports are stored under `figures_gseapy131/`.

The earlier simulation outputs, metrics, and article figures are not modified.

## Real versus FEAST GO bubble plots

Run `plot_go_bubbles.py` with the existing Python environment to render the
completed, corrected enrichment tables without rerunning GSEA:

```bash
/maiziezhou_lab2/yiru/miniconda3/bin/python \
  05_2d_conditional_transfer/pathway_analysis/plot_go_bubbles.py
```

The combined PDF is
[`figures_gseapy131/bubble_plots/study05_go_enrichment_bubbles_real_vs_FEAST.pdf`](figures_gseapy131/bubble_plots/study05_go_enrichment_bubbles_real_vs_FEAST.pdf).
Each of the 13 tasks also has an individual PDF, editable SVG, and 600-DPI PNG.
All text uses Arial. Only Real and FEAST are plotted.

Each page pairs Real and FEAST with identical GO term rows grouped by layer
or cell class. The displayed terms are the union of the top three positive
terms from each method within each label (NES > 0, FDR q < 0.05), ordered by
q ascending, NES descending, then term name. Real-selected terms appear first,
followed by additional FEAST-selected terms. This allows both shared and
method-specific enrichment to appear; it is a descriptive selection rather
than an unbiased recovery metric. No GO redundancy filtering is applied.
Labels without positive terms in either method are retained with an explicit
empty-results note.

Horizontal position is signed NES, bubble size encodes the leading-edge
gene count, and fill color is −log10(FDR q-value). Bubble area increases
linearly with count, with a visible minimum area of 18 points squared
(`18 + 210 × count / dataset_count_limit`);
the size legend uses the same mapping, recorded in `plot_settings.json`. These are
GSEA plots; the horizontal axis is not an overrepresentation-test gene ratio.
All available counterparts, including q ≥ 0.05, use filled bubbles with
continuous FDR colors. The colorbar marks q = 0.05 at −log10(q) = 1.30.
Unavailable NES/q estimates are marked NA. Negative NES indicates enrichment toward the
other supported labels. Color saturates at 50 where applicable, including
reported q = 0; raw q-values remain in the plotting CSV. Axis, bubble-area,
and color scales are shared across all tasks within each technology.
The compact layout uses tighter row and section spacing, neutral section
headers, adaptive count legends, and a wrapped footer.

`bubble_plot_data.csv`, `term_selection.csv`, `layer_class_inventory.csv`, and
`plot_settings.json` accompany the figures. These plots use no resampling
values or resampling-dependent term availability filter. They retain the
existing analysis scope and interpretation limits described above, including
the targeted MERFISH panel and supplied conditional labels.
