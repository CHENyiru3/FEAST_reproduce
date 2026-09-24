# Study 06 class-marker and GO Biological Process preservation

This is a descriptive recovery analysis, comparing each original broad cell
class against all other classes in observed and FEAST-generated expression.
It uses matched held-out target cells for each density separately; retained
real reference slices are excluded from the ranking population. All 34 classes
are retained. The source volume has one donor, `Zhuang-ABCA-1`.

## Run

Use the existing pathway environment; no FEAST environment is modified:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /maiziezhou_lab2/yiru/miniconda3/envs/sctm-pub-20260718/bin/python \
  06_3d_stack/pathway_analysis/run.py \
  --gene-sets 05_2d_conditional_transfer/pathway_analysis/gene_sets/m5.go.bp.v2025.1.Mm.symbols.gmt

/maiziezhou_lab2/yiru/miniconda3/envs/sctm-pub-20260718/bin/python \
  06_3d_stack/pathway_analysis/summarize.py
```

These commands run from `FEAST_reproduce`. The active launcher is in
`.work/study06-volume-biology-20260913/pathways.sh`; its session is
`feast-s06-go-class-cached`. Successful ranking and enrichment files can be
reused when continuing the same analysis. Use a fresh output directory if
scientific settings or inputs change.

## Methods

The established Study 05 pathway method is reused: counts normalized to 10,000
per cell, log1p, Scanpy Wilcoxon signed z scores with tie correction, and all
1,122 measured genes. No additional QC, HVG selection, regression, class
merging, cell subsampling, or DEG p-value filtering is applied. Rank ties are
ordered by gene symbol without jitter. Scores and Scanpy log2 fold changes are
saved; cell-level DEG p-values are not used to make biological-replicate claims.

The supplied mouse GO BP 2025.1 GMT is intersected with the measured panel.
Of the 1,122 genes, 1,005 have annotations in this GMT; 1,019 GO terms have
15–500 measured genes and are eligible. Both full and measured term sizes are
recorded in `results/gene_set_coverage.csv`. The same universe and terms are
used for observed and simulated rankings.

GSEApy 1.3.0 preranked enrichment uses weight 1, 1,000 simple permutations for
normalization, multilevel probability estimation (sample size 101, epsilon
1e-50), seed 2026 and one thread. These settings follow the current Study 05
workflow. Software and parameters are recorded in `results/parameters.json`.

A method-preserving optimization caches Scanpy's identical tie correction once
per immutable rank block instead of recomputing it for every class. It is local
to this analysis process and does not modify the installed Scanpy package.
A 34-class parity check reproduced every score and fold change exactly,
reducing correction calls from 34 to one and test runtime from 4.79 to 0.68 s.
The check is in `.work/study06-volume-biology-20260913/rank_cache_verification.json`.

## Outputs and interpretation

The complete analysis comprises 204 enrichment results: 34 classes × 3 density
arms × observed/FEAST. Per-class summaries report gene-rank correlation, GO NES
correlation and absolute error, enrichment-direction agreement, and descriptive
recall/precision of observed positive terms (NES > 0, within-ranking q < 0.05).
Example terms are selected from observed gap-3 enrichment only, without using
FEAST performance to choose them. All term-level results are retained.

- `results/completion.json`: completion status.
- `results/preservation_metrics.csv`: per-class recovery metrics.
- `results/term_comparison.csv.gz`: matched observed/FEAST GO results.
- `figures/study06_biological_preservation.pdf`: figure bundle.
- `figures/REPORT.md`: numerical summary and interpretation.

The labels themselves condition FEAST generation. This analysis tests whether
conditional gene and functional signatures are preserved; it is not independent
cell-type discovery. Cells, ordered sections, and simulations are not donor
replicates. GO q-values describe enrichment within a ranking, not donor-level
DE or study-wide biological reproducibility. Targeted-panel coverage limits
which processes can be evaluated.

Relevant methodology: [GSEA/MSigDB guidance](https://docs.gsea-msigdb.org/GSEA/GSEA_FAQ/),
[GSEApy implementation](https://gseapy.readthedocs.io/en/latest/run.html), and
[biological replication in single-cell DE](https://www.nature.com/articles/s41467-021-25960-2).
