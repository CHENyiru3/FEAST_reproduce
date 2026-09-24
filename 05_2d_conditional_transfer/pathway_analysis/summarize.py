"""Compare layer/class GO enrichment and export an Arial figure/report bundle."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sys
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_multilevel_gseapy131"
FIGURES = HERE / "figures_gseapy131"
FDR = 0.05
GROUPS = ["DLPFC cross-slice\nwithin donor", "DLPFC cross-slice\ncross donor", "DLPFC half-held-out",
          "MERFISH cross-slice", "MERFISH half-held-out"]
COLORS = {"real": "#333333", "feast": "#0072B2", "resampling": "#999999"}


def group_name(row):
    if row.dataset == "dlpfc" and row["mode"] == "cross_slice":
        return GROUPS[0] if row.donor_stratum == "within_donor" else GROUPS[1]
    if row.dataset == "dlpfc":
        return GROUPS[2]
    return GROUPS[3] if row["mode"] == "cross_slice" else GROUPS[4]


def read_enrichment(path):
    frame = pd.read_csv(path).set_index("term").sort_index()
    assert frame.index.is_unique
    return frame


def pair_metrics(real, candidate):
    assert real.index.equals(candidate.index)
    real_positive = (real.nes > 0) & (real.gsea_fdr_q < FDR)
    candidate_positive = (candidate.nes > 0) & (candidate.gsea_fdr_q < FDR)
    intersection = int((real_positive & candidate_positive).sum())
    n_real, n_candidate = int(real_positive.sum()), int(candidate_positive.sum())
    recall = intersection / n_real if n_real else np.nan
    precision = intersection / n_candidate if n_candidate else np.nan
    return dict(n_terms=len(real), nes_spearman=float(spearmanr(real.nes, candidate.nes).statistic),
        nes_mae=float(np.mean(np.abs(real.nes - candidate.nes))),
        direction_agreement=float(np.mean(np.sign(real.nes) == np.sign(candidate.nes))),
        real_positive_terms=n_real, candidate_positive_terms=n_candidate,
        recovered_positive_terms=intersection, positive_recall=recall, positive_precision=precision,
        positive_jaccard=intersection / int((real_positive | candidate_positive).sum())
                         if (real_positive | candidate_positive).any() else np.nan)


def compare_layer_enrichment(inventory):
    rows, comparison_rows, selected_terms = [], [], []
    term_path = RESULTS / "term_level_comparison.csv.gz"
    with gzip.open(term_path, "wt") as handle:
        first = True
        for (dataset, direction, label), group in inventory.groupby(["dataset", "direction", "label"], sort=False):
            real_meta = group.loc[group.method.eq("real")].iloc[0]
            identity = {k: real_meta[k] for k in ["dataset", "mode", "direction", "source", "target", "donor_stratum", "label", "n_label_spots"]}
            real = read_enrichment(real_meta.result_file)
            candidate_tables = {}
            common_valid = np.isfinite(real[["nes", "gsea_fdr_q"]]).all(axis=1)
            for _, meta in group.loc[group.method.ne("real")].iterrows():
                frame = read_enrichment(meta.result_file)
                candidate_tables[meta.column] = frame
                common_valid &= np.isfinite(frame[["nes", "gsea_fdr_q"]]).all(axis=1)
            for _, meta in group.loc[group.method.ne("real")].iterrows():
                frame = candidate_tables[meta.column]
                rows.append(dict(**identity, method=meta.method, replicate=int(meta.replicate),
                                 n_total_terms=len(real), n_unavailable_common=int((~common_valid).sum()),
                                 real_unavailable_terms=int(real.nes.isna().sum()),
                                 candidate_unavailable_terms=int(frame.nes.isna().sum()),
                                 **pair_metrics(real.loc[common_valid], frame.loc[common_valid])))
            feast_meta = group.loc[group.method.eq("feast")].iloc[0]
            feast = candidate_tables[feast_meta.column]
            baselines = [candidate_tables[r.column] for _, r in group.loc[group.method.eq("resampling")].iterrows()]
            assert len(baselines) == 10
            baseline_nes = np.column_stack([x.nes.to_numpy() for x in baselines])
            baseline_positive = np.column_stack([np.where(np.isfinite(x.nes) & np.isfinite(x.gsea_fdr_q),
                (x.nes.gt(0) & x.gsea_fdr_q.lt(FDR)).to_numpy().astype(float), np.nan) for x in baselines])
            terms = pd.DataFrame(dict(term=real.index, real_nes=real.nes.to_numpy(),
                real_fdr=real.gsea_fdr_q.to_numpy(), feast_nes=feast.nes.to_numpy(),
                feast_fdr=feast.gsea_fdr_q.to_numpy(),
                resampling_nes_mean=baseline_nes.mean(axis=1),
                resampling_nes_p05=np.quantile(baseline_nes, .05, axis=1),
                resampling_nes_p95=np.quantile(baseline_nes, .95, axis=1),
                resampling_positive_fraction=baseline_positive.mean(axis=1),
                real_leading_edge_genes=real.leading_edge_genes.to_numpy(),
                feast_leading_edge_genes=feast.leading_edge_genes.to_numpy()))
            real_positive = terms.real_nes.gt(0) & terms.real_fdr.lt(FDR)
            feast_positive = terms.feast_nes.gt(0) & terms.feast_fdr.lt(FDR)
            terms["positive_term_status"] = np.select(
                [~np.isfinite(terms[["real_nes", "real_fdr", "feast_nes", "feast_fdr"]]).all(axis=1),
                 real_positive & feast_positive, real_positive, feast_positive],
                ["unavailable", "recovered", "real_only", "feast_only"], default="neither")
            terms["jointly_testable_all_methods"] = common_valid.to_numpy()
            for key, value in identity.items():
                terms[key] = value
            terms.to_csv(handle, index=False, header=first)
            first = False
            # Figure selection uses the real data only: two positive GO terms per label.
            chosen = terms.loc[terms.real_nes.gt(0) & terms.real_fdr.lt(FDR)].sort_values(
                ["real_fdr", "real_nes", "term"], ascending=[True, False, True]).head(2)
            selected_terms.extend(chosen.to_dict("records"))
            print(f"Compared {dataset} {direction} {label}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics["group"] = metrics.apply(group_name, axis=1)
    metrics.to_csv(RESULTS / "recovery_metrics.csv", index=False)
    metrics.loc[metrics.method.eq("feast"), ["dataset", "mode", "direction", "label", "n_total_terms",
        "n_terms", "n_unavailable_common", "real_unavailable_terms", "candidate_unavailable_terms"]].to_csv(
        RESULTS / "testability_audit.csv", index=False)
    keys = ["dataset", "mode", "direction", "source", "target", "donor_stratum", "label", "n_label_spots", "group"]
    for _, row in metrics.loc[metrics.method.eq("feast")].iterrows():
        baseline = metrics.loc[metrics.direction.eq(row.direction) & metrics.label.eq(row.label) & metrics.method.eq("resampling")]
        for metric in ["nes_spearman", "nes_mae", "direction_agreement", "positive_recall", "positive_precision", "positive_jaccard"]:
            values = baseline[metric].dropna().to_numpy()
            comparison_rows.append(dict(**{k: row[k] for k in keys}, metric=metric, feast=row[metric],
                baseline_mean=float(np.mean(values)) if len(values) else np.nan,
                baseline_p05=float(np.quantile(values, .05)) if len(values) else np.nan,
                baseline_p95=float(np.quantile(values, .95)) if len(values) else np.nan,
                baseline_valid_replicates=len(values)))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS / "feast_vs_resampling.csv", index=False)
    pd.DataFrame(selected_terms).to_csv(RESULTS / "figure_term_selection.csv", index=False)
    return metrics, comparison


def configure_style():
    # Fonts are shared with the existing article figures, outside this interpreter's prefix.
    font_dir = Path("/maiziezhou_lab2/yiru/miniconda3/fonts")
    for name in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        font_manager.fontManager.addfont(font_dir / name)
    mpl.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white"})


def save_figure(fig, stem, bundle):
    bundle.savefig(fig, bbox_inches="tight")
    for suffix in ["pdf", "svg", "png"]:
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {}
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=600, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def plot_recovery(comparison, bundle):
    specs = [("nes_spearman", "GO enrichment profile\nSpearman correlation ↑"),
             ("positive_recall", "Recovery of real enriched terms\nRecall ↑"),
             ("positive_precision", "Agreement of called enriched terms\nPrecision ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    fig.subplots_adjust(left=.06, right=.99, top=.80, bottom=.24, wspace=.24)
    for ax, (metric, heading) in zip(axes, specs):
        for i, group in enumerate(GROUPS):
            rows = comparison.loc[comparison.metric.eq(metric) & comparison.group.eq(group)].sort_values(["direction", "label"])
            offsets = np.linspace(-.07, .07, len(rows))
            for offset, (_, row) in zip(offsets, rows.iterrows()):
                if np.isfinite(row.baseline_mean) and np.isfinite(row.feast):
                    ax.plot([i-.16+offset, i+.16+offset], [row.baseline_mean, row.feast], color="#CCCCCC", lw=.7, zorder=1)
            ax.scatter(i-.16+offsets, rows.baseline_mean, color=COLORS["resampling"], marker="s", s=20, zorder=2)
            ax.scatter(i+.16+offsets, rows.feast, color=COLORS["feast"], edgecolors="white", linewidths=.4, s=26, zorder=3)
        ax.set_title(heading, weight="bold")
        ax.set_xticks(range(5), GROUPS, fontsize=8, rotation=25, ha="right")
        ax.set_ylim(-1.03 if metric == "nes_spearman" else -.03, 1.03)
        if metric == "nes_spearman":
            ax.axhline(0, color="#888888", lw=.8, ls="--")
        ax.grid(axis="y", color="#E4E4E4", lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Recovery of layer/class GO Biological Process enrichment", fontsize=17, weight="bold", y=.99)
    fig.text(.5, .915, "AR = 0.3 · each pair is one task × layer/class · real positive terms: NES > 0 and GSEA FDR < 0.05",
             ha="center", fontsize=10, color="#555555")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS["feast"], label="FEAST"),
                        Line2D([], [], marker="s", ls="", color=COLORS["resampling"], label="Ten-draw resampling mean")],
               ncol=2, loc="lower center", bbox_to_anchor=(.5, -.015))
    save_figure(fig, "go_recovery_summary", bundle)


def plot_task_heatmaps(inventory, bundle):
    chosen = pd.read_csv(RESULTS / "figure_term_selection.csv")
    for (dataset, direction), task in inventory.groupby(["dataset", "direction"], sort=False):
        terms = chosen.loc[chosen.direction.eq(direction), "term"].drop_duplicates().tolist()
        if not terms:
            # An explicit empty-results panel preserves tasks without positive real terms.
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis("off")
            ax.text(.5, .5, f"{dataset.upper()} {direction}\nNo positive real-data GO terms at FDR < 0.05", ha="center")
            save_figure(fig, f"go_{dataset}_{direction}", bundle)
            continue
        labels = sorted(task.label.unique())
        values, significant = [], []
        for label in labels:
            group = task.loc[task.label.eq(label)]
            for method in ["real", "feast", "resampling"]:
                frames = [read_enrichment(r.result_file).loc[terms] for _, r in group.loc[group.method.eq(method)].iterrows()]
                values.append(np.mean([frame.nes.to_numpy() for frame in frames], axis=0))
                significant.append(frames[0].gsea_fdr_q.lt(FDR).to_numpy() if method != "resampling" else np.zeros(len(terms), dtype=bool))
        matrix, dots = np.asarray(values).T, np.asarray(significant).T
        limit = float(np.nanmax(np.abs(matrix)))
        fig, ax = plt.subplots(figsize=(16, max(5, len(terms) * .36 + 2)))
        fig.subplots_adjust(left=.37, right=.94, bottom=.19, top=.78)
        cmap = mpl.colormaps["RdBu_r"].copy()
        cmap.set_bad("#CCCCCC")
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        y, x = np.where(dots)
        ax.scatter(x, y, color="black", s=6)
        ax.set_yticks(range(len(terms)), [textwrap.fill(t.removeprefix("GOBP_").replace("_", " ").lower(), 45) for t in terms], fontsize=8)
        ax.set_xticks(range(matrix.shape[1]), ["Real", "FEAST", "Resamp."] * len(labels), rotation=60, ha="right", fontsize=8)
        for i, label in enumerate(labels):
            ax.text(i*3+1, 1.025, label.replace("Layer_", "L"), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", rotation=25 if dataset == "merfish" else 0, fontsize=8, weight="bold")
            if i:
                ax.axvline(i*3-.5, color="white", lw=2)
        cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
        cb.set_label("Normalized enrichment score (NES)")
        header = direction.replace("Zhuang-ABCA-1.", "").replace("_to_", " → ").replace("_low_x → high_x", " half-held-out")
        fig.suptitle(f"{dataset.upper()}  {header}", y=.99, fontsize=16, weight="bold")
        fig.text(.5, .93, "Terms selected from real data only: up to two per layer/class. Black dots: real/FEAST FDR < 0.05.",
                 ha="center", fontsize=9, color="#555555")
        fig.text(.5, .045, "Resampling colors show the mean NES across ten draws; gray = unavailable, including a missing baseline draw.\n"
                 "Target labels are supplied to generation. Enrichment agreement describes functional-profile recovery, not independent biological validation.",
                 ha="center", fontsize=8, color="#555555")
        save_figure(fig, f"go_{dataset}_{direction}", bundle)


def write_report(inventory, metrics, comparison):
    records = []
    for (group, metric), rows in comparison.groupby(["group", "metric"], sort=False):
        valid = rows.loc[rows.feast.notna() & rows.baseline_mean.notna()]
        positive = valid.feast < valid.baseline_mean if metric == "nes_mae" else valid.feast > valid.baseline_mean
        records.append(dict(group=group.replace("\n", " "), metric=metric, task_layers=len(rows),
            valid_pairs=len(valid), feast_median=float(valid.feast.median()),
            resampling_mean_median=float(valid.baseline_mean.median()),
            feast_better_pairs=int(positive.sum())))
    summary = pd.DataFrame(records)
    summary.to_csv(RESULTS / "group_summary.csv", index=False)
    lines = ["# Study 05: GO Biological Process recovery", "",
        "**Final run: GSEApy 1.3.1.** This run supersedes the provisional GSEApy 1.3.0 multilevel outputs. "
        "The older release had an upstream multilevel p-value bug and could return unnormalized ES in the NES field. "
        "Original outputs are preserved under `../results_multilevel/` with a validation note. "
        "The corrected run uses unchanged gene rankings and statistical settings.", "",
        f"Completed {inventory.direction.nunique()} primary tasks and {len(inventory)} enrichment runs.", "",
        "The analysis ranks each supported layer/class against the remaining supported target spots separately in real data, FEAST, "
        "and ten label-conditioned resampling draws. Rankings use total-count normalization to 10,000, log1p, and tie-corrected Wilcoxon scores. "
        "The full fixed gene panel is retained. MSigDB 2025.1 human/mouse GO-BP sets are matched by exact species-specific symbols; "
        "sets with 15–500 matched genes are tested with adaptive multilevel GSEA (sample size 101, eps 1e-50, "
        "1,000 simple permutations for NES normalization, seed 2026).", "",
        "## Enrichment-profile agreement", "",
        "| Task stratum | Task × label pairs | FEAST median Spearman | Resampling median of ten-draw means | FEAST higher pairs |",
        "|---|---:|---:|---:|---:|"]
    for _, r in summary.loc[summary.metric.eq("nes_spearman")].iterrows():
        lines.append(f"| {r.group} | {r.task_layers} | {r.feast_median:.3f} | {r.resampling_mean_median:.3f} | {r.feast_better_pairs}/{r.valid_pairs} |")
    concordance = comparison.loc[comparison.metric.eq("nes_spearman")]
    valid = concordance.loc[concordance.feast.notna() & concordance.baseline_mean.notna()]
    negatives = concordance.loc[concordance.feast.lt(0)]
    lines.extend(["", f"FEAST has higher NES-profile Spearman correlation than the resampling mean in "
                  f"{int((valid.feast > valid.baseline_mean).sum())}/{len(valid)} task/layer comparisons. "
                  f"Negative FEAST correlations occur in {len(negatives)} comparisons. "
                  "These are descriptive counts; task/layer comparisons share tissue and are not independent replicates.", ""])
    if len(negatives):
        lines.extend(["Negative-correlation comparisons:", ""])
        lines.extend(f"- {r.dataset} {r.direction}, {r.label}: FEAST {r.feast:.3f}; resampling mean {r.baseline_mean:.3f}."
                     for _, r in negatives.iterrows())
    lines.extend(["", "## Recovery of positive real-data terms", "",
        "All recovery metrics use the same jointly testable GO terms across real, FEAST, and all ten resampling draws within a task/layer. "
        "Terms with unavailable NES/significance in any of those profiles are excluded from that comparison and counted in `testability_audit.csv`. "
        "Positive enriched terms have NES > 0 and GSEA FDR q < 0.05. Recall is the fraction of real positive terms also positive "
        "in the candidate; precision is the fraction of candidate positive terms also positive in real data. Empty denominators are NA.", "",
        "| Task stratum | FEAST median recall | Resampling median recall | FEAST median precision | Resampling median precision |",
        "|---|---:|---:|---:|---:|"])
    for group in summary.group.unique():
        recall = summary.loc[summary.group.eq(group) & summary.metric.eq("positive_recall")].iloc[0]
        precision = summary.loc[summary.group.eq(group) & summary.metric.eq("positive_precision")].iloc[0]
        lines.append(f"| {group} | {recall.feast_median:.3f} | {recall.resampling_mean_median:.3f} | {precision.feast_median:.3f} | {precision.resampling_mean_median:.3f} |")
    examples = pd.read_csv(RESULTS / "figure_term_selection.csv")
    lines.extend(["", "## Functional examples from the existing four-panel study selection", "",
        "These tasks were already used in the study's spatial figures. For each label, the first real-data-selected positive term "
        "is shown, whether FEAST recovers it or not. The complete 13-task results remain the basis for performance assessment."])
    for direction in ["151676_to_151675", "151676_low_x_to_high_x",
                      "Zhuang-ABCA-1.006_to_Zhuang-ABCA-1.007", "Zhuang-ABCA-1.007_low_x_to_high_x"]:
        subset = examples.loc[examples.direction.eq(direction)].drop_duplicates("label")
        lines.extend(["", f"### {direction}", "",
            "| Layer/class | Real-data GO term | Real NES | FEAST NES | FEAST FDR | Resampling mean NES | Positive baseline draws |",
            "|---|---|---:|---:|---:|---:|---:|"])
        for _, r in subset.iterrows():
            term = r.term.removeprefix("GOBP_").replace("_", " ").lower()
            positive_draws = f"{round(r.resampling_positive_fraction*10)}/10" if np.isfinite(r.resampling_positive_fraction) else "NA"
            feast_nes = f"{r.feast_nes:.2f}" if np.isfinite(r.feast_nes) else "NA"
            feast_fdr = f"{r.feast_fdr:.2g}" if np.isfinite(r.feast_fdr) else "NA"
            baseline_nes = f"{r.resampling_nes_mean:.2f}" if np.isfinite(r.resampling_nes_mean) else "NA"
            lines.append(f"| {r.label} | {term} | {r.real_nes:.2f} | {feast_nes} | {feast_fdr} | "
                         f"{baseline_nes} | {positive_draws} |")
    lines.extend(["", "## Scope and interpretation", "",
        "- GSEApy 1.3.1 marks NES, p-value, FDR, and log2err unavailable when there are fewer than ten same-sign null scores. "
        "These values remain NA. They are never converted to zero or counted as enriched. Heatmaps show unavailable values in gray; "
        "a baseline mean is unavailable if any of its ten draws is unavailable.",
        "- High agreement indicates preservation of layer/class-associated functional rankings. Because the labels are provided to generation, "
        "the resampling comparison is needed to assess improvement beyond label-conditioned reference expression.",
        "- DLPFC has three slices from two donors; task directions share slices. Spots, simulated samples, overlapping GO terms, and baseline draws "
        "are not independent biological replicates. No donor-level hypothesis test is performed.",
        "- GSEA q-values describe gene-set permutation enrichment within each ranking, not biological reproducibility or a globally corrected "
        "test across all layers, tasks, and methods. Adaptive p-values have an eps floor of 1e-50; log2err is retained in per-ranking tables.",
        "- No spot QC filtering, HVG restriction, reclustering, regression, or DEG-significance filtering was applied. "
        "All-zero genes receive zero Wilcoxon scores through Scanpy's implementation; tied scores are ordered by gene symbol, with no jitter. "
        "Tie fractions are recorded in ranking_inventory.csv and should be considered when interpreting enrichment.",
        "- MERFISH uses a targeted 1,122-gene panel and cell classes; DLPFC uses 17,391 genes and cortical layer/WM labels. "
        "Gene-set coverage is reported per dataset and should not be treated as equivalent across technologies.",
        "- Only supported target spots are evaluated. Half-held-out analyses contain only the held-out half, not the observed reference context.",
        "- Heatmap terms were selected using real-data positive enrichment only. Complete results, including absent/extra terms and negative enrichment, "
        "remain in the per-ranking files and term_level_comparison.csv.gz.", "",
        "## Outputs", "",
        "- [Figure bundle](../figures_gseapy131/study05_go_pathway_analysis.pdf)",
        "- `recovery_metrics.csv`: all FEAST and baseline task/layer recovery measurements.",
        "- `feast_vs_resampling.csv`: paired FEAST versus baseline mean and empirical 5–95% range.",
        "- `term_level_comparison.csv.gz`: real, FEAST, and baseline term-level scores and leading-edge genes.",
        "- `*_gene_set_coverage.csv`: annotation/panel coverage for every downloaded GO term.",
        "- `figure_term_selection.csv`: complete real-data-only selection records for the heatmaps.", "",
        "## Annotation sources", "",
        "Human: https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/", "",
        "Mouse: https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Mm/", ""])
    lines.extend(["Corrected software: https://github.com/zqfang/GSEApy/releases/tag/v1.3.1", ""])
    (RESULTS / "REPORT.md").write_text("\n".join(lines))


def main():
    complete = json.loads((RESULTS / "completion.json").read_text())
    inventory = pd.read_csv(RESULTS / "ranking_inventory.csv", dtype={"source": str, "target": str})
    assert len(inventory) == complete["rankings"]
    metrics, comparison = compare_layer_enrichment(inventory)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    with PdfPages(FIGURES / "study05_go_pathway_analysis.pdf") as bundle:
        plot_recovery(comparison, bundle)
        plot_task_heatmaps(inventory, bundle)
    write_report(inventory, metrics, comparison)
    print(f"Wrote {RESULTS / 'REPORT.md'} and {FIGURES / 'study05_go_pathway_analysis.pdf'}", flush=True)


if __name__ == "__main__":
    main()
