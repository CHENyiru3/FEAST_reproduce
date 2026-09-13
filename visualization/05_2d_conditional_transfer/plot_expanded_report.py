"""Expand Study 05 into a direction-resolved PDF using existing simulations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance

VIS = Path(__file__).resolve().parent
STUDY = VIS.parents[1] / "05_2d_conditional_transfer"
OUT = VIS / "figures" / "article_style"
sys.path.insert(0, str(STUDY))
from conditional_resampling_baseline import conditional_donor_indices, selected_jobs
from workflow import direction_id, expected_job_path, load_config

BLUE, GRAY, TRUTH = "#0072B2", "#929292", "#333333"
CAPTIONS = []
MARKERS = {"dlpfc": ("SCGB2A2", "GFAP"), "merfish": ("Igfbp4", "Frzb")}
METRICS = (
    ("conditional_expression_wasserstein", "Expression\nWasserstein ↓"),
    ("conditional_zero_fraction_wasserstein", "Zero-fraction\nWasserstein ↓"),
    ("moran_profile_correlation", "Moran I profile\ncorrelation ↑"),
    ("residual_moran_profile_correlation", "Label-residual Moran I\ncorrelation ↑"),
)


def short(value):
    return str(value).replace("Zhuang-ABCA-1.", "")


def label(row):
    dataset = "DLPFC" if row["dataset"] == "dlpfc" else "MERFISH"
    direction = (f"{short(row['source'])} → {short(row['target'])}"
                 if row["mode"] == "cross_slice" else f"{short(row['source'])} half")
    return f"{dataset}  {direction}"


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=0.5, y=0.99, ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=9, color="#555555")


def footer(fig, text):
    fig._article_caption = text


def save(fig, stem, pages):
    path = OUT / f"{stem}.pdf"
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{stem}.svg", dpi=600, bbox_inches="tight", pad_inches=0.08,
                metadata={"Date": None})
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    CAPTIONS.append((stem, getattr(fig, "_article_caption", "")))
    plt.close(fig)
    pages.append(path)
    print(f"Rendered {stem}", flush=True)


def order_tasks(summary):
    frame = summary.loc[summary.primary_setting].copy()
    frame["stratum_order"] = frame.donor_stratum.map(
        {"within_donor": 0, "cross_donor": 1, "donor_not_declared": 2, "within_slice": 3}
    )
    return frame.sort_values(["dataset", "stratum_order", "source", "target"])


def metric_page(comparison, tasks, mode, pages):
    selected = tasks.loc[tasks["mode"].eq(mode)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.2 if mode == "cross_slice" else 4.8), sharey=True)
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.14, top=0.78, wspace=0.24)
    title(fig, "Cross-slice conditional transfer" if mode == "cross_slice" else
          "Half-slice conditional generation", "AR = 0.3 · supported target spots · FEAST versus label-conditioned resampling")
    y = np.arange(len(selected))
    for panel, (ax, (metric, heading)) in enumerate(zip(axes, METRICS)):
        rows = comparison.loc[comparison.metric.eq(metric)].set_index("direction").loc[selected.direction]
        ax.hlines(y, rows.baseline_p05, rows.baseline_p95, color=GRAY, lw=3, alpha=0.5)
        for yi, (_, row) in zip(y, rows.iterrows()):
            ax.plot([row.baseline_mean, row.feast], [yi, yi], color="#CCCCCC", lw=1)
        ax.scatter(rows.baseline_mean, y, c=GRAY, s=24, marker="s", zorder=3)
        ax.scatter(rows.feast, y, c=BLUE, s=33, edgecolors="white", linewidths=0.5, zorder=4)
        ax.set_title(heading, fontsize=11, weight="bold", pad=9)
        ax.text(-0.04, 1.15, chr(65 + panel), transform=ax.transAxes, fontsize=13, weight="bold")
        ax.grid(axis="x", color="#E5E5E5")
        ax.set_axisbelow(True)
        for i in range(1, len(selected)):
            a, b = selected.iloc[i - 1], selected.iloc[i]
            if a.dataset != b.dataset or a.donor_stratum != b.donor_stratum:
                ax.axhline(i - 0.5, color="#BBBBBB", lw=0.8)
    names = [label(r) + ("\nwithin donor" if r.donor_stratum == "within_donor" else
                        "\ncross donor" if r.donor_stratum == "cross_donor" else "")
             for _, r in selected.iterrows()]
    axes[0].set_yticks(y, names, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="FEAST"),
                        Line2D([], [], marker="s", color=GRAY, label="Resampling mean (5–95% range)")],
               loc="lower center", bbox_to_anchor=(0.59, 0.005), ncol=2, fontsize=9)
    footer(fig, "Ten resampling draws describe Monte Carlo variability; these ranges are not confidence intervals.\n"
           "Directions share slices and are not independent donor replicates. MERFISH donor identity is unspecified.")
    save(fig, f"metrics_{mode}", pages)


def support_page(tasks, support, pages):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    fig.subplots_adjust(left=0.245, right=0.97, top=0.83, bottom=0.1, wspace=0.28)
    title(fig, "Target coverage and reference gene evidence",
          "Source support ≥20 spots per label · unsupported target spots are excluded")
    axes[0].set_title("A  Target support", loc="left", weight="bold")
    axes[1].set_title("B  Genes without source evidence", loc="left", weight="bold")
    rows = support.loc[np.isclose(support.assignment_randomness, 0.3)].set_index("direction").loc[tasks.direction]
    y = np.arange(len(rows))
    axes[0].barh(y, rows.target_retained_fraction * 100, color=BLUE, height=0.6)
    axes[0].barh(y, (1 - rows.target_retained_fraction) * 100,
                 left=rows.target_retained_fraction * 100, color="#DDDDDD", height=0.6)
    for yi, (_, row) in zip(y, rows.iterrows()):
        axes[0].text(101, yi, f"{row.target_retained_spots:,}/{row.target_candidate_spots:,}", va="center", fontsize=9)
    axes[0].set_xlim(0, 132)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_yticks(y, [label(r) for _, r in tasks.iterrows()], fontsize=8.5)
    axes[0].set_xlabel("Target spots retained (%)")
    axes[1].barh(y, rows.reference_zero_gene_count, color="#D97946", height=0.6)
    for yi, (_, row) in zip(y, rows.iterrows()):
        axes[1].text(row.reference_zero_gene_count + 2, yi,
                     f"{row.reference_zero_gene_count:,} / {row.dataset_gene_panel_count:,}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(rows.reference_zero_gene_count) * 1.5)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Genes with zero source counts")
    for ax in axes:
        ax.invert_yaxis()
    footer(fig, "151670 → 151675 and 151670 → 151676 exclude Layers 1–2: coverage is 83.1% and 84.2%.\n"
           "Zero source evidence and unsupported target labels are distinct limitations.")
    save(fig, "target_coverage", pages)


def sensitivity_page(summary, pages):
    cross = summary.loc[summary["mode"].eq("cross_slice")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.18, hspace=0.32, wspace=0.32)
    title(fig, "Assignment-randomness sensitivity",
          "40 cross-slice runs · seed 2026 · dashed line: primary AR = 0.3")
    strata = [("dlpfc", "within_donor", "DLPFC within donor"),
              ("dlpfc", "cross_donor", "DLPFC cross donor"),
              ("merfish", "donor_not_declared", "MERFISH")]
    for col, (dataset, stratum, heading) in enumerate(strata):
        subset = cross.loc[cross.dataset.eq(dataset) & cross.donor_stratum.eq(stratum)]
        for index, (direction, rows) in enumerate(subset.groupby("direction", sort=False)):
            rows = rows.sort_values("assignment_randomness")
            color = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"][index]
            for row, metric in enumerate(["moran_corr", "median_gene_pearson"]):
                axes[row, col].plot(rows.assignment_randomness, rows[metric], color=color,
                                    marker=["o", "s", "^", "D"][index], markerfacecolor="white",
                                    markeredgewidth=1, linewidth=1.7, ms=4,
                                    label=f"{short(rows.iloc[0].source)} → {short(rows.iloc[0].target)}")
        axes[0, col].set_title(f"{chr(65 + col)}  {heading}", weight="bold", fontsize=11)
        axes[1, col].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8)
        axes[1, col].set_xlabel("Assignment randomness")
        for ax in axes[:, col]:
            ax.axvline(0.3, color=GRAY, ls="--", lw=1)
            ax.grid(axis="y", color="#E5E5E5")
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.5])
    axes[0, 0].set_ylabel("Moran I profile\ncorrelation", weight="bold")
    axes[1, 0].set_ylabel("Median gene\nspotwise Pearson r", weight="bold")
    footer(fig, "Parameter sensitivity is not seed uncertainty. A high correlation of gene-level spatial statistics does not establish spotwise recovery.")
    save(fig, "assignment_randomness", pages)


def qc_page(methods, tasks, pages):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    fig.subplots_adjust(left=0.245, right=0.97, top=0.83, bottom=0.18, wspace=0.24)
    title(fig, "Count-distribution diagnostics",
          "Wasserstein distance to the held-out target · lower is better")
    y = np.arange(len(tasks))
    for ax, metric, heading in zip(axes, ["library_size_wasserstein", "detected_genes_wasserstein"],
                                   ["A  Library size", "B  Detected genes per spot"]):
        for yi, (_, task) in zip(y, tasks.iterrows()):
            rows = methods.loc[methods.direction.eq(task.direction)]
            baseline = rows.loc[rows.method.ne("feast"), metric].to_numpy()
            feast = float(rows.loc[rows.method.eq("feast"), metric].iloc[0])
            ax.hlines(yi, *np.quantile(baseline, [0.05, 0.95]), color=GRAY, lw=3, alpha=0.6)
            ax.scatter(np.mean(baseline), yi, color=GRAY, s=24, marker="s")
            ax.scatter(feast, yi, color=BLUE, s=38)
        ax.set_title(heading, loc="left", weight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("Wasserstein distance (logarithmic axis)")
        ax.grid(axis="x", color="#E5E5E5")
    axes[0].set_yticks(y, [label(r) for _, r in tasks.iterrows()], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="FEAST"),
                        Line2D([], [], marker="s", color=GRAY, label="Resampling mean (5–95% range)")],
               loc="lower center", bbox_to_anchor=(0.6, 0.005), ncol=2, fontsize=9)
    footer(fig, "Blue: FEAST. Gray: mean and empirical 5–95% range across ten resampling draws.\n"
           "Technologies have different count depths and gene panels; compare methods within a task.")
    save(fig, "count_distribution_diagnostics", pages)


def vector(adata, gene):
    values = adata.layers["counts"][:, adata.var_names.get_loc(gene)]
    return np.asarray(values.toarray() if sparse.issparse(values) else values).reshape(-1)


def map_axis(ax, xy):
    span = np.ptp(xy, axis=0)
    ax.set_xlim(xy[:, 0].min() - span[0] * 0.03, xy[:, 0].max() + span[0] * 0.03)
    ax.set_ylim(xy[:, 1].max() + span[1] * 0.03, xy[:, 1].min() - span[1] * 0.03)
    ax.set_aspect("equal")
    ax.set_axis_off()


def load_spatial(config, summary, methods):
    inputs = {(dataset, str(slice_id)): ad.read_h5ad(STUDY / "data/local" / dataset / f"{slice_id}.h5ad")
              for dataset, spec in config["datasets"].items() for slice_id in spec["slices"]}
    rng = np.random.default_rng(config["public_seed"])
    cases, records = [], []
    # Advance the RNG through every primary task in the original evaluator order,
    # including half-slice tasks before MERFISH. This reproduces baseline draw 0.
    for job in selected_jobs(config):
        source, target = inputs[job.dataset, job.source], inputs[job.dataset, job.target]
        key = config["datasets"][job.dataset]["annotation_key"]
        path = expected_job_path(STUDY / "outputs/final", job)
        provenance = json.loads((path / "provenance.json").read_text())
        assert provenance["target_expression_loaded_by_generation"] is False
        support = provenance["support"]
        sx, tx = np.asarray(source.obsm["spatial"])[:, :2], np.asarray(target.obsm["spatial"])[:, :2]
        sl, tl = source.obs[key].astype(str).to_numpy(), target.obs[key].astype(str).to_numpy()
        sm = np.isin(sl, support["eligible_labels"])
        tm = np.isin(tl, support["eligible_labels"])
        if job.mode == "mask_half":
            cut = np.quantile(sx[:, 0], config["mask_half"]["split_quantile"])
            sm &= sx[:, 0] <= cut
            tm &= tx[:, 0] > cut
        assert sm.sum() == support["source_retained_spots"]
        assert tm.sum() == support["target_retained_spots"]
        donor = np.flatnonzero(sm)[conditional_donor_indices(sl[sm], tl[tm], rng)]
        if job.mode != "cross_slice":
            continue
        generated = ad.read_h5ad(path / "generated.h5ad")
        assert list(generated.obs_names) == list(target.obs_names[tm])
        np.testing.assert_allclose(generated.obsm["spatial"], tx[tm], rtol=0, atol=0)
        assert np.array_equal(generated.obs[key].astype(str).to_numpy(), tl[tm])
        # Verify this reconstructed draw against its already-scored count metric.
        panel = list(generated.var_names)
        target_library = np.asarray(target[tm, panel].layers["counts"].sum(axis=1)).ravel()
        source_library = np.asarray(source[:, panel].layers["counts"].sum(axis=1)).ravel()
        expected = methods.loc[methods.direction.eq(direction_id(job)) & methods.replicate.eq(0)
                               & methods.method.ne("feast"), "library_size_wasserstein"].iloc[0]
        np.testing.assert_allclose(wasserstein_distance(target_library, source_library[donor]), expected, rtol=1e-10)
        markers = {}
        for gene in MARKERS[job.dataset]:
            source_values, target_values = vector(source, gene), vector(target, gene)
            values = [np.log1p(source_values[sm]), np.log1p(target_values[tm]),
                      np.log1p(vector(generated, gene)), np.log1p(source_values[donor])]
            markers[gene] = values
            for method, ids, xy, data in zip(["source", "truth", "feast", "resampling_draw_0"],
                    [source.obs_names[sm], target.obs_names[tm], target.obs_names[tm], target.obs_names[tm]],
                    [sx[sm], tx[tm], tx[tm], tx[tm]], values):
                records.append(pd.DataFrame({"dataset": job.dataset, "direction": direction_id(job),
                    "gene": gene, "method": method, "spot_id": ids, "x": xy[:, 0], "y": xy[:, 1], "log1p_counts": data}))
        row = summary.loc[summary.direction.eq(direction_id(job)) & summary.primary_setting].iloc[0]
        cases.append(dict(job=job, row=row, sx=sx, tx=tx, sl=sl, tl=tl, sm=sm, tm=tm,
                          support=support, markers=markers))
        del generated
    pd.concat(records, ignore_index=True).to_csv(OUT / "spatial_plot_data.csv.gz", index=False)
    return cases, inputs


def context_page(cases, dataset, palette, pages):
    selected = [c for c in cases if c["row"].dataset == dataset]
    n = len(selected)
    fig = plt.figure(figsize=(16 if n > 2 else 9, 6))
    gs = fig.add_gridspec(2, n, left=0.095, right=0.985, bottom=0.2,
                         top=0.78, hspace=0.05, wspace=0.04)
    title(fig, f"{dataset.upper()}: source and target spatial context",
          "Known conditional labels · gray: excluded / unsupported spots")
    size = 3.8 if dataset == "dlpfc" else 1.0
    for col, case in enumerate(selected):
        for ri, prefix in enumerate(["s", "t"]):
            ax = fig.add_subplot(gs[ri, col])
            xy, labels, mask = case[prefix + "x"], case[prefix + "l"], case[prefix + "m"]
            ax.scatter(*xy.T, c="#DDDDDD", s=size, linewidths=0, rasterized=True)
            ax.scatter(*xy[mask].T, c=[palette[v] for v in labels[mask]],
                       s=size, linewidths=0, rasterized=True)
            map_axis(ax, xy)
            if ri == 0:
                ax.set_title(f"{short(case['row'].source)} → {short(case['row'].target)}",
                             weight="bold", fontsize=10, pad=10)
            else:
                ax.text(0.5, -0.04, f"{100 * case['support']['target_retained_fraction']:.1f}% retained",
                        transform=ax.transAxes, ha="center", fontsize=9)
            if col == 0:
                ax.text(-0.12, 0.5, ["Source", "Target"][ri], rotation=90,
                        transform=ax.transAxes, va="center", ha="center", weight="bold", fontsize=11)
    present = sorted({v for c in selected for v in c["sl"][c["sm"]]})
    handles = [Line2D([], [], marker="o", ls="", color=palette[v], label=v) for v in present]
    handles.append(Line2D([], [], marker="o", ls="", color="#DDDDDD", label="Unsupported"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.015),
               ncol=4, fontsize=8.5, columnspacing=1.2)
    footer(fig, "Source and target coordinates are shown in their native two-dimensional frames. "
           "Only labels with at least 20 source spots are retained. Target expression is withheld during generation. "
           "DLPFC 151675 ↔ 151676 is within donor; all directions involving 151670 are cross donor. "
           "MERFISH donor identity is unspecified. Colors remain consistent within each dataset.")
    save(fig, f"spatial_context_{dataset}", pages)


def spatial_matrix(columns, heading, stem, pages, scale_records):
    """Article layout: method rows and direction/gene columns, as in Study 00."""
    fig = plt.figure(figsize=(13.5, 10.8))
    gs = fig.add_gridspec(4, len(columns), left=0.10, right=0.985, bottom=0.1,
                         top=0.83, wspace=0.035, hspace=0.025)
    title(fig, heading, "AR = 0.3 · shared expression scale within each column · gray: unsupported spots")
    method_labels = ["Source\nreference", "Held-out\ntarget", "FEAST", "Conditional\nresampling"]
    for col, (case, gene) in enumerate(columns):
        row = case["row"]
        values = case["markers"][gene]
        vmax = float(max(np.max(v) for v in values))
        norm = mpl.colors.Normalize(0, vmax)
        scale_records.append(dict(dataset=row.dataset, direction=row.direction, gene=gene,
                                  vmin=0, vmax=vmax, transform="log1p(raw counts)", clipped=False))
        size = 4.0 if row.dataset == "dlpfc" else 1.0
        # Identical geometry across all three target rows; source keeps its own frame.
        for ri, value in enumerate(values):
            ax = fig.add_subplot(gs[ri, col])
            prefix = "s" if ri == 0 else "t"
            xy, mask = case[prefix + "x"], case[prefix + "m"]
            ax.scatter(*xy.T, c="#DDDDDD", s=size, linewidths=0, rasterized=True)
            artist = ax.scatter(*xy[mask].T, c=value, cmap="viridis", norm=norm,
                                s=size, linewidths=0, rasterized=True)
            map_axis(ax, xy)
            if ri == 0:
                bounds = gs[0, col].get_position(fig)
                center = bounds.x0 + bounds.width / 2
                fig.text(center, 0.89, f"{short(row.source)} → {short(row.target)}",
                         ha="center", weight="bold", fontsize=11)
                fig.add_artist(Line2D([center - bounds.width * 0.27, center + bounds.width * 0.27],
                                     [0.877, 0.877], transform=fig.transFigure, color="#777777", lw=0.8))
                fig.text(center, 0.85, gene, ha="center", fontsize=11, fontstyle="italic")
            if col == 0:
                ax.text(-0.16, 0.5, method_labels[ri], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=11,
                        weight="bold", color=BLUE if ri == 2 else TRUTH)
        bounds = gs[3, col].get_position(fig)
        cax = fig.add_axes([bounds.x0 + 0.13 * bounds.width, 0.055, 0.74 * bounds.width, 0.010])
        cb = fig.colorbar(artist, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=8, length=2)
        cb.set_label("log1p(counts)", fontsize=8, labelpad=1)
    footer(fig, "All maps use the primary assignment randomness 0.3. One shared, unclipped expression scale per column "
           "covers source, held-out target, FEAST and resampling. Scales vary between columns. "
           "SCGB2A2 / Igfbp4 are existing half-slice markers; GFAP / Frzb were previously selected best-case markers. "
           "All eight directions are included; these genes are illustrations, not a typical-gene accuracy estimate. "
           "Conditional resampling uses the original seed-2026 draw 0; quantitative panels show all ten draws. "
           "The target geometry and labels are known inputs, and target expression is used only for evaluation.")
    save(fig, stem, pages)


def expression_distribution_page(cases, dataset, pages):
    selected = [c for c in cases if c["row"].dataset == dataset]
    fig, axes = plt.subplots(2, len(selected), figsize=(15 if len(selected) > 2 else 8, 5.6), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.81, bottom=0.17, wspace=0.28, hspace=0.44)
    title(fig, f"{dataset.upper()}: marker expression distributions",
          "Selected marker genes · supported target spots · empirical cumulative distributions")
    for col, case in enumerate(selected):
        for ri, (gene, values) in enumerate(case["markers"].items()):
            ax = axes[ri, col]
            for val, color, heading, ls in zip(values[1:], [TRUTH, BLUE, GRAY],
                    ["Held-out truth", "FEAST", "Resampling draw 0"], ["-", "-", "--"]):
                unique, counts = np.unique(val, return_counts=True)
                ax.step(unique, np.cumsum(counts) / len(val), where="post", color=color,
                        label=heading, lw=1.6, linestyle=ls, zorder=3 if heading == "FEAST" else 2)
            ax.set_ylim(0, 1.03)
            if ri == 1:
                ax.set_xlabel("log1p(counts)", fontsize=9)
            if ri == 0:
                ax.set_title(f"{short(case['row'].source)} → {short(case['row'].target)}", fontsize=10, weight="bold")
            if col == 0:
                ax.set_ylabel(f"{gene}\nCumulative fraction", fontsize=10, weight="bold")
            else:
                ax.tick_params(labelleft=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=9)
    footer(fig, "These are marginal distributions for the displayed genes; the direction-resolved metric pages report the existing label-conditioned metrics.")
    save(fig, f"marker_distributions_{dataset}", pages)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.linewidth": 1.1, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.edgecolor": "#444444",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    config = load_config(STUDY / "config.yaml")
    score_root = STUDY / "outputs/scores"
    metric_root = STUDY / "outputs/baselines/conditional_metric_comparison"
    summary = pd.read_csv(score_root / "summary.csv", dtype={"source": str, "target": str})
    support = pd.read_csv(score_root / "support_audit.csv", dtype={"source": str, "target": str})
    comparison = pd.read_csv(metric_root / "feast_comparison.csv", dtype={"source": str, "target": str})
    methods = pd.read_csv(metric_root / "method_metrics.csv", dtype={"source": str, "target": str})
    assert len(summary) == 45 and summary.job_id.is_unique
    assert json.loads((score_root / "provenance.json").read_text())["status"] == "ok"
    assert json.loads((metric_root / "provenance.json").read_text())["status"] == "complete"
    tasks = order_tasks(summary)
    assert len(tasks) == 13
    pages = [VIS / "figures/conditional_transfer_simulation_design.pdf"]
    metric_page(comparison, tasks, "cross_slice", pages)
    cases, inputs = load_spatial(config, summary, methods)
    assert len(cases) == 8
    palettes = {}
    for dataset, spec in config["datasets"].items():
        labels = sorted({str(v) for (ds, _), a in inputs.items() if ds == dataset for v in a.obs[spec["annotation_key"]]})
        palettes[dataset] = {v: mpl.colors.to_hex(mpl.colormaps["tab20"](i % 20)) for i, v in enumerate(labels)}
    scales = []
    by_direction = {c["row"].direction: c for c in cases}
    cases = [by_direction[d] for d in tasks.loc[tasks["mode"].eq("cross_slice"), "direction"]]
    for dataset in config["datasets"]:
        context_page(cases, dataset, palettes[dataset], pages)
    within = [c for c in cases if c["row"].donor_stratum == "within_donor"]
    cross = [c for c in cases if c["row"].donor_stratum == "cross_donor"]
    merfish = [c for c in cases if c["row"].dataset == "merfish"]
    spatial_matrix([(c, g) for g in MARKERS["dlpfc"] for c in within],
                   "DLPFC: within-donor cross-slice simulation", "cross_slice_within_donor", pages, scales)
    for gene in MARKERS["dlpfc"]:
        spatial_matrix([(c, gene) for c in cross], "DLPFC: cross-donor cross-slice simulation",
                       f"cross_slice_cross_donor_{gene}", pages, scales)
    spatial_matrix([(c, g) for g in MARKERS["merfish"] for c in merfish],
                   "MERFISH: cross-slice simulation", "cross_slice_merfish", pages, scales)
    for dataset in config["datasets"]:
        expression_distribution_page(cases, dataset, pages)
    support_page(tasks, support, pages)
    sensitivity_page(summary, pages)
    qc_page(methods, tasks, pages)
    pages.append(VIS / "figures/conditional_transfer_expression_matrix.pdf")
    metric_page(comparison, tasks, "mask_half", pages)
    pd.DataFrame(scales).to_csv(OUT / "expression_scales.csv", index=False)
    comparison.to_csv(OUT / "metric_plot_data.csv", index=False)
    summary.to_csv(OUT / "sensitivity_plot_data.csv", index=False)
    support.to_csv(OUT / "support_plot_data.csv", index=False)
    methods.to_csv(OUT / "count_diagnostic_plot_data.csv", index=False)
    combined = OUT / "study05_expanded_visualization.pdf"
    subprocess.run(["pdfunite", *map(str, pages), str(combined)], check=True)
    provenance = dict(entry_point=str(Path(__file__).relative_to(VIS.parents[1])),
        config=str(STUDY / "config.yaml"), score_inputs=str(score_root), baseline_inputs=str(metric_root),
        primary_assignment_randomness=0.3, seed=2026, cross_slice_directions=8,
        markers=MARKERS, marker_selection="Reuse four existing figure markers; no new outcome-based selection",
        baseline_display="Original seed 2026 draw 0, preserving full primary task RNG sequence",
        baseline_check="Reconstructed draw library-size Wasserstein matches existing replicate-0 score in all eight directions",
        expression_scale="Per direction and gene: zero to maximum log1p count across all four displayed methods; no clipping",
        label_palette=palettes, pages=[p.name for p in pages], combined_pdf=combined.name,
        style_references=["visualization/00_simulator_benchmark/plot_spatial_comparison.py",
                          "visualization/01_clustering/plot_main_figure.py"],
        export=dict(dpi=600, pdf_fonttype=42, svg_text="editable", spatial_points="rasterized"),
        limitations=["Known target labels and coordinates are supplied", "Selected genes are illustrative",
                     "AR sweep uses one FEAST seed", "Directions share slices", "Unsupported target spots excluded"])
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "captions.md").write_text("# Study 05 figure captions\n\n" + "\n\n".join(
        f"## {stem}\n\n{caption}" for stem, caption in CAPTIONS) + "\n")
    (OUT / "README.md").write_text("# Study 05 article-style figures\n\n"
        "Combined PDF: `study05_expanded_visualization.pdf`. Existing schematic and half-slice matrix are included unchanged.\n\n"
        + "\n".join(f"{i}. `{p.name}`" for i, p in enumerate(pages, 1))
        + "\n\nRebuild from the repository root: `python visualization/05_2d_conditional_transfer/plot_expanded_report.py`.\n"
        "Requires the existing Python analysis environment and Poppler's `pdfunite`. No FEAST simulations are rerun.\n"
        "Style follows Study 00 spatial matrices and Study 01 compact panels: DejaVu Sans, bold panel headings, "
        "blue FEAST, gray resampling, viridis expression and external legends.\n"
        "Each new figure has PDF, editable SVG and 600-DPI PNG exports. Dense spatial points are rasterized at 600 DPI.\n"
        "Full methodological notes and interpretation limits are in `captions.md`.\n"
        "Plotting tables and provenance are stored beside the PDF. The previous expanded report remains in `../expanded/`.\n")
    print(f"Wrote {len(pages)} pages to {combined}", flush=True)


if __name__ == "__main__":
    main()
