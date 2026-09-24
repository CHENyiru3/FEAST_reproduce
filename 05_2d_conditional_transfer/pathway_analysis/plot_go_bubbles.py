"""Compact paired GO GSEA bubble plots from the completed GSEApy 1.3.1 run."""
from pathlib import Path
import json
import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors, ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_multilevel_gseapy131"
OUTPUT = HERE / "figures_gseapy131" / "bubble_plots"
TOP_N = 3
FDR = 0.05
COLOR_CAP = 50
METHODS = ["real", "feast"]
METHOD_TITLES = {"real": "Real", "feast": "FEAST simulated"}


def configure_style():
    for name in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        font_path = Path("/maiziezhou_lab2/yiru/miniconda3/fonts") / name
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 9.5, "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "xtick.major.width": .7, "ytick.major.width": .7,
    })


def load_plot_data():
    inventory = pd.read_csv(RESULTS / "ranking_inventory.csv")
    inventory = inventory.loc[inventory.method.isin(METHODS)].copy()
    rows, selection, contexts = [], [], []
    keys = ["dataset", "mode", "direction", "source", "target", "donor_stratum", "label", "n_label_spots"]
    for _, group in inventory.groupby(["dataset", "direction", "label"], sort=False):
        identity = {key: group.iloc[0][key] for key in keys}
        frames, picks = {}, {}
        for method in METHODS:
            meta = group.loc[group.method.eq(method)]
            assert len(meta) == 1
            frame = pd.read_csv(meta.iloc[0].result_file).set_index("term")
            assert frame.index.is_unique
            frames[method] = frame
            positive = frame.loc[frame.nes.gt(0) & frame.gsea_fdr_q.lt(FDR)]
            picks[method] = positive.reset_index().sort_values(
                ["gsea_fdr_q", "nes", "term"], ascending=[True, False, True]
            ).head(TOP_N).term.tolist()
        assert frames["real"].index.equals(frames["feast"].index)
        terms = list(dict.fromkeys(picks["real"] + picks["feast"]))
        contexts.append(dict(**identity, n_selected_terms=len(terms)))
        for order, term in enumerate(terms):
            selected_by = ";".join(method for method in picks if term in picks[method])
            selection.append(dict(**identity, term=term, order=order, selected_by=selected_by))
            for method, frame in frames.items():
                result = frame.loc[term]
                genes = [] if pd.isna(result.leading_edge_genes) else result.leading_edge_genes.split(";")
                tag_count = int(result.tag_fraction.split("/")[0])
                assert len(genes) == tag_count
                valid = bool(np.isfinite(result.nes) and np.isfinite(result.gsea_fdr_q))
                rows.append(dict(**identity, term=term, order=order, selected_by=selected_by,
                    method=method, nes=result.nes, gsea_fdr_q=result.gsea_fdr_q,
                    leading_edge_count=len(genes), leading_edge_genes=result.leading_edge_genes,
                    available=valid, significant=valid and result.gsea_fdr_q < FDR,
                    color_value=min(-np.log10(max(result.gsea_fdr_q, 1e-300)), COLOR_CAP) if valid else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(selection), pd.DataFrame(contexts)


def dataset_scales(data):
    scales = {}
    for dataset, group in data.groupby("dataset", sort=False):
        finite_nes = group.loc[np.isfinite(group.nes), "nes"]
        lower = np.floor(min(0, finite_nes.min()) * 2) / 2
        upper = np.ceil(max(0, finite_nes.max()) * 2) / 2
        count_limit = max(10, int(np.ceil(group.leading_edge_count.max() / 10) * 10))
        finite_color = group.loc[np.isfinite(group.color_value), "color_value"]
        max_color = finite_color.max() if len(finite_color) else 5
        color_limit = min(COLOR_CAP, max(5, int(np.ceil(max_color / 5) * 5)))
        scales[dataset] = dict(nes_min=float(lower), nes_max=float(upper),
                               count_limit=count_limit, color_limit=color_limit)
    return scales


def bubble_area(count, scale):
    # A visible minimum preserves small gene sets; the legend uses this exact mapping.
    return 18 + 210 * np.asarray(count, dtype=float) / scale["count_limit"]


def format_go_term(term):
    return str(term).removeprefix("GOBP_").replace("_", " ").lower()


def wrapped_go_term(term, width=51):
    return textwrap.fill(format_go_term(term), width=width,
                         break_long_words=False, break_on_hyphens=False)


def size_legend_values(scale):
    maximum = scale["count_limit"]
    if maximum <= 30:
        candidates = [5, 15, 30]
    elif maximum <= 60:
        candidates = [10, 30, 60]
    elif maximum <= 120:
        candidates = [25, 60, 120]
    elif maximum <= 300:
        candidates = [50, 150, 300]
    else:
        candidates = [int(round(maximum * .25 / 10) * 10),
                      int(round(maximum * .50 / 10) * 10), maximum]
    values = sorted({int(x) for x in candidates if 0 < x <= maximum})
    if maximum not in values:
        values.append(maximum)
    if len(values) > 3:
        values = [values[0], values[len(values) // 2], values[-1]]
    return values


def task_title(meta):
    source = str(meta.source).replace("Zhuang-ABCA-1.", "")
    target = str(meta.target).replace("Zhuang-ABCA-1.", "")
    if meta["mode"] == "cross_slice":
        title = f"{meta.dataset.upper()}  |  {source} → {target}"
        detail = "Cross-slice simulation"
        if meta.dataset == "dlpfc":
            detail += " · " + meta.donor_stratum.replace("_", " ")
    else:
        title = f"{meta.dataset.upper()}  |  {source}"
        detail = "Half-held-out simulation · evaluation on held-out half"
    return title, detail


def plot_task(data, selection, contexts, scale, bundle):
    meta = contexts.iloc[0]
    layout, headers, empty = [], [], []
    cursor = 0
    for _, context in contexts.iterrows():
        label = context.label.replace("Layer_", "Layer ")
        headers.append((cursor, f"{label}   ·   {context.n_label_spots:,} spots"))
        cursor += .62
        chosen = selection.loc[selection.label.eq(context.label)].sort_values("order")
        if chosen.empty:
            empty.append((cursor, "No positive GO terms at FDR < 0.05 in either method"))
            cursor += .75
        for _, term in chosen.iterrows():
            wrapped = wrapped_go_term(term.term, width=51)
            n_lines = len(wrapped.splitlines())
            row_height = .78 if n_lines <= 1 else 1.02 if n_lines == 2 else 1.26
            layout.append((context.label, term.term, cursor + row_height / 2, wrapped))
            cursor += row_height
        cursor += .34

    # Small extra footer allowance prevents colorbar/note collisions.
    height = max(5.8, cursor * .235 + 2.33)
    fig = plt.figure(figsize=(12.4, height))
    bottom, top = 1.40 / height, 1 - .82 / height
    label_ax = fig.add_axes([.035, bottom, .455, top - bottom])
    axes = [fig.add_axes([x, bottom, .205, top - bottom]) for x in [.505, .755]]
    for ax in [label_ax] + axes:
        ax.set_ylim(cursor - .12, -.45)
    label_ax.set_xlim(0, 1)
    label_ax.axis("off")
    for y, title in headers:
        label_ax.text(0, y, title, va="center", fontsize=9.3, weight="bold", color="#333333")
        for ax in axes:
            ax.axhspan(y - .24, y + .24, color="#F4F4F4", zorder=0, linewidth=0)
    for _, _, y, name in layout:
        label_ax.text(.012, y, name, va="center", fontsize=8.25, color="#222222", linespacing=1.08)
    for y, title in empty:
        label_ax.text(.012, y, title, va="center", fontsize=8, color="#777777")

    norm = colors.Normalize(0, scale["color_limit"])
    cmap = matplotlib.colormaps["YlOrRd"]
    positions = {(label, term): y for label, term, y, _ in layout}
    for ax, method in zip(axes, METHODS):
        frame = data.loc[data.method.eq(method)].copy()
        frame["y"] = [positions[(row.label, row.term)] for _, row in frame.iterrows()]
        ax.set_title(METHOD_TITLES[method], fontsize=11.5, weight="bold", pad=9, color="#333333")
        available = frame.loc[frame.available]
        if len(available):
            ax.scatter(available.nes, available.y,
                s=bubble_area(available.leading_edge_count, scale),
                c=available.color_value, cmap=cmap, norm=norm,
                edgecolors="#3F3F3F", linewidths=.42, alpha=.96, zorder=3)
        for y in frame.loc[~frame.available, "y"]:
            ax.text(.98, y, "NA", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=7.5, color="#888888")
        ax.set_xlim(scale["nes_min"] - .12, scale["nes_max"] + .12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune=None))
        ax.axvline(0, lw=.7, color="#999999", linestyle="--", zorder=1)
        ax.grid(axis="x", color="#EAEAEA", lw=.55, zorder=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(.7)
        ax.set_xlabel("Normalized enrichment score (NES)", fontsize=8.4, labelpad=6)
        ax.tick_params(axis="x", labelsize=8.1, length=3, width=.6)

    title, detail = task_title(meta)
    fig.text(.035, 1 - .17 / height, title, fontsize=15.5, weight="bold", va="top")
    fig.text(.035, 1 - .46 / height,
             detail + " · GO Biological Process · AR = 0.3", fontsize=9, color="#555555", va="top")

    legend_ax = fig.add_axes([.035, .64 / height, .46, .34 / height])
    legend_ax.axis("off")
    counts = size_legend_values(scale)
    handles = [legend_ax.scatter([], [], s=bubble_area(n, scale),
                facecolor="#B8B8B8", edgecolors="#404040", linewidths=.42) for n in counts]
    legend_ax.legend(handles, [str(n) for n in counts], title="Leading-edge gene count",
                     loc="center left", ncol=len(counts), fontsize=8.2, title_fontsize=8.2,
                     handletextpad=.55, columnspacing=1.45, borderaxespad=0)
    cax = fig.add_axes([.64, .88 / height, .30, .075 / height])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("−log10(FDR q-value)", fontsize=8.2, labelpad=2)
    fdr_cutoff = -np.log10(FDR)
    unique_ticks = []
    for value in [0., fdr_cutoff, scale["color_limit"]]:
        if not any(np.isclose(value, existing, atol=.05) for existing in unique_ticks):
            unique_ticks.append(value)
    cb.set_ticks(unique_ticks)
    cb.set_ticklabels([
        "1.30\nFDR=.05" if np.isclose(value, fdr_cutoff, atol=.05)
        else "≥50" if scale["color_limit"] == COLOR_CAP and np.isclose(value, COLOR_CAP)
        else f"{value:g}" for value in unique_ticks
    ])
    # Near-zero ticks need opposing alignment to keep their text separate.
    cb.ax.get_xticklabels()[0].set_ha("right")
    cb.ax.get_xticklabels()[1].set_ha("left")
    cb.ax.tick_params(labelsize=7.3, length=2, width=.5)
    cb.outline.set_linewidth(.45)
    footer = fig.text(.035, .08 / height,
        f"Rows are the union of the top {TOP_N} positively enriched GO terms selected separately "
        "from Real and FEAST within each layer/class (FDR < 0.05). Identical rows are shown in both panels.\n"
        "Bubble area indicates leading-edge gene count; color indicates FDR continuously. "
        "NES < 0 indicates enrichment in the opposite group. NA: unavailable estimate.",
        fontsize=7.15, color="#606060", va="bottom", linespacing=1.3)

    # Detect page-edge clipping before exporting; no analysis values are altered.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in label_ax.texts:
        box = text.get_window_extent(renderer)
        assert box.x1 < axes[0].bbox.x0 - 2, f"Term/header label overlaps plot: {text.get_text()}"
    assert footer.get_window_extent(renderer).x1 < fig.bbox.x1, "Footer extends beyond page"
    assert footer.get_window_extent(renderer).y1 < cb.ax.xaxis.label.get_window_extent(renderer).y0, "Footer overlaps colorbar label"
    stem = f"go_bubbles_{meta.dataset}_{meta.direction}"
    bundle.savefig(fig, bbox_inches="tight")
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}: {len(contexts)} layers/classes, {len(layout)} term rows", flush=True)


def main():
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, selection, contexts = load_plot_data()
    scales = dataset_scales(data)
    data.to_csv(OUTPUT / "bubble_plot_data.csv", index=False)
    selection.to_csv(OUTPUT / "term_selection.csv", index=False)
    contexts.to_csv(OUTPUT / "layer_class_inventory.csv", index=False)
    with PdfPages(OUTPUT / "study05_go_enrichment_bubbles_real_vs_FEAST.pdf") as bundle:
        for (dataset, direction), group in contexts.groupby(["dataset", "direction"], sort=False):
            plot_task(data.loc[data.dataset.eq(dataset) & data.direction.eq(direction)],
                selection.loc[selection.dataset.eq(dataset) & selection.direction.eq(direction)],
                group, scales[dataset], bundle)
    (OUTPUT / "plot_settings.json").write_text(json.dumps(dict(
        source_results=str(RESULTS), entry_point=str(Path(__file__).resolve()),
        methods=METHODS, selection=f"Union of top {TOP_N} positive terms per method and label; q ascending, NES descending, term ascending",
        fdr=FDR, color_cap=COLOR_CAP, scales=scales, font="Arial", png_dpi=600,
        visual_encoding=dict(row="GO Biological Process", section="layer/class",
            column="Real versus FEAST", x="normalized enrichment score",
            bubble_area="leading-edge gene count", bubble_color="-log10(FDR q-value)"),
        bubble_area_points_squared="18 + 210 * leading_edge_count / dataset_count_limit",
        significance_display="Continuous color only; FDR=0.05 marked on colorbar",
        tasks=int(contexts.groupby(["dataset", "direction"]).ngroups),
        layer_class_contexts=len(contexts), term_rows=len(selection)), indent=2) + "\n")
    print(f"Saved combined PDF: {OUTPUT / 'study05_go_enrichment_bubbles_real_vs_FEAST.pdf'}", flush=True)


if __name__ == "__main__":
    main()
