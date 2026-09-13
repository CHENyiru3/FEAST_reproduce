"""Select a compact 4 × 4 cross-slice figure from the existing article atlas."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_expanded_report import MARKERS, STUDY, VIS, map_axis, short

OUT = VIS / "figures/article_style"
STEM = "cross_slice_selected_4x4"
SELECTION_RULE = (
    "Use only the four existing atlas markers at primary AR 0.3. Select the "
    "highest saved gene-wise Pearson agreement for GFAP within DLPFC donor, "
    "SCGB2A2 across DLPFC donors, and Frzb in MERFISH. Use Igfbp4 in the "
    "opposite MERFISH direction to retain both directions and both markers. "
    "This is an editorial selection of strong examples, not typical-gene performance."
)


def select_columns():
    parts = []
    for frame in pd.read_csv(STUDY / "outputs/scores/per_gene_metrics.csv", chunksize=100_000,
                             dtype={"source": str, "target": str}):
        keep = frame["mode"].eq("cross_slice") & frame.assignment_randomness.eq(0.3)
        keep &= ((frame.dataset.eq("dlpfc") & frame.gene.isin(MARKERS["dlpfc"]))
                 | (frame.dataset.eq("merfish") & frame.gene.isin(MARKERS["merfish"])))
        parts.append(frame.loc[keep])
    candidates = pd.concat(parts, ignore_index=True)
    assert len(candidates) == 16

    def best(dataset, gene, stratum=None):
        rows = candidates.loc[candidates.dataset.eq(dataset) & candidates.gene.eq(gene)]
        if stratum is not None:
            rows = rows.loc[rows.donor_stratum.eq(stratum)]
        return rows.sort_values(["pearson", "direction"], ascending=[False, True]).iloc[0]

    within = best("dlpfc", "GFAP", "within_donor")
    cross = best("dlpfc", "SCGB2A2", "cross_donor")
    merfish = best("merfish", "Frzb")
    reverse = candidates.loc[candidates.dataset.eq("merfish") & candidates.gene.eq("Igfbp4")
                             & candidates.source.eq(merfish.target) & candidates.target.eq(merfish.source)]
    assert len(reverse) == 1
    selected = pd.DataFrame([within, cross, merfish, reverse.iloc[0]]).reset_index(drop=True)
    selected.insert(0, "column", list("ABCD"))
    return selected, candidates


def load_columns(selected):
    points = pd.read_csv(OUT / "spatial_plot_data.csv.gz", dtype={"spot_id": str})
    scales = pd.read_csv(OUT / "expression_scales.csv")
    columns = []
    displayed = []
    for _, row in selected.iterrows():
        subset = points.loc[points.direction.eq(row.direction) & points.gene.eq(row.gene)]
        methods = [subset.loc[subset.method.eq(name)] for name in
                   ["source", "truth", "feast", "resampling_draw_0"]]
        for frame in methods[2:]:
            assert np.array_equal(frame.spot_id.to_numpy(), methods[1].spot_id.to_numpy())
            np.testing.assert_array_equal(frame[["x", "y"]], methods[1][["x", "y"]])
        xy, masks = [], []
        for slice_id, frame in zip([row.source, row.target], methods[:2]):
            path = STUDY / "data/local" / row.dataset / f"{slice_id}.h5ad"
            data = ad.read_h5ad(path, backed="r")
            try:
                coords = np.asarray(data.obsm["spatial"])[:, :2]
                ids = data.obs_names.astype(str)
                mask = ids.isin(frame.spot_id)
                assert list(ids[mask]) == frame.spot_id.tolist()
                np.testing.assert_allclose(coords[mask], frame[["x", "y"]], rtol=0, atol=1e-8)
                xy.append(coords)
                masks.append(mask)
            finally:
                data.file.close()
        scale = scales.loc[scales.direction.eq(row.direction) & scales.gene.eq(row.gene)]
        assert len(scale) == 1
        columns.append(dict(row=row, xy=xy, masks=masks,
                            values=[frame.log1p_counts.to_numpy() for frame in methods],
                            vmax=float(scale.iloc[0].vmax), target_fraction=float(masks[1].mean())))
        displayed.append(subset.assign(column=row.column))
    pd.concat(displayed, ignore_index=True).to_csv(OUT / f"{STEM}_plot_data.csv.gz", index=False)
    return columns


def draw(columns):
    shared_vmax = max(float(np.max(values)) for case in columns for values in case["values"])
    for filename in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        font_manager.fontManager.addfont(Path(sys.prefix) / "fonts" / filename)
    mpl.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 10.2))
    grid = fig.add_gridspec(4, 4, left=0.08, right=0.985, bottom=0.12, top=0.82,
                           width_ratios=[1, 1, 1.4, 1.4], wspace=0.06, hspace=0.035)
    fig.suptitle("Cross-slice conditional simulation", y=0.995, fontsize=17, weight="bold")
    fig.text(0.5, 0.957, "Selected marker examples · primary AR = 0.3 · shared expression scale across all panels",
             ha="center", fontsize=10, color="#555555")
    row_labels = ["Source\nreference", "Held-out\ntarget", "FEAST", "Conditional\nresampling"]
    for col, case in enumerate(columns):
        row = case["row"]
        bounds = grid[0, col].get_position(fig)
        center = bounds.x0 + bounds.width / 2
        dataset = "DLPFC" if row.dataset == "dlpfc" else "MERFISH"
        detail = "within donor" if row.donor_stratum == "within_donor" else (
            "cross donor" if row.donor_stratum == "cross_donor" else "cross slice")
        fig.text(center, 0.911, f"{row.column}  {dataset} · {detail}", ha="center", weight="bold", fontsize=10.5)
        fig.text(center, 0.885, f"{short(row.source)} → {short(row.target)}", ha="center", fontsize=10)
        fig.add_artist(Line2D([center - bounds.width * 0.29, center + bounds.width * 0.29],
                             [0.873, 0.873], transform=fig.transFigure, color="#888888", lw=0.8))
        fig.text(center, 0.847, row.gene, ha="center", fontstyle="italic", fontsize=11.5)
        size = 3.6 if row.dataset == "dlpfc" else 1.0
        for ri, values in enumerate(case["values"]):
            ax = fig.add_subplot(grid[ri, col])
            index = 0 if ri == 0 else 1
            xy, mask = case["xy"][index], case["masks"][index]
            ax.scatter(*xy.T, c="#DDDDDD", s=size, linewidths=0, rasterized=True)
            artist = ax.scatter(*xy[mask].T, c=values, cmap="viridis", vmin=0,
                                vmax=shared_vmax, s=size, linewidths=0, rasterized=True)
            map_axis(ax, xy)
            if col == 0:
                ax.text(-0.15, 0.5, row_labels[ri], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", weight="bold", fontsize=11,
                        color="#0072B2" if ri == 2 else "#333333")
    cax = fig.add_axes([0.34, 0.078, 0.38, 0.014])
    colorbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=9, length=2)
    colorbar.set_label("Expression · log1p(counts)", fontsize=9.5, labelpad=2)
    fig.text(0.5, 0.015, "Gray: unsupported locations. Conditional resampling: fixed draw 0. Selected examples; see caption for selection criteria.",
             ha="center", fontsize=8.5, color="#555555")
    for suffix in ("pdf", "svg", "png"):
        metadata = ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else
                    {"Date": None} if suffix == "svg" else {})
        path = OUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
        print(f"Saved {path}", flush=True)
    plt.close(fig)
    return shared_vmax


def main():
    selected, candidates = select_columns()
    columns = load_columns(selected)
    selected["target_retained_fraction"] = [c["target_fraction"] for c in columns]
    selected.to_csv(OUT / f"{STEM}_selection.csv", index=False)
    candidates.to_csv(OUT / f"{STEM}_candidates.csv", index=False)
    shared_vmax = draw(columns)
    descriptions = [f"- **{r.column}: {r.dataset.upper()} {short(r.source)} → {short(r.target)}, {r.gene}.** "
                    f"Saved raw-count gene Pearson r = {r.pearson:.3f}; retained target coverage = {100*r.target_retained_fraction:.1f}%."
                    for _, r in selected.iterrows()]
    caption = ("# Selected 4 × 4 cross-slice simulation panel\n\n"
        "Rows show source reference, held-out target truth, FEAST simulation, and label-conditioned whole-spot resampling. "
        "Columns cover DLPFC within-donor transfer, DLPFC cross-donor transfer, and both MERFISH directions. "
        "All examples use primary AR = 0.3.\n\n" + "\n".join(descriptions) + "\n\n"
        + SELECTION_RULE + "\n\n"
        f"One shared expression colorbar spans 0 to {shared_vmax:.6f} log1p(counts), the maximum across all 16 panels. "
        "All panels use this common unclipped scale; expression values are unchanged. Gray spots are outside source-supported labels; "
        "column B excludes target Layers 1–2. Target labels and coordinates are supplied to generation; "
        "target expression is used only for evaluation. The baseline is the already-verified seed-2026 draw 0 "
        "from the original task sequence. A single draw is illustrative; the full atlas reports ten-draw baseline ranges. "
        "MERFISH donor identity is not declared.\n\n"
        "The complete eight-direction atlas and quantitative panels remain the evidence for overall performance. "
        "Selection used held-out agreement for presentation and does not establish typical-gene recovery.\n\n"
        "Rebuild: `python visualization/05_2d_conditional_transfer/plot_selected_cross_slice.py`.\n")
    (OUT / f"{STEM}_caption.md").write_text(caption)
    provenance = dict(entry_point=str(Path(__file__).relative_to(VIS.parents[1])), selection_rule=SELECTION_RULE,
        source_atlas="spatial_plot_data.csv.gz", source_scales="expression_scales.csv",
        metric_input="05_2d_conditional_transfer/outputs/scores/per_gene_metrics.csv",
        assignment_randomness=0.3, baseline_seed=2026, baseline_draw=0,
        layout="4 method rows × 4 direction/gene columns", font="Arial", dpi=600,
        shared_expression_scale=dict(vmin=0, vmax=shared_vmax, transform="log1p(raw counts)", clipped=False),
        colorbar_count=1,
        selected=selected[["dataset", "direction", "gene"]].to_dict("records"))
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
