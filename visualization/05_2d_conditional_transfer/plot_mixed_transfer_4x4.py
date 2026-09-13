"""Combine cross-slice and half-held-out examples in one 4 × 4 figure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from plot_expanded_report import STUDY, VIS, map_axis, short, vector

sys.path.insert(0, str(STUDY))
from conditional_resampling_baseline import conditional_donor_indices, selected_jobs
from workflow import direction_id, expected_job_path, load_config

OUT = VIS / "figures/article_style"
STEM = "conditional_transfer_cross_and_half_4x4"
SPECS = [("dlpfc", "cross_slice", "GFAP", "within_donor"),
         ("dlpfc", "mask_half", "MBP", "within_slice"),
         ("merfish", "cross_slice", "Frzb", "donor_not_declared"),
         ("merfish", "mask_half", "Igfbp4", "within_slice")]
SELECTION_RULE = (
    "Four columns cover dataset × task: DLPFC cross-slice GFAP (within donor), "
    "DLPFC half-held-out MBP, MERFISH cross-slice Frzb, and MERFISH half-held-out Igfbp4. "
    "At AR 0.3, retain the prior GFAP, Frzb and Igfbp4 selections based on highest saved "
    "raw-count gene Pearson agreement within their specified groups. For the DLPFC half-held-out "
    "column, keep slice 151676 and replace SCGB2A2 with MBP following visual comparison of "
    "SCGB2A2, MBP, TUBA1B and MT-CO2 on that same held-out target. MBP provides a clearer spatial gradient. "
    "These are selected examples, not estimates of typical performance."
)


def select_examples():
    parts = []
    for frame in pd.read_csv(STUDY / "outputs/scores/per_gene_metrics.csv", chunksize=100_000,
                             dtype={"source": str, "target": str}):
        parts.append(frame.loc[frame.assignment_randomness.eq(0.3)
                               & frame.gene.isin([spec[2] for spec in SPECS])])
    candidates = pd.concat(parts, ignore_index=True)
    selected = []
    eligible = []
    for dataset, mode, gene, stratum in SPECS:
        rows = candidates.loc[candidates.dataset.eq(dataset) & candidates["mode"].eq(mode)
                              & candidates.gene.eq(gene) & candidates.donor_stratum.eq(stratum)]
        if dataset == "dlpfc" and mode == "mask_half":
            rows = rows.loc[rows.source.eq("151676")]
        eligible.append(rows)
        selected.append(rows.sort_values(["pearson", "direction"], ascending=[False, True]).iloc[0])
    selected = pd.DataFrame(selected).reset_index(drop=True)
    selected.insert(0, "column", list("ABCD"))
    return selected, pd.concat(eligible, ignore_index=True)


def load_examples(selected):
    config = load_config(STUDY / "config.yaml")
    inputs = {(dataset, str(slice_id)): ad.read_h5ad(STUDY / "data/local" / dataset / f"{slice_id}.h5ad")
              for dataset, spec in config["datasets"].items() for slice_id in spec["slices"]}
    baseline = pd.read_csv(STUDY / "outputs/baselines/conditional_metric_comparison/method_metrics.csv")
    rng = np.random.default_rng(config["public_seed"])
    by_direction = {r.direction: r for _, r in selected.iterrows()}
    cases, plot_frames = {}, []
    # Preserve the evaluator's entire primary-task RNG sequence, including tasks
    # that are not displayed. This reproduces the existing baseline draw 0.
    for job in selected_jobs(config):
        name = direction_id(job)
        source, target = inputs[job.dataset, job.source], inputs[job.dataset, job.target]
        key = config["datasets"][job.dataset]["annotation_key"]
        path = expected_job_path(STUDY / "outputs/final", job)
        provenance = json.loads((path / "provenance.json").read_text())
        assert provenance["target_expression_loaded_by_generation"] is False
        support = provenance["support"]
        sx, tx = np.asarray(source.obsm["spatial"])[:, :2], np.asarray(target.obsm["spatial"])[:, :2]
        sl, tl = source.obs[key].astype(str).to_numpy(), target.obs[key].astype(str).to_numpy()
        sm, tm = np.isin(sl, support["eligible_labels"]), np.isin(tl, support["eligible_labels"])
        split = None
        if job.mode == "mask_half":
            split = float(np.median(sx[:, 0]))
            sm &= sx[:, 0] <= split
            tm &= tx[:, 0] > split
            assert not np.any(sm & tm)
        assert sm.sum() == support["source_retained_spots"]
        assert tm.sum() == support["target_retained_spots"]
        donor = np.flatnonzero(sm)[conditional_donor_indices(sl[sm], tl[tm], rng)]
        if name not in by_direction:
            continue
        row = by_direction[name]
        generated = ad.read_h5ad(path / "generated.h5ad")
        assert list(generated.obs_names) == list(target.obs_names[tm])
        np.testing.assert_allclose(generated.obsm["spatial"], tx[tm], rtol=0, atol=0)
        assert np.array_equal(generated.obs[key].astype(str).to_numpy(), tl[tm])
        panel = list(generated.var_names)
        source_library = np.asarray(source[:, panel].layers["counts"].sum(axis=1)).ravel()
        target_library = np.asarray(target[tm, panel].layers["counts"].sum(axis=1)).ravel()
        expected = baseline.loc[baseline.direction.eq(name) & baseline.replicate.eq(0)
                                & baseline.method.ne("feast"), "library_size_wasserstein"].iloc[0]
        np.testing.assert_allclose(wasserstein_distance(target_library, source_library[donor]), expected, rtol=1e-10)
        source_gene = vector(source, row.gene)
        values = [np.log1p(source_gene[sm]), np.log1p(vector(target, row.gene)[tm]),
                  np.log1p(vector(generated, row.gene)), np.log1p(source_gene[donor])]
        cases[name] = dict(row=row, xy=[sx, tx], masks=[sm, tm], values=values, split=split,
                           support=support)
        for method, ids, xy, value in zip(["source", "truth", "feast", "resampling_draw_0"],
                [source.obs_names[sm], target.obs_names[tm], target.obs_names[tm], target.obs_names[tm]],
                [sx[sm], tx[tm], tx[tm], tx[tm]], values):
            plot_frames.append(pd.DataFrame(dict(column=row.column, dataset=row.dataset, mode=job.mode,
                direction=name, gene=row.gene, method=method, spot_id=ids,
                x=xy[:, 0], y=xy[:, 1], log1p_counts=value)))
        del generated
    pd.concat(plot_frames, ignore_index=True).to_csv(OUT / f"{STEM}_plot_data.csv.gz", index=False)
    return [cases[r.direction] for _, r in selected.iterrows()]


def draw(cases):
    for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        font_manager.fontManager.addfont(Path(sys.prefix) / "fonts" / name)
    mpl.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white"})
    vmax = max(float(np.max(v)) for case in cases for v in case["values"])
    norm = mpl.colors.Normalize(0, vmax)
    fig = plt.figure(figsize=(13.5, 10.8))
    grid = fig.add_gridspec(4, 4, left=0.08, right=0.985, bottom=0.145, top=0.81,
                           width_ratios=[1, 1, 1.4, 1.4], wspace=0.075, hspace=0.065)
    fig.suptitle("Conditional simulation across slices and held-out halves", y=0.995, fontsize=17, weight="bold")
    fig.text(0.5, 0.956, "DLPFC and MERFISH · selected marker examples · primary AR = 0.3",
             ha="center", fontsize=10, color="#555555")
    row_labels = ["Observed\nreference", "Held-out\ntruth", "FEAST", "Conditional\nresampling"]
    for col, case in enumerate(cases):
        row, is_half = case["row"], case["split"] is not None
        bounds = grid[0, col].get_position(fig)
        center = bounds.x0 + bounds.width / 2
        dataset = "DLPFC" if row.dataset == "dlpfc" else "MERFISH"
        task = "half-held-out" if is_half else "cross slice"
        fig.text(center, 0.915, f"{row.column}  {dataset} · {task}", ha="center", weight="bold", fontsize=10.5)
        detail = f"{short(row.source)} · left → right" if is_half else f"{short(row.source)} → {short(row.target)}"
        fig.text(center, 0.887, detail, ha="center", fontsize=10)
        fig.add_artist(Line2D([center - bounds.width * 0.3, center + bounds.width * 0.3],
                             [0.875, 0.875], transform=fig.transFigure, color="#888888", lw=0.8))
        fig.text(center, 0.848, row.gene, ha="center", fontstyle="italic", fontsize=11.5)
        size = 3.6 if row.dataset == "dlpfc" else 1.0
        for ri, values in enumerate(case["values"]):
            ax = fig.add_subplot(grid[ri, col])
            index = 0 if ri == 0 else 1
            xy, mask = case["xy"][index], case["masks"][index]
            ax.scatter(*xy.T, c="#DDDDDD", s=size, linewidths=0, rasterized=True)
            if is_half and ri > 0:
                # Observed reference context uses unchanged source expression, with
                # its role distinguished from the target by a muted color treatment.
                colors = mpl.colormaps["viridis"](norm(case["values"][0]))
                colors[:, :3] = 0.35 * colors[:, :3] + 0.65 * np.asarray(mpl.colors.to_rgb("#D8D8D8"))
                ax.scatter(*case["xy"][0][case["masks"][0]].T, c=colors, s=size,
                           linewidths=0, rasterized=True)
            artist = ax.scatter(*xy[mask].T, c=values, cmap="viridis", norm=norm,
                                s=size, linewidths=0, rasterized=True)
            map_axis(ax, xy)
            if is_half:
                ax.axvline(case["split"], color="#555555", ls=(0, (3, 3)), lw=0.8)
                if ri == 0:
                    left = (xy[:, 0].min() + case["split"]) / 2
                    right = (xy[:, 0].max() + case["split"]) / 2
                    ax.text(left, 1.01, "Observed", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8, color="#555555")
                    ax.text(right, 1.01, "Held out", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8, color="#555555")
            if col == 0:
                ax.text(-0.16, 0.5, row_labels[ri], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", weight="bold", fontsize=11,
                        color="#0072B2" if ri == 2 else "#333333")
    cax = fig.add_axes([0.34, 0.092, 0.38, 0.014])
    cb = fig.colorbar(artist, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=9, length=2)
    cb.set_label("Expression · log1p(counts)", fontsize=9.5, labelpad=2)
    fig.text(0.53, 0.04, "Half-held-out columns: dashed line = split; muted left half = observed context; full-color right half = evaluated target.",
             ha="center", fontsize=8.5, color="#555555")
    fig.text(0.53, 0.018, "One shared expression scale. Gray = hidden or unsupported locations. Resampling = fixed draw 0.",
             ha="center", fontsize=8.5, color="#555555")
    for suffix in ("pdf", "svg", "png"):
        metadata = ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else
                    {"Date": None} if suffix == "svg" else {})
        path = OUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
        print(f"Saved {path}", flush=True)
    plt.close(fig)
    return vmax


def main():
    selected, candidates = select_examples()
    cases = load_examples(selected)
    vmax = draw(cases)
    selected["target_retained_fraction"] = [c["support"]["target_retained_fraction"] for c in cases]
    selected.to_csv(OUT / f"{STEM}_selection.csv", index=False)
    candidates.to_csv(OUT / f"{STEM}_candidates.csv", index=False)
    details = [f"- **{r.column}: {r.dataset.upper()}, {r['mode']}, {short(r.source)}"
               + (f" → {short(r.target)}" if r['mode'] == 'cross_slice' else " left → right")
               + f", {r.gene}.** Raw-count gene Pearson r = {r.pearson:.3f}; target support = {100*r.target_retained_fraction:.1f}%."
               for _, r in selected.iterrows()]
    caption = ("# Cross-slice and half-held-out conditional simulation: 4 × 4 panel\n\n"
        "Columns represent the complete two-dataset × two-task combination. Rows are observed reference, "
        "held-out truth, FEAST, and label-conditioned whole-spot resampling.\n\n"
        + "\n".join(details) + "\n\n" + SELECTION_RULE + "\n\n"
        "Half-held-out tasks use the existing median-x split: x ≤ median is observed and x > median is held out. "
        "In the reference row, the held-out half is gray. In the three target rows, the observed left half "
        "is repeated in muted colors for context, while the right half shows held-out truth or simulated expression. "
        "Only supported right-half spots are evaluated; observed context is never counted as prediction. "
        "The dashed line marks the split in all four half-held-out panels. Unsupported labels remain gray.\n\n"
        f"All 16 panels use one unclipped scale from 0 to {vmax:.6f} log1p(counts). "
        "This replaces the separate q99.5 display normalization of the earlier half-slice figure; "
        "the underlying raw expression and existing scores are unchanged. The muted observed-context colors "
        "are a display treatment, not rescaled expression.\n\n"
        "All FEAST examples use primary AR = 0.3. Target labels and coordinates are known inputs; target expression "
        "is reserved for evaluation. The displayed baseline reproduces seed 2026, draw 0, with RNG advanced "
        "through the original complete primary-task order. Its library-size Wasserstein distance matches the "
        "saved replicate-0 metric for each displayed task. DLPFC cross-slice transfer here is within donor; "
        "cross-donor directions remain in the full atlas. MERFISH donor identity is not declared.\n\n"
        "Selection uses held-out agreement for presentation. These examples do not estimate typical-gene or "
        "whole-study performance; retain the complete quantitative figures for those claims.\n\n"
        "Rebuild: `python visualization/05_2d_conditional_transfer/plot_mixed_transfer_4x4.py`.\n")
    (OUT / f"{STEM}_caption.md").write_text(caption)
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(dict(
        entry_point=str(Path(__file__).relative_to(VIS.parents[1])), selection_rule=SELECTION_RULE,
        metric_input="05_2d_conditional_transfer/outputs/scores/per_gene_metrics.csv",
        baseline_input="05_2d_conditional_transfer/outputs/baselines/conditional_metric_comparison/method_metrics.csv",
        assignment_randomness=0.3, seed=2026, baseline_draw=0, baseline_score_check="passed for all four tasks",
        layout="4 method rows × 2 datasets × 2 tasks", font="Arial", dpi=600, colorbar_count=1,
        shared_expression_scale=dict(vmin=0, vmax=vmax, transform="log1p(raw counts)", clipped=False),
        half_context="Observed source expression, muted to distinguish from evaluated target",
        selected=selected[["dataset", "mode", "direction", "gene"]].to_dict("records")), indent=2) + "\n")


if __name__ == "__main__":
    main()
