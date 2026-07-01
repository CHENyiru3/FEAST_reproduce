# %%
"""Robustness metric curves for clustering benchmark across alteration types.

4x4 grid: left text/legend + 3 alteration columns (mean, variance, sparsity)
× 4 metric rows (ARI, NMI, PAS, CHAOS).
Methods: STAGATE and GraphST. Data aggregated across 3 DLPFC slices.
"""

# %%
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
CLUSTERING_CSV = EXPERIMENT_ROOT / "01_clustering" / "results" / "clustering_benchmark_results.csv"
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/clustering")
OUTPUT_STEM = "robustness_alteration_metrics"

METHOD_MAP = {
    "STAGATE_mclust": "STAGATE",
    "GraphST": "GraphST",
}

PALETTE = {
    "STAGATE": "#ef7890",
    "GraphST": "#b99a4a",
}

ALTERATION_TYPES = ["mean", "variance", "sparsity_logit_shift"]
ALTERATION_TITLES = {
    "mean": "Mean\nAlteration",
    "variance": "Variance\nAlteration",
    "sparsity_logit_shift": "Sparsity\nAlteration (logit)",
}
LOG_SCALE_ALTERATIONS = {"mean", "variance"}

METRIC_ORDER = ["ARI", "NMI", "PAS", "CHAOS"]
METRIC_TITLES = {
    "ARI": "ARI ↑",
    "NMI": "NMI ↑",
    "PAS": "PAS ↓",
    "CHAOS": "CHAOS ↓",
}
METRIC_YLABELS = {
    "ARI": "ARI Score",
    "NMI": "NMI Score",
    "PAS": "PAS Score",
    "CHAOS": "CHAOS Score",
}


# %%
def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load clustering results for all alteration types + baseline, map method names."""
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()

    alter_rows = ok[ok["alteration_type"].isin(ALTERATION_TYPES)].copy()
    baseline_rows = ok[ok["alteration_type"] == "baseline"].copy()

    # Baseline applies to all alteration types — replicate baseline rows for each
    replicated = []
    for atype in ALTERATION_TYPES:
        b = baseline_rows.copy()
        b["alteration_type"] = atype
        replicated.append(b)
    baseline_all = pd.concat(replicated, ignore_index=True)

    combined = pd.concat([alter_rows, baseline_all], ignore_index=True)
    combined["method"] = combined["method"].map(METHOD_MAP)
    combined = combined.dropna(subset=["method"])
    combined["alteration_factor"] = combined["fold_change"].astype(float)
    # Baseline at fold_change=1.0 is correct for mean/variance, but for
    # sparsity_logit_shift the no-op baseline is 0.0 (no logit shift).
    mask = (combined["alteration_type"] == "sparsity_logit_shift") & (combined["alteration_factor"] == 1.0)
    combined.loc[mask, "alteration_factor"] = 0.0
    return combined


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across slices: mean and sd per (method, factor, metric, alteration_type)."""
    rows = []
    for atype in ALTERATION_TYPES:
        sub_a = df[df["alteration_type"] == atype]
        for metric in METRIC_ORDER:
            for method in METHOD_MAP.values():
                sub_m = sub_a[sub_a["method"] == method]
                grouped = sub_m.groupby("alteration_factor")[metric]
                for factor, grp in grouped:
                    rows.append({
                        "alteration_type": atype,
                        "method": method,
                        "alteration_factor": factor,
                        "metric": metric,
                        "mean": grp.mean(),
                        "sd": grp.std(ddof=1) if len(grp) > 1 else 0.0,
                    })
    return pd.DataFrame(rows)


def compute_ylim(plot_df: pd.DataFrame, metric: str, atype: str, pad: float = 0.12) -> tuple[float, float]:
    """Compute tight y-limits per metric and alteration type."""
    sub = plot_df[(plot_df["metric"] == metric) & (plot_df["alteration_type"] == atype)]
    if sub.empty:
        return 0.0, 1.0
    lo = (sub["mean"] - sub["sd"]).min()
    hi = (sub["mean"] + sub["sd"]).max()
    span = hi - lo
    if span <= 0:
        span = max(abs(hi) * 0.1, 0.01)
    return float(lo - pad * span), float(hi + pad * span)


# %%
def draw_figure(plot_df: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(METHOD_MAP.values())
    n_alters = len(ALTERATION_TYPES)
    n_metrics = len(METRIC_ORDER)

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(
        nrows=n_metrics, ncols=n_alters + 1,
        width_ratios=[0.75, 1.0, 1.0, 1.0],
        hspace=0.32, wspace=0.28,
    )

    text_ax = fig.add_subplot(gs[:, 0])

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # -- Left text / legend panel --
    text_ax.axis("off")
    text_ax.text(
        0.03, 0.94,
        "Robustness\ntest",
        ha="left", va="top",
        fontsize=30, color="#2f3292",
    )

    legend_y = 0.55
    for i, method in enumerate(methods):
        y = legend_y - i * 0.055
        text_ax.errorbar(
            [0.32], [y],
            xerr=None, yerr=[[0.012], [0.012]],
            fmt="o-", color=PALETTE[method],
            markersize=8, linewidth=2.2, capsize=5,
            transform=text_ax.transAxes, clip_on=False,
        )
        text_ax.text(
            0.44, y, method,
            transform=text_ax.transAxes,
            ha="left", va="center",
            fontsize=18, color="#222222",
        )

    # -- Metric panels --
    for row_idx, metric in enumerate(METRIC_ORDER):
        for col_idx, atype in enumerate(ALTERATION_TYPES):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            sub = plot_df[(plot_df["alteration_type"] == atype) & (plot_df["metric"] == metric)]

            for method in methods:
                sm = sub[sub["method"] == method].sort_values("alteration_factor")
                if sm.empty:
                    continue
                ax.errorbar(
                    sm["alteration_factor"], sm["mean"], yerr=sm["sd"],
                    fmt="o-", color=PALETTE[method],
                    markersize=5.5, linewidth=1.8, elinewidth=1.2,
                    capsize=4, capthick=1.2, alpha=0.95,
                )

            # Column titles on top row only
            if row_idx == 0:
                ax.set_title(
                    ALTERATION_TITLES[atype],
                    fontsize=22, fontweight="bold", pad=10,
                    color="#2f3292",
                )

            # Row labels
            if col_idx == 0:
                ax.set_ylabel(METRIC_YLABELS[metric], fontsize=15)

            # X-axis label on bottom row only
            if row_idx == n_metrics - 1:
                ax.set_xlabel("Alteration Factor", fontsize=16)
            else:
                ax.set_xlabel("")

            if atype in LOG_SCALE_ALTERATIONS:
                ax.set_xscale("log")
            ax.set_ylim(*compute_ylim(plot_df, metric, atype))

            ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.55)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#555555")

    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clustering-csv", type=Path, default=CLUSTERING_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.clustering_csv)
    plot_df = build_summary(df)

    summary_path = args.output_dir / f"{args.output_stem}_summary.csv"
    plot_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    draw_figure(plot_df, args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
