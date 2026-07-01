# %%
"""Deconvolution benchmark — grouped bar chart comparing cell2location vs RCTD.

Three panels: JSD mean, Pearson mean, Summed RMSE.
Each panel shows method means aggregated across slices, with resolution
as a subgroup (0.10 and 0.25).
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
RESULTS_CSV = EXPERIMENT_ROOT / "03_deconvolution" / "results" / "deconvolution_benchmark_results.csv"
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/deconvolution")
OUTPUT_STEM = "deconvolution_benchmark"

METRIC_SPECS = [
    ("jsd_mean", "JSD Mean ↑", "JSD"),
    ("pearson_mean", "Pearson Mean ↑", "Pearson r"),
    ("summed_rmse", "Summed RMSE ↓", "RMSE"),
]

METHOD_DISPLAY = {"cell2location": "cell2location", "rctd": "RCTD"}
METHOD_PALETTE = {"cell2location": "#4c72b0", "RCTD": "#c44e52"}
RES_HATCH = {0.1: "", 0.25: "//"}


# %%
def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_DISPLAY.values():
        for res in [0.1, 0.25]:
            sub = df[(df["method_display"] == method) & (df["resolution"] == res)]
            if sub.empty:
                continue
            row = {"method": method, "resolution": res, "n_slices": len(sub)}
            for metric_col, _, _ in METRIC_SPECS:
                vals = sub[metric_col]
                row[f"{metric_col}_mean"] = vals.mean()
                row[f"{metric_col}_sem"] = vals.std(ddof=1) / np.sqrt(len(vals))
            rows.append(row)
    return pd.DataFrame(rows)


# %%
def draw_figure(summary: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
    })

    n_metrics = len(METRIC_SPECS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 3.8))
    if n_metrics == 1:
        axes = [axes]

    methods = list(METHOD_DISPLAY.values())
    resolutions = [0.1, 0.25]
    n_methods = len(methods)
    n_res = len(resolutions)
    total_bars = n_methods * n_res
    bar_width = 0.35
    group_spacing = 0.18

    x = np.arange(n_methods)
    offsets = np.linspace(-bar_width, bar_width, n_res)

    for ax_idx, (metric_col, title, ylabel) in enumerate(METRIC_SPECS):
        ax = axes[ax_idx]
        mean_col = f"{metric_col}_mean"
        sem_col = f"{metric_col}_sem"

        for res_idx, res in enumerate(resolutions):
            vals = []
            errs = []
            for method in methods:
                row = summary[(summary["method"] == method) & (summary["resolution"] == res)]
                if row.empty:
                    vals.append(0)
                    errs.append(0)
                else:
                    vals.append(row[mean_col].values[0])
                    errs.append(row[sem_col].values[0])
            bars = ax.bar(
                x + offsets[res_idx], vals, bar_width * 0.9,
                yerr=errs, capsize=4, error_kw={"linewidth": 1.2},
                label=f"res={res:.2f}",
                color=[METHOD_PALETTE[m] for m in methods],
                edgecolor="white", linewidth=0.6,
            )
            for bar, method in zip(bars, methods):
                if res in RES_HATCH:
                    bar.set_hatch(RES_HATCH[res])

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    axes[0].legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    plt.subplots_adjust(
        left=0.09, right=0.97, top=0.88, bottom=0.14,
        wspace=0.30,
    )

    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_aggregate(RESULTS_CSV)
    summary = build_summary(df)
    summary_path = args.output_dir / f"{args.output_stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    draw_figure(summary, args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
