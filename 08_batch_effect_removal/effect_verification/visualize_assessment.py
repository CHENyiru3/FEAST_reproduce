#!/usr/bin/env python
"""Generate diagnostic plots from a batch assessment output directory.

Usage:
    python visualize_assessment.py outputs/20250101_120000/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _load_or_none(path: Path):
    return path if path.exists() else None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col=0, **kwargs) if path.exists() else None


def plot_cloud_pca(out_dir: Path, centroids_path: Path):
    """PCA of per-slice centroid coordinates (2D)."""
    with open(centroids_path) as f:
        data = json.load(f)
    centroids = np.array(data["centroids"])
    slices = data["slice_order"]

    fig, ax = plt.subplots(figsize=(8, 6))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(centroids)
    ax.scatter(proj[:, 0], proj[:, 1], c=range(len(slices)), cmap="tab10", s=80)
    for i, s in enumerate(slices):
        ax.annotate(s, (proj[i, 0], proj[i, 1]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Slice Centroid PCA")
    fig.tight_layout()
    fig.savefig(out_dir / "cloud_pca.png", dpi=150)
    plt.close(fig)


def plot_centroid_heatmap(out_dir: Path, shifts_path: Path):
    """Heatmap of centroid shifts across slices."""
    shifts = pd.read_csv(shifts_path, index_col=0)
    fig, ax = plt.subplots(figsize=(max(4, shifts.shape[0] * 0.5), 3))
    vmax = max(abs(shifts.values.min()), abs(shifts.values.max()))
    im = ax.imshow(shifts.T.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(shifts.shape[0]))
    ax.set_xticklabels(shifts.index, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Δ log μ", "Δ log ω", "Δ logit π₀"], fontsize=9)
    plt.colorbar(im, ax=ax, label="Shift")
    ax.set_title("Centroid Batch Shifts")
    fig.tight_layout()
    fig.savefig(out_dir / "centroid_heatmap.png", dpi=150)
    plt.close(fig)


def plot_distance_heatmaps(out_dir: Path):
    """Heatmaps for each distance matrix found."""
    for npz_path in sorted(out_dir.glob("distance_*.npz")):
        metric = npz_path.stem.replace("distance_", "")
        data = np.load(npz_path)
        matrix = data["matrix"]
        slices = data["slice_order"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(slices)))
        ax.set_xticklabels(slices, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(slices)))
        ax.set_yticklabels(slices, fontsize=7)
        plt.colorbar(im, ax=ax, label="Distance")
        ax.set_title(f"Pairwise Distance: {metric}")
        fig.tight_layout()
        fig.savefig(out_dir / f"distance_{metric}.png", dpi=150)
        plt.close(fig)


def plot_covariance_ellipsoids(out_dir: Path, cov_path: Path):
    """2D projections of covariance ellipsoids for each slice."""
    data = np.load(cov_path)
    covs = data["covariances"]
    slices = data.get("slice_order", [f"S{i}" for i in range(covs.shape[0])])

    pairs = [(0, 1, "log μ", "log ω"), (0, 2, "log μ", "logit π₀"), (1, 2, "log ω", "logit π₀")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, covs.shape[0]))

    for ax_idx, (di, dj, label_i, label_j) in enumerate(pairs):
        ax = axes[ax_idx]
        for s in range(covs.shape[0]):
            cov_2d = covs[s][[di, dj]][:, [di, dj]]
            vals, vecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * np.sqrt(vals)
            ell = Ellipse(xy=(0, 0), width=w, height=h, angle=angle,
                         facecolor=colors[s], alpha=0.3, edgecolor=colors[s], linewidth=1.5)
            ax.add_patch(ell)
        all_vals = [2 * np.sqrt(max(np.max(np.linalg.eigvalsh(covs[s][[di, dj]][:, [di, dj]])), 0))
                    for s in range(covs.shape[0])]
        lim = max(all_vals) * 0.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(label_i)
        ax.set_ylabel(label_j)
        ax.set_title(f"{label_i} vs {label_j}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

    fig.legend(slices, loc="center right", fontsize=7, ncol=1)
    fig.suptitle("Covariance Ellipsoids (2D projections)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(out_dir / "covariance_ellipsoids.png", dpi=150)
    plt.close(fig)


def plot_affine_diagnostics(out_dir: Path, params_path: Path):
    """Bar charts of D (scaling), b (shift), and r² per slice."""
    params = pd.read_csv(params_path, index_col=0)
    ref = params.index[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # D (scaling)
    ax = axes[0]
    d_cols = ["d_mu", "d_omega", "d_pi0"]
    x = np.arange(len(params))
    w = 0.25
    for i, col in enumerate(d_cols):
        ax.bar(x + i * w, params[col], w, label=col)
    ax.set_xticks(x + w)
    ax.set_xticklabels(params.index, rotation=45, ha="right", fontsize=7)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_title("Scaling (D)")
    ax.legend(fontsize=7)

    # b (shift)
    ax = axes[1]
    b_cols = ["b_mu", "b_omega", "b_pi0"]
    for i, col in enumerate(b_cols):
        ax.bar(x + i * w, params[col], w, label=col)
    ax.set_xticks(x + w)
    ax.set_xticklabels(params.index, rotation=45, ha="right", fontsize=7)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_title("Shift (b)")
    ax.legend(fontsize=7)

    # r²
    ax = axes[2]
    r2_cols = ["r2_mu", "r2_omega", "r2_pi0"]
    for i, col in enumerate(r2_cols):
        ax.bar(x + i * w, params[col], w, label=col)
    ax.set_xticks(x + w)
    ax.set_xticklabels(params.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("R²")
    ax.legend(fontsize=7)

    fig.suptitle(f"Affine Deformation Diagnostics (ref = {ref})")
    fig.tight_layout()
    fig.savefig(out_dir / "affine_diagnostics.png", dpi=150)
    plt.close(fig)


def plot_qc_dashboard(out_dir: Path, qc_path: Path):
    """Multi-panel QC summary."""
    qc = pd.read_csv(qc_path, index_col=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    slices = qc.index.tolist()
    x = range(len(slices))

    metrics = [
        ("total_counts_mean", "Mean Total Counts"),
        ("detected_genes", "Detected Genes"),
        ("zero_fraction", "Zero Fraction"),
        ("mean_of_means", "Mean of Gene Means"),
        ("mean_of_variances", "Mean of Gene Variances"),
        ("mean_of_zero_prop", "Mean of Zero Proportions"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(x, qc[col].values, color=plt.cm.tab10(np.linspace(0, 1, len(slices))))
        ax.set_xticks(x)
        ax.set_xticklabels(slices, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)

    fig.suptitle("Slice-Level QC Summary")
    fig.tight_layout()
    fig.savefig(out_dir / "qc_dashboard.png", dpi=150)
    plt.close(fig)


def plot_domain_shift_heatmap(out_dir: Path, domain_path: Path):
    """Heatmap of per-domain centroid shifts."""
    df = pd.read_csv(domain_path)
    if df.empty:
        return

    pivot = df.pivot_table(
        index="domain", columns="slice_id", values="shift_log_mu", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 0.6), max(3, pivot.shape[0] * 0.5)))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Shift in log μ")
    ax.set_title("Per-Domain Centroid Shifts (log μ)")
    fig.tight_layout()
    fig.savefig(out_dir / "domain_shift_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from batch assessment outputs."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to assessment output directory (e.g., outputs/20250101_120000/)",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    # Check what files are available
    centroids_json = _load_or_none(out_dir / "centroids.json")
    shifts_csv = _load_or_none(out_dir / "centroid_shifts.csv")
    cov_npz = _load_or_none(out_dir / "covariances.npz")
    affine_csv = _load_or_none(out_dir / "affine_params.csv")
    qc_csv = _load_or_none(out_dir / "qc_summary.csv")
    domain_csv = _load_or_none(out_dir / "domain_shifts.csv")

    if centroids_json:
        print("  cloud_pca ...")
        plot_cloud_pca(out_dir, centroids_json)

    if shifts_csv:
        print("  centroid_heatmap ...")
        plot_centroid_heatmap(out_dir, shifts_csv)

    print("  distance_heatmaps ...")
    plot_distance_heatmaps(out_dir)

    if cov_npz:
        print("  covariance_ellipsoids ...")
        plot_covariance_ellipsoids(out_dir, cov_npz)

    if affine_csv:
        print("  affine_diagnostics ...")
        plot_affine_diagnostics(out_dir, affine_csv)

    if qc_csv:
        print("  qc_dashboard ...")
        plot_qc_dashboard(out_dir, qc_csv)

    if domain_csv and domain_csv.stat().st_size > 0:
        print("  domain_shift_heatmap ...")
        plot_domain_shift_heatmap(out_dir, domain_csv)

    print(f"\nVisualization complete. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
