#!/usr/bin/env python
"""
Stage 4b: Visualize alignment results in 2D spatial context.

For each method and angle, plots original, rotated, and aligned coordinates
overlaid, showing the displacement from rotated to aligned positions.

Usage:
    conda run -p ${FEAST_ENV} python scripts/visualize_alignment_3d.py \
      --simulation-manifest outputs/simulation_manifest.csv \
      --methods-dir outputs/methods \
      --output-dir outputs/plots/alignment_3d
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Visualize alignment results")
    parser.add_argument("--simulation-manifest", type=str, required=True)
    parser.add_argument("--methods-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = pd.read_csv(args.simulation_manifest)
    orig_path = manifest[manifest["type"] == "reference"]["file_path"].values[0]
    orig_adata = sc.read_h5ad(orig_path)
    orig_coords = orig_adata.obsm["spatial"]

    methods_dir = Path(args.methods_dir)
    method_names = [
        d.name for d in methods_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for method in sorted(method_names):
        method_dir = methods_dir / method
        for angle_dir in sorted(method_dir.iterdir()):
            if not angle_dir.is_dir() or not angle_dir.name.startswith("angle_"):
                continue

            angle_tag = angle_dir.name.replace("angle_", "")
            coord_file = angle_dir / "aligned_coordinates.csv"
            if not coord_file.exists():
                print(f"SKIP {method}/angle_{angle_tag} (no coordinates)")
                continue

            coords_df = pd.read_csv(coord_file)
            rot_x = coords_df["x_original"].values
            rot_y = coords_df["y_original"].values
            aligned_x = coords_df["x_aligned"].values
            aligned_y = coords_df["y_aligned"].values

            # Subsample for arrows to avoid overplotting
            n_arrows = min(300, len(coords_df))
            indices = np.linspace(0, len(coords_df) - 1, n_arrows, dtype=int)

            fig, ax = plt.subplots(figsize=(8, 8))

            # Original (reference)
            ax.scatter(orig_coords[:, 0], orig_coords[:, 1],
                       c="black", s=1, alpha=0.25, label="Original (ref)")

            # Rotated (input to alignment)
            ax.scatter(rot_x, rot_y,
                       c="red", s=2, alpha=0.5, label="Rotated (input)")

            # Aligned (method output)
            ax.scatter(aligned_x, aligned_y,
                       c="blue", s=2, alpha=0.6, label="Aligned (output)")

            # Displacement arrows
            ax.quiver(rot_x[indices], rot_y[indices],
                      aligned_x[indices] - rot_x[indices],
                      aligned_y[indices] - rot_y[indices],
                      color="blue", alpha=0.4, angles="xy", scale_units="xy",
                      scale=1, width=0.3, headwidth=3)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{method.capitalize()} — Angle {angle_tag}°")
            ax.legend(markerscale=6, fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)

            fig.tight_layout()
            out_path = out_dir / f"{method}_angle_{angle_tag}.pdf"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

    print(f"\nDone. {len(list(out_dir.glob('*.pdf')))} PDFs written.")


if __name__ == "__main__":
    main()
