#!/usr/bin/env python3
"""GraphST clustering for Figure 2 clustering benchmark.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/GraphST \\
    python scripts/methods/clustering_graphST.py \\
      --input outputs/hvg_inputs/151670/mean_0.10.h5ad \\
      --output-dir outputs/methods/GraphST/151670/mean_0.10 \\
      --n-clusters auto \\
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Point rpy2 at the conda env's R (which has mclust), not the system R
if "R_HOME" not in os.environ:
    os.environ["R_HOME"] = "/maiziezhou_lab2/yiru/envs/GraphST/lib/R"

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")


def run_graphst(
    adata: sc.AnnData,
    n_clusters: int | None = None,
    device: str = "cpu",
    seed: int = 2026,
    epochs: int = 600,
    radius: int = 50,
    refinement: bool = True,
    datatype: str = "10X",
) -> tuple[np.ndarray, dict, sc.AnnData]:
    """Run GraphST clustering. Returns (labels, metadata)."""
    import torch
    from GraphST import GraphST as graphst_model
    from GraphST.utils import clustering

    torch_device = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    if n_clusters is None:
        if "ground_truth" in adata.obs.columns:
            n_clusters = len(adata.obs["ground_truth"].unique())
        else:
            n_clusters = 7

    # Inputs are already HVG-filtered and log-normalized by the shared Figure 2
    # preprocessing stage. Mark all retained genes as HVGs, then apply the
    # scaling step used by GraphST.preprocess before training the model.
    adata = adata.copy()
    adata.var["highly_variable"] = True
    sc.pp.scale(adata, zero_center=False, max_value=10)

    model = graphst_model.GraphST(
        adata,
        device=torch_device,
        epochs=epochs,
        dim_input=adata.n_vars,
        random_seed=seed,
        datatype=datatype,
    )
    adata = model.train()

    clustering(
        adata,
        n_clusters=n_clusters,
        radius=radius,
        method="mclust",
        refinement=refinement,
    )

    labels = adata.obs["domain"].astype(str).values

    elapsed = time.time() - t0
    meta = {
        "method": "GraphST_mclust",
        "n_clusters_target": n_clusters,
        "n_clusters_actual": len(np.unique(labels)),
        "device": device,
        "epochs": epochs,
        "radius": radius,
        "refinement": refinement,
        "datatype": datatype,
        "embedding_key": "emb",
        "elapsed_seconds": round(elapsed, 2),
    }

    return labels, meta, adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--datatype", type=str, default="10X")
    parser.add_argument("--no-refinement", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    sc.settings.verbosity = 0
    adata = sc.read_h5ad(str(args.input))
    print(f"GraphST: {args.input.name} ({adata.n_obs} spots x {adata.n_vars} genes)")

    n_clusters = None if args.n_clusters == "auto" else int(args.n_clusters)

    try:
        labels, meta, adata_result = run_graphst(
            adata,
            n_clusters=n_clusters,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            radius=args.radius,
            refinement=not args.no_refinement,
            datatype=args.datatype,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        adata_result.obs["predicted_cluster"] = labels
        adata_result.obs["predicted_cluster"] = adata_result.obs["predicted_cluster"].astype("category")

        clusters = pd.DataFrame({
            "spot_barcode": adata_result.obs.index,
            "predicted_cluster": labels,
        })
        clusters.to_csv(args.output_dir / "clusters.csv", index=False)

        # Clean uns of non-serializable keys before writing h5ad
        for k in list(adata_result.uns.keys()):
            try:
                del adata_result.uns[k]
            except Exception:
                pass

        try:
            adata_result.write_h5ad(str(args.output_dir / "result.h5ad"), compression="gzip")
        except Exception:
            print(f"  -> WARNING: result.h5ad write failed, clusters.csv is fine")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> {meta['n_clusters_actual']} clusters in {meta['elapsed_seconds']:.1f}s")
        return 0

    except Exception as e:
        import traceback
        fail_file = args.output_dir / "FAILED.txt"
        fail_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_file, "w") as f:
            f.write(traceback.format_exc())
        print(f"  -> FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
