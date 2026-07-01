#!/usr/bin/env python3
"""Compute clustering benchmark metrics across all methods and simulations.

Reads method outputs (clusters.csv) and simulation files (ground_truth),
then computes ARI, NMI, AMI, homogeneity, completeness, V-measure,
CHAOS, PAS, and n_predicted_clusters.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/clustering_benchmark_corrected.py \\
      --input-dir outputs/methods \\
      --simulation-manifest outputs/simulation_manifest.csv \\
      --label-column ground_truth \\
      --output outputs/benchmarks/clustering_benchmark_results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = StandardScaler().fit_transform(location)
    unique_clusters = np.unique(clusterlabel)
    if len(unique_clusters) < 2:
        return np.nan
    total_min_dist = 0.0
    for k in unique_clusters:
        cluster_points = location[clusterlabel == k]
        if len(cluster_points) < 2:
            continue
        dist_matrix = squareform(pdist(cluster_points))
        np.fill_diagonal(dist_matrix, np.inf)
        total_min_dist += np.sum(np.min(dist_matrix, axis=1))
    return total_min_dist / len(clusterlabel)


def _compute_PAS(clusterlabel, location, k=10):
    clusterlabel = np.array(clusterlabel)
    location = StandardScaler().fit_transform(location)
    n_samples = location.shape[0]
    if n_samples <= k:
        return np.nan
    dist_matrix = squareform(pdist(location))
    pas_count = 0
    for i in range(n_samples):
        k_nearest_indices = np.argsort(dist_matrix[i])[1:k + 1]
        k_nearest_labels = clusterlabel[k_nearest_indices]
        if np.sum(k_nearest_labels != clusterlabel[i]) > (k / 2):
            pas_count += 1
    return pas_count / n_samples


METHODS = ["STAGATE_mclust", "GraphST", "Leiden"]


def compute_metrics(gt_labels, pred_labels, spatial_coords):
    n_gt = len(np.unique(gt_labels))
    n_pred = len(np.unique(pred_labels))

    if n_pred < 2 or n_gt < 2:
        return {
            "ARI": np.nan, "NMI": np.nan, "AMI": np.nan,
            "Homogeneity": np.nan, "Completeness": np.nan,
            "V_measure": np.nan, "n_predicted_clusters": n_pred,
            "n_true_clusters": n_gt, "CHAOS": np.nan, "PAS": np.nan,
        }

    chaos_val = np.nan
    pas_val = np.nan
    try:
        chaos_val = _compute_CHAOS(pred_labels, spatial_coords)
        pas_val = _compute_PAS(pred_labels, spatial_coords)
    except Exception:
        pass

    try:
        return {
            "ARI": adjusted_rand_score(gt_labels, pred_labels),
            "NMI": normalized_mutual_info_score(gt_labels, pred_labels),
            "AMI": adjusted_mutual_info_score(gt_labels, pred_labels),
            "Homogeneity": homogeneity_score(gt_labels, pred_labels),
            "Completeness": completeness_score(gt_labels, pred_labels),
            "V_measure": v_measure_score(gt_labels, pred_labels),
            "n_predicted_clusters": n_pred,
            "n_true_clusters": n_gt,
            "CHAOS": chaos_val,
            "PAS": pas_val,
        }
    except Exception:
        return {
            "ARI": np.nan, "NMI": np.nan, "AMI": np.nan,
            "Homogeneity": np.nan, "Completeness": np.nan,
            "V_measure": np.nan, "n_predicted_clusters": n_pred,
            "n_true_clusters": n_gt, "CHAOS": chaos_val, "PAS": pas_val,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/methods"),
                        help="Directory with method outputs")
    parser.add_argument("--simulation-manifest", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="ground_truth")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/clustering_benchmark_results.csv"))
    args = parser.parse_args()

    if not args.simulation_manifest.exists():
        print(f"ERROR: simulation manifest not found: {args.simulation_manifest}", file=sys.stderr)
        return 1

    sim_manifest = pd.read_csv(args.simulation_manifest)
    if "status" in sim_manifest.columns:
        sim_ok = sim_manifest[sim_manifest["status"].isin(["ok", "skipped"])]
    else:
        sim_ok = sim_manifest

    t0 = time.time()
    rows = []
    total = len(sim_ok) * len(METHODS)
    done = 0

    for _, sim_row in sim_ok.iterrows():
        slice_id = str(sim_row["slice_id"])
        sim_id = str(sim_row["simulation_id"])
        sim_path = Path(sim_row["file_path"])

        if not sim_path.exists():
            for method in METHODS:
                rows.append({
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    "status": "sim_missing", "error_message": f"sim file not found: {sim_path}",
                })
                done += 1
            continue

        try:
            sim_adata = sc.read_h5ad(str(sim_path))
            gt_labels = sim_adata.obs[args.label_column].values
            spatial = sim_adata.obsm["spatial"]
        except Exception as e:
            for method in METHODS:
                rows.append({
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    "status": "sim_load_error", "error_message": str(e)[:200],
                })
                done += 1
            continue

        for method in METHODS:
            done += 1
            cluster_csv = args.input_dir / method / slice_id / sim_id / "clusters.csv"
            fail_txt = args.input_dir / method / slice_id / sim_id / "FAILED.txt"

            if not cluster_csv.exists() and fail_txt.exists():
                fail_msg = fail_txt.read_text().strip()[:200]
                rows.append({
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    "ARI": np.nan, "NMI": np.nan, "AMI": np.nan,
                    "Homogeneity": np.nan, "Completeness": np.nan,
                    "V_measure": np.nan, "n_predicted_clusters": np.nan,
                    "n_true_clusters": np.nan, "CHAOS": np.nan, "PAS": np.nan,
                    "status": "failed", "error_message": fail_msg,
                })
                print(f"  [{done}/{total}] {method}/{slice_id}/{sim_id} — FAILED (method)")
                continue

            if not cluster_csv.exists():
                rows.append({
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    "ARI": np.nan, "NMI": np.nan, "AMI": np.nan,
                    "Homogeneity": np.nan, "Completeness": np.nan,
                    "V_measure": np.nan, "n_predicted_clusters": np.nan,
                    "n_true_clusters": np.nan, "CHAOS": np.nan, "PAS": np.nan,
                    "status": "missing", "error_message": "clusters.csv not found",
                })
                print(f"  [{done}/{total}] {method}/{slice_id}/{sim_id} — MISSING")
                continue

            try:
                clusters = pd.read_csv(cluster_csv)
                if "spot_barcode" not in clusters.columns:
                    raise ValueError("clusters.csv missing spot_barcode column")
                if "predicted_cluster" not in clusters.columns:
                    raise ValueError("clusters.csv missing predicted_cluster column")

                cluster_series = clusters.set_index("spot_barcode")["predicted_cluster"]
                missing = sim_adata.obs_names.difference(cluster_series.index)
                extra = cluster_series.index.difference(sim_adata.obs_names)
                if len(missing) > 0 or len(extra) > 0:
                    raise ValueError(
                        f"spot_barcode mismatch: {len(missing)} missing, {len(extra)} extra"
                    )
                pred_labels = cluster_series.loc[sim_adata.obs_names].values

                metrics = compute_metrics(gt_labels, pred_labels, spatial)
                row = {
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    **metrics,
                    "status": "ok", "error_message": "",
                }
                rows.append(row)
                print(f"  [{done}/{total}] {method}/{slice_id}/{sim_id} — ARI={metrics['ARI']:.4f}")
            except Exception as e:
                rows.append({
                    "slice_id": slice_id, "method": method,
                    "simulation_id": sim_id,
                    "alteration_type": sim_row["alteration_type"],
                    "fold_change": sim_row["fold_change"],
                    "ARI": np.nan, "NMI": np.nan, "AMI": np.nan,
                    "Homogeneity": np.nan, "Completeness": np.nan,
                    "V_measure": np.nan, "n_predicted_clusters": np.nan,
                    "n_true_clusters": np.nan, "CHAOS": np.nan, "PAS": np.nan,
                    "status": "error", "error_message": str(e)[:200],
                })
                print(f"  [{done}/{total}] {method}/{slice_id}/{sim_id} — ERROR: {e}")

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    ok_count = (results["status"] == "ok").sum()
    elapsed = time.time() - t0
    print(f"\nBenchmark done: {ok_count}/{len(results)} ok rows in {elapsed:.1f}s")
    print(f"Output: {args.output}")

    # Write row count notes
    notes_path = args.output.parent / "row_count_notes.md"
    notes_path.write_text(f"""# Clustering Benchmark Row Count Notes

Generated: {pd.Timestamp.now().isoformat()}

- slices: {sim_ok['slice_id'].nunique()}
- simulations per slice: {sim_ok['simulation_id'].nunique()}
- methods: {len(METHODS)} ({', '.join(METHODS)})
- expected rows: {sim_ok['slice_id'].nunique()} x {sim_ok['simulation_id'].nunique()} x {len(METHODS)} = {sim_ok['slice_id'].nunique() * sim_ok['simulation_id'].nunique() * len(METHODS)}
- actual rows: {len(results)}
- ok rows: {ok_count}
- failed/missing: {(results['status'] != 'ok').sum()}

Legacy SPEC target was 276 rows (1 slice x 23 sims x 2 methods x ~6 permutations).
Current architecture differs (3 slices, 3 methods, per-sim benchmarking).
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
