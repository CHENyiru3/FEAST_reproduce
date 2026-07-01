#!/usr/bin/env python3
"""Benchmark deconvolution predictions against ground truth.

Metrics: Jensen-Shannon divergence (squared, base 2), Pearson correlation,
distance correlation (optional), summed RMSE, per-cell-type RMSE.

Handles missing prediction files gracefully by reporting status rows
instead of aborting. Aligns cell-type columns and spot rows between
truth and prediction automatically.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/deconv_benchmark.py \\
      --truth-dir outputs/ground_truth \\
      --prediction-dirs outputs/cell2location outputs/rctd \\
      --slices 007 050 100 \\
      --resolutions 0.1 0.25 \\
      --output outputs/benchmarks/deconvolution_benchmark_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr


def _jsd_per_spot(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    eps = 1e-10
    truth = np.clip(truth, eps, 1.0)
    pred = np.clip(pred, eps, 1.0)
    truth = truth / truth.sum(axis=1, keepdims=True)
    pred = pred / pred.sum(axis=1, keepdims=True)
    jsd = np.zeros(truth.shape[0])
    for i in range(truth.shape[0]):
        jsd[i] = jensenshannon(truth[i], pred[i], base=2.0) ** 2
    return jsd


def _pearson_per_cell_type(truth: np.ndarray, pred: np.ndarray) -> dict:
    results = {}
    for j in range(truth.shape[1]):
        mask = ~(np.isnan(truth[:, j]) | np.isnan(pred[:, j]))
        if mask.sum() < 3:
            results[j] = np.nan
        else:
            r, _ = pearsonr(truth[mask, j], pred[mask, j])
            results[j] = r
    return results


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    try:
        import dcor
        return dcor.distance_correlation(x, y)
    except ImportError:
        print("WARNING: dcor package not installed. Install with: pip install dcor")
        return np.nan


def _summed_rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((truth - pred) ** 2)))


def _per_cell_type_rmse(truth: np.ndarray, pred: np.ndarray) -> dict:
    results = {}
    for j in range(truth.shape[1]):
        diff = truth[:, j] - pred[:, j]
        results[j] = float(np.sqrt(np.nanmean(diff ** 2)))
    return results


def benchmark_one(
    truth_path: Path,
    pred_path: Path,
    method: str,
    slice_id: str,
    resolution: str,
) -> dict:
    row = {
        "method": method, "slice_id": slice_id, "resolution": resolution,
    }

    if not pred_path.exists():
        row["status"] = "failed: file not found"
        return row

    try:
        truth = pd.read_csv(truth_path, index_col=0)
        pred = pd.read_csv(pred_path, index_col=0)
    except Exception as e:
        row["status"] = f"failed: {e}"
        return row

    # Strip Cell2location internal prefix from column names
    C2L_PREFIX = "meanscell_abundance_w_sf_means_per_cluster_mu_fg_"
    if any(str(c).startswith(C2L_PREFIX) for c in pred.columns[:1]):
        pred.columns = [str(c).replace(C2L_PREFIX, "") for c in pred.columns]

    common_cts = truth.columns.intersection(pred.columns)
    if len(common_cts) == 0:
        row["status"] = "failed: no common cell types"
        row["n_truth_ct"] = len(truth.columns)
        row["n_pred_ct"] = len(pred.columns)
        return row

    missing_cts = set(truth.columns) - set(common_cts)
    if missing_cts:
        print(f"  WARNING: {len(missing_cts)} cell types missing from prediction"
              f" for {method}/{slice_id}/resolution_{resolution}")

    truth = truth[common_cts]
    pred = pred[common_cts]

    common_spots = truth.index.intersection(pred.index)
    spot_diff = max(len(truth.index), len(pred.index)) - min(len(truth.index), len(pred.index))
    if spot_diff > 0.1 * max(len(truth.index), len(pred.index)):
        print(f"  WARNING: spot count differs by {spot_diff} for "
              f"{method}/{slice_id}/resolution_{resolution}")

    truth = truth.loc[common_spots]
    pred = pred.loc[common_spots]

    if len(common_spots) == 0:
        row["status"] = "failed: no common spots"
        return row

    T = truth.values.astype(float)
    P = pred.values.astype(float)

    if np.all(np.isnan(P)):
        row["status"] = "failed: all-NaN predictions"
        return row

    jsd_vals = _jsd_per_spot(T, P)
    pearson_vals = _pearson_per_cell_type(T, P)
    dcor_val = _distance_correlation(T.ravel(), P.ravel())
    summed_rmse = _summed_rmse(T, P)
    per_ct_rmse = _per_cell_type_rmse(T, P)

    row.update({
        "n_spots": len(common_spots),
        "n_cell_types": len(common_cts),
        "jsd_mean": float(np.mean(jsd_vals)),
        "jsd_median": float(np.median(jsd_vals)),
        "jsd_std": float(np.std(jsd_vals)),
        "pearson_mean": float(np.nanmean(list(pearson_vals.values()))),
        "pearson_median": float(np.nanmedian(list(pearson_vals.values()))),
        "pearson_std": float(np.nanstd(list(pearson_vals.values()))),
        "distance_correlation": float(dcor_val) if not np.isnan(dcor_val) else np.nan,
        "summed_rmse": summed_rmse,
        "per_cell_type_rmse": json.dumps(
            {str(list(common_cts)[k]): v for k, v in per_ct_rmse.items()}
        ),
        "cell_types": ",".join(common_cts),
        "status": "ok",
    })
    return row


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metrics = [
        ("jsd_mean", "lower"),
        ("pearson_mean", "higher"),
        ("distance_correlation", "higher"),
        ("summed_rmse", "lower"),
    ]
    for metric, direction in metrics:
        if metric not in results_df.columns:
            continue
        vals = results_df[results_df["status"] == "ok"][metric].dropna()
        if len(vals) == 0:
            continue
        if direction == "lower":
            best_idx = vals.idxmin()
        else:
            best_idx = vals.idxmax()
        best_row = results_df.loc[best_idx] if len(vals) > 0 else None
        summary_rows.append({
            "metric": metric,
            "n_valid": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "best_method": best_row["method"] if best_row is not None else "N/A",
            "best_slice": best_row["slice_id"] if best_row is not None else "N/A",
            "best_resolution": best_row["resolution"] if best_row is not None else "N/A",
        })
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True,
                        help="Directories for each method (e.g. outputs/cell2location outputs/rctd)")
    parser.add_argument("--slices", type=str, nargs="+",
                        default=["007", "050", "100"])
    parser.add_argument("--resolutions", type=str, nargs="+",
                        default=["0.1", "0.25"])
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/"
                                     "deconvolution_benchmark_results.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"SKIP — benchmark results exist: {args.output}")
        return 0

    if not args.truth_dir.exists():
        print(f"ERROR: truth directory not found: {args.truth_dir}", file=sys.stderr)
        return 1

    all_rows = []
    total = len(args.slices) * len(args.resolutions) * len(args.prediction_dirs)
    count = 0

    for slice_id in args.slices:
        truth_slice_dir = args.truth_dir / slice_id
        if not truth_slice_dir.exists():
            print(f"WARNING: truth dir missing for slice {slice_id}")
            for res in args.resolutions:
                for pred_dir in args.prediction_dirs:
                    all_rows.append({
                        "method": pred_dir.name, "slice_id": slice_id,
                        "resolution": res, "status": "failed: truth dir not found",
                    })
                    count += 1
            continue

        for resolution in args.resolutions:
            truth_path = truth_slice_dir / f"resolution_{resolution}_proportions.csv"
            if not truth_path.exists():
                print(f"WARNING: truth file missing: {truth_path}")
                for pred_dir in args.prediction_dirs:
                    all_rows.append({
                        "method": pred_dir.name, "slice_id": slice_id,
                        "resolution": resolution,
                        "status": "failed: truth file not found",
                    })
                    count += 1
                continue

            for pred_dir in args.prediction_dirs:
                method = pred_dir.name
                pred_path = pred_dir / slice_id / f"resolution_{resolution}_proportions.csv"
                count += 1

                status_label = pred_path.exists() and "found" or "not found"
                print(f"  [{count}/{total}] {method} / {slice_id} / resolution_{resolution} ({status_label})")

                row = benchmark_one(truth_path, pred_path, method, slice_id, resolution)
                all_rows.append(row)

                if row["status"] == "ok":
                    print(f"    JSD={row['jsd_mean']:.4f}, "
                          f"Pearson={row['pearson_mean']:.3f}, "
                          f"RMSE={row['summed_rmse']:.4f}")
                else:
                    print(f"    status={row['status']}")

    results_df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    summary_df = build_summary(results_df)
    summary_path = args.output.parent / "deconvolution_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ok_count = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"\nDone: {ok_count}/{len(all_rows)} benchmark rows ok")
    print(f"Results: {args.output}")
    print(f"Summary: {summary_path}")

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
