#!/usr/bin/env python3
"""Build simulator quality metrics from saved outputs.

Compares each matched simulated output against its reference input and
reports 10 interpretable atomic metrics — no composite scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT))

IDENTITY_MEAN_CORR_THRESHOLD = 0.995
IDENTITY_VAR_CORR_THRESHOLD = 0.95
IDENTITY_ZERO_JACCARD_THRESHOLD = 0.95

METRIC_VERSION = "figure2_simulator_quality_v3"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _gene_means(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=0)


def _gene_variances(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=0)


def _gene_zero_fractions(X: np.ndarray) -> np.ndarray:
    return np.mean(X == 0, axis=0)


def _library_sizes(X: np.ndarray) -> np.ndarray:
    return np.sum(X, axis=1)


def _wasserstein(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if len(vec1) == 0 or len(vec2) == 0:
        return float("nan")
    return float(wasserstein_distance(vec1, vec2))


def _safe_read(path: str) -> ad.AnnData | None:
    try:
        return ad.read_h5ad(path)
    except Exception as exc:
        print(f"  ERROR reading {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Metrics (10)
# ---------------------------------------------------------------------------


def input_mean_corr(ref: ad.AnnData, sim: ad.AnnData) -> float:
    """log1p Pearson correlation of per-gene means."""
    import scipy.stats as st

    real_m = _gene_means(_dense(ref.X))
    sim_m = _gene_means(_dense(sim.X))
    mask = (real_m > 1e-10) & (sim_m > 1e-10)
    if mask.sum() < 10:
        return 0.0
    c, _ = st.pearsonr(np.log1p(real_m[mask]), np.log1p(sim_m[mask]))
    return float(c) if np.isfinite(c) else 0.0


def input_variance_corr(ref: ad.AnnData, sim: ad.AnnData) -> float:
    """log1p Pearson correlation of per-gene variances."""
    import scipy.stats as st

    real_v = _gene_variances(_dense(ref.X))
    sim_v = _gene_variances(_dense(sim.X))
    mask = (real_v > 1e-10) & (sim_v > 1e-10)
    if mask.sum() < 10:
        return 0.0
    c, _ = st.pearsonr(np.log1p(real_v[mask]), np.log1p(sim_v[mask]))
    return float(c) if np.isfinite(c) else 0.0


def zero_mask_jaccard(ref: ad.AnnData, sim: ad.AnnData) -> float:
    """Jaccard index of boolean zero-entry positions."""
    real_z = _dense(ref.X) == 0
    sim_z = _dense(sim.X) == 0
    inter = (real_z & sim_z).sum()
    union = (real_z | sim_z).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def cosine_divergence(ref: ad.AnnData, sim: ad.AnnData) -> float:
    """Median per-spot 1 - cosine_similarity on log1p-transformed counts."""
    ref_X = np.log1p(_dense(ref.X))
    sim_X = np.log1p(_dense(sim.X))
    dots = np.sum(ref_X * sim_X, axis=1)
    norms = np.linalg.norm(ref_X, axis=1) * np.linalg.norm(sim_X, axis=1)
    div = 1.0 - dots / np.maximum(norms, 1e-12)
    return float(np.median(div))


def relative_error_mean(ref: ad.AnnData, sim: ad.AnnData) -> float:
    """Mean absolute relative error of per-gene means."""
    real_m = _gene_means(_dense(ref.X))
    sim_m = _gene_means(_dense(sim.X))
    re = np.abs(real_m - sim_m) / (real_m + 1e-10)
    val = float(np.mean(re))
    return val if np.isfinite(val) else 1.0


def gene_zero_fraction_wasserstein(ref: ad.AnnData, sim: ad.AnnData) -> float:
    return _wasserstein(
        _gene_zero_fractions(_dense(ref.X)), _gene_zero_fractions(_dense(sim.X))
    )


def gene_mean_wasserstein(ref: ad.AnnData, sim: ad.AnnData) -> float:
    return _wasserstein(
        _gene_means(_dense(ref.X)), _gene_means(_dense(sim.X))
    )


def gene_variance_wasserstein(ref: ad.AnnData, sim: ad.AnnData) -> float:
    return _wasserstein(
        _gene_variances(_dense(ref.X)), _gene_variances(_dense(sim.X))
    )


def library_size_wasserstein(ref: ad.AnnData, sim: ad.AnnData) -> float:
    return _wasserstein(
        _library_sizes(_dense(ref.X)), _library_sizes(_dense(sim.X))
    )


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


def _moran_i_per_gene(
    adata: ad.AnnData, max_genes: int = 300
) -> Optional[np.ndarray]:
    """Return per-gene Moran's I for top-expressed genes using squidpy, or None."""
    try:
        import squidpy as sq
    except ImportError:
        return None

    if "spatial" not in adata.obsm:
        return None

    X = _dense(adata.X)
    gene_means = np.mean(X, axis=0)
    n_genes = min(max_genes, adata.n_vars)
    if n_genes < 5:
        return None
    top_idx = np.argsort(gene_means)[-n_genes:]
    top_genes = [adata.var_names[i] for i in top_idx]

    import warnings
    work = adata.copy()
    # Ensure spatial neighbors graph exists (some external simulators lack it)
    if 'spatial_connectivities' not in work.obsp:
        try:
            sq.gr.spatial_neighbors(work, n_neighs=6, copy=False)
        except Exception:
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sq.gr.spatial_autocorr(
            work, mode="moran", genes=top_genes, n_perms=10, n_jobs=1, copy=False,
        )
    moran_df = work.uns.get("moranI")
    if moran_df is None:
        return None
    values = moran_df["I"].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    return values


def moran_i_correlation(ref: ad.AnnData, sim: ad.AnnData) -> float:
    import scipy.stats as st

    real_m = _moran_i_per_gene(ref)
    sim_m = _moran_i_per_gene(sim)
    if real_m is None or sim_m is None:
        return float("nan")
    if len(real_m) < 5 or len(sim_m) < 5:
        return float("nan")
    c, _ = st.pearsonr(real_m, sim_m)
    return float(c) if np.isfinite(c) else float("nan")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "simulator",
    "sample",
    "identity_flag",
    "input_mean_corr",
    "input_variance_corr",
    "cosine_divergence",
    "relative_error_mean",
    "zero_mask_jaccard",
    "gene_zero_fraction_wasserstein",
    "gene_mean_wasserstein",
    "gene_variance_wasserstein",
    "library_size_wasserstein",
    "moran_i_correlation",
]


def compute_metrics(
    inventory_path: Path,
    exper_data: Path,
    metrics_path: Path,
    metadata_path: Path | None = None,
) -> int:
    inventory = pd.read_csv(inventory_path)
    matched = inventory[inventory["status"] == "matched"]

    if matched.empty:
        print("No matched simulator-sample pairs found.")
        return 1

    rows = []
    for _, rec in matched.iterrows():
        sim_label = rec["simulator"]
        sample = rec["sample"]
        sim_path = Path(rec["file_path"])
        ref_path = exper_data / f"{sample}.h5ad"

        print(f"  {sim_label}:{sample} ...", end=" ")

        if not ref_path.exists():
            print("reference missing")
            continue

        ref = _safe_read(str(ref_path))
        sim = _safe_read(str(sim_path))
        if ref is None or sim is None:
            continue

        try:
            imc = input_mean_corr(ref, sim)
            ivc = input_variance_corr(ref, sim)
            zmj = zero_mask_jaccard(ref, sim)
            identity_flag = bool(
                imc >= IDENTITY_MEAN_CORR_THRESHOLD
                and ivc >= IDENTITY_VAR_CORR_THRESHOLD
                and zmj >= IDENTITY_ZERO_JACCARD_THRESHOLD
            )

            spatial_ok = (
                "spatial" in ref.obsm
                and "spatial" in sim.obsm
                and ref.n_obs >= 6
            )

            row = {
                "simulator": sim_label,
                "sample": sample,
                "identity_flag": identity_flag,
                "input_mean_corr": round(imc, 10),
                "input_variance_corr": round(ivc, 10),
                "cosine_divergence": round(cosine_divergence(ref, sim), 10),
                "relative_error_mean": round(relative_error_mean(ref, sim), 10),
                "zero_mask_jaccard": round(zmj, 10),
                "gene_zero_fraction_wasserstein": round(
                    gene_zero_fraction_wasserstein(ref, sim), 10
                ),
                "gene_mean_wasserstein": round(gene_mean_wasserstein(ref, sim), 10),
                "gene_variance_wasserstein": round(
                    gene_variance_wasserstein(ref, sim), 10
                ),
                "library_size_wasserstein": round(
                    library_size_wasserstein(ref, sim), 10
                ),
                "moran_i_correlation": round(moran_i_correlation(ref, sim), 10)
                if spatial_ok
                else float("nan"),
            }
            rows.append(row)

            flag = " IDENTITY" if identity_flag else ""
            moran_str = " moran=OK" if (spatial_ok and np.isfinite(row["moran_i_correlation"])) else ""
            print(f"mean_corr={imc:.4f} var_corr={ivc:.4f}{flag}{moran_str} OK")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            # Still write row without Moran's I so other metrics are preserved
            try:
                imc = input_mean_corr(ref, sim)
                ivc = input_variance_corr(ref, sim)
                zmj = zero_mask_jaccard(ref, sim)
                identity_flag = bool(
                    imc >= IDENTITY_MEAN_CORR_THRESHOLD
                    and ivc >= IDENTITY_VAR_CORR_THRESHOLD
                    and zmj >= IDENTITY_ZERO_JACCARD_THRESHOLD
                )
                row = {
                    "simulator": sim_label,
                    "sample": sample,
                    "identity_flag": identity_flag,
                    "input_mean_corr": round(imc, 10),
                    "input_variance_corr": round(ivc, 10),
                    "cosine_divergence": round(cosine_divergence(ref, sim), 10),
                    "relative_error_mean": round(relative_error_mean(ref, sim), 10),
                    "zero_mask_jaccard": round(zmj, 10),
                    "gene_zero_fraction_wasserstein": round(gene_zero_fraction_wasserstein(ref, sim), 10),
                    "gene_mean_wasserstein": round(gene_mean_wasserstein(ref, sim), 10),
                    "gene_variance_wasserstein": round(gene_variance_wasserstein(ref, sim), 10),
                    "library_size_wasserstein": round(library_size_wasserstein(ref, sim), 10),
                    "moran_i_correlation": float("nan"),
                }
                rows.append(row)
                print(f"  recovered (moran=nan) OK")
            except Exception as exc2:
                print(f"  unrecoverable: {exc2}")

    if not rows:
        print("No metrics computed.")
        return 1

    metrics = pd.DataFrame(rows)[OUTPUT_COLUMNS]

    # --- Composite scores ---
    m = metrics
    m['gene_zero_preservation'] = np.clip(1.0 - m['gene_zero_fraction_wasserstein'] / max(m['gene_zero_fraction_wasserstein'].quantile(0.90), 1e-10), 0, 1)
    m['spot_zero_preservation'] = m['zero_mask_jaccard']
    m['zero_preservation_score'] = 0.5 * m['gene_zero_preservation'] + 0.5 * m['spot_zero_preservation']

    m['mean_structure_score'] = np.clip(m['input_mean_corr'], 0, 1)
    m['variance_structure_score'] = np.clip(m['input_variance_corr'], 0, 1)
    lib_max = max(m['library_size_wasserstein'].quantile(0.90), 1e-10)
    m['library_structure_score'] = np.clip(1.0 - m['library_size_wasserstein'] / lib_max, 0, 1)
    m['statistical_structure_score'] = 0.40 * m['mean_structure_score'] + 0.35 * m['variance_structure_score'] + 0.25 * m['library_structure_score']

    m['raw_novelty'] = np.clip(1.0 - m['input_mean_corr'], 0, 1)
    structure_retention = 0.5 * m['input_variance_corr'] + 0.5 * m['spot_zero_preservation']
    m['structured_novelty_score'] = m['raw_novelty'] * structure_retention
    m['structured_novelty_score'] = m['structured_novelty_score'] * np.where(m['identity_flag'], 0.05, 1.0)

    m['composite_quality_score'] = (
        0.30 * m['zero_preservation_score'] +
        0.35 * m['statistical_structure_score'] +
        0.35 * m['structured_novelty_score']
    )

    composite_cols = ['zero_preservation_score', 'statistical_structure_score',
                      'structured_novelty_score', 'composite_quality_score']
    for c in composite_cols:
        if c not in OUTPUT_COLUMNS:
            OUTPUT_COLUMNS.append(c)

    metrics = metrics[OUTPUT_COLUMNS]
    metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics written to {metrics_path} ({len(metrics)} rows)")

    # Metadata
    _metadata_path = metadata_path or Path(
        str(metrics_path).replace(".csv", "_metadata.json")
    )
    metadata = {
        "metric_version": METRIC_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "identity_thresholds": {
            "mean_corr": IDENTITY_MEAN_CORR_THRESHOLD,
            "variance_corr": IDENTITY_VAR_CORR_THRESHOLD,
            "zero_jaccard": IDENTITY_ZERO_JACCARD_THRESHOLD,
        },
        "n_inventory_rows": len(inventory),
        "n_matched_rows": len(matched),
        "n_metric_rows": len(metrics),
        "columns": OUTPUT_COLUMNS,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {_metadata_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, required=True,
        help="Path to simulator_inventory.csv",
    )
    parser.add_argument(
        "--exper-data", type=Path, required=True,
        help="Path to exper_data/ with reference .h5ad files",
    )
    parser.add_argument(
        "--metrics", type=Path, required=True,
        help="Output path for simulator_quality_metrics.csv",
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Output path for metadata JSON",
    )
    args = parser.parse_args()
    return compute_metrics(
        args.inventory, args.exper_data, args.metrics,
        metadata_path=args.metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
