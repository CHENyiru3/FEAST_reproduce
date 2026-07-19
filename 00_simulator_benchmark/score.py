#!/usr/bin/env python3
"""Compute corrected atomic Study 00 simulator metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_stats(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64)
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        squared = matrix.copy()
        squared.data **= 2
        mean_squared = np.asarray(squared.mean(axis=0)).ravel()
        zero = 1.0 - matrix.getnnz(axis=0) / matrix.shape[0]
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        mean = matrix.mean(axis=0)
        mean_squared = np.square(matrix).mean(axis=0)
        zero = np.mean(matrix == 0, axis=0)
    return mean, np.maximum(mean_squared - mean**2, 0.0), np.asarray(zero)


def log_corr(left: np.ndarray, right: np.ndarray) -> float:
    keep = (left > 1e-10) & (right > 1e-10)
    if keep.sum() < 10:
        return 0.0
    value = pearsonr(np.log1p(left[keep]), np.log1p(right[keep])).statistic
    return float(value) if np.isfinite(value) else 0.0


def zero_jaccard(left, right) -> float:
    total = left.shape[0] * left.shape[1]
    left = sp.csr_matrix(left).astype(bool)
    right = sp.csr_matrix(right).astype(bool)
    zero_intersection = total - left.maximum(right).nnz
    zero_union = total - left.multiply(right).nnz
    return 1.0 if zero_union == 0 else float(zero_intersection / zero_union)


def cosine_divergence(left, right) -> float:
    left = sp.csr_matrix(left).astype(np.float64)
    right = sp.csr_matrix(right).astype(np.float64)
    left.data = np.log1p(left.data)
    right.data = np.log1p(right.data)
    dot = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    return float(np.median(1.0 - dot / np.maximum(left_norm * right_norm, 1e-12)))


def graph(coords: np.ndarray, neighbors: int) -> sp.csr_matrix:
    model = NearestNeighbors(n_neighbors=neighbors + 1).fit(coords)
    raw = model.kneighbors(coords, return_distance=False)
    selected = np.vstack([row[row != index][:neighbors] for index, row in enumerate(raw)])
    rows = np.repeat(np.arange(len(coords)), neighbors)
    return sp.csr_matrix(
        (np.full(rows.size, 1.0 / neighbors), (rows, selected.ravel())),
        shape=(len(coords), len(coords)),
    )


def moran(matrix, indices: np.ndarray, weights: sp.csr_matrix) -> np.ndarray:
    result = []
    for start in range(0, len(indices), 25):
        block = matrix[:, indices[start : start + 25]]
        values = block.toarray() if sp.issparse(block) else np.asarray(block)
        centered = values - values.mean(axis=0)
        denominator = np.square(centered).sum(axis=0)
        numerator = np.sum(centered * (weights @ centered), axis=0)
        result.extend(np.where(denominator > 0, numerator / denominator, np.nan))
    return np.asarray(result)


def pair_metrics(ref: ad.AnnData, sim: ad.AnnData, max_genes: int, neighbors: int):
    if not ref.var_names.is_unique or not sim.var_names.is_unique:
        raise ValueError("gene identifiers must be unique")
    genes = ref.var_names[ref.var_names.isin(sim.var_names)]
    if genes.empty:
        raise ValueError("no exact shared gene identifiers")
    ri = ref.var_names.get_indexer(genes)
    si = sim.var_names.get_indexer(genes)
    rm, rv, rz = gene_stats(ref.X[:, ri])
    sm, sv, sz = gene_stats(sim.X[:, si])
    spots = ref.obs_names[ref.obs_names.isin(sim.obs_names)]
    if len(spots):
        ro = ref.obs_names.get_indexer(spots)
        so = sim.obs_names.get_indexer(spots)
        paired_ref = ref.X[ro, :][:, ri]
        paired_sim = sim.X[so, :][:, si]
        jaccard = zero_jaccard(paired_ref, paired_sim)
        cosine = cosine_divergence(paired_ref, paired_sim)
    else:
        jaccard = cosine = np.nan
    top = np.argsort(rm, kind="stable")[-min(max_genes, len(genes)) :][::-1]
    panel = genes[top]
    moran_corr = np.nan
    ref_moran = sim_moran = np.full(len(panel), np.nan)
    if len(panel) >= 5 and "spatial" in ref.obsm and "spatial" in sim.obsm:
        ref_weights = graph(np.asarray(ref.obsm["spatial"]), neighbors)
        sim_weights = (
            ref_weights
            if ref.obs_names.equals(sim.obs_names)
            and np.array_equal(ref.obsm["spatial"], sim.obsm["spatial"])
            else graph(np.asarray(sim.obsm["spatial"]), neighbors)
        )
        ref_moran = moran(ref.X, ref.var_names.get_indexer(panel), ref_weights)
        sim_moran = moran(sim.X, sim.var_names.get_indexer(panel), sim_weights)
        valid = np.isfinite(ref_moran) & np.isfinite(sim_moran)
        if valid.sum() >= 5:
            moran_corr = float(pearsonr(ref_moran[valid], sim_moran[valid]).statistic)
    mean_corr = log_corr(rm, sm)
    variance_corr = log_corr(rv, sv)
    metrics = {
        "identity_flag": bool(
            len(spots)
            and mean_corr >= 0.995
            and variance_corr >= 0.95
            and jaccard >= 0.95
        ),
        "n_common_genes": len(genes),
        "n_paired_spots": len(spots),
        "spot_pairing": "exact_identifier" if len(spots) else "unavailable",
        "n_moran_genes": len(panel),
        "moran_panel_sha256": hashlib.sha256("\n".join(map(str, panel)).encode()).hexdigest(),
        "input_mean_corr": mean_corr,
        "input_variance_corr": variance_corr,
        "cosine_divergence": cosine,
        "relative_error_mean": float(np.mean(np.abs(rm - sm) / (rm + 1e-10))),
        "zero_mask_jaccard": jaccard,
        "gene_zero_fraction_wasserstein": wasserstein_distance(rz, sz),
        "gene_mean_wasserstein": wasserstein_distance(rm, sm),
        "gene_variance_wasserstein": wasserstein_distance(rv, sv),
        "library_size_wasserstein": wasserstein_distance(
            np.asarray(ref.X[:, ri].sum(axis=1)).ravel(),
            np.asarray(sim.X[:, si].sum(axis=1)).ravel(),
        ),
        "moran_i_correlation": moran_corr,
    }
    panels = [
        {"rank": rank + 1, "gene_id": str(gene), "reference_moran_i": ref_moran[rank], "simulated_moran_i": sim_moran[rank]}
        for rank, gene in enumerate(panel)
    ]
    return metrics, panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--simulation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = pd.read_csv(Path(__file__).parent / "data/input_checksums.csv")
    simulations = pd.read_csv(args.simulation_dir / "simulation_manifest.csv")
    if len(simulations) != 12 or not simulations["status"].eq("ok").all():
        raise RuntimeError("simulation manifest must contain 12 successful fresh jobs")
    simulations = simulations.set_index("sample")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, panels = [], []
    for sample in config["samples"]:
        ref_record = manifest[
            (manifest["role"] == "reference")
            & (manifest["sample"] == sample)
        ].iloc[0]
        ref_path = args.data_root / ref_record.relative_path
        if sha256(ref_path) != ref_record.sha256:
            raise RuntimeError(f"reference checksum mismatch: {ref_path}")
        ref = ad.read_h5ad(ref_path)
        feast_path = args.simulation_dir / "simulations" / f"{sample}.h5ad"
        if sha256(feast_path) != simulations.loc[sample, "output_sha256"]:
            raise RuntimeError(f"fresh FEAST checksum mismatch: {feast_path}")
        candidates = [(config["feast"]["simulator_name"], feast_path)]
        for simulator in config["external_simulators"]:
            record = manifest[
                (manifest["simulator"] == simulator)
                & (manifest["sample"] == sample)
            ].iloc[0]
            external_path = args.data_root / record.relative_path
            if sha256(external_path) != record.sha256:
                raise RuntimeError(f"external checksum mismatch: {external_path}")
            candidates.append((simulator, external_path))
        for simulator, path in candidates:
            sim = ad.read_h5ad(path)
            metric, panel = pair_metrics(ref, sim, int(config["metrics"]["max_moran_genes"]), int(config["metrics"]["spatial_neighbors"]))
            rows.append({"simulator": simulator, "sample": sample, **metric})
            panels.extend({"simulator": simulator, "sample": sample, **item} for item in panel)
            print(f"{sample}: {simulator}", flush=True)
    metrics_path = args.output_dir / "simulator_quality_metrics.csv"
    panel_path = args.output_dir / "moran_gene_panel.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(panels).to_csv(panel_path, index=False)
    provenance = {
        "configuration_id": config["configuration_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "feast_version": str(simulations["feast_version"].iloc[0]),
        "feast_commit": config["required_feast_commit"],
        "public_seed": config["public_seed"],
        "calculation_randomness": "none",
        "rules": "exact gene joins; exact named spot joins; ordered reference-derived Moran panel; no composite score",
        "row_count": len(rows),
        "input_manifest_sha256": sha256(Path(__file__).parent / "data/input_checksums.csv"),
        "simulation_manifest_sha256": sha256(args.simulation_dir / "simulation_manifest.csv"),
        "outputs": {metrics_path.name: sha256(metrics_path), panel_path.name: sha256(panel_path)},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
