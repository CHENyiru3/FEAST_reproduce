#!/usr/bin/env python3
"""Build the prespecified 60-row non-composite Study 04 metric table."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

import run as workflow


STUDY_ROOT = Path(__file__).resolve().parent
K_NEIGHBORS = 30
MMD_SAMPLE_PER_BATCH = 1000
METRIC_COLUMNS = (
    "batch_asw",
    "batch_mixing_entropy_k30",
    "paired_retrieval_top1",
    "paired_retrieval_median_rank",
    "domain_asw",
    "domain_purity_k30",
    "centroid_distance",
    "covariance_distance",
    "mmd2_rbf_biased",
    "paired_expression_correlation",
)


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def standardize(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def neighbors_without_self(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[0] <= k:
        raise ValueError(f"{values.shape[0]} rows cannot support k={k}")
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(
        values,
        return_distance=False,
    )
    rows = []
    for index, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != index]
        if without_self.size < k:
            raise RuntimeError("nearest-neighbor result lacks enough non-self rows")
        rows.append(without_self[:k])
    return np.vstack(rows)


def local_batch_entropy(batch_labels: np.ndarray, neighbors: np.ndarray) -> float:
    query_fraction = (batch_labels[neighbors] == "query").mean(axis=1)
    entropy = np.zeros(query_fraction.size, dtype=np.float64)
    for probability in (query_fraction, 1.0 - query_fraction):
        nonzero = probability > 0
        entropy[nonzero] -= probability[nonzero] * np.log(probability[nonzero])
    return float(np.mean(entropy / np.log(2.0)))


def domain_purity(
    domain_labels: np.ndarray,
    valid: np.ndarray,
    neighbors: np.ndarray,
) -> float:
    neighbor_labels = domain_labels[neighbors]
    purity = (neighbor_labels == domain_labels[:, None]).mean(axis=1)
    return float(np.mean(purity[valid]))


def paired_retrieval(values: np.ndarray, n_reference: int) -> tuple[float, float]:
    reference = values[:n_reference]
    query = values[n_reference:]
    if reference.shape != query.shape:
        raise ValueError("paired retrieval requires equal reference/query support")
    distances = pairwise_distances(query, reference, metric="euclidean")
    paired = np.diag(distances)
    minimum = distances.min(axis=1)
    tolerance = 1e-12 + 1e-9 * np.maximum(1.0, np.abs(minimum))
    top1 = float(np.mean(paired <= minimum + tolerance))
    ranks = 1 + np.sum(distances < paired[:, None] - tolerance[:, None], axis=1)
    return top1, float(np.median(ranks))


def mmd2_rbf_biased(
    values: np.ndarray,
    n_reference: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    query_count = values.shape[0] - n_reference
    ref_index = rng.choice(
        n_reference,
        min(MMD_SAMPLE_PER_BATCH, n_reference),
        replace=False,
    )
    query_index = rng.choice(
        query_count,
        min(MMD_SAMPLE_PER_BATCH, query_count),
        replace=False,
    )
    x = values[ref_index]
    y = values[n_reference + query_index]
    pooled = np.vstack([x, y])
    distances = pairwise_distances(pooled, metric="euclidean")
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = (
        rbf_kernel(x, x, gamma=gamma).mean()
        + rbf_kernel(y, y, gamma=gamma).mean()
        - 2.0 * rbf_kernel(x, y, gamma=gamma).mean()
    )
    return float(max(value, 0.0)), bandwidth


def paired_expression_correlation(matrix: np.ndarray, n_reference: int) -> float:
    reference = np.asarray(matrix[:n_reference], dtype=np.float64)
    query = np.asarray(matrix[n_reference:], dtype=np.float64)
    if reference.shape != query.shape:
        raise ValueError("expression correlation requires equal paired support")
    reference -= reference.mean(axis=1, keepdims=True)
    query -= query.mean(axis=1, keepdims=True)
    numerator = np.sum(reference * query, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(query, axis=1)
    valid = denominator > 0
    correlations = numerator[valid] / denominator[valid]
    return float(np.median(correlations)) if correlations.size else float("nan")


def representations(
    method: str,
    embedding: np.ndarray,
    seed: int,
) -> list[tuple[str, bool, np.ndarray]]:
    if method != "GraphST":
        return [("native_10d", True, embedding)]
    return [
        (
            "pca20_randomized",
            True,
            PCA(n_components=20, svd_solver="randomized", random_state=seed).fit_transform(
                embedding
            ),
        ),
        (
            "pca10_randomized_sensitivity",
            False,
            PCA(n_components=10, svd_solver="randomized", random_state=seed).fit_transform(
                embedding
            ),
        ),
    ]


def metric_row(
    *,
    method: str,
    representation: str,
    primary: bool,
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    domain_labels: np.ndarray,
    n_reference: int,
    expression_correlation: float | None,
    seed: int,
) -> dict[str, object]:
    values = standardize(embedding)
    neighbors = neighbors_without_self(values, K_NEIGHBORS)
    domain_text = pd.Series(domain_labels).astype("string").fillna("<NA>").to_numpy()
    valid_domain = ~np.isin(
        np.char.lower(domain_text.astype(str)),
        ["nan", "none", "unknown", "<na>"],
    )
    if len(np.unique(domain_text[valid_domain])) < 2:
        raise ValueError("fewer than two valid domains remain")
    top1, median_rank = paired_retrieval(values, n_reference)
    mmd2, bandwidth = mmd2_rbf_biased(values, n_reference, seed)
    reference = values[:n_reference]
    query = values[n_reference:]
    return {
        "method": method,
        "representation": representation,
        "primary_representation": primary,
        "representation_dimensions": int(values.shape[1]),
        "batch_asw": float(silhouette_score(values, batch_labels)),
        "batch_mixing_entropy_k30": local_batch_entropy(batch_labels, neighbors),
        "paired_retrieval_top1": top1,
        "paired_retrieval_median_rank": median_rank,
        "domain_asw": float(silhouette_score(values[valid_domain], domain_text[valid_domain])),
        "domain_purity_k30": domain_purity(domain_text, valid_domain, neighbors),
        "centroid_distance": float(
            np.linalg.norm(reference.mean(axis=0) - query.mean(axis=0))
        ),
        "covariance_distance": float(
            np.linalg.norm(
                np.cov(reference, rowvar=False) - np.cov(query, rowvar=False),
                ord="fro",
            )
            / values.shape[1]
        ),
        "mmd2_rbf_biased": mmd2,
        "mmd_bandwidth": bandwidth,
        "paired_expression_correlation": expression_correlation,
    }


def load_panel(path: Path) -> list[str]:
    table = pd.read_csv(path)
    if list(table.columns[:2]) != ["panel_order", "gene"]:
        raise ValueError("panel must begin with panel_order,gene")
    if table["panel_order"].astype(int).tolist() != list(range(len(table))):
        raise ValueError("panel order is not contiguous")
    genes = table["gene"].astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise ValueError("panel contains duplicate genes")
    return genes


def load_support(
    reference_path: Path,
    query_path: Path,
    genes: list[str],
) -> dict[str, object]:
    reference = ad.read_h5ad(reference_path, backed="r")
    query = ad.read_h5ad(query_path, backed="r")
    try:
        if not reference.obs_names.equals(query.obs_names):
            raise ValueError("reference/query spot identity or order differs")
        if any(gene not in reference.var_names or gene not in query.var_names for gene in genes):
            raise ValueError("fixed-panel support is incomplete")
        reference_matrix = reference[:, genes].X
        query_matrix = query[:, genes].X
        reference_zero = np.asarray(reference_matrix.sum(axis=1)).reshape(-1) == 0
        query_zero = np.asarray(query_matrix.sum(axis=1)).reshape(-1) == 0
        remove = reference_zero | query_zero
        result = {
            "n_spots": int(reference.n_obs),
            "spot_ids": reference.obs_names.astype(str).to_numpy(copy=True),
            "remove_pairs": remove,
            "removed_pair_count": int(remove.sum()),
            "reference_sha256": workflow.sha256_file(reference_path),
            "query_sha256": workflow.sha256_file(query_path),
        }
    finally:
        reference.file.close()
        query.file.close()
    return result


def load_candidate(
    candidate: Path,
    method: str,
    candidate_id: str,
    config: dict,
    support: dict[str, object],
) -> dict[str, object]:
    valid, reason = workflow.candidate_is_valid(candidate, method, candidate_id, config)
    if not valid:
        raise ValueError(f"invalid candidate {candidate}: {reason}")
    with np.load(candidate / "embeddings.npz", allow_pickle=True) as payload:
        embedding = np.asarray(payload["embedding"], dtype=np.float64)
        batch_labels = np.asarray(payload["batch_labels"]).astype(str)
    n_spots = int(support["n_spots"])
    if embedding.shape[0] != 2 * n_spots or not np.isfinite(embedding).all():
        raise ValueError("candidate embedding support or finiteness failed")
    expected_batches = np.array(["ref"] * n_spots + ["query"] * n_spots)
    if not np.array_equal(batch_labels, expected_batches):
        raise ValueError("candidate batch order differs from the paired contract")

    result_path = candidate / "result.h5ad"
    result = ad.read_h5ad(result_path, backed="r")
    try:
        source_ids = result.obs["source_spot_id"].astype(str).to_numpy(copy=True)
        expected_ids = np.concatenate([support["spot_ids"], support["spot_ids"]])
        if not np.array_equal(source_ids, expected_ids):
            raise ValueError("method output changed paired spot identity/order")
        domains = result.obs["dlpfc_layer"].to_numpy(copy=True)
        normalized = None
        if method == "scVI":
            normalized = np.asarray(result.layers["scvi_normalized"])
            if normalized.shape[0] != 2 * n_spots or not np.isfinite(normalized).all():
                raise ValueError("scVI normalized expression is invalid")
    finally:
        result.file.close()
    return {
        "embedding": embedding,
        "batch_labels": batch_labels,
        "domain_labels": domains,
        "normalized": normalized,
        "embedding_sha256": workflow.sha256_file(candidate / "embeddings.npz"),
        "result_sha256": workflow.sha256_file(result_path),
        "metadata_sha256": workflow.sha256_file(candidate / "metadata.json"),
    }


def row_set(
    *,
    method: str,
    mode: str,
    alpha: float,
    candidate_id: str,
    candidate: dict[str, object],
    support: dict[str, object],
    support_variant: str,
    keep: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    n_full = int(support["n_spots"])
    reference_keep = keep[:n_full]
    query_keep = keep[n_full:]
    n_reference = int(reference_keep.sum())
    if n_reference != int(query_keep.sum()):
        raise ValueError("reference/query reduced supports differ")
    embedding = candidate["embedding"][keep]
    batches = candidate["batch_labels"][keep]
    domains = candidate["domain_labels"][keep]
    expression_correlation = None
    if method == "scVI":
        expression_correlation = paired_expression_correlation(
            candidate["normalized"][keep],
            n_reference,
        )
    rows = []
    for representation, primary, values in representations(method, embedding, seed):
        row = metric_row(
            method=method,
            representation=representation,
            primary=primary,
            embedding=values,
            batch_labels=batches,
            domain_labels=domains,
            n_reference=n_reference,
            expression_correlation=expression_correlation,
            seed=seed,
        )
        row.update(
            {
                "study_id": "04",
                "legacy_study_id": "08",
                "mode": mode,
                "alpha": f"{alpha:.2f}",
                "alpha_stratum": "interpolation" if alpha <= 1.0 else "extrapolation",
                "support_variant": support_variant,
                "removed_paired_spots": int(support["removed_pair_count"]),
                "n_reference": n_reference,
                "n_query": n_reference,
                "method_candidate_id": candidate_id,
                "public_seed": seed,
                "reference_sha256": support["reference_sha256"],
                "query_sha256": support["query_sha256"],
                "embedding_sha256": candidate["embedding_sha256"],
                "result_sha256": candidate["result_sha256"],
            }
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    run_dir = args.run_dir.resolve() if args.run_dir else config["_output_dir"]
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "metrics"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite metric output: {output_dir}")

    panel_path = run_dir / "panel" / "panel.csv"
    panel_provenance = run_dir / "panel" / "provenance.json"
    genes = load_panel(panel_path)
    if len(genes) != int(config["fixed_panel"]["n_genes"]):
        raise ValueError("fixed-panel size differs from the config")
    declared_support_cells = {
        (str(mode), float(alpha))
        for mode, alpha in config["metric_contract"]["common_support_cells"]
    }
    seed = int(config["public_seed"])
    rows: list[dict[str, object]] = []
    provenance_inputs = [
        {"path": str(panel_path.resolve()), "sha256": workflow.sha256_file(panel_path)},
        {
            "path": str(panel_provenance.resolve()),
            "sha256": workflow.sha256_file(panel_provenance),
        },
    ]
    observed_support_cells = set()
    for mode in config["simulation"]["modes"]:
        reference_path = run_dir / "simulations" / mode / "alpha_0.00.h5ad"
        for alpha_value in config["simulation"]["alpha_levels"]:
            alpha = float(alpha_value)
            if alpha == 0.0:
                continue
            query_path = run_dir / "simulations" / mode / f"alpha_{alpha:.2f}.h5ad"
            support = load_support(reference_path, query_path, genes)
            if support["removed_pair_count"]:
                observed_support_cells.add((mode, alpha))
            for method in workflow.METHOD_ARTIFACTS:
                candidate_id = (
                    f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha:.2f}"
                )
                candidate_path = run_dir / "methods" / method / mode / f"alpha_{alpha:.2f}"
                candidate = load_candidate(
                    candidate_path,
                    method,
                    candidate_id,
                    config,
                    support,
                )
                n_spots = int(support["n_spots"])
                all_keep = np.ones(2 * n_spots, dtype=bool)
                rows.extend(
                    row_set(
                        method=method,
                        mode=mode,
                        alpha=alpha,
                        candidate_id=candidate_id,
                        candidate=candidate,
                        support=support,
                        support_variant="all_spots_primary",
                        keep=all_keep,
                        seed=seed,
                    )
                )
                if (mode, alpha) in declared_support_cells:
                    remove = support["remove_pairs"]
                    if not remove.any():
                        raise ValueError(
                            f"declared common-support cell has no zero-panel pair: {mode}/{alpha:.2f}"
                        )
                    keep_pair = ~remove
                    paired_keep = np.concatenate([keep_pair, keep_pair])
                    rows.extend(
                        row_set(
                            method=method,
                            mode=mode,
                            alpha=alpha,
                            candidate_id=candidate_id,
                            candidate=candidate,
                            support=support,
                            support_variant="zero_panel_pairs_excluded_sensitivity",
                            keep=paired_keep,
                            seed=seed,
                        )
                    )
                provenance_inputs.extend(
                    [
                        {
                            "path": str((candidate_path / "metadata.json").resolve()),
                            "sha256": candidate["metadata_sha256"],
                        },
                        {
                            "path": str((candidate_path / "embeddings.npz").resolve()),
                            "sha256": candidate["embedding_sha256"],
                        },
                        {
                            "path": str((candidate_path / "result.h5ad").resolve()),
                            "sha256": candidate["result_sha256"],
                        },
                    ]
                )
            print(f"scored {mode}/alpha_{alpha:.2f}", flush=True)

    if observed_support_cells != declared_support_cells:
        raise ValueError(
            "zero-panel sensitivity cells changed: "
            f"expected={sorted(declared_support_cells)}, "
            f"observed={sorted(observed_support_cells)}"
        )
    table = pd.DataFrame(rows).sort_values(
        ["method", "mode", "alpha", "support_variant", "primary_representation"],
        ascending=[True, True, True, True, False],
    )
    expected_rows = int(config["metric_contract"]["expected_rows"])
    if len(table) != expected_rows:
        raise ValueError(f"metric row count is {len(table)}; expected {expected_rows}")
    if not np.isfinite(table[list(METRIC_COLUMNS[:-1])].to_numpy(dtype=float)).all():
        raise ValueError("a required atomic metric is non-finite")
    if table["paired_expression_correlation"].notna().sum() != 15:
        raise ValueError("scVI expression-correlation row accounting changed")

    primary = table[
        (table["support_variant"] == "all_spots_primary")
        & table["primary_representation"]
    ]
    summary = (
        primary.groupby(
            ["method", "mode", "alpha_stratum", "representation"],
            dropna=False,
        )[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    if len(summary) != 12:
        raise ValueError(f"primary summary has {len(summary)} rows; expected 12")

    sensitivity = table[
        table["support_variant"] == "zero_panel_pairs_excluded_sensitivity"
    ].merge(
        table[table["support_variant"] == "all_spots_primary"],
        on=["method", "mode", "alpha", "representation"],
        how="left",
        suffixes=("_sensitivity", "_all_spots"),
        validate="one_to_one",
    )
    delta_columns = {}
    for metric in METRIC_COLUMNS:
        delta_columns[f"delta_{metric}"] = (
            sensitivity[f"{metric}_sensitivity"]
            - sensitivity[f"{metric}_all_spots"]
        )
    sensitivity_deltas = sensitivity[
        ["method", "mode", "alpha", "representation"]
    ].assign(**delta_columns)
    if len(sensitivity_deltas) != 12:
        raise ValueError("common-support sensitivity accounting changed")

    output_dir.mkdir(parents=True)
    table_path = output_dir / "atomic_metrics.csv"
    summary_path = output_dir / "primary_summary.csv"
    sensitivity_path = output_dir / "common_support_sensitivity.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    sensitivity_deltas.to_csv(sensitivity_path, index=False)
    summary_json = {
        "schema_version": 1,
        "configuration_id": f"{config['configuration_id']}__atomic_metrics",
        "study_id": "04",
        "legacy_study_id": "08",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(table),
        "all_spots_rows": int((table["support_variant"] == "all_spots_primary").sum()),
        "common_support_rows": int(
            (table["support_variant"] == "zero_panel_pairs_excluded_sensitivity").sum()
        ),
        "primary_summary_rows": len(summary),
        "composite_score": "prohibited_not_computed",
        "winner_ranking": "prohibited_not_computed",
        "alpha_strata": {
            "interpolation": "alpha <= 1.00",
            "extrapolation": "alpha > 1.00",
        },
        "decision": "fresh_atomic_metrics_complete_pending_scientific_review",
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    output_paths = [table_path, summary_path, sensitivity_path, summary_json_path]
    provenance = {
        "schema_version": 1,
        "configuration_id": f"{config['configuration_id']}__atomic_metrics",
        "study_id": "04",
        "legacy_study_id": "08",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "feast": workflow.feast_identity(config),
        "public_seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "anndata": ad.__version__,
            "sklearn": sklearn.__version__,
        },
        "sources": [
            {"path": str(Path(__file__).resolve()), "sha256": workflow.sha256_file(Path(__file__).resolve())},
            {"path": str(config["_config_path"]), "sha256": workflow.sha256_file(config["_config_path"])},
        ],
        "inputs": provenance_inputs,
        "outputs": [
            {"path": str(path.resolve()), "sha256": workflow.sha256_file(path)}
            for path in output_paths
        ],
        "diagnostics": {
            "status": "completed",
            "rows": len(table),
            "validated_method_candidates": 36,
            "composite_score": "prohibited_not_computed",
            "winner_ranking": "prohibited_not_computed",
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(table)} rows to {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
