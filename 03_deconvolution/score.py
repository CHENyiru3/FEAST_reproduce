#!/usr/bin/env python3
"""Score matched RCTD/Cell2location outputs on common support plus __other__."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import pearsonr


METRICS = ("jsd_mean", "pearson_mean", "summed_rmse")
HISTORICAL_DIRECTION_AUDIT_SHA256 = (
    "69e0154042552f86e0cc778ac3963c2de073226a260f2b122d954febbfec58f9"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def integer_libraries(matrix, label: str) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        data = np.asarray(matrix.data, dtype=np.float64)
        libraries = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        values = np.asarray(matrix, dtype=np.float64)
        data, libraries = values.ravel(), values.sum(axis=1)
    if not np.isfinite(data).all() or data.min(initial=0) < 0:
        raise RuntimeError(f"{label} has invalid counts")
    if np.max(np.abs(data - np.rint(data)), initial=0) > 1e-6:
        raise RuntimeError(f"{label} is not raw integer count data")
    return libraries


def read_composition(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise RuntimeError(f"{label} has invalid row/column support")
    if "__other__" in frame.columns:
        raise RuntimeError(f"{label} already contains reserved __other__")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError(f"{label} has non-finite values")
    return frame


def with_other(frame: pd.DataFrame, common: pd.Index) -> np.ndarray:
    values = np.clip(frame.loc[:, common].to_numpy(np.float64), 0, None)
    residual = np.clip(1 - values.sum(axis=1), 0, None)
    result = np.column_stack([values, residual])
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals[:, 0] <= 0):
        raise RuntimeError("scored composition contains a zero-mass row")
    return result / totals


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    midpoint = 0.5 * (truth + prediction)
    jsd = 0.5 * (
        rel_entr(truth, midpoint).sum(axis=1)
        + rel_entr(prediction, midpoint).sum(axis=1)
    ) / np.log(2)
    correlations = []
    for index in range(truth.shape[1]):
        left, right = truth[:, index], prediction[:, index]
        correlations.append(
            np.nan if np.std(left) == 0 or np.std(right) == 0
            else float(pearsonr(left, right).statistic)
        )
    result = {
        "jsd_mean": float(jsd.mean()),
        "jsd_median": float(np.median(jsd)),
        "pearson_mean": float(np.nanmean(correlations)),
        "pearson_median": float(np.nanmedian(correlations)),
        "summed_rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("deconvolution scoring produced a non-finite metric")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    simulations = pd.read_csv(args.run_dir / "simulation_manifest.csv", dtype={"slice_id": str})
    methods = pd.read_csv(args.run_dir / "method_manifest.csv", dtype={"slice_id": str})
    if (
        len(simulations) != 6
        or simulations["pair_id"].duplicated().any()
        or len(methods) != 12
        or methods[["pair_id", "method"]].duplicated().any()
        or set(methods["status"]) != {"ok"}
        or not methods.groupby("pair_id")["method"].apply(
            lambda values: set(values) == {"rctd", "cell2location"}
        ).all()
    ):
        raise RuntimeError("run manifests do not contain six simulations and 12 successful methods")

    method_lookup = {
        (row.pair_id, row.method): row for row in methods.itertuples(index=False)
    }
    rows, support_rows = [], []
    for simulation_row in simulations.itertuples(index=False):
        if (
            sha256_file(Path(simulation_row.simulation_path))
            != simulation_row.simulation_sha256
            or sha256_file(Path(simulation_row.truth_path))
            != simulation_row.truth_sha256
            or sha256_file(Path(simulation_row.reference_path))
            != simulation_row.reference_sha256
        ):
            raise RuntimeError(
                f"simulation/truth/reference hash mismatch: {simulation_row.pair_id}"
            )
        simulation = ad.read_h5ad(simulation_row.simulation_path)
        reference = ad.read_h5ad(simulation_row.reference_path)
        if not simulation.var_names.equals(reference.var_names):
            raise RuntimeError(f"gene order differs: {simulation_row.pair_id}")
        if not simulation.var_names.is_unique or not reference.var_names.is_unique:
            raise RuntimeError(f"gene IDs are not unique: {simulation_row.pair_id}")
        libraries = integer_libraries(simulation.X, f"simulation {simulation_row.pair_id}")
        integer_libraries(reference.X, f"reference {simulation_row.slice_id}")
        if not simulation.obs_names.equals(pd.Index([str(i) for i in range(simulation.n_obs)])):
            raise RuntimeError(f"noncanonical simulation obs sequence: {simulation_row.pair_id}")
        embedded = simulation.uns.get("feast_reproduce", {})
        if (
            embedded.get("feast_commit") != simulation_row.feast_commit
            or embedded.get("config_sha256") != simulation_row.config_sha256
            or int(embedded.get("public_seed", -1)) != int(simulation_row.public_seed)
        ):
            raise RuntimeError(f"simulation provenance mismatch: {simulation_row.pair_id}")
        full_spots = pd.Index([f"spot_{i}" for i in range(simulation.n_obs)])
        positive_spots = full_spots[libraries > 0]
        zero_spots = full_spots[libraries <= 0]
        truth = read_composition(Path(simulation_row.truth_path), "truth")
        truth_values = np.asarray(simulation.obsm["cell_type_proportions"], np.float64)
        truth_names = pd.Index(simulation.uns["cell_type_names"]).astype(str)
        if not truth.index.equals(full_spots) or not truth.columns.equals(truth_names):
            raise RuntimeError(f"truth identity/order differs: {simulation_row.pair_id}")
        if truth_values.shape != truth.shape or not np.allclose(
            truth_values, truth.to_numpy(), atol=1e-12, rtol=0
        ):
            raise RuntimeError(f"truth CSV differs from H5AD: {simulation_row.pair_id}")
        truth_sums = truth.to_numpy().sum(axis=1)
        if not np.allclose(truth_sums[libraries > 0], 1, atol=1e-10, rtol=0):
            raise RuntimeError(f"truth positive rows do not sum to one: {simulation_row.pair_id}")
        if not np.allclose(truth_sums[libraries <= 0], 0, atol=1e-12, rtol=0):
            raise RuntimeError(f"truth zero-library rows are not zero: {simulation_row.pair_id}")

        predictions = {}
        for method in ("rctd", "cell2location"):
            method_row = method_lookup[(simulation_row.pair_id, method)]
            if method_row.simulation_sha256 != simulation_row.simulation_sha256:
                raise RuntimeError(f"{method} used a different simulation hash")
            if sha256_file(Path(method_row.output_path)) != method_row.output_sha256:
                raise RuntimeError(f"{method} output hash differs from the run manifest")
            metadata_path = Path(method_row.metadata_path)
            if sha256_file(metadata_path) != method_row.metadata_sha256:
                raise RuntimeError(f"{method} metadata hash differs from the run manifest")
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("status") != "validated_success"
                or metadata.get("input_sha256") != simulation_row.simulation_sha256
                or metadata.get("reference_sha256") != simulation_row.reference_sha256
                or metadata.get("output_sha256") != method_row.output_sha256
                or int(metadata.get("public_seed", -1)) != int(simulation_row.public_seed)
            ):
                raise RuntimeError(f"{method} metadata does not close input/output lineage")
            prediction = read_composition(Path(method_row.output_path), method)
            if sha256_file(Path(method_row.output_path)) == sha256_file(Path(simulation_row.truth_path)):
                raise RuntimeError(f"ground truth was substituted for {method}")
            predictions[method] = prediction
        rctd, cell2location = predictions["rctd"], predictions["cell2location"]
        if not rctd.index.equals(positive_spots):
            raise RuntimeError(f"RCTD spot support/order differs: {simulation_row.pair_id}")
        if not cell2location.index.equals(full_spots):
            raise RuntimeError(f"Cell2location spot support/order differs: {simulation_row.pair_id}")
        reference_types = set(reference.obs["cell_type"].astype(str))
        if set(cell2location.columns) != reference_types:
            raise RuntimeError(f"Cell2location type support differs: {simulation_row.pair_id}")
        if not set(rctd.columns).issubset(set(truth.columns)):
            raise RuntimeError(f"RCTD contains unknown cell types: {simulation_row.pair_id}")
        if rctd.to_numpy().min(initial=0) < -1e-6:
            raise RuntimeError("RCTD contains materially negative weights")
        if cell2location.to_numpy().min(initial=0) < 0:
            raise RuntimeError("Cell2location contains negative weights")
        if np.max(np.abs(rctd.sum(axis=1).to_numpy() - 1), initial=0) > 1e-5:
            raise RuntimeError("RCTD rows do not sum to one")
        if np.max(
            np.abs(cell2location.loc[positive_spots].sum(axis=1).to_numpy() - 1),
            initial=0,
        ) > 1e-6:
            raise RuntimeError("Cell2location scored rows do not sum to one")
        zero_sums = cell2location.loc[zero_spots].sum(axis=1).to_numpy()
        if not np.all(np.isclose(zero_sums, 0, atol=1e-6) | np.isclose(zero_sums, 1, atol=1e-6)):
            raise RuntimeError("Cell2location zero-library row mass is invalid")

        common = truth.columns[
            truth.columns.isin(rctd.columns) & truth.columns.isin(cell2location.columns)
        ]
        if common.empty:
            raise RuntimeError(f"no common named cell types: {simulation_row.pair_id}")
        truth_scored = with_other(truth.loc[positive_spots], common)
        for method, prediction in (
            ("rctd", rctd),
            ("cell2location", cell2location.loc[positive_spots]),
        ):
            result = metrics(truth_scored, with_other(prediction, common))
            rows.append(
                {
                    "method": method, "pair_id": simulation_row.pair_id,
                    "slice_id": simulation_row.slice_id,
                    "resolution": float(simulation_row.resolution), "status": "ok",
                    "comparison_scope": "same_expression_exact_spots_common_named_types_plus_other",
                    "n_input_spots": len(full_spots), "n_scored_spots": len(positive_spots),
                    "n_zero_library_spots_excluded": len(zero_spots),
                    "n_common_named_cell_types": len(common),
                    "n_scored_cell_types_including_other": len(common) + 1,
                    **result,
                }
            )
        support_rows.append(
            {
                "pair_id": simulation_row.pair_id,
                "same_expression_sha256": simulation_row.simulation_sha256,
                "exact_positive_spot_support": True,
                "n_full_spots": len(full_spots), "n_scored_spots": len(positive_spots),
                "n_zero_library_spots": len(zero_spots), "n_common_named_types": len(common),
                "other_residual_included": True,
            }
        )

    scores = pd.DataFrame(rows)
    if len(scores) != 12 or scores.duplicated(["method", "pair_id"]).any():
        raise RuntimeError("score table is not the exact 12-row design")
    summary = scores.groupby("method", as_index=False).agg(
        n_pairs=("pair_id", "size"), jsd_mean=("jsd_mean", "mean"),
        pearson_mean=("pearson_mean", "mean"), summed_rmse=("summed_rmse", "mean"),
    )
    indexed = summary.set_index("method")
    differences = {
        "jsd_mean_rctd_minus_cell2location": float(
            indexed.loc["rctd", "jsd_mean"]
            - indexed.loc["cell2location", "jsd_mean"]
        ),
        "pearson_mean_rctd_minus_cell2location": float(
            indexed.loc["rctd", "pearson_mean"]
            - indexed.loc["cell2location", "pearson_mean"]
        ),
        "summed_rmse_rctd_minus_cell2location": float(
            indexed.loc["rctd", "summed_rmse"]
            - indexed.loc["cell2location", "summed_rmse"]
        ),
    }
    direction_tolerance = 1e-12
    metric_winners = {
        "jsd_mean": (
            "tie_or_indeterminate"
            if abs(differences["jsd_mean_rctd_minus_cell2location"])
            <= direction_tolerance
            else (
                "rctd"
                if differences["jsd_mean_rctd_minus_cell2location"] < 0
                else "cell2location"
            )
        ),
        "pearson_mean": (
            "tie_or_indeterminate"
            if abs(differences["pearson_mean_rctd_minus_cell2location"])
            <= direction_tolerance
            else (
                "rctd"
                if differences["pearson_mean_rctd_minus_cell2location"] > 0
                else "cell2location"
            )
        ),
        "summed_rmse": (
            "tie_or_indeterminate"
            if abs(differences["summed_rmse_rctd_minus_cell2location"])
            <= direction_tolerance
            else (
                "rctd"
                if differences["summed_rmse_rctd_minus_cell2location"] < 0
                else "cell2location"
            )
        ),
    }
    historical_metric_winners = {
        "jsd_mean": "rctd",
        "pearson_mean": "cell2location",
        "summed_rmse": "cell2location",
    }
    changed_metric_directions = [
        metric
        for metric, winner in metric_winners.items()
        if winner != historical_metric_winners[metric]
    ]
    direction_changed = bool(changed_metric_directions)
    status = (
        "stop_headline_direction_changed"
        if direction_changed
        else "candidate_direction_unchanged_pending_author_review"
    )
    decision = {
        "status": status,
        "historical_direction_audit_sha256": HISTORICAL_DIRECTION_AUDIT_SHA256,
        "historical_metric_winners": historical_metric_winners,
        "metric_winners": metric_winners,
        "direction_tolerance": direction_tolerance,
        "metric_differences": differences,
        "changed_metric_directions": changed_metric_directions,
        "direction_changed": direction_changed,
        "aggregate_winner_ranking_authorized": False,
        "decision_rule": (
            "stop if any atomic metric direction changes, ties, or is indeterminate; "
            "otherwise require author review"
        ),
        "publication_claim_authorized": False,
        "claim": (
            "Fresh common-support results preserve the historical per-metric direction "
            "vector; author review is still required before publication."
            if not direction_changed
            else "At least one fresh common-support metric changes or cannot resolve its "
            "historical direction; stop and report before publication integration."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = args.output_dir / "deconvolution_scores.csv"
    summary_path = args.output_dir / "deconvolution_summary.csv"
    support_path = args.output_dir / "support_audit.csv"
    decision_path = args.output_dir / "publication_decision.json"
    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": "study03-common-support-score-v2",
                "feast_versions": sorted(set(simulations["feast_version"].astype(str))),
                "feast_commits": sorted(set(simulations["feast_commit"].astype(str))),
                "simulation_manifest_sha256": sha256_file(args.run_dir / "simulation_manifest.csv"),
                "method_manifest_sha256": sha256_file(args.run_dir / "method_manifest.csv"),
                "outputs": {
                    path.name: sha256_file(path)
                    for path in (scores_path, summary_path, support_path, decision_path)
                },
            },
            indent=2,
        ) + "\n"
    )
    return 2 if direction_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
