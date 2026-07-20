#!/usr/bin/env python3
"""Score matched RCTD/Cell2location outputs on common support plus __other__."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.special import rel_entr
from scipy.stats import pearsonr


STUDY_ROOT = Path(__file__).resolve().parent
HISTORICAL_SCORES_SHA256 = (
    "2958ac8c200974302b60f0f7443dcda3593f701ee274195e9f57d51820a827e4"
)
PRIOR_SCORES_SHA256 = (
    "bb894933e465399c38353c8d029e45aebd2378a5e204dc422ef465113ee64b59"
)
HISTORICAL_DIRECTION_AUDIT_SHA256 = (
    "69e0154042552f86e0cc778ac3963c2de073226a260f2b122d954febbfec58f9"
)

KEY_COLUMNS = ("method", "slice_id", "resolution")
METRICS = ("jsd_mean", "pearson_mean", "summed_rmse")
SCORE_METRICS = (
    "jsd_mean",
    "jsd_median",
    "pearson_mean",
    "pearson_median",
    "summed_rmse",
)
DIRECTION_TOLERANCE = 1e-12
CHANGE_TOLERANCE = 1e-12
EXPECTED_PREFERENCES = {
    "jsd_mean": "lower",
    "pearson_mean": "higher",
    "summed_rmse": "lower",
}
CHANGE_CLASSIFICATIONS = {
    "historical_to_prior": (
        "mixed_workflow_non_attributable_historical_to_prior_repair"
    ),
    "prior_to_fresh": (
        "mixed_workflow_non_attributable_prior_repair_to_fresh_full_rerun"
    ),
    "historical_to_fresh": (
        "mixed_workflow_non_attributable_historical_to_fresh_full_rerun"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}: {path}"
        )


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


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    midpoint = 0.5 * (truth + prediction)
    jsd = 0.5 * (
        rel_entr(truth, midpoint).sum(axis=1)
        + rel_entr(prediction, midpoint).sum(axis=1)
    ) / np.log(2)
    correlations = []
    for index in range(truth.shape[1]):
        left, right = truth[:, index], prediction[:, index]
        correlations.append(
            np.nan
            if np.std(left) == 0 or np.std(right) == 0
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


def expected_pair_ids(config: dict) -> set[str]:
    return {
        f"{slice_id}__resolution_{float(resolution):g}"
        for slice_id in config["slices"]
        for resolution in config["resolutions"]
    }


def score_keys(frame: pd.DataFrame) -> set[tuple[str, str, float]]:
    return set(
        zip(
            frame["method"].astype(str),
            frame["slice_id"].astype(str),
            frame["resolution"].astype(float),
        )
    )


def read_score_layer(
    path: Path,
    label: str,
    expected_keys: set[tuple[str, str, float]],
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"slice_id": str})
    required = set(KEY_COLUMNS) | set(METRICS) | {"status"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise RuntimeError(f"{label} lacks required columns: {missing}")
    if (
        len(frame) != 12
        or frame.duplicated(list(KEY_COLUMNS)).any()
        or score_keys(frame) != expected_keys
        or set(frame["method"].astype(str)) != {"rctd", "cell2location"}
        or set(frame["status"].astype(str)) != {"ok"}
    ):
        raise RuntimeError(f"{label} is not the exact successful 12-key score matrix")
    values = frame.loc[:, METRICS].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite publication metrics")
    return frame


def read_historical_winners(path: Path) -> dict[str, str]:
    audit = pd.read_csv(path)
    historical = audit.loc[audit["layer"].astype(str) == "historical"].copy()
    if (
        len(historical) != len(METRICS)
        or historical["metric"].duplicated().any()
        or set(historical["metric"].astype(str)) != set(METRICS)
    ):
        raise RuntimeError("historical direction audit lacks the exact atomic metric vector")
    historical = historical.set_index("metric")
    preferences = historical["preference"].astype(str).to_dict()
    if preferences != EXPECTED_PREFERENCES:
        raise RuntimeError("historical direction audit has unexpected metric preferences")
    winners = historical["metric_winner"].astype(str).to_dict()
    if not set(winners.values()).issubset({"rctd", "cell2location"}):
        raise RuntimeError("historical direction audit has an invalid metric winner")
    return {metric: winners[metric] for metric in METRICS}


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby("method", as_index=False).agg(
        n_pairs=("pair_id", "size"),
        jsd_mean=("jsd_mean", "mean"),
        pearson_mean=("pearson_mean", "mean"),
        summed_rmse=("summed_rmse", "mean"),
    )


def direction_details(summary: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    indexed = summary.set_index("method")
    differences = {
        f"{metric}_rctd_minus_cell2location": float(
            indexed.loc["rctd", metric] - indexed.loc["cell2location", metric]
        )
        for metric in METRICS
    }
    winners: dict[str, str] = {}
    for metric in METRICS:
        difference = differences[f"{metric}_rctd_minus_cell2location"]
        if abs(difference) <= DIRECTION_TOLERANCE:
            winners[metric] = "tie_or_indeterminate"
        elif EXPECTED_PREFERENCES[metric] == "lower":
            winners[metric] = "rctd" if difference < 0 else "cell2location"
        else:
            winners[metric] = "rctd" if difference > 0 else "cell2location"
    return differences, winners


def build_old_vs_new(
    historical: pd.DataFrame,
    prior: pd.DataFrame,
    fresh: pd.DataFrame,
) -> pd.DataFrame:
    ordered_keys = list(KEY_COLUMNS)
    layers = {
        "historical": historical,
        "prior_repaired": prior,
        "fresh": fresh,
    }
    comparison: pd.DataFrame | None = None
    for layer, frame in layers.items():
        selected = frame.loc[:, ordered_keys + list(METRICS)].rename(
            columns={metric: f"{metric}_{layer}" for metric in METRICS}
        )
        comparison = (
            selected
            if comparison is None
            else comparison.merge(selected, on=ordered_keys, how="inner", validate="one_to_one")
        )
    assert comparison is not None
    if len(comparison) != 12:
        raise RuntimeError("old-versus-new comparison did not preserve all 12 exact keys")
    for metric in METRICS:
        historical_values = comparison[f"{metric}_historical"]
        prior_values = comparison[f"{metric}_prior_repaired"]
        fresh_values = comparison[f"{metric}_fresh"]
        comparison[f"{metric}_historical_to_prior_delta"] = (
            prior_values - historical_values
        )
        comparison[f"{metric}_prior_to_fresh_delta"] = fresh_values - prior_values
        comparison[f"{metric}_historical_to_fresh_delta"] = (
            fresh_values - historical_values
        )
        comparison[f"{metric}_historical_to_prior_changed"] = (
            comparison[f"{metric}_historical_to_prior_delta"].abs() > CHANGE_TOLERANCE
        )
        comparison[f"{metric}_prior_to_fresh_changed"] = (
            comparison[f"{metric}_prior_to_fresh_delta"].abs() > CHANGE_TOLERANCE
        )
        comparison[f"{metric}_historical_to_fresh_changed"] = (
            comparison[f"{metric}_historical_to_fresh_delta"].abs() > CHANGE_TOLERANCE
        )
    for transition, classification in CHANGE_CLASSIFICATIONS.items():
        comparison[f"{transition}_classification"] = classification
    return comparison.sort_values(ordered_keys).reset_index(drop=True)


def artifact(path: Path, sha256: str, artifact_id: str) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "path": str(path.resolve()),
        "sha256": sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config.yaml")
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--prior-scores", type=Path, required=True)
    parser.add_argument("--direction-audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    require_sha256(args.historical_scores, HISTORICAL_SCORES_SHA256, "article historical scores")
    require_sha256(args.prior_scores, PRIOR_SCORES_SHA256, "prior repaired scores")
    require_sha256(
        args.direction_audit,
        HISTORICAL_DIRECTION_AUDIT_SHA256,
        "historical direction audit",
    )
    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256_file(args.config)
    simulation_manifest_path = args.run_dir / "simulation_manifest.csv"
    method_manifest_path = args.run_dir / "method_manifest.csv"
    simulations = pd.read_csv(simulation_manifest_path, dtype={"slice_id": str})
    methods = pd.read_csv(method_manifest_path, dtype={"slice_id": str})
    expected_pairs = expected_pair_ids(config)
    if (
        len(simulations) != 6
        or set(simulations["pair_id"].astype(str)) != expected_pairs
        or simulations["pair_id"].duplicated().any()
        or set(simulations["status"].astype(str)) != {"ok"}
        or len(methods) != 12
        or methods[["pair_id", "method"]].duplicated().any()
        or set(methods["status"].astype(str)) != {"ok"}
        or set(zip(methods["pair_id"].astype(str), methods["method"].astype(str)))
        != {
            (pair_id, method)
            for pair_id in expected_pairs
            for method in ("rctd", "cell2location")
        }
        or set(simulations["config_sha256"].astype(str)) != {config_hash}
        or set(methods["config_sha256"].astype(str)) != {config_hash}
        or set(simulations["feast_commit"].astype(str))
        != {str(config["required_feast_commit"])}
        or set(methods["feast_commit"].astype(str))
        != {str(config["required_feast_commit"])}
        or set(methods["feast_version"].astype(str))
        != set(simulations["feast_version"].astype(str))
        or set(simulations["public_seed"].astype(int)) != {int(config["seed"])}
        or set(methods["public_seed"].astype(int)) != {int(config["seed"])}
        or not all(
            str(row.pair_id)
            == f"{str(row.slice_id)}__resolution_{float(row.resolution):g}"
            for row in simulations.itertuples(index=False)
        )
        or not all(
            str(row.pair_id)
            == f"{str(row.slice_id)}__resolution_{float(row.resolution):g}"
            for row in methods.itertuples(index=False)
        )
    ):
        raise RuntimeError("run manifests do not contain the exact declared successful job matrix")

    method_lookup = {
        (str(row.pair_id), str(row.method)): row
        for row in methods.itertuples(index=False)
    }
    rows: list[dict] = []
    support_rows: list[dict] = []
    scoring_input_lineage: list[dict] = []
    for simulation_row in simulations.itertuples(index=False):
        pair_id = str(simulation_row.pair_id)
        simulation_path = Path(simulation_row.simulation_path)
        truth_path = Path(simulation_row.truth_path)
        reference_path = Path(simulation_row.reference_path)
        if (
            sha256_file(simulation_path) != simulation_row.simulation_sha256
            or sha256_file(truth_path) != simulation_row.truth_sha256
            or sha256_file(reference_path) != simulation_row.reference_sha256
        ):
            raise RuntimeError(f"simulation/truth/reference hash mismatch: {pair_id}")
        simulation = ad.read_h5ad(simulation_path)
        reference = ad.read_h5ad(reference_path)
        if not simulation.var_names.equals(reference.var_names):
            raise RuntimeError(f"gene order differs: {pair_id}")
        if not simulation.var_names.is_unique or not reference.var_names.is_unique:
            raise RuntimeError(f"gene IDs are not unique: {pair_id}")
        libraries = integer_libraries(simulation.X, f"simulation {pair_id}")
        integer_libraries(reference.X, f"reference {simulation_row.slice_id}")
        if not simulation.obs_names.equals(pd.Index([str(i) for i in range(simulation.n_obs)])):
            raise RuntimeError(f"noncanonical simulation obs sequence: {pair_id}")
        embedded = simulation.uns.get("feast_reproduce", {})
        if (
            embedded.get("feast_commit") != simulation_row.feast_commit
            or embedded.get("config_sha256") != simulation_row.config_sha256
            or int(embedded.get("public_seed", -1)) != int(simulation_row.public_seed)
        ):
            raise RuntimeError(f"simulation provenance mismatch: {pair_id}")
        full_spots = pd.Index([f"spot_{i}" for i in range(simulation.n_obs)])
        positive_spots = full_spots[libraries > 0]
        zero_spots = full_spots[libraries <= 0]
        truth = read_composition(truth_path, "truth")
        if truth.to_numpy(np.float64).min(initial=0) < -1e-12:
            raise RuntimeError(f"truth contains negative proportions: {pair_id}")
        truth_values = np.asarray(simulation.obsm["cell_type_proportions"], np.float64)
        truth_names = pd.Index(simulation.uns["cell_type_names"]).astype(str)
        if not truth.index.equals(full_spots) or not truth.columns.equals(truth_names):
            raise RuntimeError(f"truth identity/order differs: {pair_id}")
        if truth_values.shape != truth.shape or not np.allclose(
            truth_values, truth.to_numpy(), atol=1e-12, rtol=0
        ):
            raise RuntimeError(f"truth CSV differs from H5AD: {pair_id}")
        truth_sums = truth.to_numpy().sum(axis=1)
        if not np.allclose(truth_sums[libraries > 0], 1, atol=1e-10, rtol=0):
            raise RuntimeError(f"truth positive rows do not sum to one: {pair_id}")
        if not np.allclose(truth_sums[libraries <= 0], 0, atol=1e-12, rtol=0):
            raise RuntimeError(f"truth zero-library rows are not zero: {pair_id}")

        predictions: dict[str, pd.DataFrame] = {}
        method_lineage: dict[str, dict] = {}
        for method in ("rctd", "cell2location"):
            method_row = method_lookup[(pair_id, method)]
            output_path = Path(method_row.output_path)
            metadata_path = Path(method_row.metadata_path)
            if (
                str(method_row.slice_id) != str(simulation_row.slice_id)
                or not np.isclose(
                    float(method_row.resolution),
                    float(simulation_row.resolution),
                    atol=0,
                    rtol=0,
                )
                or Path(method_row.simulation_path).resolve()
                != simulation_path.resolve()
                or method_row.simulation_sha256 != simulation_row.simulation_sha256
                or Path(method_row.reference_path).resolve()
                != reference_path.resolve()
                or method_row.reference_sha256 != simulation_row.reference_sha256
                or str(method_row.feast_commit) != str(simulation_row.feast_commit)
                or str(method_row.feast_version) != str(simulation_row.feast_version)
                or str(method_row.config_sha256) != str(simulation_row.config_sha256)
                or int(method_row.public_seed) != int(simulation_row.public_seed)
            ):
                raise RuntimeError(f"{method} manifest lineage differs: {pair_id}")
            if sha256_file(output_path) != method_row.output_sha256:
                raise RuntimeError(f"{method} output hash differs from the run manifest")
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
            prediction = read_composition(output_path, method)
            if method_row.output_sha256 == simulation_row.truth_sha256:
                raise RuntimeError(f"ground truth was substituted for {method}")
            predictions[method] = prediction
            method_lineage[method] = {
                "prediction": artifact(
                    output_path, method_row.output_sha256, f"{pair_id}:{method}:prediction"
                ),
                "metadata": artifact(
                    metadata_path, method_row.metadata_sha256, f"{pair_id}:{method}:metadata"
                ),
                "method_source_sha256": str(method_row.method_source_sha256),
                "configuration_id": str(method_row.configuration_id),
            }

        rctd, cell2location = predictions["rctd"], predictions["cell2location"]
        if not rctd.index.equals(positive_spots):
            raise RuntimeError(f"RCTD spot support/order differs: {pair_id}")
        if not cell2location.index.equals(full_spots):
            raise RuntimeError(f"Cell2location spot support/order differs: {pair_id}")
        reference_types = set(reference.obs["cell_type"].astype(str))
        truth_types = set(truth.columns)
        cell2location_types = set(cell2location.columns)
        if truth_types != reference_types or truth_types != cell2location_types:
            raise RuntimeError(
                f"truth/reference/Cell2location full type sets differ: {pair_id}"
            )
        if not set(rctd.columns).issubset(truth_types):
            raise RuntimeError(f"RCTD contains unknown cell types: {pair_id}")
        common = truth.columns[truth.columns.isin(rctd.columns)]
        if common.empty or set(common) != set(rctd.columns):
            raise RuntimeError(f"RCTD set is not the exact common named support: {pair_id}")
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
        if not np.all(
            np.isclose(zero_sums, 0, atol=1e-6)
            | np.isclose(zero_sums, 1, atol=1e-6)
        ):
            raise RuntimeError("Cell2location zero-library row mass is invalid")

        truth_scored = with_other(truth.loc[positive_spots], common)
        for method, prediction in (
            ("rctd", rctd),
            ("cell2location", cell2location.loc[positive_spots]),
        ):
            result = metrics(truth_scored, with_other(prediction, common))
            rows.append(
                {
                    "method": method,
                    "pair_id": pair_id,
                    "slice_id": str(simulation_row.slice_id),
                    "resolution": float(simulation_row.resolution),
                    "status": "ok",
                    "comparison_scope": (
                        "same_expression_exact_spots_common_named_types_plus_other"
                    ),
                    "n_input_spots": len(full_spots),
                    "n_scored_spots": len(positive_spots),
                    "n_zero_library_spots_excluded": len(zero_spots),
                    "n_common_named_cell_types": len(common),
                    "n_scored_cell_types_including_other": len(common) + 1,
                    **result,
                }
            )
        support_rows.append(
            {
                "pair_id": pair_id,
                "slice_id": str(simulation_row.slice_id),
                "resolution": float(simulation_row.resolution),
                "same_expression_sha256": simulation_row.simulation_sha256,
                "truth_sha256": simulation_row.truth_sha256,
                "reference_sha256": simulation_row.reference_sha256,
                "rctd_output_sha256": method_lookup[(pair_id, "rctd")].output_sha256,
                "cell2location_output_sha256": method_lookup[
                    (pair_id, "cell2location")
                ].output_sha256,
                "exact_positive_spot_support_between_methods": True,
                "cell2location_full_spot_support_exact": True,
                "n_full_spots": len(full_spots),
                "n_scored_spots": len(positive_spots),
                "n_zero_library_spots": len(zero_spots),
                "zero_library_disposition": "excluded_from_both_methods_scoring",
                "n_truth_cell_types": len(truth.columns),
                "n_reference_cell_types": len(reference_types),
                "n_rctd_cell_types": len(rctd.columns),
                "n_cell2location_cell_types": len(cell2location.columns),
                "n_common_named_types": len(common),
                "truth_type_set_equals_reference_type_set": True,
                "truth_type_set_equals_cell2location_type_set": True,
                "rctd_type_set_equals_truth_ordered_common_type_set": True,
                "reference_common_type_set_equals_canonical_common_type_set": (
                    set(common).issubset(reference_types)
                ),
                "cell2location_common_type_set_equals_canonical_common_type_set": (
                    set(common).issubset(cell2location_types)
                ),
                "canonical_common_order_source": "truth_columns",
                "truth_cell_type_order_json": json.dumps(list(truth.columns)),
                "canonical_common_cell_type_order_json": json.dumps(list(common)),
                "rctd_raw_column_order_matches_canonical_common_order": (
                    rctd.columns.equals(common)
                ),
                "rctd_raw_column_order_required": False,
                "other_residual_included": True,
            }
        )
        scoring_input_lineage.append(
            {
                "pair_id": pair_id,
                "slice_id": str(simulation_row.slice_id),
                "resolution": float(simulation_row.resolution),
                "simulation": artifact(
                    simulation_path,
                    simulation_row.simulation_sha256,
                    f"{pair_id}:simulation",
                ),
                "truth": artifact(
                    truth_path, simulation_row.truth_sha256, f"{pair_id}:truth"
                ),
                "reference": artifact(
                    reference_path,
                    simulation_row.reference_sha256,
                    f"{pair_id}:reference",
                ),
                "methods": method_lineage,
            }
        )

    scores = pd.DataFrame(rows)
    if (
        len(scores) != 12
        or scores.duplicated(["method", "pair_id"]).any()
        or not np.isfinite(scores.loc[:, SCORE_METRICS].to_numpy(np.float64)).all()
    ):
        raise RuntimeError("score table is not the finite exact 12-row design")
    expected_keys = score_keys(scores)
    historical = read_score_layer(
        args.historical_scores, "article historical scores", expected_keys
    )
    prior = read_score_layer(args.prior_scores, "prior repaired scores", expected_keys)
    old_vs_new = build_old_vs_new(historical, prior, scores)
    summary = summarize(scores)
    differences, metric_winners = direction_details(summary)
    historical_metric_winners = read_historical_winners(args.direction_audit)
    changed_metric_directions = [
        metric
        for metric in METRICS
        if metric_winners[metric] != historical_metric_winners[metric]
    ]
    direction_changed = bool(changed_metric_directions)
    status = (
        "stop_headline_direction_changed"
        if direction_changed
        else "candidate_direction_unchanged_pending_author_review"
    )
    decision = {
        "status": status,
        "historical_direction_audit_path": str(args.direction_audit.resolve()),
        "historical_direction_audit_sha256": HISTORICAL_DIRECTION_AUDIT_SHA256,
        "historical_metric_winners": historical_metric_winners,
        "metric_winners": metric_winners,
        "direction_tolerance": DIRECTION_TOLERANCE,
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
    old_vs_new_path = args.output_dir / "old_vs_new_deconvolution_scores.csv"
    decision_path = args.output_dir / "publication_decision.json"
    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    old_vs_new.to_csv(old_vs_new_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")

    provenance_path = args.output_dir / "provenance.json"
    provenance = {
        "schema_version": 3,
        "configuration_id": "study03-common-support-score-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "public_seed": int(config["seed"]),
        "config": artifact(args.config, config_hash, "study03:config"),
        "scorer": artifact(
            Path(__file__), sha256_file(Path(__file__)), "study03:common-support-scorer"
        ),
        "feast_versions": sorted(set(simulations["feast_version"].astype(str))),
        "feast_commits": sorted(set(simulations["feast_commit"].astype(str))),
        "run_lineage": {
            "simulation_manifest": artifact(
                simulation_manifest_path,
                sha256_file(simulation_manifest_path),
                "study03:simulation-manifest",
            ),
            "method_manifest": artifact(
                method_manifest_path,
                sha256_file(method_manifest_path),
                "study03:method-manifest",
            ),
            "input_manifest_sha256": sorted(
                set(simulations["input_manifest_sha256"].astype(str))
            ),
            "orchestration_runner_sha256": sorted(
                set(simulations["orchestration_runner_sha256"].astype(str))
            ),
            "simulation_configuration_ids": sorted(
                set(simulations["configuration_id"].astype(str))
            ),
            "method_configuration_ids": sorted(
                set(methods["configuration_id"].astype(str))
            ),
        },
        "historical_inputs": {
            "article_historical_scores": artifact(
                args.historical_scores,
                HISTORICAL_SCORES_SHA256,
                "study03:article-historical-scores",
            ),
            "prior_repaired_scores": artifact(
                args.prior_scores,
                PRIOR_SCORES_SHA256,
                "study03:prior-repaired-scores",
            ),
            "historical_direction_audit": artifact(
                args.direction_audit,
                HISTORICAL_DIRECTION_AUDIT_SHA256,
                "study03:historical-direction-audit",
            ),
        },
        "scoring_input_lineage": scoring_input_lineage,
        "numerical_change_policy": {
            "delta_definition": "target_minus_source",
            "change_tolerance": CHANGE_TOLERANCE,
            "classifications": CHANGE_CLASSIFICATIONS,
            "interpretation": (
                "All cross-layer deltas mix workflow changes and are not attributed to a "
                "single scientific or engineering cause."
            ),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (
                scores_path,
                summary_path,
                support_path,
                old_vs_new_path,
                decision_path,
            )
        },
        "provenance_self_hash_omitted_reason": (
            "A file cannot contain its own final cryptographic hash."
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return 2 if direction_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
