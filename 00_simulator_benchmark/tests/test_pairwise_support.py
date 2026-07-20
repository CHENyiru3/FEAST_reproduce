from __future__ import annotations

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


SCORE_PATH = Path(__file__).resolve().parents[1] / "score.py"
SPEC = importlib.util.spec_from_file_location("study00_score", SCORE_PATH)
assert SPEC is not None and SPEC.loader is not None
SCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORE)


def make_data(names: list[str], coordinates: np.ndarray) -> ad.AnnData:
    matrix = sp.csr_matrix(np.arange(1, len(names) * 12 + 1).reshape(len(names), 12))
    data = ad.AnnData(matrix)
    data.obs_names = names
    data.var_names = [f"gene_{index}" for index in range(12)]
    data.obsm["spatial"] = np.asarray(coordinates, dtype=float)
    return data


def test_exact_identifiers_are_preferred() -> None:
    reference = make_data(["a", "b"], np.array([[0, 0], [1, 1]]))
    simulated = make_data(["b", "a"], np.array([[50, 50], [40, 40]]))
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated
    )
    assert reference_indices.tolist() == [0, 1]
    assert simulated_indices.tolist() == [1, 0]
    assert diagnostics["spot_pairing"] == "exact_identifier"


def test_partial_identifier_support_is_not_supplemented_by_coordinates() -> None:
    reference = make_data(["a", "b"], np.array([[0, 0], [1, 1]]))
    simulated = make_data(["a", "C_2"], np.array([[1, 1], [0, 0]]))
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated
    )
    assert reference_indices.tolist() == [0]
    assert simulated_indices.tolist() == [0]
    assert diagnostics["spot_pairing"] == "exact_identifier"


def test_complete_coordinate_bijection_recovers_permutation() -> None:
    reference = make_data(["a", "b", "c"], np.array([[0, 0], [1, 1], [2, 2]]))
    simulated = make_data(["C_1", "C_2", "C_3"], np.array([[2, 2], [0, 0], [1, 1]]))
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated
    )
    assert reference_indices.tolist() == [0, 1, 2]
    assert simulated_indices.tolist() == [1, 2, 0]
    assert diagnostics["spot_pairing"] == "unique_spatial_coordinate"
    assert diagnostics["coordinate_bijection"] is True
    assert diagnostics["pair_map_sha256"]


def test_coordinate_rounding_tolerates_serialization_noise() -> None:
    reference = make_data(["a", "b"], np.array([[0.123456789, 1], [2, 3]]))
    simulated = make_data(
        ["C_1", "C_2"], np.array([[2 + 1e-12, 3], [0.123456789, 1]])
    )
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated, coordinate_decimals=8
    )
    assert reference_indices.tolist() == [0, 1]
    assert simulated_indices.tolist() == [1, 0]
    assert diagnostics["coordinate_max_abs_delta"] <= 1e-8


def test_duplicate_coordinates_are_rejected() -> None:
    reference = make_data(["a", "b"], np.array([[0, 0], [1, 1]]))
    simulated = make_data(["C_1", "C_2"], np.array([[0, 0], [0, 0]]))
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated
    )
    assert len(reference_indices) == len(simulated_indices) == 0
    assert diagnostics["spot_pairing"] == "unavailable"
    assert diagnostics["pairing_issue"] == "nonunique_spatial_coordinates"
    assert diagnostics["n_simulated_unique_coordinates"] == 1
    assert diagnostics["n_simulated_duplicate_coordinate_rows"] == 2


def test_incomplete_coordinate_support_is_rejected() -> None:
    reference = make_data(["a", "b"], np.array([[0, 0], [1, 1]]))
    simulated = make_data(["C_1", "C_2"], np.array([[0, 0], [2, 2]]))
    reference_indices, simulated_indices, diagnostics = SCORE.paired_spot_indices(
        reference, simulated
    )
    assert len(reference_indices) == len(simulated_indices) == 0
    assert diagnostics["spot_pairing"] == "unavailable"
    assert diagnostics["pairing_issue"] == "incomplete_spatial_coordinate_bijection"


def test_nonunique_observation_identifiers_fail_with_complete_diagnostics() -> None:
    reference = make_data(["a", "b"], np.array([[0, 0], [1, 1]]))
    simulated = make_data(["C_1", "C_1"], np.array([[0, 0], [1, 1]]))
    _, _, diagnostics = SCORE.paired_spot_indices(reference, simulated)
    assert diagnostics["spot_pairing"] == "unavailable"
    assert diagnostics["pairing_issue"] == "nonunique_observation_identifier"
    assert diagnostics["pair_map_sha256"] == ""


def test_pair_metrics_respects_coordinate_permutation() -> None:
    coordinates = np.array([[0, 0], [1, 0], [0, 1]])
    reference = make_data(["a", "b", "c"], coordinates)
    order = np.array([2, 0, 1])
    simulated = make_data(["C_1", "C_2", "C_3"], coordinates[order])
    simulated.X = reference.X[order].copy()
    metrics, _ = SCORE.pair_metrics(
        reference, simulated, max_genes=12, neighbors=1, coordinate_decimals=8
    )
    assert metrics["spot_pairing"] == "unique_spatial_coordinate"
    assert metrics["n_paired_spots"] == 3
    assert metrics["zero_mask_jaccard"] == 1.0
    assert abs(metrics["cosine_divergence"]) < 1e-12
