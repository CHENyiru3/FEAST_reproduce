from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


STUDY_DIR = Path(__file__).resolve().parents[1]


def load_metrics():
    path = STUDY_DIR / "metric_utils.py"
    spec = importlib.util.spec_from_file_location("study02_metric_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_perfect_partial_transport_does_not_force_zero_rows_to_match():
    metrics = load_metrics()
    matrix = np.array(
        [[0.0, 0.0], [0.25, 0.0], [0.0, 0.0], [0.0, 0.25]],
        dtype=np.float64,
    )

    result, active, prediction = metrics.partial_transport_metrics(
        matrix, np.array([1, 3])
    )

    assert active.tolist() == [True, True]
    assert prediction.tolist() == [1, 3]
    assert result["moving_transport_coverage"] == 1.0
    assert result["transport_moving_exact_spot_recovery_rate"] == 1.0
    assert result["reference_overlap_detection_precision"] == 1.0
    assert result["reference_overlap_detection_recall"] == 1.0
    assert result["reference_exact_pair_f1"] == 1.0
    assert result["active_transport_mutual_consistency"] == 1.0
    assert result["ground_truth_transport_mass_fraction"] == 1.0


def test_inactive_moving_column_is_an_abstention_and_a_missed_recovery():
    metrics = load_metrics()
    matrix = np.array(
        [[0.0, 0.0], [0.25, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    result, active, prediction = metrics.partial_transport_metrics(
        matrix, np.array([1, 3])
    )

    assert active.tolist() == [True, False]
    assert prediction.tolist() == [1, -1]
    assert result["moving_transport_coverage"] == 0.5
    assert result["transport_moving_exact_spot_recovery_rate"] == 0.5
    assert result["transport_moving_conditional_identity_accuracy"] == 1.0
    assert result["reference_overlap_detection_recall"] == 0.5
    assert result["ground_truth_transport_mass_fraction"] == 1.0


def test_active_wrong_nonoverlap_row_reduces_precision():
    metrics = load_metrics()
    matrix = np.array(
        [[0.30, 0.0], [0.20, 0.0], [0.0, 0.0], [0.0, 0.25]],
        dtype=np.float64,
    )

    result, _, _ = metrics.partial_transport_metrics(
        matrix, np.array([1, 3])
    )

    assert result["reference_overlap_detection_precision"] == pytest.approx(2 / 3)
    assert result["reference_overlap_detection_recall"] == 1.0
    assert result["reference_exact_pair_precision"] == pytest.approx(2 / 3)
    assert result["reference_exact_pair_recall"] == 1.0
    assert result["transport_moving_exact_spot_recovery_rate"] == 0.5
    assert result["active_transport_mutual_consistency"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.array([[np.nan]]), "finite"),
        (np.array([[-1.0e-3]]), "negative"),
        (np.array([[0.0]]), "positive mass"),
    ],
)
def test_invalid_transport_is_rejected(matrix: np.ndarray, message: str):
    metrics = load_metrics()

    with pytest.raises(ValueError, match=message):
        metrics.partial_transport_metrics(matrix, np.array([0]))


def test_label_metrics_count_abstentions_as_unrecovered():
    metrics = load_metrics()

    result = metrics.mapped_label_metrics(
        reference_labels=np.array(["L1", "L2", "L3"]),
        moving_labels=np.array(["L2", "L3"]),
        active_moving=np.array([True, False]),
        moving_to_reference=np.array([1, -1]),
    )

    assert result["transport_moving_region_recovery_rate"] == 0.5
    assert result["transport_moving_conditional_region_accuracy"] == 1.0
