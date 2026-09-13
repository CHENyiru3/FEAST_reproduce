from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STUDY_DIR = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"study02_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_method_matrix_validation_accepts_rectangular_support(tmp_path: Path):
    run = load_script("run")
    matrix = np.array(
        [[0.3, 0.0], [0.2, 0.0], [0.0, 0.1], [0.0, 0.4]],
        dtype=np.float64,
    )
    job = {"reference_n_spots": 4, "moving_n_spots": 2}
    config = {"spateo": {"mapping_negative_tolerance": 1.0e-10}}

    np.save(tmp_path / "mapping_matrix.npy", matrix)
    spateo_diagnostics = {
        "mapping_shape": [4, 2],
        "mapping_all_finite": True,
        "mapping_minimum": 0.0,
        "mapping_mass": 1.0,
        "mapping_negative_tolerance": 1.0e-10,
    }
    assert run.method_matrix_is_valid(
        tmp_path,
        {**job, "method": "spateo"},
        spateo_diagnostics,
        config,
    )

    np.save(tmp_path / "coupling_matrix.npy", matrix)
    paste_diagnostics = {"coupling": {"shape": [4, 2], "mass": 1.0}}
    assert run.method_matrix_is_valid(
        tmp_path,
        {**job, "method": "paste"},
        paste_diagnostics,
        config,
    )


def test_transport_metrics_use_moving_to_reference_columns():
    metrics_module = load_script("metric_utils")
    reference_names = pd.Index(["a", "b", "c", "d"])
    moving_names = pd.Index(["b", "d"])
    positions = metrics_module.ordered_reference_positions(
        reference_names, moving_names
    )
    matrix = np.array(
        [[0.0, 0.0], [0.6, 0.0], [0.1, 0.1], [0.0, 0.7]],
        dtype=np.float64,
    )

    metrics, active, prediction = metrics_module.partial_transport_metrics(
        matrix, positions
    )

    assert positions.tolist() == [1, 3]
    assert active.tolist() == [True, True]
    assert prediction.tolist() == [1, 3]
    assert metrics["transport_moving_exact_spot_recovery_rate"] == 1.0
    assert metrics["active_transport_mutual_consistency"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "moving_names",
    [pd.Index(["d", "b"]), pd.Index(["b", "missing"])],
)
def test_ordered_reference_positions_rejects_invalid_support(moving_names: pd.Index):
    metrics_module = load_script("metric_utils")

    with pytest.raises(ValueError, match="ordered reference subset"):
        metrics_module.ordered_reference_positions(
            pd.Index(["a", "b", "c", "d"]), moving_names
        )
