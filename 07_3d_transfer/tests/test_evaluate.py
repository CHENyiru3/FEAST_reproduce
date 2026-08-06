from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT))

import evaluate  # noqa: E402


def test_finite_pearson_handles_exact_linear_and_constant_inputs():
    assert evaluate.finite_pearson(np.array([1, 2, 3]), np.array([2, 4, 6])) == pytest.approx(1.0)
    assert np.isnan(evaluate.finite_pearson(np.ones(3), np.arange(3)))


def test_adjacent_metrics_uses_only_shared_regions():
    previous = {
        "z_index": 2,
        "z_world": -2.0,
        "gene_mean": np.array([1.0, 2.0, 3.0]),
        "region_means": {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([2.0, 3.0, 4.0]),
        },
    }
    current = {
        "z_index": 3,
        "z_world": -1.5,
        "gene_mean": np.array([2.0, 4.0, 6.0]),
        "region_means": {
            "B": np.array([4.0, 6.0, 8.0]),
            "C": np.array([9.0, 9.0, 9.0]),
        },
    }
    result = evaluate.adjacent_metrics(previous, current)
    assert result["shared_regions"] == 1
    assert result["gene_mean_pearson"] == pytest.approx(1.0)
    assert result["region_gene_mean_pearson"] == pytest.approx(1.0)
    assert result["midpoint_z"] == pytest.approx(-1.75)


def test_solver_summary_uses_study07_record_schema_and_requires_convergence():
    records, error = evaluate.solver_summary({
        "all_converged": True,
        "transport_records": 10,
        "max_final_error": 9.9e-6,
    })
    assert records == 10
    assert error == pytest.approx(9.9e-6)
    with pytest.raises(ValueError, match="positive convergence"):
        evaluate.solver_summary({
            "all_converged": False,
            "transport_records": 10,
            "max_final_error": 9.9e-6,
        })
