from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from linear_baseline import fit_label_means  # noqa: E402


def test_label_mean_is_one_hot_ols_prediction() -> None:
    matrix = np.array([[1.0, 3.0], [3.0, 5.0], [10.0, 2.0]])
    labels = np.array(["a", "a", "b"])
    fitted = fit_label_means(matrix, labels)
    np.testing.assert_allclose(fitted["a"], [2.0, 4.0])
    np.testing.assert_allclose(fitted["b"], [10.0, 2.0])
