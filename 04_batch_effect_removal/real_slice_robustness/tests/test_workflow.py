from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


STUDY_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(alias: str, filename: str):
    spec = importlib.util.spec_from_file_location(alias, STUDY_ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} as {alias}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Study 04A also has top-level ``run`` and ``validate`` modules. Load the 04B
# modules under test-only aliases so a combined pytest process cannot reuse the
# already-imported 04A modules. Temporarily provide the absolute dependency
# names expected by the standalone 04B scripts while they are imported.
workflow = _load_local_module("study04b_workflow_test", "workflow.py")
metrics = _load_local_module("study04b_metrics_test", "metrics.py")
_missing = object()
_saved_dependencies = {
    name: sys.modules.get(name, _missing) for name in ("workflow", "metrics")
}
sys.modules["workflow"] = workflow
sys.modules["metrics"] = metrics
try:
    run_workflow = _load_local_module("study04b_run_test", "run.py")
    validate = _load_local_module("study04b_validate_test", "validate.py")
finally:
    for name, previous in _saved_dependencies.items():
        if previous is _missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def test_frozen_config_declares_exact_45_job_matrix() -> None:
    config = workflow.load_config(STUDY_ROOT / "config.yaml")
    cells = workflow.production_cells(config)
    assert len(cells) == 45
    assert len(set(cells)) == 45
    assert set(condition for condition, _, _ in cells) == {
        "raw",
        "sim_0.00",
        "sim_0.50",
        "sim_1.00",
        "sim_1.50",
    }
    assert set(method for _, method, _ in cells) == {"GraphST", "STAMP", "scVI"}
    assert set(seed for _, _, seed in cells) == {42, 43, 44}


def test_section_prefix_disambiguates_reused_visium_barcodes() -> None:
    obs = pd.DataFrame(index=pd.Index(["AAAC-1", "TTTG-1"]))
    var = pd.DataFrame(index=pd.Index(["gene-a", "gene-b"]))
    source = ad.AnnData(X=np.ones((2, 2)), obs=obs, var=var)
    left = workflow.prefix_spot_ids(source, "151675")
    right = workflow.prefix_spot_ids(source, "151676")
    assert set(left.obs_names).isdisjoint(set(right.obs_names))
    assert left.obs["source_barcode"].tolist() == ["AAAC-1", "TTTG-1"]
    assert right.obs["source_section_id"].tolist() == ["151676", "151676"]


def test_validator_metric_formulas_independently_match_scorer() -> None:
    rng = np.random.default_rng(42)
    n_per_batch = 80
    values = rng.normal(size=(2 * n_per_batch, 10))
    baseline = values + rng.normal(scale=0.05, size=values.shape)
    spatial = rng.uniform(size=(2 * n_per_batch, 2))
    batch = np.array(["ref"] * n_per_batch + ["query"] * n_per_batch)
    layer_pattern = np.repeat(np.array(["L1", "L2", "L3", "L4"]), 20)
    layers = np.concatenate([layer_pattern, layer_pattern])
    observed = metrics.full_metric_row(
        values,
        baseline=baseline,
        spatial=spatial,
        batch=batch,
        layers=layers,
        k=5,
        seed=42,
    )
    reconstructed = validate.validator_full_metric_row(
        values,
        baseline=baseline,
        spatial=spatial,
        batch=batch,
        layers=layers,
        k=5,
        seed=42,
    )
    assert observed.keys() == reconstructed.keys()
    for key in observed:
        assert observed[key] == pytest.approx(reconstructed[key], rel=1e-12, abs=1e-12)


def test_stability_is_invariant_to_orthogonal_feature_rotation() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(120, 10))
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    rotated = values @ q
    assert metrics.mean_knn_jaccard(values, rotated, k=10) == pytest.approx(1.0)
    assert metrics.linear_cka(values, rotated) == pytest.approx(1.0)
    assert metrics.sampled_pairwise_distance_spearman(
        values, rotated, max_pairs=10000, seed=42
    ) == pytest.approx(1.0)


def test_batch_removal_metric_set_excludes_exact_spot_retrieval() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(80, 10))
    batch = np.array(["ref"] * 40 + ["query"] * 40)
    layers = np.tile(np.array(["L1", "L2", "L3", "L4"]), 20)
    row = metrics.full_metric_row(
        values,
        baseline=values,
        spatial=rng.uniform(size=(80, 2)),
        batch=batch,
        layers=layers,
        k=5,
        seed=42,
    )
    assert "paired_retrieval_top1" not in row
    assert "paired_retrieval_median_rank" not in row
    assert "composite_score" not in row


def test_count_validation_rejects_fractional_or_negative_values() -> None:
    for matrix in (
        np.array([[1.5, 0.0]]),
        np.array([[-1.0, 2.0]]),
        np.array([[np.nan, 1.0]]),
    ):
        try:
            workflow.validate_count_matrix(matrix, "invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count matrix was accepted")


def test_uncorrected_pca_accepts_sparse_count_inputs() -> None:
    from scipy import sparse

    rng = np.random.default_rng(11)
    genes = [f"gene-{index}" for index in range(15)]
    var = pd.DataFrame(index=pd.Index(genes))
    left = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(2, size=(12, 15))),
        obs=pd.DataFrame(
            {"ground_truth": np.tile(["L1", "L2", "L3"], 4)},
            index=[f"left-{index}" for index in range(12)],
        ),
        var=var,
        obsm={"spatial": rng.uniform(size=(12, 2))},
    )
    right_counts = rng.poisson(2, size=(11, 15))
    right_counts[0] = 0
    right = ad.AnnData(
        X=sparse.csr_matrix(right_counts),
        obs=pd.DataFrame(
            {"ground_truth": ["L1", "L2", "L3", "L1", "L2", "L3", "L1", "L2", "L3", "L1", "L2"]},
            index=[f"right-{index}" for index in range(11)],
        ),
        var=var.copy(),
        obsm={"spatial": rng.uniform(size=(11, 2))},
    )
    embedding, batch, layers, source_ids, spatial = run_workflow._input_embedding(
        left, right, genes, seed=42
    )
    assert embedding.shape == (23, 10)
    assert np.isfinite(embedding).all()
    assert batch.shape == layers.shape == source_ids.shape == (23,)
    assert spatial.shape == (23, 2)
