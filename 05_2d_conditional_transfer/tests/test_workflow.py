from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from workflow import (  # noqa: E402
    Job,
    declared_jobs,
    expected_job_path,
    inspect_inputs,
    load_config,
    prepare_job,
    require_declared_job,
    sha256_file,
    sha256_lines,
    validate_generated,
)
from score import (  # noqa: E402
    compute_per_gene_metrics,
    donor_stratum,
    moran_i,
    neighbor_indices,
    panel_summary,
)
from validate import recompute_panel_summary  # noqa: E402
from run import cache_reference_counts_dense_float32  # noqa: E402


def make_slice(
    obs_prefix: str,
    genes: list[str],
    labels: list[str] | None = None,
    annotation_key: str = "ground_truth",
) -> ad.AnnData:
    labels = labels or ["a", "a", "b", "b"]
    n_obs = len(labels)
    base = np.arange(n_obs * len(genes), dtype=np.int32).reshape(n_obs, len(genes)) % 6
    obs = pd.DataFrame(
        {annotation_key: labels},
        index=[f"{obs_prefix}_{index}" for index in range(n_obs)],
    )
    result = ad.AnnData(X=base.copy(), obs=obs, var=pd.DataFrame(index=genes))
    result.layers["counts"] = base.copy()
    result.obsm["spatial"] = np.column_stack(
        [np.arange(n_obs, dtype=float), np.arange(n_obs, dtype=float) % 3]
    )
    return result


def fixture_config(
    genes: list[str],
    slices: list[str] | None = None,
    directions: list[list[str]] | None = None,
) -> dict:
    slices = slices or ["a", "b"]
    directions = [["a", "b"], ["b", "a"]] if directions is None else directions
    randomness = [0.0, 0.3]
    cross_count = len(directions) * len(randomness)
    return {
        "configuration_id": "fixture",
        "datasets": {
            "dlpfc": {
                "artifact_prefix": "DLPFC",
                "annotation_key": "ground_truth",
                "slices": slices,
                "directions": directions,
                "gene_panel": {
                    "expected_genes": len(genes),
                    "sha256": sha256_lines(genes),
                },
            }
        },
        "assignment_randomness": randomness,
        "primary_assignment_randomness": 0.3,
        "expected_cross_slice_jobs": cross_count,
        "expected_mask_half_jobs": len(slices),
        "expected_jobs": cross_count + len(slices),
        "label_support": {
            "min_source_spots": 20,
            "policy": "retain_target_labels_with_minimum_source_support",
        },
        "mask_half": {
            "split_axis": 0,
            "split_quantile": 0.5,
            "observed_relation": "less_than_or_equal",
            "masked_relation": "greater_than",
            "assignment_randomness": 0.3,
            "direction": "low_x_to_high_x",
        },
        "blueprint": {"spot_id_column": "target_obs_id"},
        "coordinates": {
            "reference_key": "spatial",
            "target_key": "spatial",
            "dimensions": 2,
            "remove_reference_keys": ["spatial_3d"],
        },
        "transport": {"sinkhorn_tol": 1e-5, "sinkhorn_iter": 1000},
    }


def write_fixture_inputs(
    tmp_path: Path,
    config: dict,
    slices: dict[str, ad.AnnData],
) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    rows = []
    for slice_id, adata in slices.items():
        path = input_dir / f"{slice_id}.h5ad"
        adata.write_h5ad(path)
        rows.append(
            {
                "dataset": "dlpfc",
                "slice_id": slice_id,
                "relative_path": path.name,
                "artifact_id": f"DLPFC_{slice_id}",
                "sha256": sha256_file(path),
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
            }
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return config_path, manifest_path, input_dir


def diagnostics(labels: tuple[str, ...], converged: str = "True", error: str = "1e-7"):
    result = {}
    for label in labels:
        result[label] = {
            "format": "columnar_records_v1",
            "n_records": 1,
            "transport_converged": [converged],
            "transport_final_error": [error],
            "transport_mass": ["1.0"],
            "transport_iterations": ["20"],
            "transport_stop_threshold": ["1e-5"],
            "transport_max_iterations": ["1000"],
            "transport_nonconvergence_policy": ["raise"],
        }
    return result


def test_public_config_declares_exact_45_unique_jobs_and_paths(tmp_path: Path):
    config = load_config(STUDY_DIR / "config.yaml")
    jobs = declared_jobs(config)
    assert len(jobs) == 45
    assert sum(job.mode == "cross_slice" for job in jobs) == 40
    assert sum(job.mode == "mask_half" for job in jobs) == 5
    assert sum(job.mode == "cross_slice" and job.dataset == "dlpfc" for job in jobs) == 30
    assert sum(job.mode == "cross_slice" and job.dataset == "merfish" for job in jobs) == 10
    paths = [expected_job_path(tmp_path, job) for job in jobs]
    assert len(set(paths)) == 45


def test_input_preflight_groups_panel_by_dataset(tmp_path: Path):
    genes = ["g1", "g2", "g3"]
    config = fixture_config(genes)
    first = make_slice("a", genes)
    second = make_slice("b", list(reversed(genes)))
    config_path, manifest_path, input_dir = write_fixture_inputs(
        tmp_path, config, {"a": first, "b": second}
    )
    _, paths, panels = inspect_inputs(config_path, manifest_path, input_dir, check_counts=True)
    assert set(paths) == {("dlpfc", "a"), ("dlpfc", "b")}
    assert panels == {"dlpfc": genes}


def test_label_support_boundary_excludes_19_and_retains_20(tmp_path: Path):
    genes = ["g1", "g2"]
    source_labels = ["kept"] * 20 + ["dropped"] * 19
    target_labels = ["kept"] * 4 + ["dropped"] * 3 + ["target_only"] * 2
    config = fixture_config(genes)
    config_path, manifest_path, input_dir = write_fixture_inputs(
        tmp_path,
        config,
        {
            "a": make_slice("a", genes, source_labels),
            "b": make_slice("b", genes, target_labels),
        },
    )
    loaded, paths, panels = inspect_inputs(config_path, manifest_path, input_dir)
    prepared = prepare_job(loaded, paths, panels, Job("cross_slice", "dlpfc", "a", "b", 0.3))
    assert prepared.support["eligible_labels"] == ["kept"]
    assert prepared.support["target_retained_spots"] == 4
    assert prepared.support["excluded_target_label_counts"] == {
        "dropped": 3,
        "target_only": 2,
    }


def test_half_split_is_fixed_and_masked_expression_cannot_change_generation_input(
    tmp_path: Path,
):
    genes = ["g1", "g2"]
    labels = ["a"] * 40
    adata = make_slice("slice", genes, labels)
    config = fixture_config(genes, ["slice"], [])
    config_path, manifest_path, input_dir = write_fixture_inputs(
        tmp_path, config, {"slice": adata}
    )
    loaded, paths, panels = inspect_inputs(config_path, manifest_path, input_dir)
    job = Job("mask_half", "dlpfc", "slice", "slice", 0.3)
    first = prepare_job(loaded, paths, panels, job)
    first_reference = np.asarray(first.reference.layers["counts"]).copy()
    assert first.target_contract.X is None
    assert first.support["source_candidate_spots"] == 20
    assert first.support["target_candidate_spots"] == 20

    changed = ad.read_h5ad(input_dir / "slice.h5ad")
    changed.X[20:, :] = 99
    changed.layers["counts"][20:, :] = 99
    changed.write_h5ad(input_dir / "slice_changed.h5ad")
    manifest = pd.read_csv(manifest_path, dtype={"slice_id": str})
    manifest.loc[0, "relative_path"] = "slice_changed.h5ad"
    manifest.loc[0, "sha256"] = sha256_file(input_dir / "slice_changed.h5ad")
    manifest.to_csv(manifest_path, index=False)
    loaded, paths, panels = inspect_inputs(config_path, manifest_path, input_dir)
    second = prepare_job(loaded, paths, panels, job)
    np.testing.assert_array_equal(first_reference, second.reference.layers["counts"])
    assert second.target_contract.X is None
    assert first.support == second.support


def test_half_split_keeps_fixed_panel_and_records_zero_evidence_genes(tmp_path: Path):
    genes = ["g1", "g2"]
    adata = make_slice("slice", genes, ["a"] * 40)
    adata.X[:20, 1] = 0
    adata.layers["counts"][:20, 1] = 0
    config = fixture_config(genes, ["slice"], [])
    config_path, manifest_path, input_dir = write_fixture_inputs(
        tmp_path, config, {"slice": adata}
    )
    loaded, paths, panels = inspect_inputs(config_path, manifest_path, input_dir)
    prepared = prepare_job(
        loaded,
        paths,
        panels,
        Job("mask_half", "dlpfc", "slice", "slice", 0.3),
    )
    assert prepared.genes == genes
    assert prepared.support["reference_zero_genes"] == ["g2"]
    assert prepared.support["reference_observed_gene_count"] == 1


def test_preparation_enforces_two_dimensional_reference_contract(tmp_path: Path):
    genes = ["g1", "g2"]
    source = make_slice("a", genes, ["a"] * 20)
    source.obsm["spatial_3d"] = np.column_stack(
        [source.obsm["spatial"], np.ones(source.n_obs)]
    )
    target = make_slice("b", genes, ["a"] * 20)
    target.obsm["spatial_3d"] = np.column_stack(
        [target.obsm["spatial"], np.ones(target.n_obs)]
    )
    config = fixture_config(genes)
    config_path, manifest_path, input_dir = write_fixture_inputs(
        tmp_path, config, {"a": source, "b": target}
    )
    loaded, paths, panels = inspect_inputs(config_path, manifest_path, input_dir)
    prepared = prepare_job(
        loaded,
        paths,
        panels,
        Job("cross_slice", "dlpfc", "a", "b", 0.3),
    )

    assert "spatial_3d" not in prepared.reference.obsm
    assert prepared.reference.obsm["spatial"].shape == (20, 2)
    assert prepared.target_contract.obsm["spatial"].shape == (20, 2)
    assert prepared.support["coordinate_contract"] == {
        "reference_key": "spatial",
        "target_key": "spatial",
        "dimensions": 2,
        "removed_reference_keys": ["spatial_3d"],
    }


def test_public_reference_fit_retains_declared_zero_evidence_gene():
    from FEAST.de_novo import ReferenceFitConfig, fit_reference

    reference = make_slice("reference", ["g1", "g2"], ["a"] * 20)
    reference.X[:, 1] = 0
    reference.layers["counts"][:, 1] = 0
    model = fit_reference(
        reference,
        "ground_truth",
        ReferenceFitConfig(
            min_gene_spots=0,
            min_gene_mean=0.0,
            max_gene_zero_prop=1.0,
            boundary_neighbors=6,
        ),
    )
    assert model.gene_names == ["g1", "g2"]


def test_reference_counts_cache_preserves_exact_values_and_public_fit():
    from FEAST.de_novo import ReferenceFitConfig, fit_reference

    reference = make_slice("reference", ["g1", "g2"], ["a"] * 20)
    expected = np.asarray(reference.layers["counts"], dtype=np.float32)
    reference.layers["counts"] = sparse.csr_matrix(reference.layers["counts"])
    cache = cache_reference_counts_dense_float32(reference)

    assert isinstance(reference.layers["counts"], np.ndarray)
    assert reference.layers["counts"].dtype == np.float32
    np.testing.assert_array_equal(reference.layers["counts"], expected)
    assert cache["value_transform"] == "none"
    assert cache["shape"] == [20, 2]
    model = fit_reference(
        reference,
        "ground_truth",
        ReferenceFitConfig(
            min_gene_spots=0,
            min_gene_mean=0.0,
            max_gene_zero_prop=1.0,
            boundary_neighbors=6,
        ),
    )
    assert model.gene_names == ["g1", "g2"]


def test_reference_counts_cache_is_bit_exact_for_public_simulation():
    from FEAST.de_novo import (
        ReferenceFitConfig,
        SimulationBlueprint,
        SimulationConfig,
        fit_reference,
        simulate_from_reference,
    )

    sparse_reference = make_slice("reference", ["g1", "g2"], ["a"] * 20)
    sparse_reference.layers["counts"] = sparse.csr_matrix(
        sparse_reference.layers["counts"]
    )
    dense_reference = sparse_reference.copy()
    cache_reference_counts_dense_float32(dense_reference)
    target_obs = pd.DataFrame(
        {"target_obs_id": [f"target_{index}" for index in range(6)]}
    )
    blueprint = SimulationBlueprint(
        coordinates=np.column_stack(
            [np.arange(6, dtype=float), np.arange(6, dtype=float) % 3]
        ),
        domain_map=np.asarray(["a"] * 6),
        obs=target_obs,
    )
    fit_config = ReferenceFitConfig(
        min_gene_spots=0,
        min_gene_mean=0.0,
        max_gene_zero_prop=1.0,
        boundary_neighbors=6,
    )
    simulation_config = SimulationConfig(
        assignment_randomness=0.3,
        sinkhorn_iter=1000,
        sinkhorn_tol=1e-5,
        transport_nonconvergence="raise",
    )
    sparse_generated = simulate_from_reference(
        fit_reference(sparse_reference, "ground_truth", fit_config),
        blueprint,
        config=simulation_config,
        random_seed=2026,
        marginal_model="empirical_reference",
    )
    dense_generated = simulate_from_reference(
        fit_reference(dense_reference, "ground_truth", fit_config),
        blueprint,
        config=simulation_config,
        random_seed=2026,
        marginal_model="empirical_reference",
    )

    np.testing.assert_array_equal(
        sparse_generated.layers["counts"], dense_generated.layers["counts"]
    )
    np.testing.assert_array_equal(sparse_generated.X, dense_generated.X)
    np.testing.assert_array_equal(
        sparse_generated.var_names.to_numpy(), dense_generated.var_names.to_numpy()
    )


def test_generated_contract_checks_blueprint_identity_before_rebinding():
    genes = ["g1", "g2", "g3"]
    config = fixture_config(genes)
    target = make_slice("target", genes)
    generated = target.copy()
    generated.obs_names = [str(index) for index in range(target.n_obs)]
    generated.obs["target_obs_id"] = target.obs_names.astype(str)
    generated.obs["domain"] = target.obs["ground_truth"].astype(str).to_numpy()
    generated.uns["de_novo"] = {
        "transport_diagnostics": diagnostics(("a", "b"))
    }
    records = validate_generated(
        generated,
        target,
        genes,
        config,
        "dlpfc",
        require_bound_obs_names=False,
    )
    assert len(records) == 2
    generated.obs.loc["0", "target_obs_id"] = "wrong"
    with pytest.raises(AssertionError, match="target IDs"):
        validate_generated(
            generated,
            target,
            genes,
            config,
            "dlpfc",
            require_bound_obs_names=False,
        )


@pytest.mark.parametrize(
    ("converged", "final_error"),
    [("False", "1e-7"), ("True", "1e-5"), ("True", "nan")],
)
def test_generated_contract_rejects_missing_positive_evidence(
    converged: str,
    final_error: str,
):
    genes = ["g1", "g2", "g3"]
    config = fixture_config(genes)
    target = make_slice("target", genes)
    generated = target.copy()
    generated.obs["target_obs_id"] = target.obs_names.astype(str)
    generated.obs["domain"] = target.obs["ground_truth"].astype(str).to_numpy()
    generated.uns["de_novo"] = {
        "transport_diagnostics": diagnostics(("a", "b"), converged, final_error)
    }
    with pytest.raises(RuntimeError):
        validate_generated(generated, target, genes, config, "dlpfc")


def test_declared_job_rejects_wrong_mask_direction_or_randomness():
    config = fixture_config(["g1"])
    require_declared_job(config, Job("mask_half", "dlpfc", "a", "a", 0.3))
    with pytest.raises(ValueError, match="undeclared"):
        require_declared_job(config, Job("mask_half", "dlpfc", "b", "a", 0.3))
    with pytest.raises(ValueError, match="undeclared"):
        require_declared_job(config, Job("mask_half", "dlpfc", "a", "a", 0.1))


def test_declared_donor_strata_are_explicit():
    assert donor_stratum(Job("cross_slice", "dlpfc", "151675", "151676", 0.3)) == (
        "within_donor"
    )
    assert donor_stratum(Job("cross_slice", "dlpfc", "151670", "151675", 0.3)) == (
        "cross_donor"
    )
    assert donor_stratum(Job("mask_half", "dlpfc", "151675", "151675", 0.3)) == (
        "within_slice"
    )
    assert donor_stratum(
        Job(
            "cross_slice",
            "merfish",
            "Zhuang-ABCA-1.006",
            "Zhuang-ABCA-1.007",
            0.3,
        )
    ) == "donor_not_declared"


def test_vectorized_scoring_reproduces_historical_scalar_metrics():
    genes = ["g1", "g2", "g3"]
    target = make_slice("target", genes)
    target_counts = np.array(
        [[4, 0, 1], [2, 3, 0], [0, 1, 5], [3, 2, 1]], dtype=np.int32
    )
    target.X = target_counts.copy()
    target.layers["counts"] = target_counts.copy()
    generated = target.copy()
    generated.layers["counts"] = sparse.csr_matrix(
        np.array([[3, 0, 2], [1, 4, 0], [0, 2, 4], [4, 1, 1]], dtype=np.int32)
    )
    generated.X = generated.layers["counts"].copy()
    neighbors = neighbor_indices(target.obsm["spatial"], 3)
    metrics = compute_per_gene_metrics(generated, target, neighbors, chunk_size=2)
    observed = np.asarray(target.layers["counts"])
    simulated = generated.layers["counts"].toarray()
    for index in range(len(genes)):
        assert metrics.loc[index, "pearson"] == pytest.approx(
            pearsonr(simulated[:, index], observed[:, index]).statistic
        )
        assert metrics.loc[index, "spearman"] == pytest.approx(
            spearmanr(simulated[:, index], observed[:, index]).statistic
        )
        assert metrics.loc[index, "generated_moran_i"] == pytest.approx(
            moran_i(simulated[:, index], neighbors)
        )
        assert metrics.loc[index, "target_moran_i"] == pytest.approx(
            moran_i(observed[:, index], neighbors)
        )
    summary = panel_summary(metrics)
    recomputed = recompute_panel_summary(metrics)
    assert recomputed == pytest.approx(summary, abs=1e-12)
    assert set(summary) == {
        "mean_corr",
        "var_corr",
        "moran_corr",
        "zero_ks",
        "median_gene_pearson",
        "median_gene_spearman",
    }
    assert np.isfinite(list(summary.values())).all()
