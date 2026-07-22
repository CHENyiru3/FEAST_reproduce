from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT))

import run  # noqa: E402
import validate  # noqa: E402
import aggregate  # noqa: E402
import compare_historical  # noqa: E402
from workflow import (  # noqa: E402
    SliceInfo,
    build_assignments,
    choose_donor,
    density_specs,
    load_config,
    representative_reference_ids,
    validate_fixed_contract,
)


def synthetic_slice_infos() -> dict[int, SliceInfo]:
    absent = {75, 94, 112}
    return {
        slice_id: SliceInfo(
            slice_id=slice_id,
            path=Path(f"slice-{slice_id:03d}.h5ad"),
            z=0.0 if slice_id <= 4 else float(slice_id - 4) / 10.0,
            n_obs=100,
            n_vars=1122,
        )
        for slice_id in range(1, 151)
        if slice_id not in absent
    }


def test_frozen_config_declares_strict_public_contract() -> None:
    config = load_config()
    assert config["generation"]["public_api"] == (
        "FEAST.de_novo.fit_reference + FEAST.de_novo.simulate_from_reference"
    )
    assert config["generation"]["target_expression_access"] == "evaluation_only"
    assert config["generation"]["donor_support_scope"] == "full density-specific reference pool"
    assert config["generation"]["smoothing"] is False
    assert config["generation"]["z_regularization"] is False
    assert config["transport"]["sinkhorn_iter"] == 1000
    assert config["transport"]["sinkhorn_tol"] == 1.0e-5
    assert config["transport"]["transport_nonconvergence"] == "raise"
    assert config["transport"]["max_transport_pairs"] == 25_000_000
    assert config["transport"]["sinkhorn_method"] == "sinkhorn_log"
    assert config["transport"]["transport_backend"] == "torch"
    assert config["transport"]["transport_device"] == "cuda:0"
    assert config["transport"]["transport_dtype"] == "float64"
    assert config["assignment_randomness_preflight"]["transport_backend"] == "torch"
    assert config["assignment_randomness_preflight"]["transport_device"] == "cpu"
    assert [
        row["expected_assignment_randomness"] for row in config["densities"]
    ] == [0.35, 0.30, 0.35]
    assert config["required_wheel_sha256"] == (
        "3ad31888faf367a91aec9d46902a5e89759837b5c7ea0e45ca93276674e68883"
    )


def test_config_rejects_a_different_feast_wheel() -> None:
    import copy

    changed = copy.deepcopy(load_config())
    changed["required_wheel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="publication wheel"):
        validate_fixed_contract(changed)


def test_evaluation_declares_split_half_trajectory_unit_not_replicates() -> None:
    evaluation = load_config()["evaluation"]
    assert evaluation["unit"] == "class-gene trajectory across ordered z levels within each density"
    assert evaluation["targets_are_biological_replicates"] is False
    assert evaluation["inferential_statistics_across_targets"] == "prohibited"
    assert evaluation["real_continuity_baseline"] == {
        "method": "deterministic within-slice within-class split-half means, correlated across z",
        "seed_formula": "public_seed + target_slice",
        "minimum_class_spots": 2,
        "minimum_z_levels": 3,
    }


@pytest.mark.parametrize(
    ("gap", "expected_targets", "expected_references", "representatives", "first_seed"),
    [
        (3, 49, 95, [4, 76, 150], 32026),
        (5, 29, 57, [4, 74, 148], 52026),
        (10, 15, 16, [1, 81, 150], 102026),
    ],
)
def test_exact_target_plan_and_seed_indices(
    gap: int,
    expected_targets: int,
    expected_references: int,
    representatives: list[int],
    first_seed: int,
) -> None:
    config = load_config()
    spec = next(item for item in density_specs(config) if item.gap == gap)
    assignments, references = build_assignments(synthetic_slice_infos(), spec, 2026)
    assert len(assignments) == expected_targets
    assert len(references) == expected_references
    assert representative_reference_ids(references, 3) == representatives
    assert assignments[0].seed == first_seed
    assert assignments[-1].seed == first_seed + expected_targets - 1
    assert [row.target_index for row in assignments] == list(range(expected_targets))


def test_donor_selection_is_reference_only_nearest_z_then_slice_id() -> None:
    infos = synthetic_slice_infos()
    support = {"rare": [10, 20]}
    # z=1.1 is equidistant from slice 10 (0.6) and 20 (1.6); lower ID wins.
    assert choose_donor("rare", 15, 1.1, support, infos) == 10
    with pytest.raises(ValueError, match="no declared reference donor"):
        choose_donor("missing", 15, 1.1, support, infos)


def test_shards_preserve_original_target_indices_and_seeds() -> None:
    rows = [
        {"target_index": index, "target_slice": 5 + index * 3, "seed": 32026 + index}
        for index in range(8)
    ]
    selected = run.select_targets(rows, None, shard_index=1, shard_count=3)
    assert [(item["target_index"], item["seed"]) for item in selected] == [(1, 32027), (4, 32030), (7, 32033)]
    selected_one = run.select_targets(rows, [17], shard_index=1, shard_count=3)
    assert [item["target_slice"] for item in selected_one] == [17]
    with pytest.raises(ValueError, match="not in this density"):
        run.select_targets(rows, [999], shard_index=0, shard_count=1)


def valid_transport_records() -> dict[str, object]:
    return {
        "format": "columnar_records_v1",
        "n_records": 2,
        "transport_converged": np.array(["True", "True"]),
        "transport_final_error": np.array(["1e-7", "2e-7"]),
        "transport_stop_threshold": np.array(["1e-5", "1e-5"]),
        "transport_iterations": np.array(["30", "40"]),
        "transport_max_iterations": np.array(["1000", "1000"]),
        "transport_nonconvergence_policy": np.array(["raise", "raise"]),
        "transport_solver_method": np.array(
            ["sinkhorn_log", "sinkhorn_log"]
        ),
        "transport_backend": np.array(["torch", "torch"]),
        "transport_device": np.array(["cuda:0", "cuda:0"]),
        "transport_dtype": np.array(["float64", "float64"]),
        "reference_name": np.array(["ref-a", "ref-b"]),
        "transport_mass": np.array(["0.8", "0.9"]),
    }


def test_validator_requires_positive_transport_evidence() -> None:
    transport = load_config()["transport"]
    diagnostics = {"class-a": valid_transport_records()}
    count, maximum = validate.validate_transport_diagnostics(diagnostics, transport, {"ref-a", "ref-b"})
    assert count == 2
    assert maximum == pytest.approx(2.0e-7)

    diagnostics["class-a"]["transport_converged"][1] = "False"
    with pytest.raises(ValueError, match="not converged"):
        validate.validate_transport_diagnostics(diagnostics, transport, {"ref-a", "ref-b"})


def test_validator_rejects_finite_but_above_tolerance_transport() -> None:
    transport = load_config()["transport"]
    records = valid_transport_records()
    records["transport_final_error"][0] = "1e-4"
    with pytest.raises(ValueError, match="sub-tolerance"):
        validate.validate_transport_diagnostics({"class-a": records}, transport, {"ref-a", "ref-b"})

    records = valid_transport_records()
    records["transport_mass"][0] = "0"
    with pytest.raises(ValueError, match="nonpositive transport mass"):
        validate.validate_transport_diagnostics({"class-a": records}, transport, {"ref-a", "ref-b"})


def test_count_validation_is_integral_nonnegative_and_finite() -> None:
    matrix = np.array([[0, 1], [2, 3]], dtype=np.int32)
    np.testing.assert_array_equal(validate.validate_count_matrix(matrix, (2, 2)), matrix)
    with pytest.raises(ValueError, match="negative"):
        validate.validate_count_matrix(np.array([[0.0, -1.0]]), (1, 2))
    with pytest.raises(ValueError, match="non-integral"):
        validate.validate_count_matrix(np.array([[0.0, 1.5]]), (1, 2))
    with pytest.raises(ValueError, match="non-finite"):
        validate.validate_count_matrix(np.array([[0.0, np.nan]]), (1, 2))


def test_blueprint_preserves_only_declared_target_metadata() -> None:
    metadata = {
        "obs_names": ["spot-a", "spot-b"],
        "var_names": ["g1", "g2"],
        "labels": np.array(["a", "b"]),
        "z": np.array([1.5, 1.5]),
        "spatial": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "spatial_3d": np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 1.5]]),
        "grid_type": "generic",
        "technology": None,
    }
    blueprint = run.make_blueprint(metadata, "class")
    assert blueprint.obs["_target_obs_name"].tolist() == metadata["obs_names"]
    np.testing.assert_array_equal(blueprint.coordinates, metadata["spatial_3d"])
    np.testing.assert_array_equal(blueprint.domain_map, metadata["labels"])
    assert set(blueprint.obs) == {"_target_obs_name", "class", "domain"}


def test_atomic_metrics_are_identity_exact_without_composite() -> None:
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 1.0],
            [3.0, 0.0, 5.0],
            [2.0, 6.0, 1.0],
        ]
    )
    metrics = aggregate.metric_row(matrix, matrix.copy())
    for name, value in metrics.items():
        assert name in aggregate.METRICS
        assert value == pytest.approx(0.0 if name == "relative_error_gene_mean" else 1.0)
    assert "composite" not in " ".join(metrics).lower()


def test_real_split_half_baseline_is_deterministic_within_class() -> None:
    matrix = np.arange(24, dtype=float).reshape(8, 3)
    labels = np.array(["a"] * 4 + ["b"] * 4)
    first = aggregate.split_half_class_means(matrix, labels, seed=2031)
    second = aggregate.split_half_class_means(matrix, labels, seed=2031)
    assert set(first) == {"a", "b"}
    for label in first:
        np.testing.assert_array_equal(first[label][0], second[label][0])
        np.testing.assert_array_equal(first[label][1], second[label][1])


def test_z_coherence_uses_class_gene_trajectories_across_levels() -> None:
    trajectories = {
        "class-a": [
            {
                "z": float(z),
                "generated": np.array([z, 2 * z], dtype=float),
                "target": np.array([z, 2 * z], dtype=float),
                "split_a": np.array([z, 2 * z], dtype=float),
                "split_b": np.array([z, 2 * z], dtype=float),
            }
            for z in range(4)
        ]
    }
    rows = aggregate.z_coherence_rows(trajectories, ["g1", "g2"], "split_a", "split_b", 3)
    assert len(rows) == 2
    assert {row["n_z_levels"] for row in rows} == {4}
    assert all(row["z_coherence"] == pytest.approx(1.0) for row in rows)
    frame = __import__("pandas").DataFrame(rows)
    summary = aggregate.continuity_summary("dense_gap3", 3, frame, frame)
    assert summary["targets_are_biological_replicates"] is False
    assert summary["normalized_z_coherence"] == pytest.approx(1.0)


def test_old_vs_new_entry_point_separates_solver_change() -> None:
    import pandas as pd

    config = load_config()
    historical_rows = []
    fresh_rows = []
    continuity_rows = []
    for gap, targets in ((3, 49), (5, 29), (10, 15)):
        old = {"density_gap": gap, "n_targets": targets}
        fresh = {"gap": gap, "targets": targets}
        continuity = {"gap": gap}
        for _, _, old_column, fresh_column in compare_historical.COMPARABLE_METRICS:
            old[old_column] = 0.5
            fresh[fresh_column] = 0.6
        for _, _, old_column, fresh_column in compare_historical.CONTINUITY_METRICS:
            old[old_column] = 0.4
            continuity[fresh_column] = 0.5
        historical_rows.append(old)
        fresh_rows.append(fresh)
        continuity_rows.append(continuity)
    comparison = compare_historical.build_comparison(
        pd.DataFrame(historical_rows),
        pd.DataFrame(fresh_rows),
        pd.DataFrame(continuity_rows),
        config,
    )
    assert len(comparison) == 34
    solver = comparison.loc[comparison["comparison_kind"] == "solver_configuration"].iloc[0]
    assert solver["historical"] == 200
    assert solver["fresh"] == 1000
    assert solver["comparable_formula"] == False  # noqa: E712
    assert set(comparison["publication_decision"]) == {"pending_author_review"}
