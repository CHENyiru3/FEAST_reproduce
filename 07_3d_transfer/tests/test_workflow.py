from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import anndata as ad
from scipy import sparse


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT))

import run  # noqa: E402
import consolidate  # noqa: E402
import workflow  # noqa: E402


def test_reference_counts_cache_preserves_exact_values():
    reference = ad.AnnData(np.asarray([[1, 0], [0, 2]], dtype=np.int32))
    reference.layers["counts"] = sparse.csr_matrix(reference.X)
    expected = np.asarray(reference.X, dtype=np.float32)

    record = run.cache_reference_counts_dense_float32(reference)

    assert isinstance(reference.layers["counts"], np.ndarray)
    assert reference.layers["counts"].dtype == np.float32
    np.testing.assert_array_equal(reference.layers["counts"], expected)
    assert record["value_transform"] == "none"
    assert record["shape"] == [2, 2]


def test_shard_id_accepts_documented_separators_only():
    assert run.valid_shard_id("canary_v2")
    assert run.valid_shard_id("age-low-01")
    assert not run.valid_shard_id("")
    assert not run.valid_shard_id("../escape")


def test_full_axis_configuration_is_exact():
    config = workflow.load_config()
    assert workflow.expected_z_values(config, "E15.5") == [round(-4.12 + 0.02 * i, 2) for i in range(158)]
    assert workflow.expected_z_values(config, "E18.5") == [round(-5.56 + 0.02 * i, 2) for i in range(202)]
    assert config["reference_fit"] == {
        "label_key": "region",
        "expected_genes": 550,
        "min_gene_spots": 1,
        "min_gene_mean": 0.0,
        "max_gene_zero_prop": 1.0,
        "boundary_neighbors": 6,
        "coordinate_scale": None,
    }
    assert config["transport"]["transport_nonconvergence"] == "raise"
    assert config["transport"]["sinkhorn_iter"] == 1000
    assert config["transport"]["sinkhorn_tol"] == 1e-5
    assert config["transport"]["max_transport_pairs"] == 25_000_000
    assert config["transport"]["sinkhorn_method"] == "sinkhorn_log"
    assert config["transport"]["transport_backend"] == "torch"
    assert config["transport"]["transport_device"] == "cuda:0"
    assert config["transport"]["transport_dtype"] == "float64"
    assert config["assignment_randomness_calibration"]["transport_backend"] == "torch"
    assert config["assignment_randomness_calibration"]["transport_device"] == "cpu"


def test_synthetic_extraction_masks_other_sorts_z_and_assigns_stable_ids():
    data = np.zeros((3, 3, 2), dtype=np.uint8)
    data[0, 0, 0] = 1
    data[1, 0, 1] = 2  # excluded Other
    data[2, 2, 1] = 1
    affine = np.array([
        [0.5, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.25, 20.0],
        [0.0, -0.5, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    schema = {
        0: {"label": "Background", "include": False},
        1: {"label": "CP", "include": True},
        2: {"label": "Other", "include": False},
    }
    payload = workflow.extract_blueprint_payload(data, affine, schema, "E15.5")
    entries = payload["slices"]
    assert [entry["z_world"] for entry in entries] == [-2.0, -1.0]
    assert [entry["n_spots"] for entry in entries] == [1, 1]
    assert entries[0]["region"] == ["CP"]
    assert entries[0]["x"] == [11.0]
    assert entries[0]["y"] == [20.25]
    assert workflow.make_spot_ids("E15.5", entries[0]) == [
        "devccf-e15_5-z000-j002-i002-k001"
    ]
    assert workflow.make_spot_ids("E15.5", entries[0]) == workflow.make_spot_ids("E15.5", entries[0])


@pytest.mark.parametrize(
    "age,expected_levels,expected_start,expected_end,expected_spots",
    [
        ("E15.5", 158, -4.12, -0.98, 2_330_927),
        ("E18.5", 202, -5.56, -1.54, 5_213_461),
    ],
)
def test_real_nifti_reader_and_extraction_scope_regression(
    age, expected_levels, expected_start, expected_end, expected_spots
):
    config = workflow.load_config()
    volume = config["_devccf_dir"] / config["ages"][age]["volume"]
    schema_path = config["_devccf_dir"] / config["blueprint"]["region_schema"]
    if not volume.is_file() or not schema_path.is_file():
        pytest.skip("site DevCCF inputs are not materialized")
    data, affine = workflow.read_nifti_gz(volume)
    summary = workflow.summarize_volume_contract(data, affine, workflow.load_region_schema(schema_path))
    assert summary["n_z_levels"] == expected_levels
    assert summary["z_values"][0] == expected_start
    assert summary["z_values"][-1] == expected_end
    assert summary["z_values"] == [round(expected_start + 0.02 * i, 2) for i in range(expected_levels)]
    assert summary["n_spots"] == expected_spots


def test_original_indices_define_selection_and_public_seeds():
    args = argparse.Namespace(z_indices=[157, 0, 79], start_index=None, stop_index=None, exclude_indices=None)
    assert run.selected_indices(args, 158) == [157, 0, 79]
    assert [2026 + index for index in run.selected_indices(args, 158)] == [2183, 2026, 2105]
    with pytest.raises(ValueError, match="duplicates"):
        run.selected_indices(argparse.Namespace(z_indices=[1, 1], start_index=None, stop_index=None, exclude_indices=None), 3)


def test_canary_selection_is_executable_and_contains_failure_middle_densest_endpoints():
    e15 = [
        {"z_world": round(-4.12 + 0.02 * index, 2), "n_spots": 100}
        for index in range(158)
    ]
    e15[47]["n_spots"] = 999
    assert run.canary_indices("E15.5", e15) == [12, 0, 78, 47, 157]
    e18 = [
        {"z_world": round(-5.56 + 0.02 * index, 2), "n_spots": 100}
        for index in range(202)
    ]
    e18[62]["n_spots"] = 999
    assert run.canary_indices("E18.5", e18) == [0, 100, 62, 201]


def test_transport_requires_positive_convergence_evidence():
    class Result:
        uns = {
            "de_novo": {
                "transport_diagnostics": {
                    "CP": {
                        "format": "columnar_records_v1",
                        "n_records": 1,
                        "transport_converged": ["True"],
                        "transport_final_error": ["1e-7"],
                        "transport_stop_threshold": ["1e-5"],
                        "transport_max_iterations": ["1000"],
                        "transport_mass": ["0.99"],
                        "transport_nonconvergence_policy": ["raise"],
                        "transport_solver_method": ["sinkhorn_log"],
                        "transport_backend": ["torch"],
                        "transport_device": ["cuda:0"],
                        "transport_dtype": ["float64"],
                        "transport_iterations": ["42"],
                    }
                }
            }
        }

    contract = workflow.frozen_transport_config(workflow.load_config(), 0.4)
    assert workflow.transport_summary(Result(), contract)["all_converged"] is True
    Result.uns["de_novo"]["transport_diagnostics"]["CP"]["transport_converged"] = ["False"]
    with pytest.raises(ValueError, match="nonconverged"):
        workflow.transport_summary(Result(), contract)


def test_transport_rejects_relaxed_threshold_and_wheel_identity_is_pinned():
    config = workflow.load_config()
    contract = workflow.frozen_transport_config(config, 0.4)

    class Result:
        uns = {
            "de_novo": {
                "transport_diagnostics": {
                    "CP": {
                        "format": "columnar_records_v1",
                        "n_records": 1,
                        "transport_converged": ["True"],
                        "transport_final_error": ["1e-7"],
                        "transport_stop_threshold": ["1e-4"],
                        "transport_max_iterations": ["1000"],
                        "transport_mass": ["1.0"],
                        "transport_nonconvergence_policy": ["raise"],
                        "transport_solver_method": ["sinkhorn_log"],
                        "transport_backend": ["torch"],
                        "transport_device": ["cuda:0"],
                        "transport_dtype": ["float64"],
                        "transport_iterations": ["42"],
                    }
                }
            }
        }

    with pytest.raises(ValueError, match="threshold"):
        workflow.transport_summary(Result(), contract)
    assert inspect.signature(workflow.feast_identity).parameters[
        "require_cuda"
    ].default is True
    identity = workflow.feast_identity(config, require_cuda=False)
    assert identity["wheel_sha256"] == config["required_wheel_sha256"]
    assert "site-packages" in identity["import_path"]
    assert identity["stage_requires_cuda"] == "false"


def test_materialized_blueprint_provenance_is_fail_closed():
    config = workflow.load_config()
    for age in workflow.AGE_ORDER:
        assert workflow.verify_prepared_blueprint(config, age)[
            "blueprint_configuration_id"
        ] == config["blueprint_configuration_id"]
    changed = dict(config)
    changed["blueprint_configuration_id"] = "changed-blueprint-contract"
    with pytest.raises(ValueError, match="blueprint_configuration_id"):
        workflow.verify_prepared_blueprint(changed, "E15.5")


def test_per_slice_lineage_survives_h5ad_roundtrip(tmp_path):
    config = workflow.load_config()
    entry = {"z_index": 0, "z_world": -4.12}
    contract = workflow.frozen_transport_config(config, 0.4)
    diagnostics = {
        "CP": {
            "format": "columnar_records_v1",
            "n_records": 1,
            "transport_converged": ["True"],
            "transport_final_error": ["1e-7"],
            "transport_stop_threshold": ["1e-5"],
            "transport_max_iterations": ["1000"],
            "transport_mass": ["0.99"],
            "transport_nonconvergence_policy": ["raise"],
            "transport_solver_method": ["sinkhorn_log"],
            "transport_backend": ["torch"],
            "transport_device": ["cuda:0"],
            "transport_dtype": ["float64"],
            "transport_iterations": ["42"],
        }
    }
    result = ad.AnnData(np.zeros((1, 1), dtype=np.int32))
    result.uns["de_novo"] = {"transport_diagnostics": diagnostics}
    summary = workflow.transport_summary(result, contract)
    references = {"synthetic-reference": {"filename": "reference.h5ad", "sha256": "a" * 64}}
    result.uns["publication_reproduction"] = {
        "configuration_id": config["configuration_id"],
        "study_id": "07",
        "legacy_study_id": "06",
        "age": "E15.5",
        "z_index": 0,
        "z_world": -4.12,
        "public_seed": 2026,
        "assignment_randomness": 0.4,
        "assignment_randomness_provenance": {
            "mode": "fixed",
            "basis": config["ages"]["E15.5"]["assignment_randomness"]["basis"],
        },
        "config_sha256": workflow.sha256_file(config["_config_path"]),
        "blueprint_sha256": workflow.sha256_file(workflow.blueprint_path(config, "E15.5")),
        "blueprint_manifest_sha256": workflow.sha256_file(workflow.blueprint_manifest_path(config, "E15.5")),
        "runner_sha256": workflow.sha256_file(workflow.STUDY_ROOT / "run.py"),
        "feast_version": config["required_feast_version"],
        "feast_commit": config["required_feast_commit"],
        "feast_wheel_sha256": config["required_wheel_sha256"],
        "feast_source_patch_sha256": config["required_source_patch_sha256"],
        "feast_candidate_provenance_sha256": config[
            "required_feast_provenance_sha256"
        ],
        "reference_artifacts": references,
        "transport_config": contract,
        "solver_diagnostics": summary,
    }
    path = tmp_path / "lineage.h5ad"
    result.write_h5ad(path)
    restored = ad.read_h5ad(path)
    workflow.validate_publication_lineage(
        restored, config, "E15.5", entry, contract, references
    )


def test_synthetic_shard_records_reject_duplicate_active_z_indices(tmp_path):
    config = {"_work_dir": tmp_path}
    first = tmp_path / "E15.5" / "first"
    first.mkdir(parents=True)
    (first / "z000.h5ad").write_bytes(b"candidate-one")
    (first / "z000.record.json").write_text(
        '{"z_index": 0, "filename": "z000.h5ad", "output_sha256": "unused"}\n',
        encoding="utf-8",
    )
    candidates = consolidate.collect_candidates(config, "E15.5")
    assert list(candidates) == [0]

    second = tmp_path / "E15.5" / "second"
    second.mkdir()
    (second / "z000.h5ad").write_bytes(b"candidate-two")
    (second / "z000.record.json").write_text(
        '{"z_index": 0, "filename": "z000.h5ad", "output_sha256": "unused"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate active candidate"):
        consolidate.collect_candidates(config, "E15.5")
