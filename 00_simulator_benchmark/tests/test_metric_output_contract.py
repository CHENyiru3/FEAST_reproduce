"""Output contract tests for the metrics builder.

Covers:
  - metric builder outputs expected columns
  - metric builder only uses matched inventory rows
  - metric builder writes metadata
  - edge cases (empty inventory, missing reference)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "scripts"))

from build_simulator_quality_metrics import (
    compute_metrics,
    OUTPUT_COLUMNS,
    METRIC_VERSION,
)


def _write_fake_inventory(tmp_path, rows, sample_names):
    """Write a minimal inventory CSV and matching exper_data .h5ad files."""
    import anndata as ad

    inv_path = tmp_path / "inventory.csv"
    exper_data = tmp_path / "exper_data"
    exper_data.mkdir()
    recs = []
    for i, (sim_label, status) in enumerate(rows):
        sample = sample_names[i % len(sample_names)]
        sim_dir = tmp_path / "sim_outputs" / sim_label
        sim_dir.mkdir(parents=True, exist_ok=True)
        sim_path = sim_dir / f"{sample}.h5ad"
        ref_path = exper_data / f"{sample}.h5ad"
        rng = np.random.default_rng(42 + i)
        X = rng.poisson(3, size=(30, 10)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obsm["spatial"] = np.column_stack([
            np.arange(30, dtype=np.float32),
            np.zeros(30, dtype=np.float32),
        ])
        adata.write_h5ad(sim_path)
        if not ref_path.exists():
            ref_adata = ad.AnnData(
                X=rng.poisson(5, size=(30, 10)).astype(np.float32)
            )
            ref_adata.obsm["spatial"] = np.column_stack([
                np.arange(30, dtype=np.float32),
                np.zeros(30, dtype=np.float32),
            ])
            ref_adata.write_h5ad(ref_path)
        recs.append({
            "simulator": sim_label,
            "sample": sample,
            "file_path": str(sim_path),
            "status": status,
        })
    pd.DataFrame(recs).to_csv(inv_path, index=False)
    return inv_path, exper_data


class TestMetricBuilderOutputColumns:
    def test_output_has_expected_columns(self, tmp_path):
        inv_path, exper_data = _write_fake_inventory(
            tmp_path,
            [("FEAST_OT", "matched")],
            ["sample_A"],
        )
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 0
        df = pd.read_csv(out)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        # No extra columns
        assert set(df.columns) == set(OUTPUT_COLUMNS)


class TestMetricBuilderRowFiltering:
    def test_only_uses_matched_inventory_rows(self, tmp_path):
        inv_path, exper_data = _write_fake_inventory(
            tmp_path,
            [
                ("FEAST_OT", "matched"),
                ("FEAST_Rank", "matched"),
                ("SRTsim", "matched"),
                ("scCube", "missing_simulation"),
                ("Splatter", "excluded_by_config"),
            ],
            ["sample_A"],
        )
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 0
        df = pd.read_csv(out)
        sims = set(df["simulator"])
        assert "scCube" not in sims
        assert "Splatter" not in sims
        assert len(df) == 3


class TestMetricMetadata:
    def test_writes_metric_metadata(self, tmp_path):
        inv_path, exper_data = _write_fake_inventory(
            tmp_path,
            [("FEAST_OT", "matched")],
            ["sample_A"],
        )
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 0
        meta_path = tmp_path / "metrics_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["metric_version"] == METRIC_VERSION
        assert "identity_thresholds" in meta
        assert meta["identity_thresholds"]["mean_corr"] == 0.995
        assert meta["n_metric_rows"] == 1

    def test_records_counts(self, tmp_path):
        inv_path, exper_data = _write_fake_inventory(
            tmp_path,
            [("FEAST_OT", "matched"), ("FEAST_Rank", "matched")],
            ["sample_A"],
        )
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 0
        meta_path = tmp_path / "metrics_metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["n_inventory_rows"] == 2
        assert meta["n_matched_rows"] == 2
        assert meta["n_metric_rows"] == 2


class TestMetricBuilderEdgeCases:
    def test_empty_inventory_returns_1(self, tmp_path):
        inv_path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["simulator", "sample", "file_path", "status"]).to_csv(
            inv_path, index=False
        )
        exper_data = tmp_path / "exper_data"
        exper_data.mkdir()
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 1

    def test_missing_reference_skipped(self, tmp_path):
        inv_path, exper_data = _write_fake_inventory(
            tmp_path,
            [("FEAST_OT", "matched")],
            ["sample_A"],
        )
        ref_path = exper_data / "sample_A.h5ad"
        ref_path.unlink()
        out = tmp_path / "metrics.csv"
        rc = compute_metrics(inv_path, exper_data, out)
        assert rc == 1
