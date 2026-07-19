#!/usr/bin/env python3
"""Generate all fresh non-OT Study 01 FEAST simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(prefix: str, value: float) -> str:
    return f"{prefix}_{value:.2f}".replace(".", "_").replace("_00", "")


def build_alterations(config: dict) -> list[dict]:
    from FEAST import Alteration

    settings = config["alterations"]
    coupling = settings["mean"].get("variance_coupling")
    rows = []
    for value in settings["mean"]["levels"]:
        if float(value) != 1.0:
            rows.append({"simulation_id": slug("mean", value), "alteration_type": "mean", "value": value, "alteration": Alteration.mean_only(fold_change=value, variance_coupling=coupling)})
    rho = settings["variance"].get("heterogeneity_scale", 1.0)
    for value in settings["variance"]["levels"]:
        if float(value) != 1.0:
            rows.append({"simulation_id": slug("variance", value), "alteration_type": "variance", "value": value, "alteration": Alteration.variance_only(fold_change=value, dispersion=rho)})
    for value in settings["variance_heterogeneity"]["levels"]:
        if float(value) != 1.0:
            rows.append({"simulation_id": slug("var_hetero", value), "alteration_type": "variance_heterogeneity", "value": value, "alteration": Alteration.variance_heterogeneity(rho=value)})
    for value in settings["sparsity_logit_shift"]["levels"]:
        if float(value) != 0.0:
            sign = "pos" if value > 0 else "neg"
            rows.append({"simulation_id": slug(f"sparsity_{sign}", abs(value)), "alteration_type": "sparsity_logit_shift", "value": value, "alteration": Alteration.sparsity_logit(delta=value)})
    for name, values in settings["combined"].items():
        rows.append({"simulation_id": f"combined_{name}", "alteration_type": "combined", "value": None, "alteration": Alteration.comprehensive(mean_fc=values.get("mean", 1.0), var_fc=values.get("variance", 1.0), delta_z=values.get("sparsity", 0.0), mean_var_coupling=coupling)})
    rows.append({"simulation_id": "baseline", "alteration_type": "baseline", "value": 1.0, "alteration": Alteration()})
    if len(rows) != 27:
        raise RuntimeError(f"expected 27 conditions, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256(args.config)
    runner_path = Path(__file__).resolve()
    runner_hash = sha256(runner_path)
    feast_config = config["feast"]
    if feast_config.get("spatial_mode") != "reference_rank" or feast_config.get("assignment_solver") != "scipy" or feast_config.get("assignment_blocks") is not False:
        raise ValueError("Study 01 requires reference-rank global SciPy assignment")

    if args.dry_run:
        # FEAST is imported here only to construct and count the declared public alterations.
        alterations = build_alterations(config)
        for slice_id in config["slices"]:
            for alteration in alterations:
                print(f"{slice_id}/{alteration['simulation_id']}")
        print("81 fresh FEAST jobs")
        return 0

    import anndata as ad
    import FEAST

    if str(FEAST.__version__) != "1.0.2":
        raise RuntimeError(f"unexpected FEAST version: {FEAST.__version__}")

    alterations = build_alterations(config)
    sources = {str(slice_id): args.input_dir / f"{slice_id}.h5ad" for slice_id in config["slices"]}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source slices: {missing}")
    inputs = pd.read_csv(
        Path(__file__).parent / "data/input_checksums.csv", dtype={"slice_id": str}
    ).set_index("slice_id")
    input_hashes = {slice_id: sha256(path) for slice_id, path in sources.items()}
    mismatched = [
        slice_id
        for slice_id, digest in input_hashes.items()
        if digest != inputs.loc[slice_id, "sha256"]
    ]
    if mismatched:
        raise RuntimeError(f"source checksum mismatch: {mismatched}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "simulation_manifest.csv"
    previous: dict[tuple[str, str], dict] = {}
    if args.resume and manifest_path.is_file():
        previous_table = pd.read_csv(
            manifest_path, dtype={"slice_id": str, "simulation_id": str}
        )
        previous = {
            (record["slice_id"], record["simulation_id"]): record
            for record in previous_table.to_dict("records")
        }
    rows = []
    total = len(sources) * len(alterations)
    index = 0
    for slice_id, source in sources.items():
        reference = ad.read_h5ad(source)
        input_hash = input_hashes[slice_id]
        for alteration in alterations:
            index += 1
            started = time.time()
            relative = Path("simulations") / slice_id / f"{alteration['simulation_id']}.h5ad"
            output = args.output_dir / relative
            print(f"[{index}/{total}] {slice_id}/{alteration['simulation_id']}", flush=True)
            old = previous.get((slice_id, alteration["simulation_id"]), {})
            if (
                args.resume
                and old.get("status") == "ok"
                and old.get("configuration_id") == config["configuration_id"]
                and old.get("config_sha256") == config_hash
                and old.get("runner_source_sha256") == runner_hash
                and old.get("input_sha256") == input_hash
                and old.get("public_seed") == config["public_seed"]
                and old.get("feast_version") == "1.0.2"
                and old.get("feast_commit") == config["required_feast_commit"]
                and old.get("spatial_mode") == "reference_rank"
                and old.get("assignment_solver") == "scipy"
                and old.get("assignment_blocks") in (False, "False")
                and output.is_file()
                and sha256(output) == old.get("output_sha256")
            ):
                rows.append(old)
                pd.DataFrame(rows).to_csv(manifest_path, index=False)
                print("  skipped verified", flush=True)
                continue
            if output.exists():
                failure = (
                    args.output_dir
                    / "failures"
                    / slice_id
                    / f"{alteration['simulation_id']}__{int(time.time())}.h5ad"
                )
                failure.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(failure))
            try:
                simulated = FEAST.simulate(
                    reference,
                    seed=int(config["public_seed"]),
                    alteration=alteration["alteration"],
                    parameter_mode=feast_config["parameter_mode"],
                    spatial_mode="reference_rank",
                    annotation_key=config["annotation_key"],
                    assignment_method=feast_config["assignment_method"],
                    assignment_solver="scipy",
                    assignment_blocks=False,
                    use_distributional_alteration=True,
                    ppf_method=feast_config["ppf_method"],
                    beta_n_jobs=int(feast_config["beta_n_jobs"]),
                    beta_early_stopping_patience=int(feast_config["beta_early_stopping_patience"]),
                    convert_n_jobs=int(feast_config["convert_n_jobs"]),
                    n_jobs=int(feast_config["n_jobs"]),
                    use_heuristic_search=False,
                    clip_overshoot_factor=0.0,
                    boundary_multiplier=1.1,
                    verbose=True,
                )
                diagnostics = simulated.uns.get("simulation_diagnostics", {})
                assignment_diagnostics = diagnostics.get(
                    "copula_rank_diagnostics", {}
                )
                if diagnostics.get("spatial_mode") != "reference_rank":
                    raise RuntimeError("FEAST returned the wrong spatial mode")
                if assignment_diagnostics.get("assignment_blocks") is not False:
                    raise RuntimeError("FEAST did not use exact global assignment")
                simulated.uns["publication_provenance"] = {
                    "configuration_id": str(config["configuration_id"]), "config_sha256": config_hash, "runner_source_sha256": runner_hash, "feast_version": str(FEAST.__version__), "feast_commit": str(config["required_feast_commit"]), "public_seed": int(config["public_seed"]), "spatial_mode": "reference_rank", "assignment_solver": "scipy", "assignment_blocks": False, "input_sha256": input_hash,
                    "solver_diagnostics": {
                        "status": "completed",
                        "assignment_method": assignment_diagnostics.get("assignment_method"),
                        "assignment_blocks": False,
                        "assignment_solver_requested": "scipy",
                        "model_selection_counts": diagnostics.get("model_selection_counts", {}),
                    },
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                simulated.write_h5ad(output, compression="gzip")
                status, output_hash = "ok", sha256(output)
            except Exception as exc:
                status, output_hash = f"failed: {type(exc).__name__}: {exc}", ""
            rows.append({"configuration_id": config["configuration_id"], "config_sha256": config_hash, "runner_source_sha256": runner_hash, "slice_id": slice_id, "simulation_id": alteration["simulation_id"], "alteration_type": alteration["alteration_type"], "alteration_value": alteration["value"], "file_path": relative.as_posix(), "input_sha256": input_hash, "output_sha256": output_hash, "public_seed": config["public_seed"], "feast_version": FEAST.__version__, "feast_commit": config["required_feast_commit"], "spatial_mode": "reference_rank", "assignment_solver": "scipy", "assignment_blocks": False, "solver_status": "completed" if status == "ok" else "failed", "elapsed_seconds": round(time.time() - started, 2), "status": status})
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
    provenance = {"configuration_id": config["configuration_id"], "generated_utc": datetime.now(timezone.utc).isoformat(), "public_seed": config["public_seed"], "feast_version": FEAST.__version__, "feast_commit": config["required_feast_commit"], "config_sha256": config_hash, "runner_source": str(runner_path), "runner_source_sha256": runner_hash, "resume_mode": bool(args.resume), "job_count": len(rows), "successful_jobs": sum(row["status"] == "ok" for row in rows), "manifest_sha256": sha256(manifest_path)}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
