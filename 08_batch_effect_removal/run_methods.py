#!/usr/bin/env python
"""Run batch correction methods on simulated batch-effect data.

For each deformation mode and alpha level, runs GraphST, scVI, and STAMP
on the reference (alpha=0.0) vs query (alpha > 0.0) pair.

Usage:
    python run_methods.py [--config config.yaml] [--method scVI] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_method(
    method_name: str,
    method_cfg: dict,
    reference_path: Path,
    query_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single method on a reference-query pair.  Returns status dict."""

    python_bin = Path(method_cfg["env"]) / "bin" / "python"
    script = PROJECT_ROOT / method_cfg["script"]

    cmd = [
        str(python_bin),
        str(script),
        "--reference", str(reference_path),
        "--query", str(query_path),
        "--output-dir", str(output_dir),
    ]

    # Add method-specific kwargs
    for k, v in method_cfg.get("kwargs", {}).items():
        flag = "--" + k.replace("_", "-")
        cmd.extend([flag, str(v)])

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return {"status": "dry_run", "method": method_name}

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2h timeout
            cwd=str(PROJECT_ROOT),
        )

        elapsed = time.time() - t0

        if result.returncode == 0:
            return {
                "status": "ok",
                "method": method_name,
                "elapsed_seconds": round(elapsed, 2),
                "stdout": result.stdout[-500:] if result.stdout else "",
            }
        else:
            fail_msg = result.stderr[-500:] if result.stderr else "unknown error"
            return {
                "status": "failed",
                "method": method_name,
                "elapsed_seconds": round(elapsed, 2),
                "error": fail_msg,
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "method": method_name,
            "elapsed_seconds": 7200,
            "error": "2h timeout reached",
        }
    except Exception as e:
        return {
            "status": "error",
            "method": method_name,
            "elapsed_seconds": round(time.time() - t0, 2),
            "error": str(e)[:500],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="config.yaml")
    parser.add_argument("--method", type=str, default=None,
                        help="Run only this method (GraphST, scVI, STAMP)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Run only this deformation mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Sleep seconds between calls (avoid GPU OOM)")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_root = Path(config.get("output_dir", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(out_root / "run_logs" / timestamp)
    methods_out = _ensure_dir(out_root / "methods")

    sim_cfg = config["simulation"]
    modes = config["deformation_modes"]
    alphas = config["alpha_levels"]
    methods_cfg = config["methods"]
    seed = config.get("random_seed", 42)

    if args.mode:
        modes = [m for m in modes if m == args.mode]
    if args.method:
        methods_cfg = {k: v for k, v in methods_cfg.items() if k == args.method}

    print(f"Modes: {modes}")
    print(f"Alphas: {alphas}")
    print(f"Methods: {list(methods_cfg.keys())}")
    print(f"Output: {methods_out}")
    print()

    run_rows = []
    total = len(modes) * (len(alphas) - 1) * len(methods_cfg)  # alpha=0.0 is reference
    done = 0

    # Discover latest simulation output
    sim_output_dir = None
    sim_manifest_path = None
    if "manifest_dir" in sim_cfg:
        manifest_dir = Path(sim_cfg["manifest_dir"])
        if not manifest_dir.is_absolute():
            manifest_dir = Path(__file__).resolve().parent / manifest_dir
        subdirs = sorted([d for d in manifest_dir.iterdir() if d.is_dir()], reverse=True)
        for d in subdirs:
            candidate = d / "manifest.csv"
            if candidate.exists():
                sim_output_dir = d
                sim_manifest_path = candidate
                break

    if sim_manifest_path is None:
        print("ERROR: no simulation manifest found", file=sys.stderr)
        return 1

    sim_manifest = pd.read_csv(sim_manifest_path)
    print(f"Simulation data: {sim_output_dir} ({len(sim_manifest)} files)")

    # Manifest file paths are relative to the simulation script's cwd
    # (effect_simulation/), so resolve from there.
    sim_base = Path(__file__).resolve().parent / "effect_simulation"

    for mode in modes:
        ref_path = None
        ref_match = sim_manifest[(sim_manifest["mode"] == mode) & (sim_manifest["alpha"] == 0.0)]
        if len(ref_match) > 0:
            ref_path = Path(ref_match.iloc[0]["file"])
            if not ref_path.is_absolute():
                ref_path = sim_base / ref_path

        if ref_path is None or not ref_path.exists():
            print(f"ERROR: reference not found for mode {mode}: {ref_path}")
            continue

        for alpha in alphas:
            if alpha == 0.0:
                continue

            query_path = None
            match = sim_manifest[(sim_manifest["mode"] == mode) & (sim_manifest["alpha"] == alpha)]
            if len(match) > 0:
                query_path = Path(match.iloc[0]["file"])
                if not query_path.is_absolute():
                    query_path = sim_base / query_path

            if query_path is None or not query_path.exists():
                print(f"MISSING: {mode}/alpha_{alpha:.2f} — skipping")
                for m_name in methods_cfg:
                    run_rows.append({
                        "mode": mode, "alpha": alpha, "method": m_name,
                        "status": "missing_input", "error": "query h5ad not found",
                    })
                continue

            for m_name, m_cfg in methods_cfg.items():
                done += 1
                m_out = methods_out / m_name / mode / f"alpha_{alpha:.2f}"
                print(f"[{done}/{total}] {m_name} / {mode} / alpha={alpha:.2f}")

                result = run_method(
                    m_name, m_cfg, ref_path, query_path, m_out,
                    dry_run=args.dry_run,
                )
                result.update({"mode": mode, "alpha": alpha})
                run_rows.append(result)

                if result["status"] == "ok":
                    print(f"  -> OK ({result['elapsed_seconds']:.1f}s)")
                else:
                    print(f"  -> {result['status'].upper()}: {result.get('error', '')[:120]}")

                if args.sleep > 0:
                    time.sleep(args.sleep)

    # Write run manifest
    manifest_df = pd.DataFrame(run_rows)
    manifest_path = run_dir / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    ok_count = (manifest_df["status"] == "ok").sum() if "status" in manifest_df.columns else 0
    print(f"\nDone: {ok_count}/{len(run_rows)} jobs OK")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
