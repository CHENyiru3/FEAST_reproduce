"""Generate alignment simulation data using simulate_alignment_rotation API."""
import sys, argparse, csv
from pathlib import Path
import scanpy as sc

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
from FEAST.alignment import simulate_alignment_rotation

ANGLES = [1, 5, 10, 30, 45, 60]

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(args.input)
sc.pp.filter_genes(adata, min_cells=30)
print(f"Input: {adata.shape[0]} spots, {adata.shape[1]} genes")

manifest_rows = []
reference_path = None

for angle in ANGLES:
    print(f"\n--- Angle {angle}° ---")
    original_adata, transformed_adata = simulate_alignment_rotation(
        adata,
        rotation_angle=angle,
        data_type="sequencing",
        filter_edge_spots=True,
        edge_margin_ratio=0.03,
        fit_params={"spatial_mode": "reference_rank", "random_seed": args.seed, "verbose": True},
    )

    if angle == 1:
        reference_path = output_dir / "reference.h5ad"
        original_adata.write_h5ad(reference_path)
        print(f"Reference saved: {reference_path} ({original_adata.shape[0]} spots)")
        manifest_rows.append({
            "alteration_id": "reference", "alteration_type": "none",
            "fold_change": 0.0, "type": "reference", "angle": 0.0,
            "file_path": str(reference_path),
            "n_spots": original_adata.shape[0],
            "n_genes": original_adata.shape[1], "seed": args.seed,
        })

    tag = str(int(angle))
    rot_path = output_dir / f"baseline_rotated_{tag}.h5ad"
    transformed_adata.write_h5ad(rot_path)
    print(f"Rotated saved: {rot_path} ({transformed_adata.shape[0]} spots)")
    manifest_rows.append({
        "alteration_id": "baseline", "alteration_type": "baseline",
        "fold_change": 1.0, "type": "rotated", "angle": float(angle),
        "file_path": str(rot_path),
        "n_spots": transformed_adata.shape[0],
        "n_genes": transformed_adata.shape[1], "seed": args.seed,
    })

manifest_path = output_dir / "simulation_manifest.csv"
with open(manifest_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    w.writeheader()
    w.writerows(manifest_rows)
print(f"\nManifest: {manifest_path} ({len(manifest_rows)} rows)")
print("Done.")
