"""Run the 28 fixed-plate PASTE2 jobs in a new, resumable output root."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('paste2_config.yaml'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    rotations = pd.read_csv(args.rotation_dir / 'rotation_manifest.csv')
    expected = {(alteration, float(angle)) for alteration in config['alterations'] for angle in config['angles']}
    observed = set(zip(rotations['alteration'], rotations['angle_degrees'].astype(float)))
    if len(rotations) != len(expected) or observed != expected:
        raise RuntimeError('rotation manifest does not cover the configured PASTE2 grid')
    args.output_dir.mkdir(parents=True)
    worker = Path(__file__).parent / 'methods' / 'run_paste2.py'
    rows = []
    for index, rotation in enumerate(rotations.sort_values(['alteration', 'angle_degrees']).itertuples(index=False), start=1):
        output = args.output_dir / rotation.alteration / f'angle_{float(rotation.angle_degrees):g}'
        command = [str(args.python), str(worker), '--reference', str(args.reference), '--moving', str(rotation.output_path), '--output-dir', str(output), '--seed', str(config['seed']), '--configuration-id', f'study02-paste2-{rotation.alteration}-{float(rotation.angle_degrees):g}', '--overlap-fraction', repr(float(rotation.retained_fraction)), '--config', str(args.config)]
        print(f'[{index}/{len(rotations)}] {rotation.alteration} {float(rotation.angle_degrees):g}°', flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / f'angle_{float(rotation.angle_degrees):g}.runner.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f'PASTE2 failed for {rotation.alteration}/{rotation.angle_degrees:g}; see runner log')
        artifacts = [{'path': str(path.resolve())} for path in sorted(output.iterdir()) if path.is_file()]
        rows.append({'job_id': f'paste2__{rotation.alteration}__angle_{float(rotation.angle_degrees):g}', 'method': 'paste2', 'alteration': rotation.alteration, 'angle_degrees': float(rotation.angle_degrees), 'retained_fraction': float(rotation.retained_fraction), 'reference': str(args.reference.resolve()), 'moving': str(rotation.output_path), 'output_dir': str(output.resolve()), 'status': 'ok', 'return_code': completed.returncode, 'method_python': str(args.python.resolve()), 'artifacts': json.dumps(artifacts)})
        pd.DataFrame(rows).to_csv(args.output_dir / 'method_manifest.csv', index=False)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
