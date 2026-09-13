"""Full-reference calibration with exact OT-field reuse and block progress logs."""
import argparse
from pathlib import Path
import time
from FEAST.de_novo import calibrate_local_references
import workflow
from run import load_references, simulation_config, write_json_atomic
from calibration_reuse import reuse_ot_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.STUDY_ROOT / 'config.yaml')
    parser.add_argument('--age', choices=workflow.AGE_ORDER, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New calibration result path; existing results are never overwritten.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = workflow.load_config(args.config)
    feast = workflow.feast_identity(config)
    started = time.perf_counter()
    references, genes, _ = load_references(config, args.age)
    references = [ref[:, genes].copy() for ref in references]
    print(f'[preparation] loaded {len(references)} references and {len(genes)} genes '
          f'in {time.perf_counter()-started:.3f}s', flush=True)
    with reuse_ot_fields(progress=True) as counters:
        result = calibrate_local_references(references, label_key=config['reference_fit']['label_key'],
            n_references=config['generation']['n_references'], config=simulation_config(config, 0),
            random_seed=int(config['public_seed']), cache_path=config['_final_dir'] / 'reference_parameters.sqlite',
            min_positions=config['generation']['min_positions'], batch_sd=config['generation']['batch_sd'])
    result.update(configuration_id=config['configuration_id'], feast=feast,
                  reference_inputs=workflow.reference_artifact_contract(config, args.age),
                  calibration_execution={'runner': Path(__file__).name,
                    'optimization': 'unrandomized OT fields reused only within the same holdout',
                    'holdout_subsampling': False, 'gene_subsampling_for_generation': False,
                    'counters': counters, 'elapsed_seconds': time.perf_counter()-started})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    write_json_atomic(args.output.with_name(args.output.stem+'.manifest.json'), result)

if __name__ == '__main__':
    main()
