"""Calibrate both ages with three whole-reference holdouts and the production simulator."""
import argparse
import json
from pathlib import Path
from FEAST.de_novo import calibrate_local_references
import workflow
from run import load_references, simulation_config, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.STUDY_ROOT / 'config.yaml')
    parser.add_argument('--age', choices=workflow.AGE_ORDER, required=True)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    feast = workflow.feast_identity(config)
    references, genes, _ = load_references(config, args.age)
    references = [ref[:, genes].copy() for ref in references]
    result = calibrate_local_references(references, label_key=config['reference_fit']['label_key'],
        n_references=config['generation']['n_references'], config=simulation_config(config, 0),
        random_seed=int(config['public_seed']), cache_path=config['_final_dir'] / 'reference_parameters.sqlite',
        min_positions=config['generation']['min_positions'], batch_sd=config['generation']['batch_sd'])
    result.update(configuration_id=config['configuration_id'], feast=feast,
                  reference_inputs=workflow.reference_artifact_contract(config, args.age))
    output = workflow.calibration_path(config, args.age)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    write_json_atomic(workflow.calibration_manifest_path(config, args.age), result)

if __name__ == '__main__':
    main()
