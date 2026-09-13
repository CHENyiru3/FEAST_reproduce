from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transfer_workflow', ROOT / 'workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_cohorts_and_approved_estimates():
    config = workflow.load_config()
    assert config['generation']['n_references'] is None
    assert config['_final_dir'].name == 'generative_transfer_github_a79cf22_estimated_ar_v1'
    for age, levels, ar in [('E15.5', 158, 0.30), ('E18.5', 202, 0.50)]:
        provenance = workflow.assignment_randomness_provenance(config, age)
        assert provenance['mode'] == 'approved_estimate'
        assert provenance['value'] == ar
        assert provenance['basis']
        assert 'calibration_file' not in config['ages'][age]['assignment_randomness']
        assert len(workflow.expected_z_values(config, age)) == levels


def test_invalid_estimate_rejected(tmp_path):
    import pytest
    import yaml
    raw = yaml.safe_load((ROOT / 'config.yaml').read_text())
    raw['ages']['E18.5']['assignment_randomness']['value'] = float('nan')
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match='finite'):
        workflow.load_config(path)
