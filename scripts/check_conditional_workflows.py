"""Verify the active configurations for Studies 05--07."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml
ROOT = Path(__file__).resolve().parents[1]
COMMIT = '68816e5c1862a6fa2a49bc30609d617c7fa4b449'
BUILD_BY_STUDY = {'Study 05': {'version': '1.0.2', 'commit': COMMIT}, 'Study 06': {'version': '1.0.5', 'commit': '449155cb9ba3f481560ae611e18cbec567111538'}, 'Study 07': {'version': '1.1.0', 'commit': COMMIT}}
SHARED_TRANSPORT = {'epsilon': 0.05, 'sinkhorn_iter': 1000, 'sinkhorn_tol': 1e-05, 'unbalanced_transport': True, 'reg_m': 5.0, 'transport_nonconvergence': 'raise', 'geometry_weight': 1.0, 'boundary_weight': 0.25, 'max_transport_pairs': 25000000}

def read_yaml(relative: str) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / relative).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{relative} must contain a mapping')
    return payload

def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)

def check_identity(name: str, config: dict[str, Any], errors: list[str]) -> None:
    build = BUILD_BY_STUDY[name]
    version = config.get('required_feast_version', config.get('feast_version'))
    require(str(version) == build['version'], f'{name}: FEAST version changed', errors)
    require(str(config.get('required_feast_commit')) == build['commit'], f'{name}: FEAST commit changed', errors)
    require(int(config.get('public_seed', -1)) == 2026, f'{name}: public seed changed', errors)
    for key, expected in SHARED_TRANSPORT.items():
        require(config.get('transport', {}).get(key) == expected, f'{name}: transport.{key} changed', errors)

def main() -> int:
    errors: list[str] = []
    study05 = read_yaml('05_2d_conditional_transfer/config.yaml')
    study06 = read_yaml('06_3d_stack/config_reference_density.yaml')
    study07 = read_yaml('07_3d_transfer/config.yaml')
    for name, config in (('Study 05', study05), ('Study 06', study06), ('Study 07', study07)):
        check_identity(name, config, errors)
    study05_datasets = study05.get('datasets', {})
    require(set(study05_datasets) == {'dlpfc', 'merfish'}, 'Study 05: dataset scope changed', errors)
    require(len(study05_datasets.get('dlpfc', {}).get('directions', [])) == 6 and len(study05_datasets.get('merfish', {}).get('directions', [])) == 2, 'Study 05: direction count changed', errors)
    require(list(map(float, study05.get('assignment_randomness', []))) == [0.0, 0.1, 0.2, 0.3, 0.5], 'Study 05: assignment-randomness grid changed', errors)
    require(int(study05_datasets.get('dlpfc', {}).get('gene_panel', {}).get('expected_genes', -1)) == 17391 and int(study05_datasets.get('merfish', {}).get('gene_panel', {}).get('expected_genes', -1)) == 1122, 'Study 05: dataset gene-panel sizes changed', errors)
    require(int(study05.get('expected_jobs', -1)) == 45 and int(study05.get('expected_cross_slice_jobs', -1)) == 40 and (int(study05.get('expected_mask_half_jobs', -1)) == 5), 'Study 05: expected job counts changed', errors)
    mask_half = study05.get('mask_half', {})
    require(int(mask_half.get('split_axis', -1)) == 0 and float(mask_half.get('split_quantile', -1)) == 0.5 and (mask_half.get('direction') == 'low_x_to_high_x') and (float(mask_half.get('assignment_randomness', -1)) == 0.3), 'Study 05: half-slice contract changed', errors)
    require(int(study05.get('label_support', {}).get('min_source_spots', -1)) == 20, 'Study 05: source label-support threshold changed', errors)
    require(int(study05.get('reference_fit', {}).get('min_gene_spots', -1)) == 0, 'Study 05: zero-evidence gene retention changed', errors)
    density_contract = {int(item['gap']): (int(item['expected_references']), int(item['expected_targets']), tuple(item.get('extra_anchor_slices', []))) for item in study06.get('densities', [])}
    require(density_contract == {3: (49, 95, ()), 5: (30, 114, ()), 10: (17, 127, (89,))}, 'Study 06: corrected real/target partition changed', errors)
    validation = study06.get('validation', {})
    require(validation.get('expected_outputs') == 336 and validation.get('expected_positions_per_stack') == 144, 'Study 06: expected output/stack size changed', errors)
    dataset = study06.get('dataset', {})
    require((dataset.get('first_slice'), dataset.get('last_slice'), dataset.get('expected_files'), dataset.get('expected_genes')) == (4, 150, 144, 1122), 'Study 06: corrected input scope changed', errors)
    require(dataset.get('excluded_conflicting_slices') == [1, 2, 3] and dataset.get('intentionally_absent_slices') == [75, 94, 112], 'Study 06: excluded/absent slices changed', errors)
    require(study06.get('anchor_selection') == {'unit': 'rank_among_available_slices', 'phase': 'first_available_slice', 'boundary_slices': [4, 150]}, 'Study 06: rank-based anchor selection changed', errors)
    calibration = study06.get('assignment_randomness_preflight', {})
    require(calibration == {'estimator': 'FEAST.estimate_assignment_randomness', 'representative_reference_slices': 3, 'n_genes': 20, 'step': 0.05, 'maximum': 0.5, 'spatial_neighbors': 6, 'seed_formula': 'public_seed + gap * 1000 + representative_index', 'transport_backend': 'torch', 'transport_device': 'cpu', 'transport_dtype': 'float64'}, 'Study 06: reference-only calibration procedure changed', errors)
    require(all('expected_assignment_randomness' not in item for item in study06.get('densities', [])), 'Study 06: superseded calibration values must not be imposed', errors)
    for key, expected in {'sinkhorn_method': 'sinkhorn_log', 'transport_backend': 'torch', 'transport_device': 'cuda:0', 'transport_dtype': 'float64'}.items():
        require(study06.get('transport', {}).get(key) == expected, f'Study 06: transport.{key} changed', errors)
    historical = study06.get('historical_comparison', {})
    require(historical.get('status') == 'superseded_different_scientific_question' and historical.get('comparable_to_corrected_design') is False, 'Study 06: superseded design must remain noncomparable', errors)
    generation = study06.get('generation', {})
    require(generation.get('smoothing') is False, 'Study 06: smoothing was enabled', errors)
    require(generation.get('z_regularization') is False, 'Study 06: z regularization was enabled', errors)
    expected_ages = {'E15.5': (-4.12, -0.98, 0.02, 158, 2330927), 'E18.5': (-5.56, -1.54, 0.02, 202, 5213461)}
    observed_ages = {age: (float(item['z_start']), float(item['z_end']), float(item['z_step']), int(item['expected_levels']), int(item['expected_spots'])) for age, item in study07.get('ages', {}).items()}
    require(observed_ages == expected_ages, 'Study 07: full-axis scope changed', errors)
    require(study07.get('ages', {}).get('E18.5', {}).get('assignment_randomness', {}).get('mode') == 'reference_calibration', 'Study 07: E18.5 is not reference-only calibrated', errors)
    study07_fit = study07.get('reference_fit', {})
    require(int(study07_fit.get('expected_genes', -1)) == 550 and int(study07_fit.get('min_gene_spots', -1)) == 1 and (float(study07_fit.get('min_gene_mean', -1)) == 0.0) and (float(study07_fit.get('max_gene_zero_prop', -1)) == 1.0), 'Study 07: historical E15.5 550-gene reference-fit contract changed', errors)
    publication = json.loads((ROOT / 'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    studies = publication.get('studies', {})
    for study in ('05_2d_conditional_transfer', '06_3d_stack', '07_3d_transfer'):
        record = studies.get(study, {})
        require(bool(record), f'publication manifest lacks {study}', errors)
        require(record.get('historical_conditional_outputs_consumed') is False, f'{study}: historical conditional outputs were marked as consumed', errors)
    if errors:
        print('conditional-workflow contract: FAILED')
        for error in errors:
            print(f'- {error}')
        return 1
    print('conditional-workflow contract: OK (45 + 336 + 360 fresh H5ADs declared; Study 06: 144 positions per stack; legacy 93-output design excluded)')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
