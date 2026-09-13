"""Structurally validate the v2 article evidence-contract candidate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / 'configs'
REPRO_ROOT = ROOT.parents[1]
WORKSPACE_ROOT = REPRO_ROOT.parent
EXP_ROOT = WORKSPACE_ROOT / 'FEAST_experiments'
PKG_ROOT = WORKSPACE_ROOT / 'FEAST'
REQUIRED = {'contract_version', 'workflow_id', 'supersedes', 'study', 'clean_rerun_study', 'status', 'method_name', 'article_scope', 'inputs', 'observed_target_covariates', 'engine', 'rng', 'solver', 'fallback_policy', 'outputs', 'evidence_binding', 'evidence', 'claim_limits', 'claim_authorizations', 'decision_required', 'release'}
METHODS = {'generative_parameter_assignment', 'reference_rank', 'conditional_empirical_rank', 'conditional_imputation', 'batch_parameter_deformation'}
STATUSES = {'validated_candidate', 'in_progress', 'validation_required', 'decision_required'}
EXPECTED_PREDECESSORS = {'00_simulator_benchmark': 'FEAST-ART-00-SIM-v1', '01_clustering': 'FEAST-ART-01-CLUSTER-v1', '02_alignment': 'FEAST-ART-02-ALIGN-v1', '03_deconvolution': 'FEAST-ART-03-DECONV-v1', '04_2d_conditional_transfer': 'FEAST-ART-04-2D-XSLICE-v1', '05_3d_stack': 'FEAST-ART-05-3D-IMPUTE-v1', '06_3d_transfer': 'FEAST-ART-06-DEVCCF-E15-v1', '08_batch_effect_removal': 'FEAST-ART-08-BATCH-v1'}
CONDITIONAL_STUDIES = {'04_2d_conditional_transfer', '05_3d_stack', '06_3d_transfer'}

def resolve(alias_path: str) -> Path:
    roots = {'REPRO': REPRO_ROOT, 'EXP': EXP_ROOT, 'PKG': PKG_ROOT}
    alias, separator, relative = alias_path.partition(':')
    if separator != ':' or alias not in roots or (not relative):
        raise ValueError('path must use REPRO:, EXP:, or PKG: alias')
    return roots[alias] / relative

def validate_file(path: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return ({}, [f'cannot parse JSON: {exc}'])
    missing = sorted(REQUIRED - data.keys())
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    if data.get('contract_version') != '2.0-candidate':
        errors.append('contract_version must be 2.0-candidate')
    if data.get('workflow_id') != path.stem:
        errors.append('workflow_id must equal the config filename stem')
    if data.get('method_name') not in METHODS:
        errors.append(f"unknown method_name: {data.get('method_name')!r}")
    if data.get('status') not in STATUSES:
        errors.append(f"unknown status: {data.get('status')!r}")
    if not isinstance(data.get('inputs'), list) or not data.get('inputs'):
        errors.append('inputs must be a nonempty list')
    if not isinstance(data.get('evidence'), list) or len(data.get('evidence', [])) < 2:
        errors.append('evidence must contain at least predecessor and current-state records')
    if not isinstance(data.get('decision_required'), list):
        errors.append('decision_required must be a list')
    study = data.get('study')
    predecessor = data.get('supersedes', {})
    if predecessor.get('workflow_id') != EXPECTED_PREDECESSORS.get(study):
        errors.append('supersedes.workflow_id does not match the legacy study mapping')
    release = data.get('release', {})
    if release != {'authorized': False, 'tag': None, 'pypi_publication': False}:
        errors.append('release must remain entirely unauthorized')
    authorizations = data.get('claim_authorizations', {})
    expected_authorizations = {'publication_claim': False, 'composite_score': False, 'method_ranking': False, 'winner': False}
    if authorizations != expected_authorizations:
        errors.append('all claim authorizations must remain false in this candidate')
    binding = data.get('evidence_binding', {})
    artifacts = data.get('outputs', {}).get('artifacts', [])
    status = data.get('status')
    if status == 'validated_candidate':
        if binding.get('state') != 'validated' or not artifacts:
            errors.append('validated_candidate requires validated binding and bound artifacts')
    if status == 'in_progress':
        if binding.get('state') != 'unbound_active' or artifacts:
            errors.append('in_progress must remain unbound while active work continues')
    if status == 'validation_required' and binding.get('state') != 'noncanonical_frozen':
        errors.append('validation_required must identify frozen evidence as noncanonical')
    if study in CONDITIONAL_STUDIES:
        disposition = data.get('conditional_ot_disposition', {})
        expected_counts = {'active_outputs': 156, 'explicit_failure_saved': 31, 'failure_tainted': 26, 'missing_positive_convergence_evidence': 99, 'verifiably_converged': 0, 'canonical_outputs': 0}
        if status != 'validation_required':
            errors.append('legacy conditional workflows must remain validation_required')
        if any((disposition.get(key) != value for key, value in expected_counts.items())):
            errors.append('conditional-OT aggregate counts do not match the frozen audit')
        if disposition.get('publication_status') != 'noncanonical':
            errors.append('conditional OT must remain noncanonical')
        if disposition.get('transport_nonconvergence') != 'raise':
            errors.append('future conditional reruns must fail closed on nonconvergence')
    references = (('supersedes', [predecessor]), ('evidence', data.get('evidence', [])), ('outputs.artifacts', artifacts))
    for collection_name, entries in references:
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append(f'each {collection_name} entry must be an object')
                continue
            alias_path = entry.get('path')
            if not isinstance(alias_path, str):
                errors.append(f'{collection_name} entries require a path string')
                continue
            try:
                evidence_path = resolve(alias_path)
            except ValueError as exc:
                errors.append(f'{alias_path!r}: {exc}')
                continue
            if not evidence_path.is_file():
                errors.append(f'missing bound file: {alias_path}')
    return (data, errors)

def main() -> int:
    paths = sorted(CONFIG_DIR.glob('*.json'))
    errors: list[str] = []
    if len(paths) != 8:
        errors.append(f'expected 8 workflow configs, found {len(paths)}')
    ids: set[str] = set()
    studies: set[str] = set()
    for path in paths:
        data, file_errors = validate_file(path)
        workflow_id = data.get('workflow_id', path.stem)
        study = data.get('study')
        if workflow_id in ids:
            file_errors.append(f'duplicate workflow_id: {workflow_id}')
        if study in studies:
            file_errors.append(f'duplicate study: {study}')
        ids.add(workflow_id)
        if study:
            studies.add(study)
        errors.extend((f'{path.name}: {message}' for message in file_errors))
    if studies != set(EXPECTED_PREDECESSORS):
        errors.append('the exact eight legacy studies are required')
    if errors:
        print('Contract validation failed:')
        for error in errors:
            print(f'- {error}')
        return 1
    print(f'Validated {len(paths)} v2 candidate workflow contracts and evidence paths.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
