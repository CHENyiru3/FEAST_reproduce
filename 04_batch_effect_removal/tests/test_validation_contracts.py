from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT))
import score
import validate
import compare_historical as historical

def test_historical_input_records_use_explicit_resolved_paths(tmp_path: Path) -> None:
    atomic = tmp_path / 'atomic.csv'
    composite = tmp_path / 'composite.csv'

def test_validator_owned_metric_formulas_match_scorer_on_synthetic_data() -> None:
    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(80, 25))
    batch = np.array(['ref'] * 40 + ['query'] * 40)
    domain = np.tile(np.array(['1', '2', '3', '4']), 20)
    observed = score.metric_row(method='GraphST', representation='synthetic', primary=True, embedding=embedding, batch_labels=batch, domain_labels=domain, n_reference=40, expression_correlation=None, seed=42)
    independently_recomputed = validate.validator_metric_row(method='GraphST', representation='synthetic', primary=True, embedding=embedding, batch_labels=batch, domain_labels=domain, n_reference=40, expression_correlation=None, seed=42)
    pd.testing.assert_series_equal(pd.Series(observed), pd.Series(independently_recomputed), check_exact=False, rtol=1e-13, atol=1e-13)

def test_validator_rejects_named_zero_variance_expression_pair() -> None:
    matrix = np.arange(48, dtype=np.float64).reshape(8, 6)
    matrix[2] = 1.0
    matrix[6] = 2.0
    names = np.array(['spot-a', 'spot-b', 'spot-zero', 'spot-d'])
    with pytest.raises(ValueError, match='spot-zero'):
        validate.validator_paired_expression_correlation(matrix, 4, names)

def test_validator_owned_pca_and_expression_paths_match_scorer() -> None:
    rng = np.random.default_rng(2026)
    embedding = rng.normal(size=(80, 25))
    scorer_representations = score.representations('GraphST', embedding, 42)
    validator_representations = validate.validator_representations('GraphST', embedding, 42)
    for scorer_value, validator_value in zip(scorer_representations, validator_representations, strict=True):
        assert scorer_value[:2] == validator_value[:2]
        np.testing.assert_allclose(scorer_value[2], validator_value[2], rtol=1e-13, atol=1e-13)
    expression = rng.uniform(0.1, 10.0, size=(8, 12))
    names = np.array(['spot-a', 'spot-b', 'spot-c', 'spot-d'])
    assert score.paired_expression_correlation(expression, 4, names) == pytest.approx(validate.validator_paired_expression_correlation(expression, 4, names), rel=1e-13, abs=1e-13)

def test_stamp_retry_gate_fails_closed_without_explicit_binding() -> None:
    candidate = Path('unused')
    missing_retry = {'patience': 10}
    with pytest.raises(ValueError, match='lacks numerical_retry'):
        score.validate_stamp_retry(missing_retry, candidate)
    with pytest.raises(ValueError, match='lacks numerical_retry'):
        validate.validate_stamp_retry(missing_retry, candidate, 'shift_only', 1.5)
    improper_standard = {'patience': 20, 'numerical_retry': {'unexpected': True}}
    with pytest.raises(ValueError, match='carries a retry declaration'):
        score.validate_stamp_retry(improper_standard, candidate)
    with pytest.raises(ValueError, match='carries a retry declaration'):
        validate.validate_stamp_retry(improper_standard, candidate, 'shift_only', 1.5)

def test_live_retry_binding_rejects_a_second_parameter_change() -> None:
    candidate = STUDY_ROOT / 'outputs/final_rerun_20260718_v2/methods/STAMP/shift_only/alpha_1.50'
    if not (candidate / 'metadata.json').is_file():
        pytest.skip('bounded publication retry candidate is not present')
    metadata = json.loads((candidate / 'metadata.json').read_text(encoding='utf-8'))
    changed = copy.deepcopy(metadata)
    changed['learning_rate'] = 0.004
    with pytest.raises(ValueError, match='unchanged contract'):
        score.validate_stamp_retry(changed, candidate)
    with pytest.raises(ValueError, match='unchanged contract'):
        validate.validate_stamp_retry(changed, candidate, 'shift_only', 1.5)
