"""Pinned, non-destructive Study 04 historical atomic-metric comparison.

The historical composite is retained only as a named noncanonical artifact.
It is deliberately never parsed into a numerical comparison because its
metric contract is invalid and differs from the prespecified atomic contract.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
KEY_COLUMNS = ('method', 'mode', 'alpha', 'support_variant', 'representation')
ALL_SPOTS = 'all_spots_primary'
PAIRED_SUPPORT = 'paired_zero_panel_spots_excluded_sensitivity'
REPRESENTATIONS = {'GraphST': ('pca20_randomized_seed42', 'pca10_randomized_seed42_sensitivity'), 'STAMP': ('native_10d_topics',), 'scVI': ('native_10d_latent',)}
NUMERICAL_CHANGE_CLASS = 'rng_api_repair_and_full_method_rerun'
CLAIM_DISPOSITION = 'author_review_evidence_no_active_manuscript_claim'

def alpha_text(value: object) -> str:
    return f'{float(value):.2f}'

def normalized_keys(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result['method'] = result['method'].astype(str)
    result['mode'] = result['mode'].astype(str)
    result['alpha'] = result['alpha'].map(alpha_text)
    result['support_variant'] = result['support_variant'].astype(str)
    result['representation'] = result['representation'].astype(str)
    return result

def expected_key_frame(*, modes: Iterable[str], alpha_levels: Iterable[object], common_support_cells: Iterable[tuple[str, float]]) -> pd.DataFrame:
    support_cells = {(str(mode), alpha_text(alpha)) for mode, alpha in common_support_cells}
    rows: list[dict[str, str]] = []
    for method, representations in REPRESENTATIONS.items():
        for mode in modes:
            for raw_alpha in alpha_levels:
                alpha = alpha_text(raw_alpha)
                if alpha == '0.00':
                    continue
                supports = [ALL_SPOTS]
                if (str(mode), alpha) in support_cells:
                    supports.append(PAIRED_SUPPORT)
                for support in supports:
                    for representation in representations:
                        rows.append({'method': method, 'mode': str(mode), 'alpha': alpha, 'support_variant': support, 'representation': representation})
    return pd.DataFrame(rows, columns=KEY_COLUMNS).sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

def assert_exact_key_matrix(table: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    missing_columns = set(KEY_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(f'{label} lacks key columns: {sorted(missing_columns)}')
    observed = normalized_keys(table)[list(KEY_COLUMNS)]
    if observed.duplicated().any():
        duplicate = observed.loc[observed.duplicated(keep=False)].head().to_dict('records')
        raise ValueError(f'{label} contains duplicate metric keys: {duplicate}')
    observed = observed.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if not observed.equals(expected):
        merged = expected.merge(observed, on=list(KEY_COLUMNS), how='outer', indicator=True)
        mismatch = merged.loc[merged['_merge'] != 'both'].head(20).to_dict('records')
        raise ValueError(f'{label} does not match the exact 60-row key matrix: {mismatch}')

def load_pinned_historical_atomic(historical_atomic_path: Path, expected_keys: pd.DataFrame) -> pd.DataFrame:
    if not historical_atomic_path.is_file():
        raise FileNotFoundError(f'pinned historical atomic table is missing: {historical_atomic_path}')
    historical = normalized_keys(pd.read_csv(historical_atomic_path))
    assert_exact_key_matrix(historical, expected_keys, 'historical atomic table')
    return historical

def _numeric_direction(old: float, new: float, tolerance: float=1e-12) -> str:
    if np.isnan(old) and np.isnan(new):
        return 'not_applicable'
    if not np.isfinite(old) or not np.isfinite(new):
        raise ValueError('old-versus-new comparison contains asymmetric or non-finite values')
    delta = new - old
    if abs(delta) <= tolerance:
        return 'unchanged_within_1e-12'
    return 'increased' if delta > 0 else 'decreased'

def build_old_vs_new(fresh: pd.DataFrame, *, historical_atomic_path: Path, expected_keys: pd.DataFrame, metric_columns: Iterable[str]) -> pd.DataFrame:
    metric_columns = tuple(metric_columns)
    fresh = normalized_keys(fresh)
    assert_exact_key_matrix(fresh, expected_keys, 'fresh atomic table')
    historical = load_pinned_historical_atomic(historical_atomic_path, expected_keys)
    columns = [*KEY_COLUMNS, *metric_columns]
    comparison = historical[columns].merge(fresh[columns], on=list(KEY_COLUMNS), how='outer', validate='one_to_one', indicator=True, suffixes=('_old', '_new'))
    if len(comparison) != 60 or not (comparison['_merge'] == 'both').all():
        raise ValueError('historical/fresh atomic tables are not one-to-one over 60 rows')
    comparison = comparison.drop(columns='_merge')
    for metric in metric_columns:
        old = pd.to_numeric(comparison[f'{metric}_old'], errors='coerce').to_numpy(float)
        new = pd.to_numeric(comparison[f'{metric}_new'], errors='coerce').to_numpy(float)
        if metric == 'paired_expression_correlation':
            expected_applicable = comparison['method'].eq('scVI').to_numpy()
            if not np.array_equal(np.isfinite(old), expected_applicable):
                raise ValueError('historical expression-correlation applicability changed')
            if not np.array_equal(np.isfinite(new), expected_applicable):
                raise ValueError('fresh expression-correlation applicability changed')
        elif not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError(f'non-finite comparable metric: {metric}')
        comparison[f'{metric}_delta_new_minus_old'] = new - old
        comparison[f'{metric}_numeric_direction'] = [_numeric_direction(old_value, new_value) for old_value, new_value in zip(old, new, strict=True)]
    comparison.insert(len(KEY_COLUMNS), 'numerical_change_class', NUMERICAL_CHANGE_CLASS)
    comparison.insert(len(KEY_COLUMNS) + 1, 'manuscript_claim_disposition', CLAIM_DISPOSITION)
    return comparison.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

def build_simulation_cell_comparison(fresh_manifest: pd.DataFrame, *, historical_atomic_path: Path, modes: Iterable[str], alpha_levels: Iterable[object], expected_keys: pd.DataFrame) -> pd.DataFrame:
    load_pinned_historical_atomic(historical_atomic_path, expected_keys)
    expected_cells = {(str(mode), alpha_text(alpha)) for mode in modes for alpha in alpha_levels}
    fresh = fresh_manifest.copy()
    fresh['mode'] = fresh['mode'].astype(str)
    fresh['alpha'] = fresh['alpha'].map(alpha_text)
    observed_cells = set(zip(fresh['mode'], fresh['alpha'], strict=True))
    if len(fresh) != 14 or fresh[['mode', 'alpha']].duplicated().any():
        raise ValueError('fresh simulation manifest is not the exact 14-cell ladder')
    if observed_cells != expected_cells:
        raise ValueError('fresh simulation manifest cells differ from the historical comparison contract')
    rows: list[dict[str, object]] = []
    for mode, alpha in sorted(expected_cells):
        fresh_row = fresh[(fresh['mode'] == mode) & (fresh['alpha'] == alpha)]
        if len(fresh_row) != 1:
            raise ValueError(f'fresh simulation cell is not unique: {mode}/{alpha}')
        rows.append({'mode': mode, 'alpha': alpha, 'status': 'present', 'numerical_change_class': NUMERICAL_CHANGE_CLASS, 'interpretation': 'cross_stage_rng_api_repair_and_full_rerun_not_repeat_drift_evidence', 'historical_atomic_source': str(historical_atomic_path.resolve()), 'manuscript_claim_disposition': CLAIM_DISPOSITION})
    result = pd.DataFrame(rows).sort_values(['mode', 'alpha']).reset_index(drop=True)
    if len(result) != 14:
        raise ValueError('simulation old-versus-new cell table must contain 14 rows')
    return result

def invalid_composite_disposition(invalid_composite_path: Path, replacement_path: Path) -> pd.DataFrame:
    if not invalid_composite_path.is_file():
        raise FileNotFoundError(f'invalid historical composite is missing: {invalid_composite_path}')
    return pd.DataFrame([{'prior_artifact': str(invalid_composite_path.resolve()), 'status': 'noncanonical_not_numerically_comparable', 'reason': 'invalid approximate kBET, constant connectivity, neutral missing-expression imputation, unapproved composite, pre-repair RNG/API inputs', 'replacement': str(replacement_path.resolve()), 'numeric_comparison_performed': False, 'numerical_change_class': NUMERICAL_CHANGE_CLASS, 'manuscript_claim_disposition': CLAIM_DISPOSITION}])

def pinned_input_records(historical_atomic_path: Path, invalid_composite_path: Path) -> list[dict[str, str]]:
    return [{'role': 'historical_comparable_atomic_metrics', 'path': str(historical_atomic_path.resolve())}, {'role': 'historical_invalid_composite_disposition_only', 'path': str(invalid_composite_path.resolve())}]
if __name__ == '__main__':
    raise SystemExit('compare_historical.py is a pinned comparison library; run score.py for Study 04.')
