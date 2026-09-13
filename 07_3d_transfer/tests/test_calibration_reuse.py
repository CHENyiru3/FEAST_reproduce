"""The calibration speedup must preserve the existing transported fields."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_reuse import reuse_ot_fields
from FEAST.de_novo import local, conditional, transport


@pytest.fixture
def field_case(monkeypatch):
    rng = np.random.default_rng(12)
    source = rng.normal(size=(24, 2)).astype(np.float32)
    quantiles = rng.uniform(.01, .99, (24, 20)).astype(np.float32)
    target = SimpleNamespace(coordinates=rng.normal(size=(18, 2)).astype(np.float32), n_spots=18)
    config = local.SimulationConfig(sinkhorn_method='sinkhorn_log')
    def generate(references, blueprint, *, config, random_seed=0):
        return conditional.transport_reference_field(source_coordinates=source,
            target_coordinates=blueprint.coordinates, source_quantiles=quantiles,
            config=config, random_seed=random_seed)
    monkeypatch.setattr(local, 'simulate_local_references', generate)
    return target, config, generate


def test_reuse_all_ar_values_and_seeds_exactly(field_case):
    target, config, generate = field_case
    values = np.round(np.arange(0, .50001, .05), 2)
    expected = [generate([], target, config=replace(config, assignment_randomness=ar), random_seed=20+i)
                for i, ar in enumerate(values)]
    with reuse_ot_fields(progress=False) as counters:
        actual = [local.simulate_local_references([], target,
                  config=replace(config, assignment_randomness=ar), random_seed=20+i)
                  for i, ar in enumerate(values)]
        assert counters['field_solves'] == 1
        assert counters['block_solves'] == 1
        assert counters['field_reuses'] == 10
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left.latent_scores, right.latent_scores)
        assert left.diagnostics == right.diagnostics
    assert local.simulate_local_references is generate


def test_reuse_scope_resets_between_holdouts_and_settings(field_case):
    target, config, generate = field_case
    other = SimpleNamespace(coordinates=target.coordinates[::-1].copy(), n_spots=18)
    expected = generate([], other, config=replace(config, geometry_weight=.75))
    with reuse_ot_fields(progress=False) as counters:
        local.simulate_local_references([], target, config=config)
        local.simulate_local_references([], other, config=config)
        actual = local.simulate_local_references([], other, config=replace(config, geometry_weight=.75))
        assert counters['field_solves'] == counters['block_solves'] == 3
    np.testing.assert_array_equal(actual.latent_scores, expected.latent_scores)


def test_mutated_holdout_is_rejected_and_functions_are_restored(field_case):
    target, config, generate = field_case
    original_transport = conditional.transport_reference_field
    original_solve = transport._solve_transport_plan
    with pytest.raises(ValueError, match='geometry changed'):
        with reuse_ot_fields(progress=False):
            local.simulate_local_references([], target, config=config)
            target.coordinates[0, 0] += 1
            local.simulate_local_references([], target, config=config)
    assert local.simulate_local_references is generate
    assert conditional.transport_reference_field is original_transport
    assert transport._solve_transport_plan is original_solve
