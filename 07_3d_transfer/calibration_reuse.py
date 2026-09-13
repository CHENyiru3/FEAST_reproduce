"""Reuse deterministic OT fields within one unchanged calibration holdout.

This scoped adapter leaves the installed FEAST wheel unchanged. The existing
solver computes each unrandomized field once; the existing AR formula is then
applied with the original seed. It is used only in the single-process Study 07
calibration runner, and restores all original functions on exit.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
import time

import numpy as np
from FEAST.de_novo import conditional, local, transport
from FEAST.de_novo.quantile_field import quantiles_to_normal_scores


@contextmanager
def reuse_ot_fields(*, progress=True):
    original_generate = local.simulate_local_references
    original_transport = conditional.transport_reference_field
    original_solve = transport._solve_transport_plan
    cache = {}
    target = None
    settings = None
    counters = {'field_solves': 0, 'field_reuses': 0, 'block_solves': 0,
                'ot_solve_seconds': 0.0}

    def generate(reference_slices, target_blueprint, **kwargs):
        nonlocal target, settings
        config = kwargs.get('config') or local.SimulationConfig()
        current_settings = (asdict(replace(config, assignment_randomness=0.0)),
                            kwargs.get('n_references'), kwargs.get('min_positions', 50))
        # calibrate_local_references constructs one blueprint per holdout and
        # reuses its fitted reference objects across AR candidates.
        if target_blueprint is not target or current_settings != settings:
            cache.clear()
            target = target_blueprint
            settings = current_settings
        if progress:
            print(f'[calibration] AR={config.assignment_randomness:.2f} '
                  f'positions={target_blueprint.n_spots} start', flush=True)
        started = time.perf_counter()
        result = original_generate(reference_slices, target_blueprint, **kwargs)
        if progress:
            print(f'[calibration] AR={config.assignment_randomness:.2f} '
                  f'generation_seconds={time.perf_counter()-started:.3f}', flush=True)
        return result

    def solve(*args, **kwargs):
        counters['block_solves'] += 1
        number = counters['block_solves']
        if progress:
            print(f'[OT block {number}] start source={len(args[0])} target={len(args[1])}', flush=True)
        started = time.perf_counter()
        result = original_solve(*args, **kwargs)
        elapsed = time.perf_counter() - started
        counters['ot_solve_seconds'] += elapsed
        if progress:
            print(f'[OT block {number}] seconds={elapsed:.3f} '
                  f'iterations={result[1].get("iterations")} '
                  f'error={result[1].get("final_error")}', flush=True)
        return result

    def field(*, source_coordinates, target_coordinates, source_quantiles,
              source_boundary=None, target_boundary=None, config=None, random_seed=0):
        cfg = config or transport.TransportConfig()
        key = id(source_quantiles)
        entry = cache.get(key)
        geometry = (source_coordinates, target_coordinates, source_boundary, target_boundary)
        if entry is not None and not all(np.array_equal(left, right)
                                        for left, right in zip(entry['geometry'], geometry)):
            raise ValueError('calibration geometry changed inside an OT field cache scope')
        if entry is None:
            base = original_transport(source_coordinates=source_coordinates,
                target_coordinates=target_coordinates, source_quantiles=source_quantiles,
                source_boundary=source_boundary, target_boundary=target_boundary,
                config=replace(cfg, assignment_randomness=0.0), random_seed=random_seed)
            entry = {'base': base,
                     # Retaining this immutable fitted array also prevents ID reuse.
                     'source_quantiles': source_quantiles,
                     'source_scores': quantiles_to_normal_scores(source_quantiles,
                         clip_eps=float(cfg.latent_clip_eps)).astype(np.float32, copy=False),
                     'geometry': tuple(None if value is None else np.array(value, copy=True) for value in geometry)}
            cache[key] = entry
            counters['field_solves'] += 1
        else:
            counters['field_reuses'] += 1
        randomness = float(cfg.assignment_randomness)
        if randomness > 0:
            rng = np.random.default_rng(random_seed)
            source_scores = entry['source_scores']
            sampled_indices = rng.integers(0, source_scores.shape[0], size=len(target_coordinates))
            values = ((1.0-randomness)*entry['base'].latent_scores
                      + randomness*source_scores[sampled_indices, :]).astype(np.float32, copy=False)
        else:
            values = entry['base'].latent_scores.copy()
        diagnostics = dict(entry['base'].diagnostics, assignment_randomness=randomness)
        return transport.TransportResult(latent_scores=values, diagnostics=diagnostics)

    local.simulate_local_references = generate
    conditional.transport_reference_field = field
    transport._solve_transport_plan = solve
    try:
        yield counters
    finally:
        local.simulate_local_references = original_generate
        conditional.transport_reference_field = original_transport
        transport._solve_transport_plan = original_solve
        cache.clear()
