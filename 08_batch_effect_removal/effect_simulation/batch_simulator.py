"""Batch effect controllable simulation using FEAST core API.

Produces a serial batch-effect ladder: one reference slice -> many
alpha-strength simulations with identical spatial quantile field Q
and progressively stronger marginal distribution distortion P.

Deformation modes are estimated from real paired slices via
FEAST.characterize_batch() (OLS in theta space).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from FEAST.FEAST_core.parameter_cloud import BatchDeformation

# Backward-compat alias
DeformationConfig = BatchDeformation

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def build_deformation_modes(
    adata_ref,
    adata_query,
    modes: List[str] = ("shift_only", "diagonal_affine"),
) -> dict:
    """Build deformation configs from real paired slices via characterize_batch.

    Parameters
    ----------
    adata_ref : AnnData   -- clean reference slice
    adata_query : AnnData -- batch-affected slice
    modes : list[str]     -- which modes to build

    Returns
    -------
    dict[str, DeformationConfig]
        'diagonal_affine' -> full (D, b) from OLS
        'shift_only'      -> (D=[1,1,1], b) from OLS (centroid shift only)
    """
    from FEAST import characterize_batch

    full = characterize_batch(adata_ref, adata_query, name="characterized")

    result = {}
    if "diagonal_affine" in modes:
        result["diagonal_affine"] = BatchDeformation(
            D=full.D.copy(), b=full.b.copy(), name="diagonal_affine",
        )
    if "shift_only" in modes:
        result["shift_only"] = BatchDeformation(
            D=np.array([1.0, 1.0, 1.0]),
            b=full.b.copy(),
            name="shift_only",
        )

    return result


def simulate_batch_ladder(
    adata_ref,
    deform: DeformationConfig,
    alphas: List[float] = None,
    *,
    random_seed: int = 42,
) -> Dict[float, "ad.AnnData"]:
    """Generate a batch-effect ladder from a reference slice.

    For each alpha, applies theta deformation and decodes counts
    preserving spatial quantile field Q.

    Parameters
    ----------
    adata_ref : AnnData  -- clean reference slice
    deform : DeformationConfig
    alphas : list[float] -- default: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    random_seed : int

    Returns
    -------
    dict[float, AnnData]  -- alpha -> batch-affected slice
    """
    from FEAST.FEAST_core.simulator import simulate_batch_effect

    if alphas is None:
        alphas = DEFAULT_ALPHAS

    results = {}
    for alpha in alphas:
        sim = simulate_batch_effect(
            adata_ref,
            D=deform.D,
            b=deform.b,
            alpha=alpha,
            random_seed=random_seed,
        )
        sim.uns["batch_alpha"] = alpha
        sim.uns["batch_config"] = deform.to_dict()
        results[alpha] = sim

    return results
