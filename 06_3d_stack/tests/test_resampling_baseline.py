from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conditional_resampling_baseline import prepare_sampling, resample_spots
from evaluate_resampling_metrics import conditional_distances, log_normalize, prepare_conditionals


def test_five_reference_sampling_preserves_classes_vectors_and_weights(tmp_path):
    inputs = {
        4: ([[1, 10], [2, 20], [3, 30]], ['a', 'a', 'b']),
        7: ([[5, 50]], ['a']),
        10: ([[9, 90]], ['x']), 13: ([[9, 90]], ['x']), 16: ([[9, 90]], ['x']),
        19: ([[8, 80], [99, 990]], ['c', 'a']),
    }
    for number, (values, labels) in inputs.items():
        data = ad.AnnData(np.array(values), obs=pd.DataFrame({'class': labels},
                         index=[str(i) for i in range(len(labels))]),
                         var=pd.DataFrame(index=['g1', 'g2']))
        data.layers['counts'] = data.X.copy()
        data.write_h5ad(tmp_path / f'ref.{number:03d}.h5ad')
    row = {'target_slice': 5, 'primary_reference_slices': [4, 7, 10, 13, 16],
           'reference_weights': {f'ref.{i:03d}': w for i, w in zip([4, 7, 10, 13, 16], [.1, .3, .2, .2, .2])},
           'donor_support': [{'label': 'c', 'donor_slice': 19}]}
    labels = np.array(['a'] * 5000 + ['b'] * 4 + ['c'] * 3)
    groups, support = prepare_sampling(row, {'reference_pool': list(inputs)}, tmp_path,
        'ref', 'class', 'counts', labels, ['g1', 'g2'])
    sampled = resample_spots(groups, (len(labels), 2), np.random.default_rng(2026))
    repeated = resample_spots(groups, (len(labels), 2), np.random.default_rng(2026))
    np.testing.assert_array_equal(sampled, repeated)
    assert set(map(tuple, sampled[:5000])) <= {(1, 10), (2, 20), (5, 50)}
    assert abs(np.mean(sampled[:5000, 0] == 5) - .75) < .025
    np.testing.assert_array_equal(sampled[5000:5004], np.tile([3, 30], (4, 1)))
    np.testing.assert_array_equal(sampled[5004:], np.tile([8, 80], (3, 1)))
    assert support[-1]['uses_supporting_donor']


def test_cached_conditional_distance_matches_independent_wasserstein():
    observed = np.array([[0., 0., 0.], [1., 3., 0.], [2., 0., 5.], [0., 4., 1.]])
    generated = np.array([[1., 0., 0.], [0., 2., 1.], [4., 0., 2.], [1., 1., 0.]])
    labels = np.array(['a', 'a', 'b', 'b'])
    x, library_x, detected_x = log_normalize(observed)
    y, library_y, detected_y = log_normalize(generated)
    gene_distances = np.zeros(3)
    zero_distance = 0.
    for label in ('a', 'b'):
        mask = labels == label
        gene_distances += .5 * np.array([wasserstein_distance(x[mask, g], y[mask, g]) for g in range(3)])
        zero_distance += .5 * wasserstein_distance((observed[mask] == 0).mean(axis=0),
                                                   (generated[mask] == 0).mean(axis=0))
    expected = [np.median(gene_distances), zero_distance,
                wasserstein_distance(library_x, library_y), wasserstein_distance(detected_x, detected_y)]
    actual = conditional_distances(observed, generated, labels, prepare_conditionals(observed, labels))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
