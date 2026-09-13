import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow import SliceInfo, build_assignments, density_specs, load_config
from preflight import plan_density


def test_density_selection_and_explicit_donor():
    config = load_config()
    available = [i for i in range(4, 151) if i not in [75, 94, 112]]
    info = {i: SliceInfo(i, Path(str(i)), i * .02, 100, 1122) for i in available}
    labels = {i: {'common'} for i in available}
    labels[5].add('rare')
    labels[89].add('rare')
    counts = {i: {label: 60 for label in labels[i]} for i in available}
    for spec in density_specs(config):
        assignments, pool = build_assignments(info, spec, 2026)
        # For this synthetic input supply the rare label on a retained reference.
        donor = min(pool, key=lambda i: abs(i-89))
        labels[donor].add('rare'); counts[donor]['rare'] = 60
        rows = plan_density(spec, assignments, pool, info, labels, counts, 'ref', 5)
        assert len(rows) == spec.expected_targets and len(pool) == spec.expected_references
        for row in rows:
            assert len(row['primary_reference_slices']) == 5
            assert row['lower_ref_slice'] in row['primary_reference_slices']
            assert row['upper_ref_slice'] in row['primary_reference_slices']
            assert np.isclose(sum(row['reference_weights'].values()), 1)
        assert rows[0]['donor_support'][0]['donor_slice'] == donor


def test_inventory_reads_only_metadata(monkeypatch, tmp_path):
    import anndata as ad
    import pandas as pd
    import preflight
    data = ad.AnnData(np.ones((3, 1122)), obs=pd.DataFrame({'class': ['a']*3, 'z': [1.]*3}, index=['x','y','z']))
    data.layers['counts'] = data.X.copy()
    data.obsm['spatial'] = np.array([[0.,0.],[1.,0.],[0.,1.]])
    data.obsm['spatial_3d'] = np.column_stack([data.obsm['spatial'], np.ones(3)])
    data.write_h5ad(tmp_path / 'Zhuang-ABCA-1.004.h5ad')
    config = load_config()
    config['dataset'].update(first_slice=4, last_slice=4, expected_files=1)
    original = preflight.read_elem
    reads = []
    def metadata_only(element):
        reads.append(element.name)
        assert element.name in ['/obs','/var','/obsm/spatial','/obsm/spatial_3d']
        return original(element)
    monkeypatch.setattr(preflight, 'read_elem', metadata_only)
    frame, *_ = preflight.inspect_inputs(tmp_path, config)
    assert len(frame) == 1 and len(reads) == 4
