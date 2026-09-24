"""Broad-class marker and GO-BP recovery for matched Study 06 held-out cells.

Reuse the established Study 05 Wilcoxon/GSEA methodology. No donor-level
inference is made from cells or serial slices of this single volume.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from importlib.metadata import version

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/study06-go-numba')
import anndata as ad
from anndata.experimental import read_elem
import gseapy as gp
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DEFAULT_ROOT = STUDY / 'outputs/generative_five_reference_1.0.6_precision_v1'
DEFAULT_GMT = STUDY.parent / '05_2d_conditional_transfer/pathway_analysis/gene_sets/m5.go.bp.v2025.1.Mm.symbols.gmt'
PARAMETERS = dict(normalization_total=10000, transformation='log1p',
    ranking='Scanpy Wilcoxon signed z: original broad class versus rest',
    tie_correct=True, use_raw=False, gene_universe='all 1122 measured genes',
    rank_ties='score descending then gene symbol ascending; no jitter',
    min_size=15, max_size=500, weight=1.0, permutations=1000,
    gsea_method='multilevel', adaptive_sample_size=101, adaptive_eps=1e-50,
    seed=2026, gsea_threads=1, descriptive_fdr_cutoff=.05,
    tie_correction_implementation='cache the unchanged Scanpy correction once per immutable gene block',
    inference='descriptive conditional biological-signature preservation; no donor-level DEG inference')


def read_matrix(path):
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs'])
        genes = list(map(str, read_elem(f['var']).index))
        matrix = sparse.csr_matrix(read_elem(f['layers/counts']))
    return obs, genes, matrix


def pooled_counts(root, plan, density, method):
    blocks, labels, expected_genes = [], [], None
    name = next(k for k, d in plan['densities'].items() if d is density)
    for index, row in enumerate(density['targets'], 1):
        target_name = f"Zhuang-ABCA-1.{int(row['target_slice']):03d}"
        observed_path = Path(plan['data_dir']) / f'{target_name}.h5ad'
        if method == 'observed':
            obs, genes, matrix = read_matrix(observed_path)
        else:
            with h5py.File(observed_path, 'r') as f:
                truth_obs = read_elem(f['obs'])
            obs, genes, matrix = read_matrix(root / 'targets' / name / target_name / 'generated.h5ad')
            if not obs.index.equals(truth_obs.index) or not np.array_equal(
                    obs['class'].astype(str).to_numpy(), truth_obs['class'].astype(str).to_numpy()):
                raise ValueError(f'{target_name}: target identities or original classes differ')
        if expected_genes is None: expected_genes = genes
        elif expected_genes != genes: raise ValueError('gene order changed between slices')
        blocks.append(matrix)
        labels.append(obs['class'].astype(str).to_numpy())
        if index % 24 == 0:
            print(f'Gap {density["spec"]["gap"]} {method}: loaded {index}/{len(density["targets"])} slices.', flush=True)
    counts = sparse.vstack(blocks, format='csr')
    return counts, np.concatenate(labels), expected_genes


def rank_classes(counts, labels, genes):
    data = ad.AnnData(X=counts.astype(np.float64),
        obs=pd.DataFrame({'class': pd.Categorical(labels)}, index=pd.Index(np.arange(len(labels)).astype(str))),
        var=pd.DataFrame(index=pd.Index(genes, name='gene')))
    sizes = data.obs['class'].value_counts()
    if len(sizes) < 2 or sizes.min() < 2:
        raise ValueError(f'class-versus-rest requires at least two cells per class: {sizes.to_dict()}')
    sc.pp.normalize_total(data, target_sum=PARAMETERS['normalization_total'])
    sc.pp.log1p(data)
    # Scanpy 1.11 repeats the identical correction for every class in each
    # immutable rank block. Cache that result locally; do not alter the library.
    from scanpy.tools import _rank_genes_groups as ranking_impl
    original_tiecorrect = ranking_impl._tiecorrect
    last_ranks = last_correction = None
    processed_genes = 0

    def cached_tiecorrect(ranks):
        nonlocal last_ranks, last_correction, processed_genes
        if ranks is not last_ranks:
            last_ranks, last_correction = ranks, original_tiecorrect(ranks)
            processed_genes += ranks.shape[1]
            if processed_genes % 200 < ranks.shape[1] or processed_genes == len(genes):
                print(f'  Wilcoxon genes: {processed_genes}/{len(genes)}', flush=True)
        return last_correction

    try:
        ranking_impl._tiecorrect = cached_tiecorrect
        sc.tl.rank_genes_groups(data, groupby='class', reference='rest', method='wilcoxon',
            tie_correct=True, use_raw=False, n_genes=len(genes), rankby_abs=False, key_added='class_ranking')
    finally:
        ranking_impl._tiecorrect = original_tiecorrect
    scores, folds = {}, {}
    for label in sorted(set(labels)):
        frame = sc.get.rank_genes_groups_df(data, group=label, key='class_ranking').set_index('names').reindex(genes)
        if not np.isfinite(frame['scores'].to_numpy()).all():
            raise ValueError(f'non-finite ranking for {label}')
        scores[label] = frame['scores'].to_numpy(float)
        folds[label] = frame['logfoldchanges'].to_numpy(float)
    return (pd.DataFrame(scores, index=pd.Index(genes, name='gene')),
            pd.DataFrame(folds, index=pd.Index(genes, name='gene')), sizes.to_dict())


def load_sets(path, genes):
    universe = set(genes); gene_sets, coverage = {}, []
    for line in path.open():
        name, description, *members = line.rstrip('\n').split('\t')
        original = set(members); matched = sorted(universe & original)
        eligible = PARAMETERS['min_size'] <= len(matched) <= PARAMETERS['max_size']
        coverage.append(dict(term=name, source=description, original_genes=len(original),
            measured_genes=len(matched), coverage_fraction=len(matched)/len(original), eligible=eligible))
        if eligible:gene_sets[name] = matched
    return gene_sets, pd.DataFrame(coverage)


def enrich(scores, gene_sets):
    ordered = scores.rename('score').rename_axis('gene').reset_index().sort_values(
        ['score', 'gene'], ascending=[False, True], kind='stable')
    result = gp.prerank(rnk=ordered, gene_sets=gene_sets, organism='mouse', outdir=None,
        ascending=None, min_size=15, max_size=500, permutation_num=1000, weight=1.,
        threads=1, seed=2026, no_plot=True, verbose=False,
        method='multilevel', sample_size=101, eps=1e-50)
    frame = result.res2d.rename(columns={'Term':'term','ES':'es','NES':'nes','NOM p-val':'nominal_p',
        'FDR q-val':'gsea_fdr_q','FWER p-val':'fwer_p','Lead_genes':'leading_edge_genes',
        'Tag %':'tag_fraction','Gene %':'gene_fraction'}).drop(columns='Name')
    for column in ('es','nes','nominal_p','gsea_fdr_q','log2err'):
        frame[column] = pd.to_numeric(frame[column])
    if set(frame.term) != set(gene_sets) or not np.isfinite(frame[['es','nes']]).all().all():
        raise ValueError('incomplete or non-finite GO enrichment results')
    return frame.sort_values('term')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gene-sets', type=Path, default=DEFAULT_GMT)
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    args = parser.parse_args();args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = json.loads((args.input_root/'validation_summary.json').read_text())
    if validation['status'] != 'passed' or validation['validated_outputs'] != 336:
        raise ValueError('completed Study 06 validation is required')
    plan = json.loads((args.input_root/'plan.json').read_text())
    metadata = dict(parameters=PARAMETERS, software={key:version(key) for key in
        ('scanpy','gseapy','numpy','pandas','scipy','anndata')}, python=sys.executable,
        input_root=str(args.input_root.resolve()), gene_sets=str(args.gene_sets.resolve()),
        population='pooled held-out target cells only within each density; paired original class labels',
        class_labels_supplied_to_simulator=True,
        prior_method='05_2d_conditional_transfer/pathway_analysis/run.py')
    (args.output_dir/'parameters.json').write_text(json.dumps(metadata, indent=2)+'\n')
    inventory = []
    for density in sorted(plan['densities'].values(), key=lambda d:int(d['spec']['gap'])):
        gap = int(density['spec']['gap']);directory=args.output_dir/f'gap{gap}';directory.mkdir(exist_ok=True)
        for method in ('observed','feast'):
            score_path = directory/f'{method}_rankings.csv.gz'
            if score_path.exists():
                ranks=pd.read_csv(score_path,index_col='gene')
                counts=json.loads((directory/f'{method}_cells.json').read_text())
            else:
                started=time.perf_counter()
                matrix,labels,genes=pooled_counts(args.input_root,plan,density,method)
                print(f'Gap {gap} {method}: ranking {matrix.shape[0]:,} cells × {matrix.shape[1]} genes.',flush=True)
                ranks,folds,counts=rank_classes(matrix,labels,genes)
                ranks.to_csv(score_path)
                folds.to_csv(directory/f'{method}_log2_fold_changes.csv.gz')
                (directory/f'{method}_cells.json').write_text(json.dumps(counts,indent=2)+'\n')
                del matrix,labels,folds;gc.collect()
                print(f'Gap {gap} {method}: all {len(ranks.columns)} class rankings complete in {time.perf_counter()-started:.1f}s.',flush=True)
            gene_sets,coverage=load_sets(args.gene_sets,list(ranks.index))
            coverage.to_csv(args.output_dir/'gene_set_coverage.csv',index=False)
            for index,label in enumerate(ranks.columns):
                output=directory/f'{method}_class_{index:02d}_go.csv.gz'
                if not output.exists():
                    started=time.perf_counter();frame=enrich(ranks[label],gene_sets)
                    temp=output.with_name(output.name+'.tmp');frame.to_csv(temp,index=False,compression='gzip');os.replace(temp,output)
                    print(f'Gap {gap} {method} {label}: {len(frame)} GO terms, {time.perf_counter()-started:.1f}s.',flush=True)
                inventory.append(dict(gap=gap,method=method,label=label,n_label_cells=int(counts[label]),
                    n_target_cells=int(sum(counts.values())),n_genes=len(ranks),ranking_file=str(score_path),
                    result_file=str(output),zero_score_fraction=float(np.mean(ranks[label]==0)),
                    tied_score_fraction=float(1-ranks[label].nunique()/len(ranks))))
                pd.DataFrame(inventory).to_csv(args.output_dir/'ranking_inventory.csv',index=False)
    (args.output_dir/'completion.json').write_text(json.dumps(dict(status='complete',
        comparisons=len(inventory),classes_per_gap=34,gaps=[3,5,10],gene_sets=len(gene_sets)),indent=2)+'\n')
    print('All Study 06 class-versus-rest rankings and GO analyses complete.',flush=True)

if __name__=='__main__':main()
