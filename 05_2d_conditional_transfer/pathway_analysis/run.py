"""Layer/class GO-BP recovery in the 13 primary Study 05 tasks.

Run with the existing sctm-pub-20260718 Python environment. Completed rankings
and enrichment tables are reused when continuing this same analysis.
"""
from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from importlib.metadata import version

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/study05-numba")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".packages/gseapy_1_3_1"))

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import wasserstein_distance

STUDY = HERE.parent
RANKINGS = HERE / "results"
RESULTS = HERE / "results_multilevel_gseapy131"
sys.path.insert(0, str(STUDY))
from conditional_resampling_baseline import conditional_donor_indices, selected_jobs
from score import donor_stratum
from workflow import direction_id, expected_job_path, inspect_inputs, prepare_job

PARAMETERS = dict(
    assignment_randomness=0.3, normalization_total=10000, transformation="log1p",
    ranking="Scanpy Wilcoxon signed z score: label versus all other supported labels",
    tie_correct=True, gene_universe="entire fixed dataset panel; exact species-specific symbols",
    rank_ties="score descending, gene symbol ascending; no random jitter",
    gene_sets="MSigDB GO Biological Process 2025.1: human C5 / mouse M5",
    permutations=1000, weight=1.0, min_size=15, max_size=500, gsea_seed=2026,
    gsea_method="multilevel", adaptive_sample_size=101, adaptive_eps=1e-50,
    descriptive_fdr_cutoff=0.05,
    baseline_seeds=list(range(2026, 2036)), workers=16, gsea_threads_per_worker=1,
    biological_inference="descriptive recovery benchmark; spots and simulations are not donor replicates",
)
GMT_FILES = {"dlpfc": "c5.go.bp.v2025.1.Hs.symbols.gmt", "merfish": "m5.go.bp.v2025.1.Mm.symbols.gmt"}


def load_gene_sets(dataset, genes):
    universe = set(genes)
    sets, audit = {}, []
    path = HERE / "gene_sets" / GMT_FILES[dataset]
    with path.open() as handle:
        for line in handle:
            name, description, *members = line.rstrip("\n").split("\t")
            members = set(members)
            matched = sorted(members & universe)
            eligible = PARAMETERS["min_size"] <= len(matched) <= PARAMETERS["max_size"]
            audit.append(dict(term=name, source=description, original_genes=len(members),
                              matched_genes=len(matched), panel_coverage=len(matched) / len(members),
                              eligible=eligible))
            if eligible:
                sets[name] = matched
    if not sets:
        raise ValueError(f"No eligible GO gene sets for {dataset}")
    return sets, pd.DataFrame(audit)


def rank_layer_genes(counts, labels, genes):
    """Use counts explicitly; no reclustering, QC exclusion, HVGs, or regression."""
    data = ad.AnnData(X=counts.astype(np.float64).copy(),
                     obs=pd.DataFrame({"label": pd.Categorical(labels)}),
                     var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    sizes = data.obs.label.value_counts()
    if len(sizes) < 2 or sizes.min() < 2:
        raise ValueError(f"Layer-versus-rest needs at least two spots per label: {sizes.to_dict()}")
    sc.pp.normalize_total(data, target_sum=PARAMETERS["normalization_total"])
    sc.pp.log1p(data)
    sc.tl.rank_genes_groups(data, groupby="label", reference="rest", method="wilcoxon",
                           tie_correct=PARAMETERS["tie_correct"], use_raw=False,
                           n_genes=len(genes), rankby_abs=False, key_added="layer_ranking")
    ranks = {}
    for label in sorted(set(labels)):
        frame = sc.get.rank_genes_groups_df(data, group=label, key="layer_ranking")
        score = frame.set_index("names")["scores"].reindex(genes)
        if not np.isfinite(score.to_numpy()).all():
            raise ValueError(f"Non-finite or missing ranking scores for {label}")
        ranks[label] = score.to_numpy(dtype=np.float64)
    return pd.DataFrame(ranks, index=pd.Index(genes, name="gene"))


def prepare_rankings():
    """Write one reusable ranking matrix per task, preserving the baseline RNG stream."""
    config, paths, panels = inspect_inputs(STUDY / "config.yaml", STUDY / "data/input_checksums.csv",
                                          STUDY / "data/local")
    jobs = selected_jobs(config)
    assert len(jobs) == 13
    targets = {key: ad.read_h5ad(path) for key, path in paths.items()}
    rngs = [np.random.default_rng(seed) for seed in PARAMETERS["baseline_seeds"]]
    scored_baselines = pd.read_csv(STUDY / "outputs/baselines/conditional_metric_comparison/method_metrics.csv")
    inventory = []
    for dataset, genes in panels.items():
        _, audit = load_gene_sets(dataset, genes)
        audit.to_csv(RESULTS / f"{dataset}_gene_set_coverage.csv", index=False)
    for job_index, job in enumerate(jobs, 1):
        started = time.monotonic()
        direction = direction_id(job)
        task_dir = RANKINGS / job.dataset / direction
        task_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_job(config, paths, panels, job)
        key = config["datasets"][job.dataset]["annotation_key"]
        target = targets[job.dataset, job.target][prepared.target_contract.obs_names, prepared.genes].copy()
        labels = target.obs[key].astype(str).to_numpy()
        source_labels = prepared.reference.obs[key].astype(str).to_numpy()
        donors = [conditional_donor_indices(source_labels, labels, rng) for rng in rngs]
        metadata_file = task_dir / "task.json"
        if metadata_file.exists():
            meta = json.loads(metadata_file.read_text())
            for rank in meta["rankings"]:
                rank["result_file"] = str(RESULTS / job.dataset / direction / "enrichment" / f"{rank['column']}.csv.gz")
            result_task_dir = RESULTS / job.dataset / direction
            result_task_dir.mkdir(parents=True, exist_ok=True)
            (result_task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n")
            inventory.extend(meta["rankings"])
            print(f"Rankings {job_index}/13 reused: {job.dataset} {direction}", flush=True)
            del prepared, target
            yield meta["rankings"]
            continue
        generated_path = expected_job_path(STUDY / "outputs/final", job) / "generated.h5ad"
        generated = ad.read_h5ad(generated_path)
        assert list(generated.obs_names) == list(target.obs_names)
        assert list(generated.var_names) == prepared.genes
        np.testing.assert_allclose(generated.obsm["spatial"], target.obsm["spatial"], rtol=0, atol=0)
        assert np.array_equal(generated.obs[key].astype(str).to_numpy(), labels)
        source_counts = prepared.reference.layers["counts"]
        truth_counts = target.layers["counts"]
        truth_library = np.asarray(truth_counts.sum(axis=1)).ravel()
        source_library = np.asarray(source_counts.sum(axis=1)).ravel()
        rank_columns, rank_meta = {}, []
        identity = dict(dataset=job.dataset, mode=job.mode, direction=direction,
                        source=job.source, target=job.target, donor_stratum=donor_stratum(job))
        for method_index in range(12):
            if method_index == 0:
                method, replicate, matrix = "real", -1, truth_counts
            elif method_index == 1:
                method, replicate, matrix = "feast", -1, generated.layers["counts"]
            else:
                method, replicate = "resampling", method_index - 2
                matrix = source_counts[donors[replicate], :]
                expected = scored_baselines.loc[scored_baselines.direction.eq(direction)
                    & scored_baselines.method.ne("feast") & scored_baselines.replicate.eq(replicate),
                    "library_size_wasserstein"].iloc[0]
                np.testing.assert_allclose(wasserstein_distance(truth_library, source_library[donors[replicate]]),
                                           expected, rtol=1e-10, atol=1e-10)
            ranks = rank_layer_genes(matrix, labels, prepared.genes)
            for group_index, label in enumerate(ranks.columns):
                column = f"{method}_{replicate}_group_{group_index:02d}"
                values = ranks[label].to_numpy()
                rank_columns[column] = values
                rank_meta.append(dict(**identity, method=method, replicate=replicate, label=label,
                    column=column, ranking_file=str(task_dir / "rankings.csv.gz"),
                    result_file=str(RESULTS / job.dataset / direction / "enrichment" / f"{column}.csv.gz"),
                    n_target_spots=len(labels), n_label_spots=int(np.sum(labels == label)),
                    n_genes=len(prepared.genes), zero_score_fraction=float(np.mean(values == 0)),
                    tied_score_fraction=float(1 - len(np.unique(values)) / len(values))))
            print(f"Ranking {job_index}/13 {direction}: {method} {replicate}", flush=True)
        pd.DataFrame(rank_columns, index=pd.Index(prepared.genes, name="gene")).to_csv(task_dir / "rankings.csv.gz")
        meta = dict(**identity, support=prepared.support, generated_input=str(generated_path),
                    target_expression_access="evaluation only", baseline_draw_check="all ten match existing library-size metrics",
                    rankings=rank_meta, ranking_seconds=time.monotonic() - started)
        metadata_file.write_text(json.dumps(meta, indent=2) + "\n")
        result_task_dir = RESULTS / job.dataset / direction
        result_task_dir.mkdir(parents=True, exist_ok=True)
        (result_task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n")
        inventory.extend(rank_meta)
        del prepared, target, generated, ranks, rank_columns, matrix
        gc.collect()
        print(f"Rankings ready {job_index}/13: {direction} ({meta['ranking_seconds']:.1f}s)", flush=True)
        yield rank_meta
    pd.DataFrame(inventory).to_csv(RESULTS / "ranking_inventory.csv", index=False)


@lru_cache(maxsize=1)
def read_rankings(path):
    return pd.read_csv(path, index_col="gene")


@lru_cache(maxsize=2)
def cached_gene_sets(dataset, genes):
    return load_gene_sets(dataset, genes)[0]


def run_layer_gsea(meta):
    path = Path(meta["result_file"])
    if path.exists():
        return dict(column=meta["column"], direction=meta["direction"], reused=True)
    started = time.monotonic()
    rankings = read_rankings(meta["ranking_file"])
    values = rankings[meta["column"]]
    ordered = values.rename("score").reset_index().sort_values(["score", "gene"], ascending=[False, True], kind="stable")
    genes = tuple(rankings.index)
    gene_sets = cached_gene_sets(meta["dataset"], genes)
    result = gp.prerank(rnk=ordered, gene_sets=gene_sets,
        organism="human" if meta["dataset"] == "dlpfc" else "mouse", outdir=None,
        ascending=None, min_size=PARAMETERS["min_size"], max_size=PARAMETERS["max_size"],
        permutation_num=PARAMETERS["permutations"], weight=PARAMETERS["weight"],
        threads=PARAMETERS["gsea_threads_per_worker"], seed=PARAMETERS["gsea_seed"],
        no_plot=True, verbose=False, method=PARAMETERS["gsea_method"],
        sample_size=PARAMETERS["adaptive_sample_size"], eps=PARAMETERS["adaptive_eps"])
    frame = result.res2d.rename(columns={"Term": "term", "ES": "es", "NES": "nes",
        "NOM p-val": "nominal_p", "FDR q-val": "gsea_fdr_q", "FWER p-val": "fwer_p",
        "Lead_genes": "leading_edge_genes", "Tag %": "tag_fraction", "Gene %": "gene_fraction"}).drop(columns="Name")
    for col in ["es", "nes", "nominal_p", "gsea_fdr_q", "log2err"]:
        frame[col] = pd.to_numeric(frame[col])
    if set(frame.term) != set(gene_sets) or not np.isfinite(frame.es).all():
        raise ValueError(f"GSEA returned incomplete/invalid results: {meta['direction']} {meta['column']}")
    unavailable = frame.nes.isna()
    if not frame.loc[unavailable, ["nominal_p", "gsea_fdr_q"]].isna().all().all():
        raise ValueError("Unavailable NES must retain unavailable significance estimates")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    frame.sort_values("term").to_csv(temp_path, index=False, compression="gzip")
    os.replace(temp_path, path)
    info = dict(column=meta["column"], direction=meta["direction"], terms=len(frame),
                positive_terms=int(((frame.nes > 0) & (frame.gsea_fdr_q < 0.05)).sum()),
                unavailable_terms=int(unavailable.sum()),
                seconds=time.monotonic() - started)
    del result, frame
    gc.collect()
    return info


def main():
    if gp.__version__ != "1.3.1":
        raise RuntimeError("This analysis requires GSEApy 1.3.1; 1.3.0 has a documented multilevel p-value bug")
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESULTS / "analysis.log", level=logging.INFO)
    sc.settings.verbosity = 0
    versions = {name: version(name) for name in ["scanpy", "gseapy", "numpy", "pandas", "scipy", "anndata"]}
    (RESULTS / "parameters.json").write_text(json.dumps(dict(parameters=PARAMETERS, software=versions,
        python=sys.executable, input_config=str(STUDY / "config.yaml")), indent=2) + "\n")
    futures = []
    with ProcessPoolExecutor(max_workers=PARAMETERS["workers"], mp_context=mp.get_context("spawn")) as pool:
        for metadata in prepare_rankings():
            # Real and FEAST are submitted before baseline draws, without changing
            # any RNG stream, input contrast, or enrichment calculation.
            futures.extend(pool.submit(run_layer_gsea, meta) for meta in metadata)
            finished = sum(f.done() for f in futures)
            print(f"Enrichment jobs queued: {len(futures)}; completed so far: {finished}", flush=True)
        inventory = pd.read_csv(RESULTS / "ranking_inventory.csv").to_dict("records")
        for index, future in enumerate(as_completed(futures), 1):
            info = future.result()
            if index % 10 == 0 or index == len(futures):
                print(f"GSEA completed {index}/{len(futures)}: {info}", flush=True)
    missing = [r["result_file"] for r in inventory if not Path(r["result_file"]).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} enrichment outputs; continue this same run to finish them")
    (RESULTS / "completion.json").write_text(json.dumps(dict(status="complete", rankings=len(inventory),
        tasks=len({r['direction'] for r in inventory}), parameters=PARAMETERS), indent=2) + "\n")
    print(f"Completed all {len(inventory)} rankings. Run summarize.py to generate the comparison report.", flush=True)


if __name__ == "__main__":
    main()
