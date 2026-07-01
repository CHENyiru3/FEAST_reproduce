#!/usr/bin/env python
"""Batch Correction Benchmark — evaluate integration quality per method/alpha.

Standard metrics for spatial batch correction:
  Batch mixing:    kBET acceptance, iLISI, batch ASW, graph connectivity
  Bio preservation: domain ASW, domain ARI, spatial autocorrelation (Moran's I)
  Expression:       per-gene Pearson r (query vs ref, pre- and post-correction)
  Combined:         overall score = 0.4 * batch_mixing + 0.4 * bio_pres + 0.2 * expr

Usage:
    /maiziezhou_lab2/yiru/miniconda3/envs/batch-eval/bin/python benchmark.py \\
      --methods-dir outputs/methods --run-manifest outputs/run_logs/latest \\
      --output outputs/benchmarks/batch_correction_results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  kBET — k-nearest neighbour batch effect test
# ═══════════════════════════════════════════════════════════════════════════

def kbet(embedding, batch_labels, k=50, n_subsample=500, seed=42):
    """kBET acceptance rate.  Higher = better mixing (max 1.0)."""
    rng = np.random.default_rng(seed)
    n = embedding.shape[0]
    unique_batches, counts = np.unique(batch_labels, return_counts=True)
    global_p = counts / counts.sum()
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    neigh_batches = batch_labels[indices[:, 1:]]

    idx = rng.choice(n, size=min(n, n_subsample), replace=False)
    accepted = 0

    for i in idx:
        observed = np.array([(neigh_batches[i] == b).sum() for b in unique_batches])
        expected = global_p * k
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = ((observed - expected) ** 2 / expected).sum()
            chi2 = chi2 if np.isfinite(chi2) else 0.0
        if chi2 < 3.84 * n_batches:
            accepted += 1

    return accepted / len(idx)


# ═══════════════════════════════════════════════════════════════════════════
#  LISI — Local Inverse Simpson's Index
# ═══════════════════════════════════════════════════════════════════════════

def lisi(embedding, labels, k=50, seed=42):
    """Local Inverse Simpson's Index.
    For batch labels: higher = better mixing (range 1 to n_batches).
    """
    rng = np.random.default_rng(seed)
    n = embedding.shape[0]
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    if n_labels < 2:
        return np.full(n, 1.0)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    neigh_labels = labels[indices[:, 1:]]

    lisi_vals = np.zeros(n)
    for i in range(n):
        counts = np.array([(neigh_labels[i] == lab).sum() for lab in unique_labels])
        props = counts / k
        simpson = (props ** 2).sum()
        lisi_vals[i] = 1.0 / simpson if simpson > 0 else 1.0

    return lisi_vals


# ═══════════════════════════════════════════════════════════════════════════
#  Graph connectivity
# ═══════════════════════════════════════════════════════════════════════════

def graph_connectivity(embedding, batch_labels, k=15, seed=42):
    """Fraction of batch pairs connected in the k-NN graph. Higher = better."""
    rng = np.random.default_rng(seed)
    unique_batches = np.unique(batch_labels)

    if len(unique_batches) < 2:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embedding)
    knn_graph = nn.kneighbors_graph(embedding, mode="connectivity")

    connected_pairs = 0
    total_pairs = 0
    for i, b1 in enumerate(unique_batches):
        for j, b2 in enumerate(unique_batches):
            if i >= j:
                continue
            total_pairs += 1
            idx1 = np.where(batch_labels == b1)[0]
            idx2 = np.where(batch_labels == b2)[0]
            subgraph = knn_graph[idx1][:, idx2]
            if subgraph.nnz > 0:
                connected_pairs += 1

    return connected_pairs / total_pairs if total_pairs > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  ASW — Average Silhouette Width
# ═══════════════════════════════════════════════════════════════════════════

def asw_scores(embedding, batch_labels, domain_labels=None,
               subsample=2000, seed=42):
    """ASW_batch (lower=better) and ASW_domain (higher=better preservation)."""
    rng = np.random.default_rng(seed)
    n = embedding.shape[0]

    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
    else:
        idx = slice(None)

    emb = embedding[idx]
    result = {}

    bl = batch_labels[idx]
    if len(np.unique(bl)) >= 2:
        result["ASW_batch"] = float(silhouette_score(emb, bl))

    if domain_labels is not None:
        dl = domain_labels[idx]
        if len(np.unique(dl)) >= 2:
            result["ASW_domain"] = float(silhouette_score(emb, dl))

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Domain clustering preservation
# ═══════════════════════════════════════════════════════════════════════════

def domain_ari(embedding, domain_labels, resolution=0.8, seed=42):
    """Cluster embedding (Leiden) → ARI with ground-truth domains."""
    if len(np.unique(domain_labels)) < 2:
        return np.nan, np.nan

    adata_tmp = sc.AnnData(X=np.ones((embedding.shape[0], 1)))
    adata_tmp.obsm["X_emb"] = embedding
    sc.pp.neighbors(adata_tmp, use_rep="X_emb", random_state=seed)
    sc.tl.leiden(adata_tmp, resolution=resolution, random_state=seed)

    pred = adata_tmp.obs["leiden"].values
    return float(adjusted_rand_score(domain_labels, pred)), len(np.unique(pred))


# ═══════════════════════════════════════════════════════════════════════════
#  Spatial autocorrelation (Moran's I) of embedding components
# ═══════════════════════════════════════════════════════════════════════════

def moran_i_embedding(embedding, spatial_coords, n_components=5):
    """Mean Moran's I of top embedding PCs. Higher = more spatial structure."""
    from sklearn.neighbors import kneighbors_graph

    n = embedding.shape[0]
    k = min(15, n - 1)
    adj = kneighbors_graph(spatial_coords, n_neighbors=k, mode="connectivity")
    adj = adj.toarray()
    adj = (adj + adj.T) > 0

    n_comp = min(n_components, embedding.shape[1])
    morans = []
    for c in range(n_comp):
        x = embedding[:, c] - embedding[:, c].mean()
        denom = (x ** 2).sum()
        if denom == 0:
            morans.append(0.0)
            continue
        num = (adj * np.outer(x, x)).sum()
        w_sum = adj.sum()
        morans.append(float(n / w_sum * num / denom) if w_sum > 0 else 0.0)

    return float(np.mean(morans))


# ═══════════════════════════════════════════════════════════════════════════
#  Expression correlation
# ═══════════════════════════════════════════════════════════════════════════

def expression_correlation(X_ref, X_query, X_corrected=None, n_genes=500, seed=42):
    """Per-gene Pearson r (median).  Returns pre, post, and delta."""
    rng = np.random.default_rng(seed)
    n_genes_total = X_ref.shape[1]
    gene_idx = rng.choice(n_genes_total, size=min(n_genes, n_genes_total), replace=False)

    corr_pre = []
    corr_post = []
    for g in gene_idx:
        c, _ = pearsonr(X_ref[:, g], X_query[:, g])
        corr_pre.append(c if np.isfinite(c) else 0.0)
        if X_corrected is not None:
            c, _ = pearsonr(X_ref[:, g], X_corrected[:, g])
            corr_post.append(c if np.isfinite(c) else 0.0)

    result = {"expr_corr_pre": float(np.median(corr_pre))}
    if X_corrected is not None:
        result["expr_corr_post"] = float(np.median(corr_post))
        result["expr_corr_delta"] = result["expr_corr_post"] - result["expr_corr_pre"]
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Combined score
# ═══════════════════════════════════════════════════════════════════════════

def combined_score(kbet_val, ilisi_mean, conn, asw_batch, asw_domain,
                   domain_ari_val, expr_delta):
    """Aggregate into a single 0-1 score. Higher = better."""
    # Batch mixing (0-1)
    vals = [kbet_val, min(ilisi_mean / 2.0, 1.0), conn]
    batch_score = np.nanmean([v for v in vals if np.isfinite(v)])

    # Bio preservation (0-1)
    bio_vals = []
    if asw_domain is not None and np.isfinite(asw_domain):
        bio_vals.append(max(0.0, min(asw_domain, 1.0)))
    if domain_ari_val is not None and np.isfinite(domain_ari_val):
        bio_vals.append(max(0.0, min(domain_ari_val, 1.0)))
    bio_score = np.mean(bio_vals) if bio_vals else 0.5

    # Expression
    if expr_delta is not None and np.isfinite(expr_delta):
        expr_score = 1.0 / (1.0 + np.exp(-5 * expr_delta))
    else:
        expr_score = 0.5

    weights = np.array([0.4, 0.4, 0.2])
    scores = np.array([batch_score, bio_score, expr_score])
    mask = np.isfinite(scores)
    if mask.sum() > 0:
        return float(np.average(scores[mask], weights=weights[mask] / weights[mask].sum()))
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════

def _to_dense(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def compute_benchmark(methods_dir, run_manifest, output_csv):
    rows = []
    ok_jobs = run_manifest[run_manifest["status"] == "ok"]

    if len(ok_jobs) == 0:
        print("No successful method runs found.")
        return pd.DataFrame()

    total = len(ok_jobs)
    t0 = time.time()

    for done, (_, job_row) in enumerate(ok_jobs.iterrows(), 1):
        method = job_row["method"]
        mode = job_row["mode"]
        alpha = job_row["alpha"]

        m_out = methods_dir / method / mode / f"alpha_{alpha:.2f}"
        emb_file = m_out / "embeddings.npz"
        result_file = m_out / "result.h5ad"

        base = {"method": method, "mode": mode, "alpha": alpha}

        if not emb_file.exists() and not result_file.exists():
            rows.append({**base, "status": "output_missing"})
            print(f"  [{done}/{total}] {method}/{mode}/alpha={alpha:.2f} — MISSING")
            continue

        try:
            # ── Load embedding ──
            if emb_file.exists():
                data = np.load(emb_file, allow_pickle=True)
                embedding = data["embedding"]
                batch_labels = data["batch_labels"]
            else:
                adata_out = sc.read_h5ad(str(result_file))
                for key in ["X_scVI", "X_STAMP", "X_GraphST"]:
                    if key in adata_out.obsm:
                        embedding = adata_out.obsm[key]
                        break
                else:
                    raise ValueError("no recognised embedding key in obsm")
                batch_labels = adata_out.obs["batch"].values

            n_spots = embedding.shape[0]

            # ── Load additional info from result h5ad ──
            adata_out = None
            domain_labels = None
            spatial_coords = None
            X_ref = X_query = X_corrected = None

            if result_file.exists():
                try:
                    adata_out = sc.read_h5ad(str(result_file))
                    if "dlpfc_layer" in adata_out.obs.columns:
                        domain_labels = adata_out.obs["dlpfc_layer"].values
                    if "spatial" in adata_out.obsm:
                        spatial_coords = adata_out.obsm["spatial"]
                except Exception:
                    pass

            # ── Batch mixing ──
            kbet_val = kbet(embedding, batch_labels)
            ilisi_vals = lisi(embedding, batch_labels)
            ilisi_mean = float(np.mean(ilisi_vals))
            conn = graph_connectivity(embedding, batch_labels)
            asw = asw_scores(embedding, batch_labels, domain_labels)
            asw_batch = asw.get("ASW_batch", np.nan)
            asw_domain = asw.get("ASW_domain", np.nan)

            # ── Bio preservation ──
            ari_val, n_clusters = (np.nan, np.nan)
            if domain_labels is not None:
                ari_val, n_clusters = domain_ari(embedding, domain_labels)

            # ── Spatial autocorrelation ──
            moran_mean = np.nan
            if spatial_coords is not None:
                moran_mean = moran_i_embedding(embedding, spatial_coords)

            # ── Expression correlation ──
            expr = {"expr_corr_pre": np.nan, "expr_corr_post": np.nan, "expr_corr_delta": np.nan}
            if adata_out is not None and "batch" in adata_out.obs.columns:
                try:
                    ref_mask = adata_out.obs["batch"].values == "ref"
                    query_mask = adata_out.obs["batch"].values == "query"
                    X_ref = _to_dense(adata_out.X[ref_mask])
                    X_query = _to_dense(adata_out.X[query_mask])
                    X_corrected = None
                    for layer_key in ["scvi_normalized"]:
                        if layer_key in adata_out.layers:
                            X_corrected = _to_dense(adata_out.layers[layer_key][query_mask])
                            break
                    expr = expression_correlation(X_ref, X_query, X_corrected)
                except Exception:
                    pass

            # ── Combined ──
            overall = combined_score(kbet_val, ilisi_mean, conn, asw_batch,
                                     asw_domain, ari_val, expr.get("expr_corr_delta", np.nan))

            row = {
                **base, "status": "ok", "n_spots": n_spots,
                "kBET": round(kbet_val, 4),
                "iLISI": round(ilisi_mean, 4),
                "graph_conn": round(conn, 4),
                "ASW_batch": round(asw_batch, 4) if np.isfinite(asw_batch) else np.nan,
                "ASW_domain": round(asw_domain, 4) if np.isfinite(asw_domain) else np.nan,
                "domain_ARI": round(ari_val, 4) if np.isfinite(ari_val) else np.nan,
                "spatial_moran": round(moran_mean, 4) if np.isfinite(moran_mean) else np.nan,
                "expr_corr_pre": round(expr["expr_corr_pre"], 4) if np.isfinite(expr["expr_corr_pre"]) else np.nan,
                "expr_corr_post": round(expr.get("expr_corr_post", np.nan), 4) if np.isfinite(expr.get("expr_corr_post", np.nan)) else np.nan,
                "expr_corr_delta": round(expr.get("expr_corr_delta", np.nan), 4) if np.isfinite(expr.get("expr_corr_delta", np.nan)) else np.nan,
                "overall_score": round(overall, 4) if np.isfinite(overall) else np.nan,
            }
            rows.append(row)
            print(f"  [{done}/{total}] {method}/{mode}/alpha={alpha:.2f} "
                  f"— kBET={kbet_val:.3f} iLISI={ilisi_mean:.2f} overall={overall:.3f}")

        except Exception as e:
            import traceback
            rows.append({**base, "status": "error", "error": traceback.format_exc()[-200:]})
            print(f"  [{done}/{total}] {method}/{mode}/alpha={alpha:.2f} — ERROR: {e}")

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    ok_count = (results["status"] == "ok").sum()
    elapsed = time.time() - t0
    print(f"\nBenchmark done: {ok_count}/{len(results)} ok in {elapsed:.1f}s → {output_csv}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods-dir", type=Path, default=Path("outputs/methods"))
    parser.add_argument("--run-manifest", type=Path, default=Path("outputs/run_logs/latest"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/batch_correction_results.csv"))
    args = parser.parse_args()

    if args.run_manifest.is_dir():
        subdirs = sorted([d for d in args.run_manifest.iterdir() if d.is_dir()])
        if not subdirs:
            print("ERROR: no run log subdirectories", file=sys.stderr)
            return 1
        run_manifest_path = subdirs[-1] / "run_manifest.csv"
    else:
        run_manifest_path = args.run_manifest

    if not run_manifest_path.exists():
        print(f"ERROR: run manifest not found: {run_manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(run_manifest_path)
    print(f"Run manifest: {len(manifest)} jobs ({(manifest['status'] == 'ok').sum()} ok)")

    results = compute_benchmark(args.methods_dir, manifest, args.output)

    if len(results) > 0 and "overall_score" in results.columns:
        ok = results[results["status"] == "ok"]
        if len(ok) > 0:
            print("\n=== Summary by method ===")
            summary = ok.groupby("method")[
                ["kBET", "iLISI", "graph_conn", "ASW_domain", "domain_ARI",
                 "expr_corr_delta", "overall_score"]
            ].mean()
            print(summary.round(4).to_string())
            print("\n=== Summary by alpha ===")
            by_alpha = ok.groupby("alpha")[
                ["kBET", "iLISI", "graph_conn", "overall_score"]
            ].mean()
            print(by_alpha.round(4).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
