"""
Alternative model: binary vector -> TruncatedSVD (latent factors) -> KDTree.

This implements the "SVD + KD-Tree" pipeline (a common latent-factor approach)
and evaluates it on the SAME offline protocol used in compare_models.py, so the
numbers line up directly with cosine / dot / jaccard / hybrid.

Pipeline
1. Build a BINARY feature vector per perfume:
   notes (multi-hot, vocab-filtered) + family/subfamily/gender (one-hot).
2. TruncatedSVD reduces the sparse binary matrix to N latent dimensions
   (works directly on sparse data; like LSA for text).
3. KDTree is built on the dense reduced matrix and queried for nearest items.
   - "euclidean" : KDTree on raw SVD components (L2 distance)
   - "cosine~"   : KDTree on L2-normalized components (Euclidean on normalized
                   vectors is rank-equivalent to cosine similarity)

Metrics (same as compare_models.py, same seed/sample so it's apples-to-apples)
- LOO hit-rate@10   : hide a perfume, query with HALF its notes (+ its
                      family/subfamily/gender), check if it returns in top-10
- Precision@10 fam  : fraction of top-10 sharing the query's family
- Precision@10 notes: fraction of top-10 sharing >= 2 of the query's notes

Run:
    python src/svd_kdt_compare.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

import feature_engineering as fe

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEED = 42
SAMPLE = 300
K = 10
SVD_DIMS = [50, 100, 200]


def _load():
    df = joblib.load(f"{MODELS_DIR}/perfume_df.pkl")
    mlb = joblib.load(f"{MODELS_DIR}/mlb_ingredients.pkl")
    ohe = joblib.load(f"{MODELS_DIR}/ohe_categories.pkl")
    return df, mlb, ohe


def build_binary_matrix(df, mlb, ohe) -> csr_matrix:
    """Binary notes (multi-hot) + binary categories (one-hot), aligned to df."""
    vocab_set = set(mlb.classes_)
    ing_filtered = fe.filter_to_vocabulary(df["ingredients"], vocab_set)
    ing_bin = mlb.transform(ing_filtered).astype(np.float64)
    cat_bin = ohe.transform(df[["family", "subfamily", "gender"]]).astype(np.float64)
    return hstack([ing_bin, cat_bin]).tocsr()


def build_query_binary(notes, row, mlb, ohe) -> csr_matrix:
    """Same binary representation for a user query."""
    known = [n.title() for n in notes if n.title() in set(mlb.classes_)]
    ing_bin = mlb.transform([known]).astype(np.float64)
    cat_input = pd.DataFrame(
        [[str(row["family"]).upper(), str(row["subfamily"]).upper(),
          str(row["gender"]).upper()]],
        columns=["family", "subfamily", "gender"],
    )
    cat_bin = ohe.transform(cat_input).astype(np.float64)
    return hstack([ing_bin, cat_bin]).tocsr()


def evaluate_neighbors(df, neighbors_fn, sample_idx, loo_idx):
    """neighbors_fn(notes, row, k) -> ordered array of catalog indices."""
    fam_prec, note_prec = [], []
    for i in sample_idx:
        row = df.iloc[i]
        notes = set(row["ingredients"])
        idx = neighbors_fn(list(notes), row, K + 1)
        top = [j for j in idx if j != i][:K]
        fam_prec.append(np.mean([df.iloc[j]["family"] == row["family"] for j in top]))
        note_prec.append(
            np.mean([len(notes & set(df.iloc[j]["ingredients"])) >= 2 for j in top])
        )

    rng = np.random.default_rng(SEED)
    hits, evaluated = 0, 0
    for i in loo_idx:
        row = df.iloc[i]
        notes = list(row["ingredients"])
        if len(notes) < 2:
            continue
        evaluated += 1
        half = rng.choice(notes, size=max(1, len(notes) // 2), replace=False).tolist()
        idx = neighbors_fn(half, row, K)
        if i in set(idx[:K]):
            hits += 1

    loo = hits / evaluated if evaluated else 0.0
    return loo, float(np.mean(fam_prec)), float(np.mean(note_prec))


def main() -> None:
    df, mlb, ohe = _load()
    binary = build_binary_matrix(df, mlb, ohe)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(df), size=min(SAMPLE, len(df)), replace=False)
    loo_idx = rng.choice(len(df), size=min(SAMPLE, len(df)), replace=False)

    print("=== SVD + KDTree comparison ===")
    print(f"Catalog size : {len(df):,}")
    print(f"Binary dims  : {binary.shape[1]} (notes + categories, 0/1)")
    print(f"Sample size  : {SAMPLE} (seed={SEED}), K={K}")
    print()

    header = (f"{'method':<26}{'LOO hit@10':>12}{'P@10 family':>14}"
              f"{'P@10 notes':>13}{'fit(s)':>9}{'eval(s)':>9}")
    print(header)
    print("-" * len(header))

    rows = []

    # --- Baseline anchor: cosine on the SAME binary vectors (no SVD) ----------
    def cosine_neighbors(notes, row, k):
        q = build_query_binary(notes, row, mlb, ohe)
        scores = cosine_similarity(q, binary).ravel()
        return np.argsort(scores)[::-1][:k]

    t0 = time.time()
    loo, fam, note = evaluate_neighbors(df, cosine_neighbors, sample_idx, loo_idx)
    dt = time.time() - t0
    rows.append(("cosine (binary, no SVD)", loo, fam, note))
    print(f"{'cosine (binary, no SVD)':<26}{loo:>12.4f}{fam:>14.4f}"
          f"{note:>13.4f}{0.0:>9.1f}{dt:>9.1f}")

    # --- SVD + KDTree at several latent dimensions ----------------------------
    for dims in SVD_DIMS:
        t0 = time.time()
        svd = TruncatedSVD(n_components=dims, random_state=SEED)
        reduced = svd.fit_transform(binary)
        evr = float(svd.explained_variance_ratio_.sum())

        # raw components -> Euclidean KDTree
        tree_euc = KDTree(reduced, metric="euclidean")
        # normalized components -> Euclidean ~ cosine
        reduced_norm = normalize(reduced)
        tree_cos = KDTree(reduced_norm, metric="euclidean")
        fit_dt = time.time() - t0

        def euc_neighbors(notes, row, k, _svd=svd, _tree=tree_euc):
            q = build_query_binary(notes, row, mlb, ohe)
            qr = _svd.transform(q)
            _, idx = _tree.query(qr, k=k)
            return idx[0]

        def cos_neighbors(notes, row, k, _svd=svd, _tree=tree_cos):
            q = build_query_binary(notes, row, mlb, ohe)
            qr = normalize(_svd.transform(q))
            _, idx = _tree.query(qr, k=k)
            return idx[0]

        t0 = time.time()
        loo_e, fam_e, note_e = evaluate_neighbors(df, euc_neighbors, sample_idx, loo_idx)
        loo_c, fam_c, note_c = evaluate_neighbors(df, cos_neighbors, sample_idx, loo_idx)
        eval_dt = time.time() - t0

        name_e = f"SVD{dims}+KDT euclidean"
        name_c = f"SVD{dims}+KDT cosine~"
        rows.append((name_e, loo_e, fam_e, note_e))
        rows.append((name_c, loo_c, fam_c, note_c))
        print(f"{name_e:<26}{loo_e:>12.4f}{fam_e:>14.4f}{note_e:>13.4f}"
              f"{fit_dt:>9.1f}{eval_dt/2:>9.1f}")
        print(f"{name_c:<26}{loo_c:>12.4f}{fam_c:>14.4f}{note_c:>13.4f}"
              f"{fit_dt:>9.1f}{eval_dt/2:>9.1f}   (explained var={evr:.2%})")

    print()
    print("Reference (from compare_models.py, same seed/sample):")
    print("  dot product            LOO 0.8949  P@10 fam 0.6687  P@10 notes 0.9720")
    print("  cosine (weighted)      LOO 0.8237  P@10 fam 0.6340  P@10 notes 0.9367")
    print("  hybrid 0.7/0.3 (prod)  LOO 0.7661  P@10 fam 0.5767  P@10 notes 0.9447")
    print()
    best_loo = max(rows, key=lambda r: r[1])
    print(f"Best LOO among SVD/KDT rows here: {best_loo[0]} ({best_loo[1]:.4f})")


if __name__ == "__main__":
    main()
