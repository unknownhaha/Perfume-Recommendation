"""
optimized.py — fast, standalone trainer for the content-based perfume recommender.

This is the plain-Python equivalent of optimized.ipynb. Run it directly instead
of converting/executing the notebook each time:

    python optimized.py

Why it is fast
--------------
The notebook evaluated leave-one-out (LOO) hit-rate by calling
`NearestNeighbors.kneighbors` once *per query* inside a Python loop. Here every
LOO query for a sample is stacked into a single sparse matrix and scored with one
batched `kneighbors` call, which is far faster for the same result.

Pipeline
--------
1. Load data/perfumes.json
2. Encode notes (MultiLabelBinarizer) + family/subfamily/gender (OneHotEncoder)
3. Compare metrics (euclidean / manhattan / cosine / jaccard) x k via LOO hit-rate
4. Quick weight sweep on the winning metric (a few candidate weight vectors)
5. Save the winning model -> models/opt_*.pkl + models/opt_matrix.npz
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# --- Config ----------------------------------------------------------------
SEED = 42
SAMPLE = 300
DATA_PATH = "data/perfumes.json"
MODELS_DIR = "models"
METRICS = ["euclidean", "manhattan", "cosine", "jaccard"]
K_VALUES = [3, 5, 7, 10]

# Candidate weight vectors [w_ing, w_family, w_subfamily, w_gender] for the sweep.
WEIGHT_CANDIDATES = [
    [1.0, 1.0, 1.0, 1.0],   # uniform (notebook winner)
    [1.0, 0.5, 0.3, 0.2],   # previous production weighting
    [2.0, 0.5, 0.3, 0.2],   # ingredient-heavy
    [1.0, 1.5, 1.0, 0.5],   # category-heavy
]


def load_data() -> pd.DataFrame:
    df = pd.read_json(DATA_PATH)
    df["ingredients"] = df["ingredients"].apply(
        lambda x: [i.strip() for i in x] if isinstance(x, list) else []
    )
    print(f"Loaded {len(df):,} perfumes")
    return df


def encode(df):
    """Sparse multi-hot notes + one-hot categories."""
    mlb = MultiLabelBinarizer(sparse_output=True)
    ing = mlb.fit_transform(df["ingredients"]).tocsr().astype(np.float64)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat = ohe.fit_transform(df[["family", "subfamily", "gender"]]).tocsr().astype(np.float64)

    n_family = len(ohe.categories_[0])
    n_subfamily = len(ohe.categories_[1])
    n_gender = len(ohe.categories_[2])
    print(f"Blocks — ingredients:{ing.shape[1]}  family:{n_family}  "
          f"subfamily:{n_subfamily}  gender:{n_gender}")
    return mlb, ohe, ing, cat, n_family, n_subfamily, n_gender


def _weight_categories(cat: csr_matrix, nf, ns, w) -> csr_matrix:
    """Scale each one-hot block (family/subfamily/gender) by its weight."""
    c = cat.tocsc(copy=True)
    c[:, :nf] *= w[1]
    c[:, nf:nf + ns] *= w[2]
    c[:, nf + ns:] *= w[3]
    return c.tocsr()


def build_matrix(ing, cat, nf, ns, w) -> csr_matrix:
    return hstack([ing * w[0], _weight_categories(cat, nf, ns, w)]).tocsr()


def build_query_matrix(df, idxs, halves, mlb, ohe, nf, ns, w) -> csr_matrix:
    """Stack one weighted query vector per sampled perfume into a single matrix."""
    ing_q = mlb.transform(halves).tocsr().astype(np.float64) * w[0]
    cats = df.iloc[idxs][["family", "subfamily", "gender"]]
    cat_q = _weight_categories(ohe.transform(cats).tocsr().astype(np.float64), nf, ns, w)
    return hstack([ing_q, cat_q]).tocsr()


def loo_hit_rate(df, mlb, ohe, matrix, nf, ns, w, metric, k,
                 sample=SAMPLE, seed=SEED) -> float:
    """Batched leave-one-out hit-rate@k.

    Hide a perfume, query with HALF of its notes (+ its profile), and check
    whether the perfume returns in the top-k. All queries are scored in one
    batched kneighbors call.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(sample, len(df)), replace=False)

    valid, halves = [], []
    for i in idxs:
        notes = list(df.iloc[i]["ingredients"])
        if len(notes) < 2:
            continue
        half = rng.choice(notes, size=max(1, len(notes) // 2), replace=False).tolist()
        valid.append(int(i))
        halves.append(half)

    if not valid:
        return 0.0

    queries = build_query_matrix(df, valid, halves, mlb, ohe, nf, ns, w)
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(matrix)
    _, neighbours = model.kneighbors(queries)

    hits = sum(1 for row, i in zip(neighbours, valid) if i in row[:k])
    return hits / len(valid)


def compare_metrics(df, mlb, ohe, ing, cat, nf, ns):
    """Metric x k comparison on the uniform (binary) matrix."""
    uniform = [1.0, 1.0, 1.0, 1.0]
    matrix = build_matrix(ing, cat, nf, ns, uniform)
    rows = []
    print("\nMetric comparison (uniform weights, LOO hit-rate):")
    for metric in METRICS:
        for k in K_VALUES:
            t0 = time.time()
            hr = loo_hit_rate(df, mlb, ohe, matrix, nf, ns, uniform, metric, k)
            rows.append({"metric": metric, "k": k, "hit_rate": round(hr, 4)})
            print(f"  metric={metric:10s} k={k:2d}  hit_rate={hr:.4f}  ({time.time()-t0:.1f}s)")
    res = pd.DataFrame(rows).sort_values("hit_rate", ascending=False).reset_index(drop=True)
    print("\nTop configs:")
    print(res.head(5).to_string(index=False))
    return res.iloc[0]


def sweep_weights(df, mlb, ohe, ing, cat, nf, ns, metric, k):
    """Quick weight sweep on the winning metric."""
    print(f"\nWeight sweep (metric={metric}, k={k}):")
    best_w, best_hr = WEIGHT_CANDIDATES[0], -1.0
    for w in WEIGHT_CANDIDATES:
        matrix = build_matrix(ing, cat, nf, ns, w)
        hr = loo_hit_rate(df, mlb, ohe, matrix, nf, ns, w, metric, k)
        marker = ""
        if hr > best_hr:
            best_hr, best_w = hr, w
            marker = "  <- best so far"
        print(f"  w={w}  hit_rate={hr:.4f}{marker}")
    print(f"Best weights: {best_w}  (hit_rate={best_hr:.4f})")
    return best_w, best_hr


def main() -> None:
    t_start = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    mlb, ohe, ing, cat, nf, ns, ng = encode(df)

    best = compare_metrics(df, mlb, ohe, ing, cat, nf, ns)
    metric, k = str(best["metric"]), int(best["k"])
    print(f"\nBest metric/k: metric={metric}, k={k}, hit_rate={best['hit_rate']}")

    best_w, best_hr = sweep_weights(df, mlb, ohe, ing, cat, nf, ns, metric, k)

    final_matrix = build_matrix(ing, cat, nf, ns, best_w)
    config = {
        "metric": metric,
        "k": k,
        "weights": [float(x) for x in best_w],
        "n_family": nf,
        "n_subfamily": ns,
        "n_gender": ng,
        "hit_rate": float(best_hr),
    }

    joblib.dump(df, f"{MODELS_DIR}/opt_perfume_df.pkl")
    joblib.dump(mlb, f"{MODELS_DIR}/opt_mlb.pkl")
    joblib.dump(ohe, f"{MODELS_DIR}/opt_ohe.pkl")
    joblib.dump(config, f"{MODELS_DIR}/opt_config.pkl")
    save_npz(f"{MODELS_DIR}/opt_matrix.npz", final_matrix)

    print(f"\nSaved optimized model -> {MODELS_DIR}/opt_*.pkl + opt_matrix.npz")
    print("Config:", config)
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
