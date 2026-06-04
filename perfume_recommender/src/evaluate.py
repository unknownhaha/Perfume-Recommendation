"""
Offline evaluation — KNN (cosine) at K=10, same scoring as recommender.py.

Metrics:
- Precision@10 (family)
- Precision@10 (>=2 shared notes)
- Leave-one-out hit-rate@10
- Light weight sweep (LOO@10 per weight candidate)

Run:
    python src/evaluate.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from scipy.sparse import load_npz

import feature_engineering as fe

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
K = fe.DEFAULT_K
RNG = np.random.default_rng(42)


def _load():
    df = joblib.load(f"{MODELS_DIR}/perfume_df.pkl")
    mlb = joblib.load(f"{MODELS_DIR}/mlb_ingredients.pkl")
    ohe = joblib.load(f"{MODELS_DIR}/ohe_categories.pkl")
    best = joblib.load(f"{MODELS_DIR}/best_approach.pkl")
    matrix = load_npz(f"{MODELS_DIR}/matrix_{best}.npz")
    return df, mlb, ohe, matrix


def precision_at_k(df, mlb, ohe, matrix, sample=300, k=K):
    idxs = RNG.choice(len(df), size=min(sample, len(df)), replace=False)
    fam_prec, note_prec = [], []
    for i in idxs:
        row = df.iloc[i]
        notes = set(row["ingredients"])
        query = fe.build_query_vector(
            list(notes), row["family"], row["subfamily"], row["gender"], mlb, ohe
        )
        top_idx, _ = fe.knn_cosine_topk(query, matrix, k=k, exclude_index=i)
        fam_prec.append(np.mean([df.iloc[j]["family"] == row["family"] for j in top_idx]))
        note_prec.append(
            np.mean([len(notes & set(df.iloc[j]["ingredients"])) >= 2 for j in top_idx])
        )
    return float(np.mean(fam_prec)), float(np.mean(note_prec))


def leave_one_out_hit_rate(df, mlb, ohe, matrix, sample=300, k=K):
    """Query with half of a perfume's notes; is the perfume back in top-K?"""
    idxs = RNG.choice(len(df), size=min(sample, len(df)), replace=False)
    hits = 0
    evaluated = 0
    for i in idxs:
        row = df.iloc[i]
        notes = list(row["ingredients"])
        if len(notes) < 2:
            continue
        evaluated += 1
        half = RNG.choice(notes, size=max(1, len(notes) // 2), replace=False).tolist()
        query = fe.build_query_vector(
            half, row["family"], row["subfamily"], row["gender"], mlb, ohe
        )
        top_idx, _ = fe.knn_cosine_topk(query, matrix, k=k)
        if i in top_idx:
            hits += 1
    return hits / evaluated if evaluated else 0.0


def weight_sweep(df, sample=200, k=K):
    """Light sweep over ingredient/family weights -> leave-one-out hit-rate."""
    from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

    vocab = fe.build_ingredient_vocabulary(df, fe.MIN_INGREDIENT_COUNT)
    vocab_set = set(vocab)
    ing_filtered = fe.filter_to_vocabulary(df["ingredients"], vocab_set)

    mlb = MultiLabelBinarizer(classes=vocab, sparse_output=True)
    ing_matrix = mlb.fit_transform(ing_filtered)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_matrix = ohe.fit_transform(df[["family", "subfamily", "gender"]])

    candidates = [
        {"ingredients": 1.0, "family": 1.0, "subfamily": 1.0, "gender": 1.0},
        {"ingredients": 1.0, "family": 0.5, "subfamily": 0.3, "gender": 0.2},
        {"ingredients": 2.0, "family": 0.3, "subfamily": 0.2, "gender": 0.15},
    ]

    original = dict(fe.WEIGHTS)
    results = []
    for w in candidates:
        fe.WEIGHTS.update(w)
        matrix = fe.build_perfume_matrix(ing_matrix, cat_matrix, ohe)
        hr = leave_one_out_hit_rate(df, mlb, ohe, matrix, sample=sample, k=k)
        results.append((w, hr))
    fe.WEIGHTS.update(original)
    return results


def main() -> None:
    df, mlb, ohe, matrix = _load()
    cfg = joblib.load(f"{MODELS_DIR}/feature_config.pkl") if os.path.exists(
        f"{MODELS_DIR}/feature_config.pkl") else {}

    print("=== KNN Cosine Recommender Evaluation (K=10) ===")
    print(f"Catalog size : {len(df):,}")
    print(f"Vocab size   : {cfg.get('vocab_size', len(mlb.classes_))}")
    print(f"Weights      : {cfg.get('weights', fe.WEIGHTS)}")
    print(f"Metric       : NearestNeighbors(metric='cosine')")
    print()

    fam_p, note_p = precision_at_k(df, mlb, ohe, matrix, sample=300, k=K)
    print(f"Precision@10 (family)        : {fam_p:.4f}")
    print(f"Precision@10 (>=2 notes)     : {note_p:.4f}")

    hr = leave_one_out_hit_rate(df, mlb, ohe, matrix, sample=300, k=K)
    print(f"Leave-one-out hit-rate@10    : {hr:.4f}")
    print()

    print("Light weight sweep (leave-one-out hit-rate@10):")
    for w, score in weight_sweep(df, sample=200, k=K):
        marker = "  <- current" if w == cfg.get("weights") else ""
        print(f"  ing={w['ingredients']}, fam={w['family']}, "
              f"sub={w['subfamily']}, gen={w['gender']}  ->  {score:.4f}{marker}")


if __name__ == "__main__":
    main()
