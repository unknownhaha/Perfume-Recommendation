"""
Offline evaluation for the hybrid content-based recommender.

Metrics (no user ratings required):
- Precision@K (family)      : fraction of top-K sharing the query's family
- Precision@K (notes)       : fraction of top-K sharing >= 2 query notes
- Leave-one-out hit-rate@K  : hide a perfume, query with HALF of its notes,
                              check whether the held-out perfume returns in top-K
- Light weight sweep        : try a few ingredient/category weight blends and
                              report the leave-one-out hit-rate for each

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
from sklearn.metrics.pairwise import cosine_similarity

import feature_engineering as fe

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RNG = np.random.default_rng(42)


def _load():
    df = joblib.load(f"{MODELS_DIR}/perfume_df.pkl")
    mlb = joblib.load(f"{MODELS_DIR}/mlb_ingredients.pkl")
    ohe = joblib.load(f"{MODELS_DIR}/ohe_categories.pkl")
    best = joblib.load(f"{MODELS_DIR}/best_approach.pkl")
    matrix = load_npz(f"{MODELS_DIR}/matrix_{best}.npz")
    return df, mlb, ohe, matrix


def precision_at_k(df, mlb, ohe, matrix, sample=300, k=10):
    idxs = RNG.choice(len(df), size=min(sample, len(df)), replace=False)
    fam_prec, note_prec = [], []
    for i in idxs:
        row = df.iloc[i]
        notes = set(row["ingredients"])
        query = fe.build_query_vector(
            list(notes), row["family"], row["subfamily"], row["gender"], mlb, ohe
        )
        cos = cosine_similarity(query, matrix).ravel()
        top = [j for j in np.argsort(cos)[::-1] if j != i][:k]
        fam_prec.append(np.mean([df.iloc[j]["family"] == row["family"] for j in top]))
        note_prec.append(
            np.mean([len(notes & set(df.iloc[j]["ingredients"])) >= 2 for j in top])
        )
    return float(np.mean(fam_prec)), float(np.mean(note_prec))


def leave_one_out_hit_rate(df, mlb, ohe, matrix, sample=300, k=10):
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
        cos = cosine_similarity(query, matrix).ravel()
        top = np.argsort(cos)[::-1][:k]
        if i in top:
            hits += 1
    return hits / evaluated if evaluated else 0.0


def weight_sweep(df, sample=200, k=10):
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
        {"ingredients": 1.0, "family": 0.2, "subfamily": 0.1, "gender": 0.1},
        {"ingredients": 1.0, "family": 0.3, "subfamily": 0.2, "gender": 0.15},
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

    print("=== Hybrid Recommender Evaluation ===")
    print(f"Catalog size : {len(df):,}")
    print(f"Vocab size   : {cfg.get('vocab_size', len(mlb.classes_))}")
    print(f"Weights      : {cfg.get('weights', fe.WEIGHTS)}")
    print(f"Hybrid blend : {cfg.get('hybrid', fe.HYBRID_WEIGHTS)}")
    print()

    fam_p, note_p = precision_at_k(df, mlb, ohe, matrix, sample=300, k=10)
    print(f"Precision@10 (family)        : {fam_p:.4f}")
    print(f"Precision@10 (>=2 notes)     : {note_p:.4f}")

    hr = leave_one_out_hit_rate(df, mlb, ohe, matrix, sample=300, k=10)
    print(f"Leave-one-out hit-rate@10    : {hr:.4f}")
    print()

    print("Light weight sweep (leave-one-out hit-rate@10):")
    for w, score in weight_sweep(df, sample=200, k=10):
        marker = "  <- current" if w == cfg.get("weights") else ""
        print(f"  ing={w['ingredients']}, fam={w['family']}, "
              f"sub={w['subfamily']}, gen={w['gender']}  ->  {score:.4f}{marker}")


if __name__ == "__main__":
    main()
