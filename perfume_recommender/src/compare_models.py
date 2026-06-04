"""
Compare several similarity/ranking methods on the SAME offline protocol so we
can see which one is actually best for this content-based perfume recommender.

Why this exists
---------------
The production ranker uses a hybrid score (0.7*cosine + 0.3*jaccard). That blend
was a design choice, not the result of a sweep. This script puts every candidate
through identical evaluation so the comparison is apples-to-apples:

Methods compared
- cosine            : cosine similarity on the full weighted vector (production base)
- euclidean         : negative Euclidean distance on the same weighted vector
- dot               : raw dot product (linear kernel) on the weighted vector
- jaccard           : pure Jaccard overlap on the user's notes only
- hybrid a/b        : a*cosine + b*jaccard, for several (a, b) blends

Metrics (no user ratings required)
- LOO hit-rate@10   : hide a perfume, query with HALF its notes (+ its
                      family/subfamily/gender), check if it returns in top-10
- Precision@10 fam  : fraction of top-10 sharing the query's family
- Precision@10 notes: fraction of top-10 sharing >= 2 of the query's notes

All methods share the same random sample and seed, so differences are real.

Run:
    python src/compare_models.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import feature_engineering as fe

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEED = 42
SAMPLE = 300
K = 10

# Hybrid blends to try: (cosine_weight, jaccard_weight)
HYBRID_BLENDS = [
    (1.0, 0.0),   # cosine only (for reference)
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),   # current production
    (0.6, 0.4),
    (0.5, 0.5),
    (0.0, 1.0),   # jaccard only (for reference)
]


def _load():
    df = joblib.load(f"{MODELS_DIR}/perfume_df.pkl")
    mlb = joblib.load(f"{MODELS_DIR}/mlb_ingredients.pkl")
    ohe = joblib.load(f"{MODELS_DIR}/ohe_categories.pkl")
    best = joblib.load(f"{MODELS_DIR}/best_approach.pkl")
    matrix = load_npz(f"{MODELS_DIR}/matrix_{best}.npz").tocsr()
    return df, mlb, ohe, matrix


def build_binary_notes_matrix(df, mlb) -> csr_matrix:
    """Binary (0/1) note matrix aligned to df row order, using mlb vocabulary."""
    vocab_set = set(mlb.classes_)
    filtered = fe.filter_to_vocabulary(df["ingredients"], vocab_set)
    return mlb.transform(filtered).astype(np.float64).tocsr()


def jaccard_scores(query_note_set, bin_matrix, perfume_counts, mlb_classes_index):
    """Vectorised Jaccard of one query note set vs every perfume."""
    cols = [mlb_classes_index[n] for n in query_note_set if n in mlb_classes_index]
    q_count = len(cols)
    if q_count == 0:
        return np.zeros(bin_matrix.shape[0])
    q_vec = np.zeros((bin_matrix.shape[1], 1))
    q_vec[cols, 0] = 1.0
    inter = (bin_matrix @ q_vec).ravel()
    union = perfume_counts + q_count - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    return jac


# --- Score providers: each returns a score array (higher = more relevant) ----

def make_scorers(df, mlb, ohe, matrix, bin_matrix):
    perfume_counts = np.asarray(bin_matrix.sum(axis=1)).ravel()
    classes_index = {c: i for i, c in enumerate(mlb.classes_)}

    def query_vec(notes, row):
        return fe.build_query_vector(
            notes, row["family"], row["subfamily"], row["gender"], mlb, ohe
        )

    def cosine_score(notes, row):
        return cosine_similarity(query_vec(notes, row), matrix).ravel()

    def euclidean_score(notes, row):
        # smaller distance = more similar -> negate so higher is better
        return -euclidean_distances(query_vec(notes, row), matrix).ravel()

    def dot_score(notes, row):
        return (matrix @ query_vec(notes, row).T).toarray().ravel()

    def jaccard_score(notes, row):
        return jaccard_scores(
            {n.title() for n in notes}, bin_matrix, perfume_counts, classes_index
        )

    return {
        "cosine": cosine_score,
        "euclidean": euclidean_score,
        "dot": dot_score,
        "jaccard": jaccard_score,
    }, cosine_score, jaccard_score


# --- Evaluation protocol (identical sample for every method) -----------------

def evaluate_scorer(df, scorer, sample_idx, loo_idx):
    """Return (loo_hit_rate, precision_family, precision_notes) for one scorer."""
    # Precision@K: query with ALL of a perfume's notes, exclude itself
    fam_prec, note_prec = [], []
    for i in sample_idx:
        row = df.iloc[i]
        notes = set(row["ingredients"])
        scores = scorer(list(notes), row)
        top = [j for j in np.argsort(scores)[::-1] if j != i][:K]
        fam_prec.append(np.mean([df.iloc[j]["family"] == row["family"] for j in top]))
        note_prec.append(
            np.mean([len(notes & set(df.iloc[j]["ingredients"])) >= 2 for j in top])
        )

    # Leave-one-out hit-rate: query with HALF the notes, must return itself
    rng = np.random.default_rng(SEED)
    hits, evaluated = 0, 0
    for i in loo_idx:
        row = df.iloc[i]
        notes = list(row["ingredients"])
        if len(notes) < 2:
            continue
        evaluated += 1
        half = rng.choice(notes, size=max(1, len(notes) // 2), replace=False).tolist()
        scores = scorer(half, row)
        top = np.argsort(scores)[::-1][:K]
        if i in top:
            hits += 1

    loo = hits / evaluated if evaluated else 0.0
    return loo, float(np.mean(fam_prec)), float(np.mean(note_prec))


def main() -> None:
    df, mlb, ohe, matrix = _load()
    bin_matrix = build_binary_notes_matrix(df, mlb)
    scorers, cosine_score, jaccard_score = make_scorers(df, mlb, ohe, matrix, bin_matrix)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(df), size=min(SAMPLE, len(df)), replace=False)
    loo_idx = rng.choice(len(df), size=min(SAMPLE, len(df)), replace=False)

    print("=== Model / Metric Comparison ===")
    print(f"Catalog size : {len(df):,}")
    print(f"Sample size  : {SAMPLE} (seed={SEED}), K={K}")
    print()

    header = f"{'method':<16}{'LOO hit@10':>12}{'P@10 family':>14}{'P@10 notes':>13}{'time(s)':>10}"
    print(header)
    print("-" * len(header))

    rows = []

    # Single-metric methods
    for name, scorer in scorers.items():
        t0 = time.time()
        loo, fam, note = evaluate_scorer(df, scorer, sample_idx, loo_idx)
        dt = time.time() - t0
        rows.append((name, loo, fam, note))
        print(f"{name:<16}{loo:>12.4f}{fam:>14.4f}{note:>13.4f}{dt:>10.1f}")

    # Hybrid blends (cosine + jaccard) -- reuse cached score arrays per query
    for a, b in HYBRID_BLENDS:
        name = f"hybrid {a:.1f}/{b:.1f}"

        def hybrid_scorer(notes, row, a=a, b=b):
            return a * cosine_score(notes, row) + b * jaccard_score(notes, row)

        t0 = time.time()
        loo, fam, note = evaluate_scorer(df, hybrid_scorer, sample_idx, loo_idx)
        dt = time.time() - t0
        rows.append((name, loo, fam, note))
        marker = "  <- production" if (a, b) == (0.7, 0.3) else ""
        print(f"{name:<16}{loo:>12.4f}{fam:>14.4f}{note:>13.4f}{dt:>10.1f}{marker}")

    print()
    best_loo = max(rows, key=lambda r: r[1])
    best_fam = max(rows, key=lambda r: r[2])
    best_note = max(rows, key=lambda r: r[3])
    print(f"Best LOO hit-rate@10 : {best_loo[0]} ({best_loo[1]:.4f})")
    print(f"Best Precision@10 fam: {best_fam[0]} ({best_fam[2]:.4f})")
    print(f"Best Precision@10 not: {best_note[0]} ({best_note[3]:.4f})")


if __name__ == "__main__":
    main()
