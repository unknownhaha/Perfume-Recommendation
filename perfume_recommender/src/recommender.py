"""
Recommendation engine — loads fitted artifacts and exposes a clean recommend()
function. Used by both notebooks and the Streamlit app.

Optimized version:
- Hybrid relevance score = cosine similarity + Jaccard note overlap.
- Gender AND brand pre-filtering.
- Returns the notes that actually matched the user's query (explainability).
- No per-request KNN refit: scores are computed with a single sparse
  cosine_similarity call, which is fast at this catalog size.
- Small in-memory cache for repeated queries.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

import feature_engineering as fe

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Module-level cache — artifacts are loaded once and reused across calls
_cache: dict = {}
# Result cache keyed by query signature
_result_cache: dict = {}
_RESULT_CACHE_MAX = 256


def _load_models():
    if _cache:
        return _cache["df"], _cache["mlb"], _cache["ohe"], _cache["matrix"]
    _cache["df"] = joblib.load(f"{_MODELS_DIR}/perfume_df.pkl")
    _cache["mlb"] = joblib.load(f"{_MODELS_DIR}/mlb_ingredients.pkl")
    _cache["ohe"] = joblib.load(f"{_MODELS_DIR}/ohe_categories.pkl")
    best = joblib.load(f"{_MODELS_DIR}/best_approach.pkl")
    _cache["matrix"] = load_npz(f"{_MODELS_DIR}/matrix_{best}.npz")
    return _cache["df"], _cache["mlb"], _cache["ohe"], _cache["matrix"]


def get_options(df):
    """Return dropdown/multiselect options from the cleaned dataframe."""
    genders = sorted(df["gender"].unique().tolist())
    families = sorted(df["family"].unique().tolist())
    subfamilies = sorted(df["subfamily"].unique().tolist())
    brands = sorted(df["brand"].unique().tolist())

    from collections import Counter
    all_ingredients = []
    for ing_list in df["ingredients"]:
        all_ingredients.extend(ing_list)
    top_ingredients = [i for i, _ in Counter(all_ingredients).most_common(200)]

    return genders, families, subfamilies, brands, top_ingredients


def _cache_key(liked_ingredients, family, subfamily, gender, brand, n):
    return (
        tuple(sorted(str(i).title() for i in liked_ingredients)),
        str(family).upper(), str(subfamily).upper(),
        str(gender).upper(), str(brand).upper() if brand else None, int(n),
    )


def recommend(liked_ingredients, family="UNKNOWN", subfamily="UNKNOWN",
              gender="UNKNOWN", brand=None, description="", n=10):
    """
    Return top-N recommended perfumes as a DataFrame, ranked by a hybrid score.

    Parameters
    ----------
    liked_ingredients : list[str]  — notes the user likes (e.g. ["Rose", "Jasmine"])
    family            : str        — preferred scent family (e.g. "FLORAL")
    subfamily         : str        — preferred sub-family (e.g. "SOFT FLORAL")
    gender            : str        — "MALE" / "FEMALE" / "UNISEX" / "UNKNOWN"
    brand             : str | None — restrict results to a single brand
    description       : str        — kept for API compatibility (unused)
    n                 : int        — number of results to return

    Returns
    -------
    pd.DataFrame with columns: name_perfume, brand, [image_name], family,
        subfamily, gender, ingredients, matched_notes, similarity,
        hybrid_score, similarity_pct
    """
    key = _cache_key(liked_ingredients, family, subfamily, gender, brand, n)
    if key in _result_cache:
        return _result_cache[key].copy()

    df, mlb, ohe, matrix = _load_models()

    # Build filter mask (gender + optional brand)
    mask = pd.Series(True, index=df.index)
    if gender.upper() not in ("UNKNOWN", ""):
        mask &= (df["gender"] == gender.upper()) | (df["gender"] == "UNISEX")
    if brand:
        mask &= df["brand"] == str(brand).upper()

    filtered_df = df[mask].reset_index(drop=True)
    if len(filtered_df) == 0:
        return filtered_df
    filtered_matrix = matrix[mask.values]

    # 1. Cosine similarity (vector space) — single sparse pass, no refit
    query = fe.build_query_vector(
        liked_ingredients, family, subfamily, gender, mlb, ohe,
    )
    cos = cosine_similarity(query, filtered_matrix).ravel()

    # 2. Jaccard overlap on the candidate shortlist (cheap re-rank)
    query_notes = {str(i).title() for i in liked_ingredients}
    shortlist = np.argsort(cos)[::-1][: max(n * 5, 50)]

    jac = np.zeros(len(shortlist))
    matched = []
    for j, idx in enumerate(shortlist):
        perfume_notes = set(filtered_df.iloc[idx]["ingredients"])
        jac[j] = fe.jaccard(query_notes, perfume_notes)
        matched.append(sorted(query_notes & perfume_notes))

    hybrid = (fe.HYBRID_WEIGHTS["cosine"] * cos[shortlist]
              + fe.HYBRID_WEIGHTS["jaccard"] * jac)

    order = np.argsort(hybrid)[::-1][:n]
    chosen = shortlist[order]

    results = filtered_df.iloc[chosen].copy()
    results["matched_notes"] = [matched[o] for o in order]
    results["similarity"] = cos[chosen].round(4)
    results["hybrid_score"] = hybrid[order].round(4)
    results["similarity_pct"] = (results["hybrid_score"] * 100).round(1).astype(str) + "%"

    cols = [
        "name_perfume", "brand", "family", "subfamily", "gender",
        "ingredients", "matched_notes", "similarity", "hybrid_score",
        "similarity_pct",
    ]
    if "image_name" in results.columns:
        cols.insert(2, "image_name")
    results = results[cols].reset_index(drop=True)

    if len(_result_cache) >= _RESULT_CACHE_MAX:
        _result_cache.clear()
    _result_cache[key] = results.copy()
    return results
