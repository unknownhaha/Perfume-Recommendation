"""
Recommendation engine — KNN with cosine similarity (no Jaccard ranking).

Loads fitted artifacts and exposes recommend() for notebooks and Streamlit.
Fits NearestNeighbors(metric='cosine') on the gender/brand-filtered catalog
subset per query (same pattern as 03_recommendation_engine.ipynb).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import feature_engineering as fe

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_K = fe.DEFAULT_K

_cache: dict = {}
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
              gender="UNKNOWN", brand=None, description="", n=DEFAULT_K):
    """
    Return top-N recommended perfumes ranked by KNN cosine similarity.

    Parameters
    ----------
    liked_ingredients : list[str]
    family, subfamily, gender : str
    brand : str | None — optional brand filter
    description : str — unused (API compatibility)
    n : int — number of neighbors (default 10)

    Returns
    -------
    pd.DataFrame with similarity, similarity_pct, matched_notes (explainability only)
    """
    n = min(int(n), DEFAULT_K) if n else DEFAULT_K
    n = max(1, n)

    key = _cache_key(liked_ingredients, family, subfamily, gender, brand, n)
    if key in _result_cache:
        return _result_cache[key].copy()

    df, mlb, ohe, matrix = _load_models()

    mask = pd.Series(True, index=df.index)
    if gender.upper() not in ("UNKNOWN", ""):
        mask &= (df["gender"] == gender.upper()) | (df["gender"] == "UNISEX")
    if brand:
        mask &= df["brand"] == str(brand).upper()

    filtered_df = df[mask].reset_index(drop=True)
    if len(filtered_df) == 0:
        return filtered_df

    filtered_matrix = matrix[mask.values]
    query = fe.build_query_vector(
        liked_ingredients, family, subfamily, gender, mlb, ohe,
    )

    indices, sims = fe.knn_cosine_topk(query, filtered_matrix, k=n)
    if len(indices) == 0:
        return filtered_df.iloc[0:0].copy()

    query_notes = {str(i).title() for i in liked_ingredients}
    results = filtered_df.iloc[indices].copy()
    matched = []
    for idx in indices:
        perfume_notes = set(filtered_df.iloc[idx]["ingredients"])
        matched.append(sorted(query_notes & perfume_notes))

    results["matched_notes"] = matched
    results["similarity"] = np.round(sims, 4)
    results["hybrid_score"] = results["similarity"]
    results["similarity_pct"] = (results["similarity"] * 100).round(1).astype(str) + "%"

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
