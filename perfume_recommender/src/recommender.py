"""
Recommendation engine — loads fitted models and exposes a clean recommend() function.
Used by both notebooks and the Streamlit app.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz
from sklearn.neighbors import NearestNeighbors

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Module-level cache — models are loaded once and reused across calls
_cache: dict = {}


def _load_models():
    if _cache:
        return (
            _cache["df"], _cache["mlb"], _cache["ohe"], _cache["tfidf"],
            _cache["best_approach"], _cache["matrix"], _cache["knn"],
        )
    _cache["df"]            = joblib.load(f"{_MODELS_DIR}/perfume_df.pkl")
    _cache["mlb"]           = joblib.load(f"{_MODELS_DIR}/mlb_ingredients.pkl")
    _cache["ohe"]           = joblib.load(f"{_MODELS_DIR}/ohe_categories.pkl")
    _cache["tfidf"]         = joblib.load(f"{_MODELS_DIR}/tfidf_description.pkl")
    _cache["best_approach"] = joblib.load(f"{_MODELS_DIR}/best_approach.pkl")
    _cache["matrix"]        = load_npz(f"{_MODELS_DIR}/matrix_{_cache['best_approach']}.npz")
    _cache["knn"]           = joblib.load(f"{_MODELS_DIR}/knn_model.pkl")
    return (
        _cache["df"], _cache["mlb"], _cache["ohe"], _cache["tfidf"],
        _cache["best_approach"], _cache["matrix"], _cache["knn"],
    )


def _build_query_vector(mlb, ohe, tfidf, approach, liked_ingredients,
                        family, subfamily, gender, description=""):
    ing_vec = mlb.transform([liked_ingredients])

    if approach == "A":
        return ing_vec

    cat_input = pd.DataFrame(
        [[family.upper(), subfamily.upper(), gender.upper()]],
        columns=["family", "subfamily", "gender"],
    )
    cat_vec = ohe.transform(cat_input)

    if approach == "B":
        return hstack([ing_vec, cat_vec])

    if approach == "C":
        desc_vec = tfidf.transform([description])
        return hstack([ing_vec, cat_vec, desc_vec])

    if approach == "D":
        return hstack([ing_vec * 2, cat_vec])

    raise ValueError(f"Unknown approach: {approach}")


def get_options(df):
    """Return dropdown/multiselect options from the cleaned dataframe."""
    genders     = sorted(df["gender"].unique().tolist())
    families    = sorted(df["family"].unique().tolist())
    subfamilies = sorted(df["subfamily"].unique().tolist())
    brands      = sorted(df["brand"].unique().tolist())

    from collections import Counter
    all_ingredients = []
    for ing_list in df["ingredients"]:
        all_ingredients.extend(ing_list)
    top_ingredients = [i for i, _ in Counter(all_ingredients).most_common(200)]

    return genders, families, subfamilies, brands, top_ingredients


def recommend(liked_ingredients, family="UNKNOWN", subfamily="UNKNOWN",
              gender="UNKNOWN", description="", n=10):
    """
    Return top-N recommended perfumes as a DataFrame.

    Parameters
    ----------
    liked_ingredients : list[str]  — notes the user likes (e.g. ["Rose", "Jasmine"])
    family            : str        — preferred scent family (e.g. "FLORAL")
    subfamily         : str        — preferred sub-family (e.g. "SOFT FLORAL")
    gender            : str        — "MALE" / "FEMALE" / "UNISEX" / "UNKNOWN"
    description       : str        — optional free-text hint (used only by approach C)
    n                 : int        — number of results to return

    Returns
    -------
    pd.DataFrame with columns: name_perfume, brand, family, subfamily, gender,
                                ingredients, similarity
    """
    df, mlb, ohe, tfidf, best_approach, matrix, _ = _load_models()

    # Gender pre-filter
    if gender.upper() not in ("UNKNOWN", ""):
        mask = (df["gender"] == gender.upper()) | (df["gender"] == "UNISEX")
    else:
        mask = pd.Series([True] * len(df))

    filtered_df     = df[mask].reset_index(drop=True)
    filtered_matrix = matrix[mask.values]

    k = min(n + 1, len(filtered_df))
    knn_local = NearestNeighbors(n_neighbors=k, metric="cosine",
                                  algorithm="brute", n_jobs=-1)
    knn_local.fit(filtered_matrix)

    query = _build_query_vector(
        mlb, ohe, tfidf, best_approach,
        liked_ingredients, family, subfamily, gender, description,
    )
    distances, indices = knn_local.kneighbors(query, n_neighbors=k)

    results = filtered_df.iloc[indices[0]].copy()
    results["similarity"] = (1 - distances[0]).round(4)
    results["similarity_pct"] = (results["similarity"] * 100).round(1).astype(str) + "%"
    cols = [
        "name_perfume", "brand", "family", "subfamily",
        "gender", "ingredients", "similarity", "similarity_pct",
    ]
    if "image_name" in results.columns:
        cols.insert(2, "image_name")
    return results[cols].head(n)
