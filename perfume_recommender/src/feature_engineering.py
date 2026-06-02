"""
Shared feature engineering — single source of truth for how perfumes and user
queries are turned into feature vectors.

Both `build_models.py` (offline) and `recommender.py` (inference) import from
here so the training matrix and the query vector can never drift apart.

Design notes:
- Ingredients are multi-hot encoded with a rare-note filter (drops the noisy
  long tail of notes seen in fewer than MIN_INGREDIENT_COUNT perfumes).
- Each feature group gets an EXPLICIT weight (see WEIGHTS) instead of the old
  raw `ingredients * 2`. Because the production metric is cosine similarity,
  these weights change the vector direction and therefore the ranking.
- No L2 normalization is applied on purpose: `metric="cosine"` already
  normalizes vectors internally, so an extra pass would be redundant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

# --- Tunable configuration -------------------------------------------------

# Notes appearing in fewer than this many perfumes are dropped from the
# vocabulary. At 394 raw notes this removes ~21% of the long tail while only
# 3 perfumes lose all of their notes.
MIN_INGREDIENT_COUNT = 5

# Relative importance of each feature group in the final vector. Ingredients
# dominate; family/sub-family/gender act as contextual tie-breakers. These
# values were chosen by the weight sweep in evaluate.py (best leave-one-out
# hit-rate@10 = 0.852).
WEIGHTS = {
    "ingredients": 1.00,
    "family": 0.50,
    "subfamily": 0.30,
    "gender": 0.20,
}

# Weight blend for the hybrid relevance score used at inference time.
HYBRID_WEIGHTS = {
    "cosine": 0.70,   # vector-space similarity
    "jaccard": 0.30,  # direct overlap of the notes the user picked
}


# --- Cleaning helpers ------------------------------------------------------

def clean_string(val, default: str = "UNKNOWN") -> str:
    if pd.isna(val) or str(val).strip() == "":
        return default
    return str(val).strip().upper()


def clean_ingredients(val) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(i).strip().title() for i in val if str(i).strip()]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise categoricals/ingredients and drop perfumes with no notes."""
    df = df.copy()
    df["family"] = df["family"].apply(lambda x: clean_string(x, "UNKNOWN"))
    df["subfamily"] = df["subfamily"].apply(lambda x: clean_string(x, "UNKNOWN"))
    df["gender"] = df["gender"].apply(lambda x: clean_string(x, "UNKNOWN"))
    df["brand"] = df["brand"].apply(lambda x: clean_string(x, "UNKNOWN"))
    df["ingredients"] = df["ingredients"].apply(clean_ingredients)
    df = df[df["ingredients"].apply(len) > 0].reset_index(drop=True)
    return df


def build_ingredient_vocabulary(df: pd.DataFrame,
                                min_count: int = MIN_INGREDIENT_COUNT) -> list[str]:
    """Return the sorted list of notes kept after the rare-note filter."""
    from collections import Counter

    counter: Counter = Counter()
    for notes in df["ingredients"]:
        counter.update(notes)
    vocab = sorted(note for note, c in counter.items() if c >= min_count)
    return vocab


def filter_to_vocabulary(ingredient_lists, vocab_set: set[str]):
    """Keep only notes that survived the vocabulary filter."""
    return [[n for n in notes if n in vocab_set] for notes in ingredient_lists]


# --- Vector assembly -------------------------------------------------------

def _category_blocks(ohe):
    """Return (start, end) column slices for family / subfamily / gender."""
    lengths = [len(cats) for cats in ohe.categories_]
    bounds = np.cumsum([0] + lengths)
    return {
        "family": (bounds[0], bounds[1]),
        "subfamily": (bounds[1], bounds[2]),
        "gender": (bounds[2], bounds[3]),
    }


def apply_category_weights(cat_matrix: csr_matrix, ohe) -> csr_matrix:
    """Scale each one-hot category block by its configured weight."""
    cat = cat_matrix.tocsc(copy=True).astype(np.float64)
    for name, (start, end) in _category_blocks(ohe).items():
        if end > start:
            cat[:, start:end] = cat[:, start:end] * WEIGHTS[name]
    return cat.tocsr()


def build_perfume_matrix(ingredient_matrix: csr_matrix,
                         cat_matrix: csr_matrix, ohe) -> csr_matrix:
    """Weighted concatenation used for the production feature matrix."""
    ing = ingredient_matrix.astype(np.float64) * WEIGHTS["ingredients"]
    cat = apply_category_weights(cat_matrix, ohe)
    return hstack([ing, cat]).tocsr()


def build_query_vector(liked_ingredients, family, subfamily, gender,
                       mlb, ohe) -> csr_matrix:
    """Encode a user query into the SAME weighted space as the catalog."""
    known = [n.title() for n in liked_ingredients if n.title() in set(mlb.classes_)]
    ing_vec = mlb.transform([known]).astype(np.float64) * WEIGHTS["ingredients"]

    cat_input = pd.DataFrame(
        [[str(family).upper(), str(subfamily).upper(), str(gender).upper()]],
        columns=["family", "subfamily", "gender"],
    )
    cat_vec = apply_category_weights(ohe.transform(cat_input), ohe)
    return hstack([ing_vec, cat_vec]).tocsr()


def jaccard(query_notes: set[str], perfume_notes: set[str]) -> float:
    """Jaccard similarity between two note sets (0..1)."""
    if not query_notes or not perfume_notes:
        return 0.0
    inter = len(query_notes & perfume_notes)
    union = len(query_notes | perfume_notes)
    return inter / union if union else 0.0
