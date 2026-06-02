"""
Rebuild all model artifacts for the (optimized) production recommender.

This is the canonical, runnable replacement for hand-running
`02_preprocessing.ipynb`. It uses `feature_engineering.py` so the saved matrix
and the inference query vector are guaranteed to share the same encoding,
weights, and vocabulary.

Run:
    python src/build_models.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

import feature_engineering as fe
from data_loader import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load + clean ------------------------------------------------------
    df = load_data()
    print(f"Loaded {len(df):,} perfumes")
    df = fe.clean_dataframe(df)
    print(f"After cleaning (notes present): {len(df):,}")

    # 2. Rare-note vocabulary filter --------------------------------------
    vocab = fe.build_ingredient_vocabulary(df, fe.MIN_INGREDIENT_COUNT)
    vocab_set = set(vocab)
    print(f"Ingredient vocabulary: {len(vocab):,} notes "
          f"(min_count={fe.MIN_INGREDIENT_COUNT})")
    df["ingredients_filtered"] = fe.filter_to_vocabulary(df["ingredients"], vocab_set)

    # 3. Encoders ----------------------------------------------------------
    mlb = MultiLabelBinarizer(classes=vocab, sparse_output=True)
    ingredient_matrix = mlb.fit_transform(df["ingredients_filtered"])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_matrix = ohe.fit_transform(df[["family", "subfamily", "gender"]])

    # 4. Weighted production matrix ---------------------------------------
    matrix = fe.build_perfume_matrix(ingredient_matrix, cat_matrix, ohe)
    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Production matrix: shape={matrix.shape}, nnz={matrix.nnz:,}, "
          f"sparsity={sparsity * 100:.2f}%")

    # 5. Persist -----------------------------------------------------------
    joblib.dump(df, f"{MODELS_DIR}/perfume_df.pkl")
    joblib.dump(mlb, f"{MODELS_DIR}/mlb_ingredients.pkl")
    joblib.dump(ohe, f"{MODELS_DIR}/ohe_categories.pkl")
    joblib.dump("HYBRID", f"{MODELS_DIR}/best_approach.pkl")
    joblib.dump(
        {"weights": fe.WEIGHTS, "hybrid": fe.HYBRID_WEIGHTS,
         "min_ingredient_count": fe.MIN_INGREDIENT_COUNT,
         "vocab_size": len(vocab)},
        f"{MODELS_DIR}/feature_config.pkl",
    )
    save_npz(f"{MODELS_DIR}/matrix_HYBRID.npz", matrix)

    print("Saved: perfume_df.pkl, mlb_ingredients.pkl, ohe_categories.pkl, "
          "best_approach.pkl, feature_config.pkl, matrix_HYBRID.npz")
    print("Done.")


if __name__ == "__main__":
    main()
