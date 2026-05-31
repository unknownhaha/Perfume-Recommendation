"""
Perfume Recommender — Streamlit Web App
Run with: streamlit run src/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import streamlit as st
from perfume_images import (
    download_and_extract_images,
    images_are_ready,
    load_image_for_display,
    resolve_image_path,
)
from recommender import recommend, get_options

st.set_page_config(
    page_title="Perfume Recommender",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_resource(show_spinner="Loading perfume catalog...")
def load_options():
    df = joblib.load(os.path.join(MODELS_DIR, "perfume_df.pkl"))
    return get_options(df)


try:
    genders, families, subfamilies, brands, top_ingredients = load_options()
    models_ready = True
except Exception:
    models_ready = False

with st.sidebar:
    st.title("🌸 Perfume Recommender")
    st.caption("Find your perfect scent using Content-Based KNN matching.")
    st.divider()

    if not models_ready:
        st.error(
            "Models not found. Please run the notebooks first:\n\n"
            "1. `02_preprocessing.ipynb`\n"
            "2. `03_recommendation_engine.ipynb`"
        )
    else:
        st.subheader("Your Preferences")

        gender = st.radio(
            "Gender",
            options=["Any"] + [g.title() for g in genders if g != "UNKNOWN"],
            horizontal=True,
        )

        family = st.selectbox(
            "Scent Family",
            options=["Any"] + [f.title() for f in families if f != "UNKNOWN"],
        )

        subfamily = st.selectbox(
            "Sub-Family (optional)",
            options=["Any"] + [s.title() for s in subfamilies if s != "UNKNOWN"],
        )

        liked_notes = st.multiselect(
            "Scent Notes You Like",
            options=[i.title() for i in top_ingredients],
            default=["Rose", "Jasmine"] if "Rose" in top_ingredients else top_ingredients[:2],
            help="Pick the notes (ingredients) you enjoy most.",
        )

        n_results = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

        st.divider()
        st.caption("Perfume photos (optional, ~835 MB one-time download)")
        if images_are_ready():
            st.success("Images ready")
        else:
            if st.button("Download perfume images", use_container_width=True):
                with st.spinner("Downloading images.zip from HuggingFace… This may take several minutes."):
                    try:
                        download_and_extract_images()
                        st.success("Images ready. Click **Find Perfumes** again to see photos.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")

        st.divider()
        search_btn = st.button("🔍 Find Perfumes", type="primary", use_container_width=True)

st.title("🌸 Perfume Recommendation System")
st.caption("Content-Based Filtering · KNN Cosine Similarity · 26k+ perfumes")

if not models_ready:
    st.warning(
        "The trained models are not ready yet. "
        "Please run `02_preprocessing.ipynb` and `03_recommendation_engine.ipynb` first."
    )
    st.stop()

if not search_btn:
    st.info("Configure your preferences in the sidebar and click **Find Perfumes**.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Perfumes", "26,000+")
    col2.metric("Unique Ingredients", "1,000+")
    col3.metric("Algorithm", "KNN Cosine")
    st.stop()

if not liked_notes:
    st.warning("Please select at least one scent note from the sidebar.")
    st.stop()

with st.spinner("Finding your perfect perfumes..."):
    results = recommend(
        liked_ingredients=liked_notes,
        family=family.upper() if family != "Any" else "UNKNOWN",
        subfamily=subfamily.upper() if subfamily != "Any" else "UNKNOWN",
        gender=gender.upper() if gender != "Any" else "UNKNOWN",
        n=n_results,
    )

st.success(f"Top {len(results)} perfumes matching your preferences")
st.subheader("Recommended Perfumes")

for rank, (_, row) in enumerate(results.iterrows(), start=1):
    with st.container():
        col_rank, col_img, col_info, col_score = st.columns([0.4, 1.2, 5.4, 1.5])

        with col_rank:
            st.markdown(f"### {rank}")

        with col_img:
            img_name = row.get("image_name") if "image_name" in row.index else None
            pil_img = load_image_for_display(img_name) if img_name else None
            img_path = resolve_image_path(img_name) if img_name else None

            if pil_img is not None:
                try:
                    st.image(pil_img, width=120)
                except Exception:
                    if img_path and os.path.isfile(img_path):
                        st.image(img_path, width=120)
                    else:
                        st.caption("Image load error")
            elif img_path and os.path.isfile(img_path):
                st.image(img_path, width=120)
            else:
                st.markdown(
                    "<div style='width:100%;aspect-ratio:1;background:#f0f2f6;"
                    "border-radius:8px;display:flex;align-items:center;"
                    "justify-content:center;color:#888;font-size:0.75rem;'>"
                    "No photo</div>",
                    unsafe_allow_html=True,
                )

        with col_info:
            st.markdown(f"**{row['name_perfume']}** — *{row['brand']}*")
            tags = f"`{row['family'].title()}` `{row['gender'].title()}`"
            if row["subfamily"] and row["subfamily"] != "UNKNOWN":
                tags += f" `{row['subfamily'].title()}`"
            st.markdown(tags)
            notes = ", ".join(row["ingredients"][:8])
            if len(row["ingredients"]) > 8:
                notes += f" +{len(row['ingredients']) - 8} more"
            st.caption(f"Notes: {notes}")

        with col_score:
            sim_val = float(row["similarity"])
            color = "green" if sim_val >= 0.7 else "orange" if sim_val >= 0.4 else "red"
            st.markdown(
                f"<div style='text-align:center; padding:8px; border-radius:8px;"
                f"background:#f0f2f6; font-size:1.2rem; font-weight:bold; color:{color}'>"
                f"{row['similarity_pct']}</div>",
                unsafe_allow_html=True,
            )
            st.caption("match")

        st.divider()

with st.expander("View as table"):
    display_df = results[["name_perfume", "brand", "family", "gender", "similarity_pct"]].copy()
    display_df.columns = ["Perfume", "Brand", "Family", "Gender", "Match"]
    st.dataframe(display_df, width="stretch", hide_index=True)

st.divider()
st.caption("Dataset: doevent/perfume · Model: KNN (cosine) · Built with Streamlit")
