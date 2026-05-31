# Perfume Recommender — Agent / Developer Guide

Read this file before editing anything in `perfume_recommender/`. It describes how the system works, which files must stay in sync, and what changes commonly break production.

## What this project is

Content-based perfume recommendation (no user ratings). Users pick gender, scent family, and liked notes; the system returns top-N similar perfumes from ~26k items using **KNN + cosine similarity** on sparse feature vectors.

- **Dataset**: [doevent/perfume](https://huggingface.co/datasets/doevent/perfume) → `data/perfumes.json`
- **UI**: Streamlit — `src/app.py`
- **Inference API**: `recommend()` in `src/recommender.py`
- **Training / EDA**: Jupyter notebooks in `notebooks/`

There is **no rating column**. Do not add supervised rating prediction without new labeled data.

---

## Architecture (do not break this flow)

```mermaid
flowchart LR
    json["data/perfumes.json"]
    nb02["02_preprocessing.ipynb"]
    artifacts["models/*.pkl + matrix_*.npz"]
    nb03["03_recommendation_engine.ipynb"]
    rec["src/recommender.py"]
    app["src/app.py"]
    imgs["data/images/images/*.jpg"]

    json --> nb02 --> artifacts
    artifacts --> nb03
    artifacts --> rec
    rec --> app
    imgs --> app
```

**Runtime path (app / `recommend()`):**

1. Load encoders + `perfume_df.pkl` + `matrix_{approach}.npz` + `knn_model.pkl`
2. Optionally filter catalog by gender (includes UNISEX)
3. Build query vector with the **same encoders** as training
4. Fit/query `NearestNeighbors(metric='cosine', algorithm='brute')` on filtered matrix
5. Return rows with `similarity` and `similarity_pct`; optional image via `perfume_images`

---

## Directory layout

```
perfume_recommender/
├── AGENTS.md                 # This file
├── requirements.txt
├── data/
│   ├── perfumes.json         # Raw catalog (~26k rows) — required
│   ├── images.zip            # Optional HF download (~835 MB)
│   └── images/               # Extracted photos (gitignored)
│       └── images/           # IMPORTANT: jpg files live HERE (nested folder)
├── models/                   # Generated artifacts — must match preprocessing
│   ├── perfume_df.pkl        # Cleaned DataFrame for display + filters
│   ├── mlb_ingredients.pkl   # MultiLabelBinarizer (ingredients)
│   ├── ohe_categories.pkl    # OneHotEncoder (family, subfamily, gender)
│   ├── tfidf_description.pkl   # TfidfVectorizer (description) — used by approach C only
│   ├── best_approach.pkl     # Single char: "A" | "B" | "C" | "D"
│   ├── matrix_A.npz … matrix_D.npz
│   ├── knn_model.pkl         # Fitted on matrix for best_approach only
│   └── model_comparison.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb      # Builds models/ artifacts
│   ├── 03_recommendation_engine.ipynb
│   └── 04_evaluation.ipynb
└── src/
    ├── data_loader.py        # HF download + load perfumes.json
    ├── perfume_images.py     # Image download/resolve — see naming rule below
    ├── recommender.py          # Core recommend() — used by app and notebooks
    └── app.py                  # Streamlit UI
```

---

## Critical rules (read before any edit)

### 1. Never import a module named `image_utils`

The file is **`src/perfume_images.py`**. Streamlit ships its own `streamlit.elements.lib.image_utils`. Importing `from image_utils import …` causes `ImportError` or wrong module shadowing.

```python
# Correct
from perfume_images import load_image_for_display, images_are_ready
```

### 2. Encoder + matrix + `best_approach` must stay aligned

| If you change… | You must also… |
|----------------|----------------|
| `MultiLabelBinarizer` / ingredient cleaning | Re-run `02_preprocessing.ipynb` and `03_recommendation_engine.ipynb` |
| `OneHotEncoder` categories | Same |
| Feature approach (A/B/C/D) | Update `best_approach.pkl`, `knn_model.pkl`, and query builder in `recommender.py` |
| `perfume_df.pkl` columns | Update `recommend()` return columns and `app.py` display |

`_build_query_vector()` in `recommender.py` must mirror how matrices were built in `02_preprocessing.ipynb`:

| Approach | Feature vector |
|----------|----------------|
| A | `MultiLabelBinarizer(ingredients)` only |
| B | ingredients + `OneHotEncoder(family, subfamily, gender)` |
| C | B + `TfidfVectorizer(description)` (max_features=500) |
| D | `ingredients * 2` + categories (production winner) |

**Production approach is D** (saved in `models/best_approach.pkl`). Changing approach without retraining breaks recommendations.

### 3. Inference refits KNN on the gender-filtered subset

`recommender.recommend()` does **not** use the saved `knn_model.pkl` for queries. It loads the matrix and fits a fresh `NearestNeighbors` on the filtered subset for correct indices. The saved `knn_model.pkl` is for notebooks / full-catalog use. Do not assume `knn.kneighbors` on the global model matches filtered `DataFrame` row order.

### 4. Image paths are nested

After extracting `images.zip`, files are under:

`data/images/images/<image_name>.jpg`

`perfume_images._files_dir()` and `resolve_image_path()` handle this. Do not assume `data/images/<filename>.jpg` at the top level.

### 5. Data column contract

Raw JSON columns: `brand`, `name_perfume`, `family`, `subfamily`, `fragrances`, `ingredients` (list), `origin`, `gender`, `years`, `description`, `image_name`.

After cleaning in preprocessing (uppercase categoricals, title-case ingredients):

- `gender`: `MALE` | `FEMALE` | `UNISEX` | `UNKNOWN`
- `family` / `subfamily`: uppercase strings
- `ingredients`: list of title-case strings

`recommend()` returns: `name_perfume`, `brand`, `image_name` (if present), `family`, `subfamily`, `gender`, `ingredients`, `similarity`, `similarity_pct`.

### 6. User input in the app vs encoding

Streamlit passes **title-case** notes (e.g. `"Rose"`, `"Jasmine"`). `MultiLabelBinarizer` was fit on **title-case** ingredients after `clean_ingredients()`. Family/gender from UI are `.upper()` before encoding. Keep that consistent if you change the UI.

### 7. Large files are gitignored

Do not commit `data/images/`, `data/images.zip`, or HuggingFace `data/.cache/`. `models/perfume_df.pkl` is large (~15 MB) but required locally for the app.

---

## Safe vs risky changes

| Safer | Risky (requires full pipeline re-run) |
|-------|--------------------------------------|
| UI text, layout, styling in `app.py` | Changing encoding logic in only one of notebook / `recommender.py` |
| More results (`n` slider max) | Renaming model files without updating loaders |
| Evaluation metrics in `04_evaluation.ipynb` | Changing `clean_*` rules without rebuilding `perfume_df.pkl` |
| New sidebar filters that only subset `df` before KNN | Switching from approach D without updating `best_approach.pkl` |
| Adding columns to display only | Removing `image_name` from `perfume_df.pkl` |

---

## How to run

```bash
cd perfume_recommender
pip install -r requirements.txt
```

**First-time data:**

```bash
python -c "from src.data_loader import download_dataset; download_dataset()"
```

**Build model artifacts (required before app):**

Run in order: `notebooks/02_preprocessing.ipynb` → `notebooks/03_recommendation_engine.ipynb`

**App:**

```bash
streamlit run src/app.py
```

**Optional images (~835 MB):** Sidebar button in app, or:

```bash
python src/perfume_images.py
```

Only one Streamlit instance on port 8501 at a time (duplicate processes caused stale code during debugging).

---

## Rebuilding artifacts from scratch

If preprocessing logic changes:

1. Run `02_preprocessing.ipynb` → regenerates `perfume_df.pkl`, encoders, `matrix_*.npz`
2. Run `03_recommendation_engine.ipynb` → picks best approach, saves `knn_model.pkl`, `best_approach.pkl`
3. Restart Streamlit (clears `@st.cache_resource` for `load_options`)
4. Smoke test: `python -c "import sys; sys.path.insert(0,'src'); from recommender import recommend; print(recommend(['Rose'], family='FLORAL', n=3).columns)"`

---

## Tech stack (do not add unless necessary)

- Python 3.10+
- `scikit-learn` — `MultiLabelBinarizer`, `OneHotEncoder`, `NearestNeighbors` (cosine, brute)
- `scipy.sparse` — feature matrices
- `joblib` — persistence
- `streamlit` — UI
- `Pillow` — image display
- `huggingface_hub` — dataset + images download

**Not used in v1:** Elasticsearch, sentence-transformers, PyTorch/TensorFlow, user rating models.

At ~26k items, full-matrix KNN in RAM is fine (<5 ms per query). No vector DB required.

---

## Common pitfalls (from past bugs)

1. **ImportError `load_image_for_display`** — Wrong module name `image_utils`; use `perfume_images`.
2. **Photos always “No photo”** — Images not extracted, or lookup path missing nested `images/images/`.
3. **Stale Streamlit** — Multiple processes on :8501; kill all `streamlit`/`python` listeners before restart.
4. **Query vector mismatch** — User likes `["Oud"]` but note not in `mlb.classes_` → silent zero vector for unknown labels (sklearn warning). Expected for OOV notes.

---

## Minimal smoke tests after edits

```bash
cd perfume_recommender
python -c "import sys; sys.path.insert(0,'src'); from recommender import recommend; r=recommend(['Rose','Jasmine'], family='FLORAL', gender='FEMALE', n=3); assert 'similarity' in r.columns; print('OK', len(r))"
python -c "import sys; sys.path.insert(0,'src'); from perfume_images import images_are_ready, resolve_image_path; import joblib; df=joblib.load('models/perfume_df.pkl'); p=resolve_image_path(df['image_name'].iloc[0]); print('images ready', images_are_ready(), 'sample', bool(p))"
python -m py_compile src/app.py src/recommender.py src/perfume_images.py src/data_loader.py
```

---

## When extending the project

- **New filters (brand, origin):** Filter `df` and `matrix` with the same boolean mask before `NearestNeighbors.fit`.
- **Free-text search:** New approach E; re-run preprocessing + comparison; update `_build_query_vector`.
- **Different dataset file:** Update `data_loader.py` filename (`perfumes.json` on HF, not `perfume.json`).

Keep this file updated when you change architecture, artifact names, or the production approach letter.
