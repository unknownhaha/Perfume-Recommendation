# Perfume Recommendation

Content-based perfume recommender (~26k perfumes) using KNN + cosine similarity on scent features (ingredients, family, gender).

**Dataset:** [doevent/perfume](https://huggingface.co/datasets/doevent/perfume)

## Quick start

```bash
cd perfume_recommender
pip install -r requirements.txt
```

If `data/perfumes.json` is missing:

```bash
python -c "from src.data_loader import download_dataset; download_dataset()"
```

Model artifacts are included under `models/`. To rebuild from scratch, run notebooks `02_preprocessing.ipynb` then `03_recommendation_engine.ipynb`.

```bash
streamlit run src/app.py
```

Optional product images (~835 MB): use the sidebar in the app or `python src/perfume_images.py`.

## Project layout

All application code lives in [`perfume_recommender/`](perfume_recommender/). See [`perfume_recommender/AGENTS.md`](perfume_recommender/AGENTS.md) for architecture, artifact contracts, and safe change guidelines.
