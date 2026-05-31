import os
import json
from huggingface_hub import hf_hub_download

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset():
    print("Downloading perfume dataset from HuggingFace...")
    json_path = hf_hub_download(
        repo_id="doevent/perfume",
        repo_type="dataset",
        filename="perfumes.json",
        local_dir=DATA_DIR,
    )
    print(f"Downloaded to: {json_path}")
    return json_path

def load_data(json_path=None):
    if json_path is None:
        json_path = os.path.join(DATA_DIR, "perfumes.json")
    if not os.path.exists(json_path):
        json_path = download_dataset()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    import pandas as pd
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    download_dataset()
