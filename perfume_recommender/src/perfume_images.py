"""
Load perfume bottle images from the HuggingFace dataset archive (images.zip).
Images are optional — the app works without them using a placeholder.
"""
import os
import zipfile
from huggingface_hub import hf_hub_download

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.abspath(os.path.join(DATA_DIR, "images"))
# HuggingFace zip extracts into a nested `images/` subfolder
IMAGES_FILES_DIR = os.path.join(IMAGES_DIR, "images")
EXTRACT_MARKER = os.path.join(IMAGES_DIR, ".extracted")
ZIP_FILENAME = "images.zip"

_ready_cache: bool | None = None


def _files_dir() -> str:
    """Directory that actually contains .jpg files after extract."""
    if os.path.isdir(IMAGES_FILES_DIR):
        for f in os.listdir(IMAGES_FILES_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return IMAGES_FILES_DIR
    return IMAGES_DIR


def images_are_ready() -> bool:
    """True if images were extracted and at least one .jpg exists."""
    global _ready_cache
    if _ready_cache is not None:
        return _ready_cache

    if not os.path.isdir(IMAGES_DIR):
        _ready_cache = False
        return False

    files_root = _files_dir()
    jpg_count = 0
    if os.path.isdir(files_root):
        for f in os.listdir(files_root):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                jpg_count += 1
                if jpg_count >= 10:
                    break

    _ready_cache = jpg_count >= 10
    return _ready_cache


def _invalidate_ready_cache():
    global _ready_cache
    _ready_cache = None


def download_and_extract_images() -> str:
    """
    Download images.zip from HuggingFace and extract to data/images/.
    Returns the images directory path. (~835 MB download, one-time.)
    """
    _invalidate_ready_cache()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    if images_are_ready() and os.path.isfile(EXTRACT_MARKER):
        return IMAGES_DIR

    zip_path = os.path.join(DATA_DIR, ZIP_FILENAME)
    if not os.path.isfile(zip_path):
        zip_path = hf_hub_download(
            repo_id="doevent/perfume",
            repo_type="dataset",
            filename=ZIP_FILENAME,
            local_dir=DATA_DIR,
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(IMAGES_DIR)

    with open(EXTRACT_MARKER, "w", encoding="utf-8") as f:
        f.write("ok")
    _invalidate_ready_cache()
    return IMAGES_DIR


def resolve_image_path(image_name) -> str | None:
    """Return local file path for image_name, or None if missing."""
    if image_name is None or (isinstance(image_name, float) and str(image_name) == "nan"):
        return None
    name = str(image_name).strip()
    if not name:
        return None

    if not images_are_ready():
        return None

    files_root = _files_dir()
    for base in (files_root, IMAGES_DIR):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return None


def load_image_for_display(image_name):
    """Return a PIL Image for Streamlit, or None if unavailable."""
    path = resolve_image_path(image_name)
    if not path:
        return None
    try:
        from PIL import Image

        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


if __name__ == "__main__":
    print("Downloading and extracting images (~835 MB)...")
    path = download_and_extract_images()
    print(f"Done. Images folder: {path}")
