"""
Robust Model Downloader for Offline PyTorch Transformer Execution.
Downloads base architectures (bert-base-uncased, roberta-base) directly
using HTTP chunk streaming to bypass Windows symlink and tempfile locking issues.
"""

from pathlib import Path
import requests
from tqdm import tqdm

from src.config import BASE_DIR

MODELS_DIR = BASE_DIR / "models_base"

MODEL_FILES = {
    "bert-base-uncased": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
    ],
    "roberta-base": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ],
}


def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[Downloader] Already exists: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    print(f"[Downloader] Downloading {dest_path.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))

    with open(temp_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    temp_path.replace(dest_path)
    print(f"[Downloader] Finished: {dest_path.name}")


def ensure_models():
    for model_name, files in MODEL_FILES.items():
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Downloader] Checking model assets for {model_name} in {model_dir}...")
        for fname in files:
            url = f"https://huggingface.co/{model_name}/resolve/main/{fname}"
            dest = model_dir / fname
            download_file(url, dest)


if __name__ == "__main__":
    ensure_models()
