"""
Configuration file for Dostoevsky Emotion & Sentiment Analysis Project.
Ensures reproducible seeds, dynamic device detection (CPU vs CUDA),
and standardized project directory paths.
"""

from pathlib import Path
import random
import numpy as np
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

# Data file paths
RAW_SENTENCES_FILE = BASE_DIR / "sentences.csv"
CLEANED_SENTENCES_FILE = DATA_DIR / "sentences_cleaned.csv"
METADATA_FILE = BASE_DIR / "metadata.csv"
TRANSLATOR_METADATA_FILE = DATA_DIR / "metadata_translators.csv"

# Reproducibility
SEED = 42

def set_seed(seed: int = SEED):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Auto-detect whether CUDA GPU is available or fallback to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU.")
    return device

# Batch size adjustments based on device
def get_batch_size(task: str = "inference") -> int:
    """Return device-adapted batch sizes."""
    if torch.cuda.is_available():
        return 32 if task == "training" else 64
    else:
        # Smaller batch size for CPU to avoid excessive memory and latency
        return 8 if task == "training" else 16
