"""
Dataset Preparation for GoEmotions Ekman 7-Class Mapping.
Maps 28 GoEmotions labels to 7 Ekman emotions:
anger, disgust, fear, joy, sadness, surprise, neutral.
Filters single-label / clear Ekman assignments and prepares
train and validation splits with fixed random seed.
"""

import sys
from collections import Counter
from pathlib import Path

BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import pandas as pd
from datasets import load_dataset
from src.config import DATA_DIR, SEED, set_seed

EMOTIONS_7 = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Ekman mapping according to Demszky et al. (2020)
# All 27 emotions + neutral mapped to Ekman 6 + neutral
GO_TO_EKMAN = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    
    "disgust": "disgust",
    
    "fear": "fear",
    "nervousness": "fear",
    
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "admiration": "joy",
    "desire": "joy",
    "caring": "joy",
    
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    
    "surprise": "surprise",
    "curiosity": "surprise",
    "confusion": "surprise",
    "realization": "surprise",
    
    "neutral": "neutral",
}


def prepare_ekman_dataset():
    set_seed(SEED)
    print("[Dataset] Loading GoEmotions (simplified) from Hugging Face...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    raw_feature_names = ds["train"].features["labels"].feature.names
    print(f"[Dataset] Raw GoEmotions classes ({len(raw_feature_names)}): {raw_feature_names}")

    splits = ["train", "validation", "test"]
    processed = {}

    for split in splits:
        split_data = ds[split]
        rows = []
        for item in split_data:
            text = item["text"]
            raw_labels = [raw_feature_names[i] for i in item["labels"]]
            
            # Map raw labels to Ekman
            ekman_labels = {GO_TO_EKMAN[l] for l in raw_labels if l in GO_TO_EKMAN}
            
            # Keep clean single-label Ekman instances to ensure clear gradient signal
            if len(ekman_labels) == 1:
                label = list(ekman_labels)[0]
                rows.append({"text": text, "label": label, "split": split})
                
        df = pd.DataFrame(rows)
        processed[split] = df
        print(f"[Dataset] Split '{split}' single Ekman label count: {len(df)}")
        print(f"  Distribution:\n{df['label'].value_counts()}\n")

    # Save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        out_path = DATA_DIR / f"goemotions_ekman_{split}.csv"
        processed[split].to_csv(out_path, index=False)
        print(f"[Dataset] Saved {out_path}")

    return processed


if __name__ == "__main__":
    prepare_ekman_dataset()
