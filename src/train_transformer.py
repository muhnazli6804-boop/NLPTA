"""
Optimized Transformer Fine-Tuning Module for 7 Ekman Emotion Classes on CPU.
Models: BERT (bert-base-uncased) & RoBERTa (roberta-base).
Classes: anger, disgust, fear, joy, sadness, surprise, neutral.

Optimizations for local CPU execution:
1. Dynamic batch padding (DataCollatorWithPadding) instead of fixed padding.
2. Multi-threaded CPU tensor execution (torch.set_num_threads).
3. Transfer learning with frozen lower layers (layers 0-7 frozen, top 4 layers + classifier trained).
4. Balanced class sampling and class-weighted CrossEntropyLoss.
5. Macro F1 & Accuracy checkpointing, saving best models and confusion matrices.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.config import DATA_DIR, RESULTS_DIR, SEED, get_device, set_seed

# CPU Thread Optimization
torch.set_num_threads(min(8, os.cpu_count() or 4))

EMOTIONS_7 = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
LABEL2ID = {label: i for i, label in enumerate(EMOTIONS_7)}
ID2LABEL = {i: label for i, label in enumerate(EMOTIONS_7)}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_balanced_data(
    train_samples_per_class: int = 300,
    val_samples_per_class: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample balanced stratified subsets from GoEmotions Ekman data."""
    set_seed(SEED)
    train_file = DATA_DIR / "goemotions_ekman_train.csv"
    val_file = DATA_DIR / "goemotions_ekman_validation.csv"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("GoEmotions processed CSVs not found. Run prepare_emotion_dataset.py first.")

    train_df_raw = pd.read_csv(train_file)
    val_df_raw = pd.read_csv(val_file)

    train_sampled = []
    val_sampled = []

    for emo in EMOTIONS_7:
        sub_train = train_df_raw[train_df_raw["label"] == emo]
        n_tr = min(len(sub_train), train_samples_per_class)
        train_sampled.append(sub_train.sample(n=n_tr, random_state=SEED))

        sub_val = val_df_raw[val_df_raw["label"] == emo]
        n_val = min(len(sub_val), val_samples_per_class)
        val_sampled.append(sub_val.sample(n=n_val, random_state=SEED))

    train_df = pd.concat(train_sampled).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat(val_sampled).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df["label_id"] = train_df["label"].map(LABEL2ID)
    val_df["label_id"] = val_df["label"].map(LABEL2ID)

    print(f"[Dataset] Balanced Train Samples: {len(train_df)} ({train_samples_per_class} per class)")
    print(f"[Dataset] Balanced Val Samples: {len(val_df)} ({val_samples_per_class} per class)")
    return train_df, val_df


def freeze_lower_layers(model, model_alias: str, unfreeze_layers: int = 4):
    """
    Freeze embeddings and lower encoder layers.
    Fine-tunes only the top `unfreeze_layers` encoder layers + classifier head.
    Significantly speeds up backprop on CPU while retaining rich features.
    """
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Unfreeze top transformer layers
    if model_alias == "bert":
        encoder_layers = model.bert.encoder.layer
        if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
            for param in model.bert.pooler.parameters():
                param.requires_grad = True
    elif model_alias == "roberta":
        encoder_layers = model.roberta.encoder.layer
    else:
        encoder_layers = []

    total_layers = len(encoder_layers)
    for i in range(total_layers - unfreeze_layers, total_layers):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{model_alias}] Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def train_single_model(
    model_name_or_path: str,
    model_alias: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
) -> Dict:
    set_seed(SEED)
    device = get_device()
    print(f"\n=======================================================")
    print(f"  Starting CPU-Optimized Fine-Tuning: {model_alias.upper()}")
    print(f"=======================================================")

    # 1. Load Data
    train_df, val_df = load_balanced_data(train_samples_per_class=300, val_samples_per_class=60)

    # Class weights for loss
    class_counts = train_df["label_id"].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (len(EMOTIONS_7) * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 2. Tokenizer and Model
    print(f"[{model_alias}] Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(EMOTIONS_7),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    
    # Freeze lower layers for CPU efficiency
    freeze_lower_layers(model, model_alias, unfreeze_layers=4)
    model.to(device)

    # 3. DataLoaders with Dynamic Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = TextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset = TextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

    # 4. Optimizer and Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # 5. Training Loop with Checkpointing
    best_macro_f1 = -1.0
    model_save_dir = RESULTS_DIR / "models" / f"{model_alias}_ekman"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_f1": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_alias}] Epoch {epoch}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"[{model_alias}] Epoch {epoch} Results -> Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        # Checkpoint best model
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            print(f"[{model_alias}] New best Macro F1 ({val_f1:.4f})! Saving model checkpoint to {model_save_dir}...")
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

    train_duration = round(time.time() - start_time, 2)
    print(f"[{model_alias}] Training finished in {train_duration} seconds.")

    # 6. Final Evaluation on Validation Set using Best Model
    print(f"[{model_alias}] Loading best checkpoint for final metrics calculation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(model_save_dir).to(device)
    best_model.eval()

    final_preds = []
    final_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(labels.cpu().numpy())

    final_acc = float(accuracy_score(final_targets, final_preds))
    final_f1 = float(f1_score(final_targets, final_preds, average="macro"))
    rep = classification_report(final_targets, final_preds, target_names=EMOTIONS_7, output_dict=True)
    cm = confusion_matrix(final_targets, final_preds)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS_7, yticklabels=EMOTIONS_7)
    plt.title(f"Confusion Matrix: {model_alias.upper()} (Macro F1: {final_f1:.3f})")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_alias}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    metrics = {
        "model_alias": model_alias,
        "base_model": model_name_or_path,
        "best_macro_f1": round(final_f1, 4),
        "accuracy": round(final_acc, 4),
        "training_time_seconds": train_duration,
        "classification_report": rep,
        "history": history,
        "saved_path": str(model_save_dir),
    }

    print(f"[{model_alias}] Complete. Best Macro F1: {final_f1:.4f}, Accuracy: {final_acc:.4f}\n")
    return metrics


def run_pipeline():
    all_metrics = {}
    
    # 1. BERT
    print("\n>>> PIPELINE STAGE 3.1: BERT Fine-Tuning <<<")
    bert_metrics = train_single_model(
        model_name_or_path="bert-base-uncased",
        model_alias="bert",
        epochs=3,
        batch_size=16,
        lr=5e-5,
    )
    all_metrics["bert"] = bert_metrics

    # 2. RoBERTa
    print("\n>>> PIPELINE STAGE 3.2: RoBERTa Fine-Tuning <<<")
    roberta_metrics = train_single_model(
        model_name_or_path="roberta-base",
        model_alias="roberta",
        epochs=3,
        batch_size=16,
        lr=5e-5,
    )
    all_metrics["roberta"] = roberta_metrics

    # Save summary metrics JSON
    metrics_file = RESULTS_DIR / "fine_tuning_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Metrics] All fine-tuning metrics saved to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "roberta", "both"], default="both")
    args = parser.parse_args()

    if args.model == "both":
        run_pipeline()
    elif args.model == "bert":
        m = train_single_model("bert-base-uncased", "bert", epochs=3, batch_size=16, lr=5e-5)
        with open(RESULTS_DIR / "fine_tuning_metrics_bert.json", "w") as f:
            json.dump(m, f, indent=2)
    elif args.model == "roberta":
        m = train_single_model("roberta-base", "roberta", epochs=3, batch_size=16, lr=5e-5)
        with open(RESULTS_DIR / "fine_tuning_metrics_roberta.json", "w") as f:
            json.dump(m, f, indent=2)
