"""
Optimized Transformer Corpus Inference Module for 7 Ekman Emotion Classes.
Performs batch inference on Dostoevsky sentences using fine-tuned models
(RoBERTa and BERT).

Optimizations for fast CPU execution:
1. Length-based bucketing (sort by length before batching, dynamic padding, restore original order).
2. Multi-threaded CPU tensor execution (torch.set_num_threads).
3. Handling of long sentences (>512 tokens) via sliding window chunking with mean pooling.
4. Softmax probability extraction for all 7 emotion classes.
5. Title-level checkpointing to prevent progress loss.
6. Aggregations per title and narrative arc (100 bins).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.config import BASE_DIR, DATA_DIR, RESULTS_DIR, RAW_SENTENCES_FILE, get_device, set_seed, SEED

torch.set_num_threads(os.cpu_count() or 8)

EMOTIONS_7 = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


class InferenceBatchDataset(Dataset):
    """Dataset for fast batched tokenization during inference."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = [str(t) if pd.notna(t) and str(t).strip() else "..." for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


class TransformerInferenceEngine:
    def __init__(self, model_alias: str = "roberta", batch_size: int = 32):
        self.device = get_device()
        self.model_alias = model_alias
        self.batch_size = batch_size
        self.model_dir = RESULTS_DIR / "models" / f"{model_alias}_ekman"
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Trained model checkpoint not found at {self.model_dir}. Run Tahap 3 first.")

        print(f"[Engine] Loading {model_alias.upper()} from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"[Engine] {model_alias.upper()} successfully loaded and ready.")

    def _infer_long_text_sliding_window(self, text: str, max_chunk: int = 512, stride: int = 128) -> np.ndarray:
        """Process extremely long sentences (>512 tokens) with sliding window pooling."""
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"][0]
        
        if len(input_ids) <= max_chunk - 2:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_chunk)
            with torch.no_grad():
                out = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device)
                )
                probs = F.softmax(out.logits, dim=1).cpu().numpy()[0]
            return probs

        # Sliding window for long text
        chunk_probs = []
        step = max_chunk - stride - 2
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        for i in range(0, len(input_ids), step):
            chunk = input_ids[i : i + max_chunk - 2]
            full_chunk = torch.tensor([cls_id] + chunk.tolist() + [sep_id], dtype=torch.long).unsqueeze(0).to(self.device)
            mask = torch.ones_like(full_chunk).to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=full_chunk, attention_mask=mask)
                p = F.softmax(out.logits, dim=1).cpu().numpy()[0]
                chunk_probs.append(p)

        avg_prob = np.mean(chunk_probs, axis=0)
        return avg_prob / np.sum(avg_prob)

    def infer_dataframe(self, df: pd.DataFrame, desc: str = "Corpus Inference") -> pd.DataFrame:
        """Run batched inference with length bucketing on a DataFrame of sentences."""
        texts = df["text"].tolist()
        n = len(texts)
        
        # 1. Identify long sentences (> 250 words) that require sliding window
        word_counts = df["word_count"].fillna(0).astype(int).tolist()
        long_indices = [i for i, wc in enumerate(word_counts) if wc >= 250]
        regular_indices = [i for i in range(n) if i not in set(long_indices)]

        all_probs = np.zeros((n, len(EMOTIONS_7)), dtype=np.float32)

        # 2. Process regular sentences with length-based sorting for minimal padding waste
        if regular_indices:
            # Sort regular sentences by text length
            reg_lengths = [len(texts[i]) for i in regular_indices]
            sorted_order = np.argsort(reg_lengths)
            sorted_reg_indices = [regular_indices[k] for k in sorted_order]
            sorted_texts = [texts[idx] for idx in sorted_reg_indices]

            reg_dataset = InferenceBatchDataset(sorted_texts, self.tokenizer, max_length=512)
            collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            reg_loader = DataLoader(
                reg_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
            )

            current_ptr = 0
            pbar = tqdm(reg_loader, desc=f"[{self.model_alias.upper()}] {desc}")
            with torch.no_grad():
                for batch in pbar:
                    b_size = batch["input_ids"].size(0)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

                    for b_i in range(b_size):
                        orig_idx = sorted_reg_indices[current_ptr + b_i]
                        all_probs[orig_idx] = probs[b_i]
                    current_ptr += b_size

        # 3. Process long sentences with sliding window
        if long_indices:
            print(f"[{self.model_alias.upper()}] Processing {len(long_indices)} long monologues via sliding window pooling...")
            for idx in tqdm(long_indices, desc="Long window pooling"):
                all_probs[idx] = self._infer_long_text_sliding_window(texts[idx])

        # 4. Construct DataFrame
        result_df = pd.DataFrame()
        for i, emo in enumerate(EMOTIONS_7):
            result_df[f"{self.model_alias}_prob_{emo}"] = np.round(all_probs[:, i], 4)

        pred_labels = [EMOTIONS_7[idx] for idx in np.argmax(all_probs, axis=1)]
        result_df[f"{self.model_alias}_pred_emotion"] = pred_labels

        return result_df


def run_corpus_inference(
    target_title: Optional[str] = None,
    model_alias: str = "roberta",
    batch_size: int = 32,
    output_filename: str = "transformer_scores_sentences.csv",
) -> pd.DataFrame:
    set_seed(SEED)
    print(f"\n=======================================================")
    print(f"  Corpus Inference: {model_alias.upper()} (Target: {target_title or 'ALL 12 TITLES'})")
    print(f"=======================================================")

    engine = TransformerInferenceEngine(model_alias=model_alias, batch_size=batch_size)

    print(f"[Inference] Loading corpus sentences from {RAW_SENTENCES_FILE}...")
    corpus_df = pd.read_csv(RAW_SENTENCES_FILE)

    if target_title:
        corpus_df = corpus_df[corpus_df["title"] == target_title].copy().reset_index(drop=True)
        print(f"[Inference] Filtered for '{target_title}': {len(corpus_df)} sentences.")

    # Checkpoint management per title
    checkpoint_dir = RESULTS_DIR / "checkpoints_inference" / model_alias
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    titles = corpus_df["title"].unique()
    processed_dfs = []

    for t_idx, title in enumerate(titles, start=1):
        clean_title_fn = "".join([c if c.isalnum() else "_" for c in title])[:40]
        ckpt_file = checkpoint_dir / f"{clean_title_fn}.csv"

        if ckpt_file.exists():
            print(f"[{t_idx}/{len(titles)}] Checkpoint found for '{title}': loading {ckpt_file.name}...")
            title_df = pd.read_csv(ckpt_file)
            processed_dfs.append(title_df)
            continue

        print(f"\n[{t_idx}/{len(titles)}] Running inference on: '{title}' ({len(corpus_df[corpus_df['title'] == title])} sentences)...")
        sub_df = corpus_df[corpus_df["title"] == title].copy().reset_index(drop=True)
        
        pred_cols = engine.infer_dataframe(sub_df, desc=f"{title[:25]}")
        merged_title_df = pd.concat([sub_df, pred_cols], axis=1)

        # Save checkpoint
        merged_title_df.to_csv(ckpt_file, index=False)
        print(f"[Inference] Saved checkpoint: {ckpt_file}")
        processed_dfs.append(merged_title_df)

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # 1. Save sentence-level results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / output_filename
    final_df.to_csv(out_path, index=False)
    print(f"\n[Inference] Complete! Full sentences output saved to {out_path} ({len(final_df)} rows).")
    if not target_title:
        standard_out = RESULTS_DIR / "transformer_scores_sentences.csv"
        final_df.to_csv(standard_out, index=False)
        print(f"[Inference] Standard sentence output saved to {standard_out}")

    # 2. Aggregation per title
    title_summary = []
    prob_cols = [f"{model_alias}_prob_{emo}" for emo in EMOTIONS_7]

    for title, group in final_df.groupby("title"):
        row = {
            "title": title,
            "total_sentences": len(group),
            "total_words": int(group["word_count"].sum()),
        }
        for emo in EMOTIONS_7:
            row[f"{model_alias}_mean_prob_{emo}"] = round(group[f"{model_alias}_prob_{emo}"].mean(), 4)
            row[f"mean_prob_{emo}"] = row[f"{model_alias}_mean_prob_{emo}"]

        pred_counts = group[f"{model_alias}_pred_emotion"].value_counts(normalize=True)
        for emo in EMOTIONS_7:
            row[f"{model_alias}_pred_pct_{emo}"] = round(pred_counts.get(emo, 0.0) * 100, 2)
            row[f"pred_pct_{emo}"] = row[f"{model_alias}_pred_pct_{emo}"]

        title_summary.append(row)

    title_df = pd.DataFrame(title_summary)
    title_out = RESULTS_DIR / (f"transformer_scores_by_title_test.csv" if target_title else f"transformer_scores_by_title_{model_alias}.csv")
    title_df.to_csv(title_out, index=False)
    if not target_title:
        title_df.to_csv(RESULTS_DIR / "transformer_scores_by_title.csv", index=False)
    print(f"[Inference] Title-level aggregation saved to {title_out}")

    # 3. Narrative Arc Progression (100 bins per book)
    arc_rows = []
    for title, group in final_df.groupby("title"):
        group = group.copy().reset_index(drop=True)
        n = len(group)
        group["narrative_pct"] = (group.index / n) * 100
        group["narrative_bin"] = pd.cut(group["narrative_pct"], bins=100, labels=False)

        agg = group.groupby("narrative_bin").agg(
            {f"{model_alias}_prob_{emo}": "mean" for emo in EMOTIONS_7}
        ).reset_index()

        agg["title"] = title
        agg["narrative_progress_pct"] = agg["narrative_bin"] + 1
        arc_rows.append(agg)

    arc_df = pd.concat(arc_rows, ignore_index=True)
    arc_out = RESULTS_DIR / (f"transformer_scores_narrative_arc_test.csv" if target_title else f"transformer_scores_narrative_arc_{model_alias}.csv")
    arc_df.to_csv(arc_out, index=False)
    if not target_title:
        arc_df.to_csv(RESULTS_DIR / "transformer_scores_narrative_arc.csv", index=False)
    print(f"[Inference] Narrative arc progression saved to {arc_out}")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test on 'Crime and Punishment' only first.")
    parser.add_argument("--full", action="store_true", help="Run full inference on all 12 titles.")
    parser.add_argument("--model", choices=["roberta", "bert"], default="roberta")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.test or not args.full:
        print("[Run Mode] Testing inference on 'Crime and Punishment'...")
        run_corpus_inference(
            target_title="Crime and Punishment",
            model_alias=args.model,
            batch_size=args.batch_size,
            output_filename=f"transformer_scores_crime_and_punishment_{args.model}.csv",
        )
    else:
        print("[Run Mode] Full corpus inference on all 12 titles...")
        run_corpus_inference(
            target_title=None,
            model_alias=args.model,
            batch_size=args.batch_size,
            output_filename=f"transformer_scores_sentences_{args.model}.csv",
        )
