"""
Helper script to prepare sliced batches for LLM Annotators (Gemini & Claude).
Splits 1,200 sentences into 4 manageable slices of 300 sentences each to ensure
flawless output within LLM max token constraints.
"""

import json
from pathlib import Path
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
BATCH_DIR = DATA_DIR / "annotation_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

def create_slices():
    # 1. Task 1 for Gemini
    t1_path = DATA_DIR / "annotation_task_annotator_1.csv"
    if t1_path.exists():
        df1 = pd.read_csv(t1_path)
        batch_size = 300
        for i in range(0, len(df1), batch_size):
            slice_idx = (i // batch_size) + 1
            sub_df = df1.iloc[i : i + batch_size]
            items = []
            for _, row in sub_df.iterrows():
                items.append({
                    "task_id": int(row["task_id"]),
                    "title": str(row["title"]),
                    "text": str(row["text"])
                })
            out_file = BATCH_DIR / f"gemini_batch_{slice_idx}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[Slice] Saved {len(items)} sentences to {out_file}")

    # 2. Task 2 for Claude
    t2_path = DATA_DIR / "annotation_task_annotator_2.csv"
    if t2_path.exists():
        df2 = pd.read_csv(t2_path)
        batch_size = 300
        for i in range(0, len(df2), batch_size):
            slice_idx = (i // batch_size) + 1
            sub_df = df2.iloc[i : i + batch_size]
            items = []
            for _, row in sub_df.iterrows():
                items.append({
                    "task_id": int(row["task_id"]),
                    "title": str(row["title"]),
                    "text": str(row["text"])
                })
            out_file = BATCH_DIR / f"claude_batch_{slice_idx}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[Slice] Saved {len(items)} sentences to {out_file}")

if __name__ == "__main__":
    create_slices()
