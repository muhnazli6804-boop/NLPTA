"""
Stratified Sampling and Annotation Task Preparation Module.
Samples 2,000 sentences proportionally across all 12 titles (seed=42).
Splits into:
- 400 overlap sentences (annotated by both Annotator 1 and Annotator 2)
- 800 sentences for Annotator 1 only
- 800 sentences for Annotator 2 only
Total per annotator = 1,200 sentences.
Generates master file and individual task CSVs with annotation guidelines.
"""

import sys
from pathlib import Path

BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RAW_SENTENCES_FILE, SEED, set_seed

TARGET_TOTAL_SAMPLE = 2000
OVERLAP_SAMPLE = 400
SOLO_PER_ANNOTATOR = 800  # 800 * 2 = 1600; 1600 + 400 = 2000


def create_stratified_sample():
    set_seed(SEED)
    print(f"[Sampling] Reading corpus from {RAW_SENTENCES_FILE}...")
    df = pd.read_csv(RAW_SENTENCES_FILE)
    total_corpus_sents = len(df)
    print(f"[Sampling] Total corpus sentences: {total_corpus_sents}")

    # 1. Proportional allocation per title
    title_counts = df["title"].value_counts()
    title_proportions = title_counts / total_corpus_sents
    
    # Calculate initial quota with largest remainder method to sum exactly to 2000
    exact_quotas = title_proportions * TARGET_TOTAL_SAMPLE
    allocated_quotas = exact_quotas.apply(np.floor).astype(int)
    remainders = exact_quotas - allocated_quotas
    
    shortfall = TARGET_TOTAL_SAMPLE - allocated_quotas.sum()
    if shortfall > 0:
        top_remainders = remainders.nlargest(shortfall).index
        for t in top_remainders:
            allocated_quotas[t] += 1

    print("\n--- Stratified Quota per Title (Total: 2,000) ---")
    quota_dict = allocated_quotas.to_dict()
    for title, q in quota_dict.items():
        print(f" - {title}: {q} sentences ({q/TARGET_TOTAL_SAMPLE*100:.1f}%)")

    # 2. Draw stratified sample
    sampled_frames = []
    for title, q in quota_dict.items():
        subset = df[df["title"] == title].sample(n=q, random_state=SEED)
        sampled_frames.append(subset)

    sample_df = pd.concat(sampled_frames, ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample_df.insert(0, "sample_id", range(1, len(sample_df) + 1))

    # 3. Stratified split for 400 overlap, 800 annotator 1, 800 annotator 2
    # We allocate 20% of each title's quota to overlap, 40% to ann1, 40% to ann2
    assignment = []
    
    # Group by title within sample_df to distribute partition evenly across books
    overlap_ids = set()
    ann1_only_ids = set()
    ann2_only_ids = set()

    for title, group in sample_df.groupby("title"):
        group_ids = group["sample_id"].tolist()
        n_group = len(group_ids)
        
        n_overlap = int(round(n_group * (OVERLAP_SAMPLE / TARGET_TOTAL_SAMPLE)))
        # Ensure at least 1 overlap if n_group >= 5
        if n_overlap == 0 and n_group >= 5:
            n_overlap = 1
            
        remaining_count = n_group - n_overlap
        n_ann1 = remaining_count // 2
        n_ann2 = remaining_count - n_ann1

        # Slice IDs
        cur_overlap = group_ids[:n_overlap]
        cur_ann1 = group_ids[n_overlap : n_overlap + n_ann1]
        cur_ann2 = group_ids[n_overlap + n_ann1 :]

        overlap_ids.update(cur_overlap)
        ann1_only_ids.update(cur_ann1)
        ann2_only_ids.update(cur_ann2)

    # Adjust exact totals if slight rounding divergence exists
    # If overlap_ids != 400, adjust
    diff_overlap = len(overlap_ids) - OVERLAP_SAMPLE
    if diff_overlap > 0:
        # Move excess from overlap to ann1/ann2
        excess = list(overlap_ids)[:diff_overlap]
        for idx, sid in enumerate(excess):
            overlap_ids.remove(sid)
            if idx % 2 == 0:
                ann1_only_ids.add(sid)
            else:
                ann2_only_ids.add(sid)
    elif diff_overlap < 0:
        needed = abs(diff_overlap)
        # Pull from ann1 and ann2
        for i in range(needed):
            if i % 2 == 0 and ann1_only_ids:
                sid = ann1_only_ids.pop()
                overlap_ids.add(sid)
            elif ann2_only_ids:
                sid = ann2_only_ids.pop()
                overlap_ids.add(sid)

    # Ensure ann1 and ann2 are exactly 800 each
    diff_ann = len(ann1_only_ids) - SOLO_PER_ANNOTATOR
    if diff_ann > 0:
        # Transfer from ann1 to ann2
        excess = list(ann1_only_ids)[:diff_ann]
        for sid in excess:
            ann1_only_ids.remove(sid)
            ann2_only_ids.add(sid)
    elif diff_ann < 0:
        needed = abs(diff_ann)
        excess = list(ann2_only_ids)[:needed]
        for sid in excess:
            ann2_only_ids.remove(sid)
            ann1_only_ids.add(sid)

    # Tag assignments in master dataframe
    def get_assignment_tag(sid):
        if sid in overlap_ids:
            return "overlap"
        elif sid in ann1_only_ids:
            return "annotator_1_only"
        elif sid in ann2_only_ids:
            return "annotator_2_only"
        return "unassigned"

    sample_df["assignment"] = sample_df["sample_id"].apply(get_assignment_tag)

    # Verification of split
    print("\n--- Final Assignment Partition ---")
    print(f"Overlap (Annotator 1 & 2): {len(overlap_ids)}")
    print(f"Annotator 1 Solo: {len(ann1_only_ids)}")
    print(f"Annotator 2 Solo: {len(ann2_only_ids)}")
    print(f"Total Unique Sentences: {len(sample_df)}")

    # 4. Save master file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = DATA_DIR / "annotation_master_2000.csv"
    sample_df.to_csv(master_path, index=False)
    print(f"[Sampling] Master sample file saved to {master_path}")

    # 5. Build Annotator 1 Task CSV (1,200 rows)
    ann1_ids = overlap_ids.union(ann1_only_ids)
    ann1_df = sample_df[sample_df["sample_id"].isin(ann1_ids)].copy()
    # Shuffle order so annotator cannot tell which sentences are overlap
    ann1_df = ann1_df.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    ann1_df.insert(0, "task_id", range(1, len(ann1_df) + 1))
    
    # Add empty columns for annotation
    ann1_df["emotion_label"] = ""
    ann1_df["confidence_1_to_5"] = ""
    ann1_df["notes"] = ""
    
    # Keep only annotator-facing columns (do not leak assignment label)
    ann1_export_cols = [
        "task_id",
        "sample_id",
        "sentence_id",
        "title",
        "text",
        "emotion_label",
        "confidence_1_to_5",
        "notes",
    ]
    ann1_task_path = DATA_DIR / "annotation_task_annotator_1.csv"
    ann1_df[ann1_export_cols].to_csv(ann1_task_path, index=False)
    print(f"[Sampling] Task file for Annotator 1 saved to {ann1_task_path} ({len(ann1_df)} rows).")

    # 6. Build Annotator 2 Task CSV (1,200 rows)
    ann2_ids = overlap_ids.union(ann2_only_ids)
    ann2_df = sample_df[sample_df["sample_id"].isin(ann2_ids)].copy()
    # Shuffle order with different seed
    ann2_df = ann2_df.sample(frac=1.0, random_state=SEED + 2).reset_index(drop=True)
    ann2_df.insert(0, "task_id", range(1, len(ann2_df) + 1))
    
    ann2_df["emotion_label"] = ""
    ann2_df["confidence_1_to_5"] = ""
    ann2_df["notes"] = ""
    
    ann2_task_path = DATA_DIR / "annotation_task_annotator_2.csv"
    ann2_df[ann1_export_cols].to_csv(ann2_task_path, index=False)
    print(f"[Sampling] Task file for Annotator 2 saved to {ann2_task_path} ({len(ann2_df)} rows).")

    return sample_df, ann1_df, ann2_df


if __name__ == "__main__":
    create_stratified_sample()
