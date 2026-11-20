import json
import csv
from pathlib import Path

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "annotation_batches"
CSV_PATH = DATA_DIR / "annotation_task_annotator_1.csv"

VALID_EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}

def verify_all():
    print("=== VERIFYING ANNOTATOR 1 (GEMINI) DELIVERABLES ===")
    
    # 1. Check all 4 batch result files
    total_batch_items = 0
    all_batch_task_ids = set()
    
    for i in range(1, 5):
        rf = BATCH_DIR / f"gemini_results_{i}.json"
        assert rf.exists(), f"Missing result file: {rf}"
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), f"{rf.name} is not a list"
        assert len(data) == 300, f"{rf.name} has {len(data)} items, expected 300"
        
        expected_start = (i - 1) * 300 + 1
        expected_end = i * 300
        assert data[0]["task_id"] == expected_start, f"Batch {i} start task_id {data[0]['task_id']} != {expected_start}"
        assert data[-1]["task_id"] == expected_end, f"Batch {i} end task_id {data[-1]['task_id']} != {expected_end}"
        
        for item in data:
            assert "task_id" in item
            assert "emotion_label" in item
            assert "confidence_1_to_5" in item
            assert item["emotion_label"] in VALID_EMOTIONS, f"Invalid emotion: {item['emotion_label']}"
            assert item["confidence_1_to_5"] in [1, 2, 3, 4, 5], f"Invalid confidence: {item['confidence_1_to_5']}"
            all_batch_task_ids.add(item["task_id"])
        
        total_batch_items += len(data)
        print(f"  [OK] {rf.name}: 300 items (task_id {expected_start} to {expected_end})")
        
    assert total_batch_items == 1200, f"Total batch items {total_batch_items} != 1200"
    assert len(all_batch_task_ids) == 1200, "Duplicate or missing task_ids in batches"

    # 2. Check CSV file
    assert CSV_PATH.exists(), f"Missing CSV: {CSV_PATH}"
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1200, f"CSV has {len(rows)} rows, expected 1200"
    
    counts = {e: 0 for e in VALID_EMOTIONS}
    conf_counts = {c: 0 for c in range(1, 6)}
    
    for r in rows:
        tid = int(r["task_id"])
        emo = r["emotion_label"]
        conf = int(r["confidence_1_to_5"])
        assert emo in VALID_EMOTIONS, f"Invalid emotion in CSV row {tid}: '{emo}'"
        assert conf in conf_counts, f"Invalid confidence in CSV row {tid}: {conf}"
        counts[emo] += 1
        conf_counts[conf] += 1

    print("\n  [OK] annotation_task_annotator_1.csv: all 1,200 rows validly annotated")
    print("\n=== FINAL EMOTION DISTRIBUTION (Annotator 1 - Gemini) ===")
    for emo, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (cnt / 1200) * 100
        print(f"  - {emo:<10}: {cnt:>4} ({pct:>5.1f}%)")

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    for c in range(5, 0, -1):
        cnt = conf_counts[c]
        pct = (cnt / 1200) * 100
        print(f"  - Level {c}: {cnt:>4} ({pct:>5.1f}%)")

    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    verify_all()
