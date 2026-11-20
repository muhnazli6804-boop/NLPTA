"""
Data Inspection and Quality Audit Script for sentences.csv.
Analyzes columns, row counts per title, empty/short sentences,
duplicates, and extremely long sentences.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "sentences.csv"
OUTPUT_CLEAN_FILE = BASE_DIR / "data" / "sentences_cleaned.csv"
REPORT_FILE = BASE_DIR / "results" / "data_inspection_report.json"

def run_inspection():
    print(f"[Inspection] Loading {INPUT_FILE}...")
    
    total_rows = 0
    title_counts = Counter()
    title_words = defaultdict(int)
    
    empty_sentences = []
    short_sentences = []  # word_count <= 2
    very_long_sentences = []  # word_count >= 100
    extreme_long_sentences = []  # word_count >= 250
    
    text_counter = Counter()
    duplicate_texts = defaultdict(list)
    
    rows_valid = []
    dropped_rows = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        
        for idx, row in enumerate(reader, start=1):
            total_rows += 1
            sid = row.get("sentence_id")
            gid = row.get("gutenberg_id")
            title = row.get("title")
            pid = row.get("paragraph_id")
            text = row.get("text", "")
            raw_wc = row.get("word_count", "0")
            
            try:
                wc = int(raw_wc)
            except ValueError:
                wc = len(text.split())
                
            stripped_text = text.strip()
            
            # 1. Title tracking
            title_counts[title] += 1
            title_words[title] += wc
            
            # 2. Empty / whitespace check
            if not stripped_text:
                empty_sentences.append({
                    "row_index": idx,
                    "sentence_id": sid,
                    "title": title,
                    "paragraph_id": pid
                })
                dropped_rows.append(row)
                continue
            
            # 3. Short sentences (word_count <= 2)
            if wc <= 2:
                short_sentences.append({
                    "sentence_id": sid,
                    "title": title,
                    "text": stripped_text,
                    "word_count": wc
                })
                
            # 4. Long sentences
            if wc >= 100:
                very_long_sentences.append({
                    "sentence_id": sid,
                    "title": title,
                    "text": stripped_text,
                    "word_count": wc
                })
            if wc >= 250:
                extreme_long_sentences.append({
                    "sentence_id": sid,
                    "title": title,
                    "text": stripped_text,
                    "word_count": wc
                })
                
            # 5. Duplicates tracking
            text_counter[stripped_text] += 1
            if text_counter[stripped_text] == 2:
                # First time we see it repeated
                duplicate_texts[stripped_text].append(sid)
            elif text_counter[stripped_text] > 2:
                duplicate_texts[stripped_text].append(sid)
                
            rows_valid.append(row)

    # Sort long sentences descending
    extreme_long_sentences.sort(key=lambda x: x["word_count"], reverse=True)
    very_long_sentences.sort(key=lambda x: x["word_count"], reverse=True)
    
    # Calculate duplicates summary
    total_duplicate_instances = sum(count - 1 for count in text_counter.values() if count > 1)
    unique_duplicated_texts = len(duplicate_texts)
    
    # Top duplicated phrases
    top_duplicates = [
        {"text": text, "count": count}
        for text, count in text_counter.most_common(20)
        if count > 1
    ]

    report = {
        "columns": fields,
        "total_rows": total_rows,
        "titles_summary": [
            {
                "title": title,
                "sentences_count": count,
                "total_words": title_words[title],
                "avg_words_per_sentence": round(title_words[title] / count, 2)
            }
            for title, count in title_counts.most_common()
        ],
        "empty_sentences_count": len(empty_sentences),
        "short_sentences_count_le_2_words": len(short_sentences),
        "long_sentences_ge_100_words": len(very_long_sentences),
        "extreme_long_sentences_ge_250_words": len(extreme_long_sentences),
        "unique_duplicated_texts": unique_duplicated_texts,
        "total_duplicate_redundant_instances": total_duplicate_instances,
        "top_20_duplicated_phrases": top_duplicates,
        "sample_short_sentences": short_sentences[:15],
        "top_5_longest_sentences": extreme_long_sentences[:5]
    }
    
    # Save JSON report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"[Inspection] Report saved to {REPORT_FILE}")
    
    # Save cleaned file if needed
    OUTPUT_CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CLEAN_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_valid)
        
    print(f"[Inspection] Valid rows ({len(rows_valid)}) saved to {OUTPUT_CLEAN_FILE}")
    print(f"[Inspection] Dropped empty rows: {len(dropped_rows)}")
    
    return report

if __name__ == "__main__":
    rep = run_inspection()
    print("\n--- SUMMARY OF INSPECTION ---")
    print(f"Total Sentences: {rep['total_rows']}")
    print(f"Empty Sentences: {rep['empty_sentences_count']}")
    print(f"Short Sentences (<= 2 words): {rep['short_sentences_count_le_2_words']}")
    print(f"Long Sentences (>= 100 words): {rep['long_sentences_ge_100_words']}")
    print(f"Extreme Long Sentences (>= 250 words): {rep['extreme_long_sentences_ge_250_words']}")
    print(f"Unique Duplicated Texts: {rep['unique_duplicated_texts']}")
    print(f"Titles count: {len(rep['titles_summary'])}")
