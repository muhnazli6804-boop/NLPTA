"""
Lexicon-Based Sentiment and Emotion Analysis Module.
Implements VADER Sentiment Intensity Analyzer and NRC Emotion Lexicon (8 Plutchik emotions).
Includes tokenization, lemmatization fallback, sentence-level scoring,
and aggregations per title, per paragraph, and normalized narrative progression.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import BASE_DIR, DATA_DIR, RESULTS_DIR, RAW_SENTENCES_FILE

NRC_LEXICON_FILE = DATA_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# 8 Plutchik emotions in NRC
NRC_EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


class LexiconAnalyzer:
    """Combines VADER Sentiment and NRC Emotion Lexicon scoring."""

    def __init__(self, nrc_file: Path = NRC_LEXICON_FILE):
        print("[LexiconAnalyzer] Initializing VADER SentimentIntensityAnalyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

        print(f"[LexiconAnalyzer] Loading NRC Emotion Lexicon from {nrc_file}...")
        self.nrc_dict: Dict[str, Set[str]] = self._load_nrc_lexicon(nrc_file)
        print(f"[LexiconAnalyzer] Loaded {len(self.nrc_dict)} words from NRC Lexicon.")

    def _load_nrc_lexicon(self, filepath: Path) -> Dict[str, Set[str]]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"NRC Lexicon file not found at {filepath}. Please ensure it is downloaded."
            )

        nrc = defaultdict(set)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    word, category, assoc = parts
                    if assoc == "1":
                        nrc[word.lower()].add(category.lower())
        return dict(nrc)

    def tokenize(self, text: str) -> List[str]:
        """Extract clean alphanumeric lowercase tokens."""
        return re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())

    def analyze_sentence(self, text: str) -> Dict[str, any]:
        """Analyze sentiment (VADER) and emotion distribution (NRC) for a single sentence."""
        # 1. VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores["compound"]
        if compound >= 0.05:
            sentiment_cat = "positive"
        elif compound <= -0.05:
            sentiment_cat = "negative"
        else:
            sentiment_cat = "neutral"

        # 2. NRC emotion counts
        tokens = self.tokenize(text)
        nrc_counts = {emotion: 0 for emotion in NRC_EMOTIONS}
        nrc_matched_words = []

        for token in tokens:
            emotions = self.nrc_dict.get(token)
            if not emotions:
                # Fallback to lemmatization
                lemma_n = self.lemmatizer.lemmatize(token, "n")
                lemma_v = self.lemmatizer.lemmatize(token, "v")
                emotions = self.nrc_dict.get(lemma_n) or self.nrc_dict.get(lemma_v)

            if emotions:
                has_emotion = False
                for emo in NRC_EMOTIONS:
                    if emo in emotions:
                        nrc_counts[emo] += 1
                        has_emotion = True
                if has_emotion:
                    nrc_matched_words.append(token)

        total_emotional_words = sum(nrc_counts.values())

        # Normalized distribution (sums to 1.0 if total_emotional_words > 0)
        nrc_norm = {}
        for emo in NRC_EMOTIONS:
            if total_emotional_words > 0:
                nrc_norm[f"nrc_norm_{emo}"] = round(nrc_counts[emo] / total_emotional_words, 4)
            else:
                nrc_norm[f"nrc_norm_{emo}"] = 0.0

        # Dominant emotion
        if total_emotional_words > 0:
            dominant_emotion = max(nrc_counts, key=nrc_counts.get)
        else:
            dominant_emotion = "none"

        result = {
            "vader_compound": round(compound, 4),
            "vader_pos": round(vader_scores["pos"], 4),
            "vader_neu": round(vader_scores["neu"], 4),
            "vader_neg": round(vader_scores["neg"], 4),
            "vader_sentiment": sentiment_cat,
            "nrc_total_emotional_words": total_emotional_words,
            "nrc_dominant_emotion": dominant_emotion,
        }
        for emo in NRC_EMOTIONS:
            result[f"nrc_count_{emo}"] = nrc_counts[emo]
        result.update(nrc_norm)

        return result


def process_dataset(
    target_title: Optional[str] = None,
    output_filename: str = "lexicon_scores_sentences.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run lexicon analysis on sentences.
    If target_title is provided, runs on that novel only (e.g., 'Crime and Punishment').
    Saves sentence scores and aggregated metrics to RESULTS_DIR.
    """
    analyzer = LexiconAnalyzer()
    
    print(f"[Lexicon] Loading dataset from {RAW_SENTENCES_FILE}...")
    df = pd.read_csv(RAW_SENTENCES_FILE)
    
    if target_title:
        df = df[df["title"] == target_title].copy()
        print(f"[Lexicon] Filtering for target title: '{target_title}' ({len(df)} sentences).")
    else:
        print(f"[Lexicon] Processing all {len(df)} sentences across all novels.")

    # Process sentences
    results = []
    texts = df["text"].fillna("").tolist()
    
    for text in tqdm(texts, desc="Analyzing lexicon sentiment & emotions"):
        res = analyzer.analyze_sentence(text)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    full_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

    # 1. Save sentence-level output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sentence_out = RESULTS_DIR / output_filename
    full_df.to_csv(sentence_out, index=False)
    print(f"[Lexicon] Sentence-level scores saved to {sentence_out} ({len(full_df)} rows).")

    # 2. Aggregation per title
    title_summary = []
    for title, group in full_df.groupby("title"):
        total_sents = len(group)
        total_words = group["word_count"].sum()
        mean_compound = group["vader_compound"].mean()
        
        # Emotion sums
        raw_emo_sums = {emo: group[f"nrc_count_{emo}"].sum() for emo in NRC_EMOTIONS}
        total_emo_words = sum(raw_emo_sums.values())
        
        row = {
            "title": title,
            "total_sentences": total_sents,
            "total_words": total_words,
            "vader_compound_mean": round(mean_compound, 4),
            "vader_pos_mean": round(group["vader_pos"].mean(), 4),
            "vader_neu_mean": round(group["vader_neu"].mean(), 4),
            "vader_neg_mean": round(group["vader_neg"].mean(), 4),
            "pct_positive": round((group["vader_sentiment"] == "positive").mean() * 100, 2),
            "pct_neutral": round((group["vader_sentiment"] == "neutral").mean() * 100, 2),
            "pct_negative": round((group["vader_sentiment"] == "negative").mean() * 100, 2),
            "total_nrc_emotional_words": total_emo_words,
        }
        
        # Add emotion distribution percentages
        for emo in NRC_EMOTIONS:
            row[f"nrc_count_{emo}"] = raw_emo_sums[emo]
            if total_emo_words > 0:
                row[f"nrc_prop_{emo}"] = round(raw_emo_sums[emo] / total_emo_words, 4)
            else:
                row[f"nrc_prop_{emo}"] = 0.0

        title_summary.append(row)

    title_df = pd.DataFrame(title_summary)
    title_out = RESULTS_DIR / ("lexicon_scores_by_title_test.csv" if target_title else "lexicon_scores_by_title.csv")
    title_df.to_csv(title_out, index=False)
    print(f"[Lexicon] Title-level aggregation saved to {title_out}")

    # 3. Aggregation by Narrative Arc Progression (100 bins per book, 0-100%)
    arc_rows = []
    for title, group in full_df.groupby("title"):
        group = group.copy().reset_index(drop=True)
        n = len(group)
        group["narrative_pct"] = (group.index / n) * 100
        group["narrative_bin"] = pd.cut(group["narrative_pct"], bins=100, labels=False)
        
        agg = group.groupby("narrative_bin").agg({
            "vader_compound": "mean",
            **{f"nrc_count_{emo}": "sum" for emo in NRC_EMOTIONS}
        }).reset_index()
        
        agg["title"] = title
        agg["narrative_progress_pct"] = agg["narrative_bin"] + 1
        
        # Calculate emotion proportions in this narrative slice
        total_in_bin = agg[[f"nrc_count_{emo}" for emo in NRC_EMOTIONS]].sum(axis=1)
        for emo in NRC_EMOTIONS:
            agg[f"nrc_prop_{emo}"] = (agg[f"nrc_count_{emo}"] / total_in_bin).fillna(0.0).round(4)
            
        arc_rows.append(agg)
        
    arc_df = pd.concat(arc_rows, ignore_index=True)
    arc_out = RESULTS_DIR / ("lexicon_scores_narrative_arc_test.csv" if target_title else "lexicon_scores_narrative_arc.csv")
    arc_df.to_csv(arc_out, index=False)
    print(f"[Lexicon] Narrative arc progression saved to {arc_out}")

    return full_df, title_df, arc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lexicon Analysis (VADER + NRC).")
    parser.add_argument("--test", action="store_true", help="Run test on 'Crime and Punishment' only first.")
    parser.add_argument("--full", action="store_true", help="Run on all 12 titles.")
    args = parser.parse_args()

    if args.test or not args.full:
        print("[Run Mode] Testing on single title: 'Crime and Punishment'...")
        process_dataset(target_title="Crime and Punishment", output_filename="lexicon_scores_crime_and_punishment.csv")
    else:
        print("[Run Mode] Full execution on entire dataset...")
        process_dataset(target_title=None, output_filename="lexicon_scores_sentences.csv")
