"""
Visualization script for Tahap 1 Lexicon Results.
Generates summary charts:
1. Emotion distribution heatmap across all 12 titles.
2. Sentiment polarity comparison (VADER compound and positive/negative percentages).
3. Narrative emotional arc for Crime and Punishment.
"""

import sys
from pathlib import Path
BASE_DIR_PATH = Path(__file__).resolve().parent.parent
if str(BASE_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_PATH))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import RESULTS_DIR

def plot_lexicon_results():
    print("[Plot] Loading lexicon results...")
    title_df = pd.read_csv(RESULTS_DIR / "lexicon_scores_by_title.csv")
    arc_df = pd.read_csv(RESULTS_DIR / "lexicon_scores_narrative_arc.csv")

    sns.set_theme(style="whitegrid")
    
    # --- 1. Emotion Proportion Heatmap ---
    emo_cols = [c for c in title_df.columns if c.startswith("nrc_prop_")]
    emo_names = [c.replace("nrc_prop_", "").capitalize() for c in emo_cols]
    
    heatmap_data = title_df.set_index("title")[emo_cols]
    heatmap_data.columns = emo_names
    
    plt.figure(figsize=(11, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'NRC Emotion Proportion'})
    plt.title("Distribusi Proporsi 8 Emosi NRC pada 12 Karya Fyodor Dostoevsky", fontsize=14, pad=15)
    plt.xlabel("Dimensi Emosi (NRC / Plutchik)", fontsize=12)
    plt.ylabel("Judul Karya", fontsize=12)
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "lexicon_emotions_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved emotion heatmap to {heatmap_path}")

    # --- 2. VADER Sentiment Comparison Barplot ---
    plt.figure(figsize=(12, 6))
    sorted_df = title_df.sort_values("vader_compound_mean", ascending=True)
    colors = ["#d9534f" if x < 0 else "#5cb85c" for x in sorted_df["vader_compound_mean"]]
    
    bars = plt.barh(sorted_df["title"], sorted_df["vader_compound_mean"], color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Rata-rata Skor Polaritas VADER Compound per Judul Buku", fontsize=14, pad=15)
    plt.xlabel("Skor VADER Compound (-1 = Sangat Negatif, +1 = Sangat Positif)", fontsize=12)
    plt.tight_layout()
    sentiment_path = RESULTS_DIR / "lexicon_sentiment_comparison.png"
    plt.savefig(sentiment_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved sentiment comparison to {sentiment_path}")

    # --- 3. Narrative Emotional Arc for Crime and Punishment ---
    cp_arc = arc_df[arc_df["title"] == "Crime and Punishment"].sort_values("narrative_progress_pct")
    
    plt.figure(figsize=(12, 5))
    plt.plot(cp_arc["narrative_progress_pct"], cp_arc["vader_compound"], color="#2b5c8f", linewidth=2, label="VADER Compound (Mean per 1% Bin)")
    # Rolling smoothing
    smoothed = cp_arc["vader_compound"].rolling(window=7, min_periods=1, center=True).mean()
    plt.plot(cp_arc["narrative_progress_pct"], smoothed, color="#d9534f", linewidth=2.5, linestyle="-", label="Smoothed Trend (Window=7)")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title("Emotional Arc: Perkembangan Sentimen Naratif Crime and Punishment (100 Bins)", fontsize=14, pad=15)
    plt.xlabel("Progres Narasi Novel (%)", fontsize=12)
    plt.ylabel("Skor VADER Compound", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    arc_path = RESULTS_DIR / "crime_and_punishment_arc.png"
    plt.savefig(arc_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved emotional arc to {arc_path}")

if __name__ == "__main__":
    plot_lexicon_results()
