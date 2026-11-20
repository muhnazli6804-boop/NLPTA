"""
Visualization Generation Module for Thesis
Project: "Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky"

Generates 5 publication-ready figures (300 DPI):
1. fig1_emotional_arc_crime_and_punishment.png: NRC vs RoBERTa narrative arc with Picca et al. checkpoints.
2. fig2_emotion_distribution_across_titles.png: RoBERTa emotion profile across all 12 novels.
3. fig3_spearman_correlation_heatmap.png: Correlation heatmap between NRC Lexicon and RoBERTa.
4. fig4_jsd_inter_novel_clustering.png: Hierarchical clustering of novels based on emotion JSD distances.
5. fig5_lexicon_vs_transformer_correlation.png: Scatter plots with regression lines for the 6 intersecting emotions.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root workspace is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import RESULTS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

EMOTION_COLORS = {
    "anger": "#d9534f",      # Red
    "disgust": "#8a6d3b",    # Brown/Olive
    "fear": "#6f42c1",       # Purple
    "joy": "#f0ad4e",        # Gold/Orange
    "sadness": "#0275d8",    # Blue
    "surprise": "#5bc0de",   # Cyan/Teal
    "neutral": "#6c757d",    # Slate Grey
    "anticipation": "#e83e8c",# Pink/Magenta
    "trust": "#20c997",      # Emerald Green
}

INTERSECTING_6 = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def plot_fig1_crime_and_punishment_arc():
    """
    Fig 1: Emotional Arc Comparison for 'Crime and Punishment' (RoBERTa vs NRC)
    with Picca et al. (2024) narrative checkpoints.
    """
    logger.info("Plotting Fig 1: Crime and Punishment Emotional Arc...")
    arc_roberta_path = RESULTS_DIR / "transformer_scores_narrative_arc.csv"
    arc_lex_path = RESULTS_DIR / "lexicon_scores_narrative_arc.csv"

    if not arc_roberta_path.exists():
        arc_roberta_path = RESULTS_DIR / "transformer_scores_narrative_arc_roberta.csv"

    df_rob = pd.read_csv(arc_roberta_path)
    df_lex = pd.read_csv(arc_lex_path)

    rob_cp = df_rob[df_rob["title"] == "Crime and Punishment"].sort_values("narrative_progress_pct")
    lex_cp = df_lex[df_lex["title"] == "Crime and Punishment"].sort_values("narrative_progress_pct")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 1. RoBERTa Arc (Top)
    ax1.plot(rob_cp["narrative_progress_pct"], rob_cp["roberta_prob_fear"].rolling(5, min_periods=1).mean() * 100,
             label="Fear (Ketakutan)", color=EMOTION_COLORS["fear"], linewidth=2.2)
    ax1.plot(rob_cp["narrative_progress_pct"], rob_cp["roberta_prob_sadness"].rolling(5, min_periods=1).mean() * 100,
             label="Sadness (Kesedihan)", color=EMOTION_COLORS["sadness"], linewidth=2.0)
    ax1.plot(rob_cp["narrative_progress_pct"], rob_cp["roberta_prob_anger"].rolling(5, min_periods=1).mean() * 100,
             label="Anger (Kemarahan)", color=EMOTION_COLORS["anger"], linewidth=1.8)
    ax1.plot(rob_cp["narrative_progress_pct"], rob_cp["roberta_prob_joy"].rolling(5, min_periods=1).mean() * 100,
             label="Joy (Kebahagiaan)", color=EMOTION_COLORS["joy"], linewidth=2.0)
    ax1.plot(rob_cp["narrative_progress_pct"], rob_cp["roberta_prob_surprise"].rolling(5, min_periods=1).mean() * 100,
             label="Surprise (Terkejut)", color=EMOTION_COLORS["surprise"], linewidth=1.6, linestyle="--")

    # Annotate Picca checkpoints on RoBERTa
    checkpoints = [
        (17, "1. Pembunuhan Alyona\n(Part I)", EMOTION_COLORS["fear"]),
        (32, "2. Kematian Marmeladov\n(Part II)", EMOTION_COLORS["sadness"]),
        (67, "3. Duel Interogasi Porfiry\n(Part IV-V)", EMOTION_COLORS["surprise"]),
        (88, "4. Bunuh Diri Svidrigailov\n& Pengakuan (Part VI)", EMOTION_COLORS["anger"]),
        (98, "5. Kelahiran Kembali\ndi Siberia (Epilog)", EMOTION_COLORS["joy"])
    ]

    for x_pos, label, col in checkpoints:
        ax1.axvline(x=x_pos, color=col, linestyle=":", alpha=0.75, linewidth=1.3)
        ax1.text(x_pos, ax1.get_ylim()[1] if hasattr(ax1, 'get_ylim') else 28, f" {label}", 
                 rotation=90, va="top", ha="right", fontsize=7.5, color=col, fontweight="bold")

    ax1.set_title("A. Pendekatan Transformer (Fine-tuned RoBERTa) - Probabilitas Emosi Kontekstual", fontweight="bold")
    ax1.set_ylabel("Probabilitas Emosi (%)")
    ax1.legend(loc="upper right", framealpha=0.9, ncol=3)

    # 2. NRC Lexicon Arc (Bottom)
    ax2.plot(lex_cp["narrative_progress_pct"], lex_cp["nrc_prop_fear"].rolling(5, min_periods=1).mean() * 100,
             label="Fear (NRC)", color=EMOTION_COLORS["fear"], linewidth=2.0)
    ax2.plot(lex_cp["narrative_progress_pct"], lex_cp["nrc_prop_sadness"].rolling(5, min_periods=1).mean() * 100,
             label="Sadness (NRC)", color=EMOTION_COLORS["sadness"], linewidth=1.8)
    ax2.plot(lex_cp["narrative_progress_pct"], lex_cp["nrc_prop_anger"].rolling(5, min_periods=1).mean() * 100,
             label="Anger (NRC)", color=EMOTION_COLORS["anger"], linewidth=1.6)
    ax2.plot(lex_cp["narrative_progress_pct"], lex_cp["nrc_prop_joy"].rolling(5, min_periods=1).mean() * 100,
             label="Joy (NRC)", color=EMOTION_COLORS["joy"], linewidth=1.8)
    ax2.plot(lex_cp["narrative_progress_pct"], lex_cp["nrc_prop_trust"].rolling(5, min_periods=1).mean() * 100,
             label="Trust (NRC)", color=EMOTION_COLORS["trust"], linewidth=1.5, linestyle=":")

    for x_pos, _, col in checkpoints:
        ax2.axvline(x=x_pos, color=col, linestyle=":", alpha=0.5, linewidth=1.0)

    ax2.set_title("B. Pendekatan Leksikon (NRC Emotion Lexicon) - Frekuensi Kata Emosi", fontweight="bold")
    ax2.set_xlabel("Progres Narasi Novel (%) [0 = Awal, 100 = Epilog Akhir]")
    ax2.set_ylabel("Kerapatan Kata Emosi (%)")
    ax2.legend(loc="upper right", framealpha=0.9, ncol=3)

    plt.suptitle("Dinamika Alur Emosi (Emotional Arc) pada 'Crime and Punishment': Transformer vs. Leksikon\n(Mengacu pada Studi Acuan Picca, Schnyder & Romele, 2024)", 
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_fig1 = RESULTS_DIR / "fig1_emotional_arc_crime_and_punishment.png"
    plt.savefig(out_fig1, dpi=300)
    plt.close()
    logger.info(f"Saved Fig 1 to {out_fig1}")


def plot_fig2_emotion_distribution_across_titles():
    """
    Fig 2: Comparative Emotion Distribution Across all 12 Dostoevsky Novels (RoBERTa).
    Grouped stacked bar chart showing the affective fingerprint of each novel.
    """
    logger.info("Plotting Fig 2: Emotion Distribution Across 12 Titles...")
    title_path = RESULTS_DIR / "transformer_scores_by_title.csv"
    if not title_path.exists():
        title_path = RESULTS_DIR / "transformer_scores_by_title_roberta.csv"

    df_title = pd.read_csv(title_path)

    # Sort titles by publication/narrative importance or sadness percentage
    df_title["sort_key"] = df_title["roberta_pred_pct_sadness"] + df_title["roberta_pred_pct_fear"]
    df_title = df_title.sort_values("sort_key", ascending=True).reset_index(drop=True)

    titles = df_title["title"].tolist()
    # Abbreviate long titles for clean display
    clean_titles = [t.replace(" :  or, The devils", "").replace(" :  or, Prison life in Siberia", "") for t in titles]

    fig, ax = plt.subplots(figsize=(13, 7))

    bottom = np.zeros(len(titles))
    emotions_to_stack = ["neutral", "surprise", "fear", "sadness", "anger", "joy", "disgust"]

    for emo in emotions_to_stack:
        vals = df_title[f"roberta_pred_pct_{emo}"].values
        ax.barh(clean_titles, vals, left=bottom, color=EMOTION_COLORS[emo], label=emo.capitalize(), edgecolor="white", alpha=0.9)
        bottom += vals

    ax.set_xlabel("Proporsi Prediksi Emosi (%)")
    ax.set_title("Distribusi Proporsi Emosi RoBERTa pada 12 Karya Fiksi Fyodor Dostoevsky\n(Diurutkan berdasarkan akumulasi intensitas emosi negatif: Fear + Sadness)", 
                 fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Emosi")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    out_fig2 = RESULTS_DIR / "fig2_emotion_distribution_across_titles.png"
    plt.savefig(out_fig2, dpi=300)
    plt.close()
    logger.info(f"Saved Fig 2 to {out_fig2}")


def plot_fig3_spearman_heatmap():
    """
    Fig 3: Spearman Correlation Heatmap across 12 Novels and 6 Intersecting Emotions.
    Shows where Lexicon and Transformer converge and diverge across narrative arcs.
    """
    logger.info("Plotting Fig 3: Spearman Correlation Heatmap...")
    corr_file = RESULTS_DIR / "spearman_correlation_narrative_arc.csv"
    if not corr_file.exists():
        logger.warning(f"{corr_file} not found. Skipping Fig 3.")
        return

    df_corr = pd.read_csv(corr_file)
    pivot_df = df_corr.pivot(index="title", columns="emotion", values="spearman_rho")

    # Clean title names
    pivot_df.index = [t.replace(" :  or, The devils", "").replace(" :  or, Prison life in Siberia", "") for t in pivot_df.index]
    pivot_df = pivot_df[INTERSECTING_6]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.3, vmax=0.7,
                cbar_kws={"label": "Koefisien Korelasi Spearman (rho)"}, ax=ax, linewidths=0.5)

    ax.set_title("Korelasi Spearman Alur Narasi: Leksikon NRC vs. Transformer RoBERTa\n(100 Bins per Judul pada 6 Emosi Beririsan)", 
                 fontweight="bold")
    ax.set_xlabel("Kelas Emosi Ekman")
    ax.set_ylabel("Karya Dostoevsky")

    plt.tight_layout()
    out_fig3 = RESULTS_DIR / "fig3_spearman_correlation_heatmap.png"
    plt.savefig(out_fig3, dpi=300)
    plt.close()
    logger.info(f"Saved Fig 3 to {out_fig3}")


def plot_fig4_jsd_clustering():
    """
    Fig 4: Hierarchical Clustering Clustermap based on Jensen-Shannon Divergence (JSD).
    Reveals thematic and affective grouping among Dostoevsky's novels.
    """
    logger.info("Plotting Fig 4: JSD Inter-Novel Clustermap...")
    jsd_file = RESULTS_DIR / "jsd_matrix_roberta.csv"
    if not jsd_file.exists():
        logger.warning(f"{jsd_file} not found. Skipping Fig 4.")
        return

    df_jsd = pd.read_csv(jsd_file, index_col=0)
    # Shorten titles
    short_titles = [t.replace(" :  or, The devils", "").replace(" :  or, Prison life in Siberia", "")[:28] for t in df_jsd.index]
    df_jsd.index = short_titles
    df_jsd.columns = short_titles

    # Convert JSD divergence to distance sqrt(JSD)
    dist_mat = np.sqrt(df_jsd.values)
    df_dist = pd.DataFrame(dist_mat, index=short_titles, columns=short_titles)

    g = sns.clustermap(df_dist, cmap="YlGnBu", annot=True, fmt=".2f", figsize=(11, 10),
                       cbar_kws={"label": "JSD Distance sqrt(JSD)"}, linewidths=0.5)
    g.fig.suptitle("Kluster Kedekatan Emosional Antar-Novel Dostoevsky Berdasarkan JSD Distance (RoBERTa)", 
                   fontweight="bold", y=1.02)

    out_fig4 = RESULTS_DIR / "fig4_jsd_inter_novel_clustering.png"
    g.savefig(out_fig4, dpi=300)
    plt.close()
    logger.info(f"Saved Fig 4 to {out_fig4}")


def plot_fig5_lexicon_vs_transformer_scatter():
    """
    Fig 5: Scatter plots with regression lines comparing novel-level emotion density:
    NRC Lexicon % vs. RoBERTa Probability %.
    """
    logger.info("Plotting Fig 5: Lexicon vs Transformer Regression Plots...")
    lex_title_path = RESULTS_DIR / "lexicon_scores_by_title.csv"
    trans_title_path = RESULTS_DIR / "transformer_scores_by_title.csv"
    if not trans_title_path.exists():
        trans_title_path = RESULTS_DIR / "transformer_scores_by_title_roberta.csv"

    df_lex = pd.read_csv(lex_title_path)
    df_trans = pd.read_csv(trans_title_path)

    merged = pd.merge(df_lex, df_trans, on="title")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, emo in enumerate(INTERSECTING_6):
        ax = axes[idx]
        x = merged[f"nrc_prop_{emo}"] * 100
        y = merged[f"roberta_pred_pct_{emo}"]

        sns.regplot(x=x, y=y, ax=ax, color=EMOTION_COLORS[emo], 
                    scatter_kws={"s": 60, "alpha": 0.8}, line_kws={"linewidth": 1.5})

        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{emo.capitalize()} (r = {corr:.2f})", fontweight="bold")
        ax.set_xlabel(f"NRC Lexicon {emo.capitalize()} (%)")
        ax.set_ylabel(f"RoBERTa {emo.capitalize()} (%)")

    plt.suptitle("Perbandingan Proporsi Emosi per Novel: Leksikon NRC vs. Transformer RoBERTa (12 Judul)", 
                 fontweight="bold", y=0.99)
    plt.tight_layout()
    out_fig5 = RESULTS_DIR / "fig5_lexicon_vs_transformer_correlation.png"
    plt.savefig(out_fig5, dpi=300)
    plt.close()
    logger.info(f"Saved Fig 5 to {out_fig5}")


def generate_all_visualizations():
    """
    Main function to generate all publication-ready figures.
    """
    logger.info("=== GENERATING ALL THESIS PUBLICATION FIGURES ===")
    plot_fig1_crime_and_punishment_arc()
    plot_fig2_emotion_distribution_across_titles()
    plot_fig3_spearman_heatmap()
    plot_fig4_jsd_clustering()
    plot_fig5_lexicon_vs_transformer_scatter()
    logger.info("=== ALL 5 FIGURES SUCCESSFULLY GENERATED ===")


if __name__ == "__main__":
    generate_all_visualizations()
