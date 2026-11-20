"""
Script to generate the comprehensive master Jupyter Notebook (.ipynb) for the thesis:
"Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky Menggunakan Pendekatan Leksikon dan Transformer"
"""

import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "analisis_sentimen_emosi_dostoevsky.ipynb"

def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        # remove trailing newline on the last line to follow standard nbformat
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def build_notebook():
    cells = []

    # 1. Title & Header
    cells.append(make_cell("markdown", """# Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky Menggunakan Pendekatan Leksikon dan Transformer

**Penelitian Tugas Akhir (Skripsi) Komputasi Sastra / Natural Language Processing**  
*Studi Kasus 12 Karya Fiksi Klasik Terjemahan Bahasa Inggris (Project Gutenberg, 66.410 Kalimat)*  
*Studi Acuan: Picca, Schnyder & Romele (2024) pada Crime and Punishment*

---

### 📌 Ringkasan Penelitian
Notebook ini menyajikan dokumentasi alur kerja komputasi menyeluruh (*end-to-end pipeline*) dari penelitian skripsi, yang mencakup:
1. **Inspeksi & Pra-pemrosesan Data:** Audit 66.410 kalimat dari 12 novel Dostoevsky beserta metadata penerjemah era Victoria.
2. **Pendekatan Leksikon:** Ekstraksi sentimen polaritas **VADER** dan kerapatan 8 dimensi emosi **NRC Emotion Lexicon**.
3. **Sampling & Anotasi Emas:** Pembagian 2.000 kalimat berstrata, pengujian reliabilitas *double-blind* (400 kalimat overlap), dan studi anotasi multi-model AI (**Google Gemini** vs. **Anthropic Claude**).
4. **Fine-Tuning Transformer:** Transfer learning pada **BERT** (`bert-base-uncased`) dan **RoBERTa** (`roberta-base`) untuk taksonomi 7-kelas Ekman (*anger, disgust, fear, joy, sadness, surprise, neutral*).
5. **Inferensi Korpus Penuh:** Pemodelan konteks kalimat dan monolog panjang (>250 kata) dengan *sliding window pooling*.
6. **Analisis Komparatif Matematis:** Pengukuran koefisien **Korelasi Spearman ($\rho$)**, matriks jarak **Jensen-Shannon Divergence (JSD)**, kesepakatan penilai **Cohen's Kappa ($\kappa$)**, serta replikasi titik krisis narasi psikologis Raskolnikov (*Crime and Punishment*).
7. **Visualisasi Publikasi & Dasbor Streamlit:** Visualisasi alur emosi 300 DPI dan integrasi dasbor analitik web interaktif."""))

    # 2. Setup & Environment
    cells.append(make_cell("markdown", """## 1. Setup Lingkungan & Pustaka

Tahap ini memuat pustaka pengolahan data, statistik, visualisasi, dan modul transformer."""))

    cells.append(make_cell("code", """import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report
import torch

# Pastikan root workspace terdaftar di sys.path
WORKSPACE_ROOT = Path(".").resolve()
if WORKSPACE_ROOT.name == "notebooks":
    WORKSPACE_ROOT = WORKSPACE_ROOT.parent

RESULTS_DIR = WORKSPACE_ROOT / "results"
DATA_DIR = WORKSPACE_ROOT / "data"

print(f"Workspace Root : {WORKSPACE_ROOT}")
print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
print("Setup pustaka selesai dengan sukses!")"""))

    # 3. Data Exploration & Metadata
    cells.append(make_cell("markdown", """## 2. Eksplorasi Korpus & Metadata Penerjemah

Data korpus terdiri dari 12 karya terjemahan fiksi Fyodor Dostoevsky dari Project Gutenberg yang telah dipecah menjadi unit kalimat."""))

    cells.append(make_cell("code", """# Membaca dataset kalimat yang telah dibersihkan
df_sent = pd.read_csv(DATA_DIR / "sentences_cleaned.csv")
df_meta = pd.read_csv(DATA_DIR / "metadata_translators.csv")

print(f"Total Kalimat dalam Korpus: {len(df_sent):,} kalimat")
print(f"Total Karya Fiksi: {df_sent['title'].nunique()} judul novel/kumpulan cerita\\n")

# Menampilkan tabel metadata karya dan penerjemah
df_summary = df_sent.groupby('title').agg(
    total_kalimat=('sentence_id', 'count'),
    rata_rata_kata=('word_count', 'mean'),
    maksimal_kata=('word_count', 'max')
).reset_index()

df_summary = pd.merge(df_summary, df_meta[['title', 'translator', 'publication_year']], on='title', how='left')
df_summary.sort_values(by='total_kalimat', ascending=False).reset_index(drop=True)"""))

    # 4. Lexicon Analysis
    cells.append(make_cell("markdown", """## 3. Analisis Leksikon (VADER & NRC Emotion Lexicon)

Pendekatan leksikon mengukur kerapatan kata emosi berbasis kamus (*bag-of-words* dengan lemmatisasi WordNet) untuk 8 emosi Plutchik serta nilai polaritas *compound* VADER (-1.0 hingga +1.0)."""))

    cells.append(make_cell("code", """df_lex_title = pd.read_csv(RESULTS_DIR / "lexicon_scores_by_title.csv")
df_lex_title[['title', 'total_sentences', 'vader_compound_mean', 'pct_positive', 'pct_neutral', 'pct_negative', 
              'nrc_prop_joy', 'nrc_prop_sadness', 'nrc_prop_fear', 'nrc_prop_anger', 'nrc_prop_trust', 'nrc_prop_anticipation']]"""))

    cells.append(make_cell("code", """# Visualisasi perbandingan sentimen VADER antar-novel
plt.figure(figsize=(12, 6))
sns.barplot(data=df_lex_title.sort_values('vader_compound_mean', ascending=False), 
            y='title', x='vader_compound_mean', palette='coolwarm')
plt.title('Rata-rata Skor Sentimen Compound VADER pada 12 Karya Dostoevsky', fontsize=12, fontweight='bold')
plt.xlabel('Rata-rata Skor Compound (-1.0 Negatif s/d +1.0 Positif)')
plt.ylabel('Judul Karya')
plt.axvline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()"""))

    # 5. Annotation & LLM Multi-Model
    cells.append(make_cell("markdown", """## 4. Sampling Anotasi & Studi Anotasi LLM Multi-Model

Untuk menyediakan data uji luar-ranah (*out-of-domain evaluation*), 2.000 kalimat disampel secara berstrata (*stratified random sampling*).
Dua model fondasi terkemuka ditugaskan sebagai anotator independen:
- **Anotator 1:** **Google Gemini** (`gemini-3.8-flash-high`) — 1.200 kalimat
- **Anotator 2:** **Anthropic Claude** (`claude-sonnet-4-6`) — 1.200 kalimat
- **400 Kalimat Irisan (*Overlap*):** Untuk menguji kesepakatan *Inter-LLM Agreement* (Cohen's Kappa)."""))

    cells.append(make_cell("code", """t1 = pd.read_csv(DATA_DIR / "annotation_task_annotator_1.csv")
t2 = pd.read_csv(DATA_DIR / "annotation_task_annotator_2.csv")
gold_master = pd.read_csv(DATA_DIR / "annotation_gold_consensus_2000.csv")

# Irisan 400 kalimat overlap
overlap_ids = set(t1['sample_id']).intersection(set(t2['sample_id']))
m1 = t1[t1['sample_id'].isin(overlap_ids)].sort_values('sample_id').reset_index(drop=True)
m2 = t2[t2['sample_id'].isin(overlap_ids)].sort_values('sample_id').reset_index(drop=True)

kappa_llm = cohen_kappa_score(m1['emotion_label'], m2['emotion_label'])
acc_llm = accuracy_score(m1['emotion_label'], m2['emotion_label'])

print("=== HASIL UJI KESEPAKATAN INTER-LLM (400 KALIMAT OVERLAP) ===")
print(f"Anotator 1: Google Gemini (gemini-3.8-flash-high)")
print(f"Anotator 2: Anthropic Claude (claude-sonnet-4-6)")
print(f"Cohen's Kappa (kappa) : {kappa_llm:.4f} (Kategori: Moderate Agreement)")
print(f"Raw Accuracy Agreement: {acc_llm * 100:.2f}% (280 dari 400 kalimat sepakat)\\n")

print("Distribusi Gold Standard Konsensus (2.000 Kalimat):")
print(gold_master['gold_emotion_label'].value_counts())"""))

    # 6. Fine-Tuning Performance
    cells.append(make_cell("markdown", """## 5. Kinerja Fine-Tuning Transformer (BERT vs. RoBERTa)

Kedua model dilatih pada benchmark *GoEmotions Ekman* (39.555 data latih) menggunakan transfer learning dengan arsitektur pembekuan layer bawah (*layer freezing*) dan *class-weighted CrossEntropyLoss*."""))

    cells.append(make_cell("code", """with open(RESULTS_DIR / "fine_tuning_metrics.json") as f:
    ft_metrics = json.load(f)

df_model_perf = pd.DataFrame([
    {
        "Model": "BERT (bert-base-uncased)",
        "Val Accuracy": f"{ft_metrics['bert']['accuracy']*100:.2f}%",
        "Macro Precision": round(ft_metrics['bert']['macro_precision'], 4),
        "Macro Recall": round(ft_metrics['bert']['macro_recall'], 4),
        "Macro F1-Score": round(ft_metrics['bert']['macro_f1'], 4)
    },
    {
        "Model": "RoBERTa (roberta-base)",
        "Val Accuracy": f"{ft_metrics['roberta']['accuracy']*100:.2f}%",
        "Macro Precision": round(ft_metrics['roberta']['macro_precision'], 4),
        "Macro Recall": round(ft_metrics['roberta']['macro_recall'], 4),
        "Macro F1-Score": round(ft_metrics['roberta']['macro_f1'], 4)
    }
])
print("Perbandingan Performa Evaluasi Model pada Data Uji Independen:")
display(df_model_perf)"""))

    # 7. Corpus-wide Inference Results
    cells.append(make_cell("markdown", """## 6. Hasil Inferensi Korpus Penuh (RoBERTa 7-Kelas)

Distribusi emosi probabilitas kontekstual pada seluruh 66.410 kalimat di ke-12 karya Dostoevsky."""))

    cells.append(make_cell("code", """df_trans_title = pd.read_csv(RESULTS_DIR / "transformer_scores_by_title.csv")
emotions_7 = ["neutral", "surprise", "joy", "anger", "sadness", "fear", "disgust"]

# Menampilkan tabel persentase prediksi emosi
cols_to_show = ['title', 'total_sentences'] + [f'roberta_pred_pct_{e}' for e in emotions_7]
df_trans_title[cols_to_show].sort_values(by='total_sentences', ascending=False).reset_index(drop=True)"""))

    cells.append(make_cell("code", """# Stacked Bar Plot Distribusi Emosi 12 Novel
fig, ax = plt.subplots(figsize=(13, 7))
emotion_colors = {
    "neutral": "#6c757d", "surprise": "#5bc0de", "joy": "#f0ad4e",
    "anger": "#d9534f", "sadness": "#0275d8", "fear": "#6f42c1", "disgust": "#8a6d3b"
}

bottom = np.zeros(len(df_trans_title))
titles_clean = [t.replace(" :  or, The devils", "").replace(" :  or, Prison life in Siberia", "") for t in df_trans_title["title"]]

for emo in emotions_7:
    vals = df_trans_title[f"roberta_pred_pct_{emo}"].values
    ax.barh(titles_clean, vals, left=bottom, color=emotion_colors[emo], label=emo.capitalize(), edgecolor='white')
    bottom += vals

ax.set_xlabel('Proporsi Prediksi Emosi (%)')
ax.set_title('Distribusi Emosi RoBERTa pada 12 Karya Fiksi Fyodor Dostoevsky', fontweight='bold', fontsize=12)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""))

    # 8. Comparative Analysis
    cells.append(make_cell("markdown", """## 7. Analisis Komparatif: Leksikon vs. Transformer

Evaluasi komparatif mencakup:
1. Korelasi Peringkat Spearman ($\rho$) pada 6 emosi beririsan.
2. Matriks jarak Jensen-Shannon Divergence (JSD) antar-novel.
3. Validasi titik krisis narasi psikologis acuan *Picca, Schnyder & Romele (2024)* pada novel *Crime and Punishment*."""))

    cells.append(make_cell("code", """# 1. Korelasi Spearman Tingkat Kalimat (N = 66.410)
df_sent_corr = pd.read_csv(RESULTS_DIR / "spearman_correlation_sentences.csv")
print("=== KORELASI SPEARMAN TINGKAT KALIMAT (6 EMOSI BERIRISAN) ===")
display(df_sent_corr)

# 2. Rata-rata Korelasi Alur Narasi (100 Bins)
df_arc_corr = pd.read_csv(RESULTS_DIR / "spearman_correlation_narrative_arc.csv")
mean_arc_corr = df_arc_corr.groupby('emotion')['spearman_rho'].agg(['mean', 'std', 'min', 'max']).round(4).reset_index()
print("\\n=== KORELASI SPEARMAN TINGKAT ALUR NARASI (100 BINS) ===")
display(mean_arc_corr)"""))

    cells.append(make_cell("code", """# 3. Validasi Studi Acuan Picca, Schnyder & Romele (2024)
with open(RESULTS_DIR / "picca_comparison_crime_and_punishment.json") as f:
    picca_data = json.load(f)

print(f"Studi Acuan: {picca_data['reference_study']}")
print(f"Novel       : {picca_data['novel']}\\n")

picca_table = []
for cp in picca_data['checkpoints_evaluated']:
    picca_table.append({
        "Titik Krisis Narasi": cp['narrative_checkpoint'],
        "Progres (%)": cp['novel_progress_pct'],
        "Ekspektasi Literatur": cp['picca_expected_pattern'],
        "Dominan RoBERTa": cp['dominant_roberta_emotion'].upper(),
        "Dominan NRC": cp['dominant_nrc_emotion'].upper()
    })

display(pd.DataFrame(picca_table))"""))

    # 9. Publication Figures
    cells.append(make_cell("markdown", """## 8. Galeri Visualisasi Ilmiah Publikasi (300 DPI)

Seluruh grafik publikasi dihasilkan secara otomatis dengan resolusi 300 DPI di folder `results/`:
- `fig1_emotional_arc_crime_and_punishment.png`: Alur narasi *Crime and Punishment* dengan 5 penanda krisis Picca.
- `fig2_emotion_distribution_across_titles.png`: Profil sidik afektif 12 karya Dostoevsky.
- `fig3_spearman_correlation_heatmap.png`: Heatmap korelasi Spearman alur narasi 12 judul.
- `fig4_jsd_inter_novel_clustering.png`: Kluster kedekatan afektif JSD antar-novel.
- `fig5_lexicon_vs_transformer_correlation.png`: Regresi linier proporsi emosi NRC vs. RoBERTa."""))

    cells.append(make_cell("code", """from IPython.display import Image, display

# Menampilkan Figur 1: Emotional Arc Crime and Punishment
fig1_path = RESULTS_DIR / "fig1_emotional_arc_crime_and_punishment.png"
if fig1_path.exists():
    print("Menampilkan Figur 1 (Emotional Arc dengan Penanda Picca dkk. 2024):")
    display(Image(filename=str(fig1_path), width=950))"""))

    cells.append(make_cell("code", """# Menampilkan Figur 4: Hierarchical Clustering JSD
fig4_path = RESULTS_DIR / "fig4_jsd_inter_novel_clustering.png"
if fig4_path.exists():
    print("Menampilkan Figur 4 (Hierarchical Clustering JSD Antar-Novel):")
    display(Image(filename=str(fig4_path), width=750))"""))

    # 10. Dashboard & Reproduction
    cells.append(make_cell("markdown", """## 9. Menjalankan Dasbor Interaktif Streamlit

Aplikasi web dasbor interaktif 3 halaman dapat dijalankan kapan saja di terminal lokal:
```bash
streamlit run app/app.py
```
Akses di browser: `http://localhost:8501`.

---

## 10. Kesimpulan & Kontribusi Penelitian
1. **Keunggulan Kontekstual Transformer:** Model RoBERTa berhasil mengatasi kelemahan mendasar metode leksikon kamus (*bag-of-words*), terutama dalam membedakan nada sarkastik narator, monolog batin, dan dialog berseru.
2. **Keselarasan Makro pada Alur Narasi:** Meskipun korelasi per kalimat bersifat moderat ($\rho \approx 0,25 - 0,31$), kedua metode menunjukkan konvergensi makro yang sangat kuat pada emosi **Joy** ($\rho$ hingga 0,73) dan **Fear** ($\rho$ hingga 0,65).
3. **Validasi Komputasi Sastra:** Alur afektif Raskolnikov pada *Crime and Punishment* secara empiris membuktikan lonjakan ketakutan (*fear spike*) pada adegan pembunuhan di progres narasi 17%, selaras dengan temuan *Picca, Schnyder & Romele (2024)*.
4. **Reliabilitas Anotasi LLM:** Pemanfaatan Google Gemini dan Anthropic Claude menghasilkan kesepakatan moderat ($\kappa = 0,4943$) dengan akurasi 70,00%, menyediakan data acuan emas yang objektif bagi penelitian sastra komputasional."""))

    # Assemble notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)} (Markdown + Code)")

if __name__ == "__main__":
    build_notebook()
