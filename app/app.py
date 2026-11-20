"""
Streamlit Interactive Dashboard for Thesis Project:
"Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky Menggunakan Pendekatan Leksikon dan Transformer"

Three Main Pages:
1. Emotional Arc per Judul: Visualisasi alur emosi dinamis sepanjang narasi novel.
2. Perbandingan Distribusi Emosi Antar-Judul: Profil emosi komparatif 12 karya Dostoevsky.
3. Perbandingan Leksikon vs. Transformer: Analisis korelasi Spearman, JSD, dan benchmark Picca dkk. (2024).

Constraint: Dasbor hanya membaca file di results/ dan data/, tidak melakukan inferensi ulang.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE_ROOT / "results"
DATA_DIR = WORKSPACE_ROOT / "data"

# Page config
st.set_page_config(
    page_title="Dostoevsky Emotion & Sentiment Analytics",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette for emotions
EMOTION_COLORS = {
    "anger": "#d9534f",
    "disgust": "#8a6d3b",
    "fear": "#6f42c1",
    "joy": "#f0ad4e",
    "sadness": "#0275d8",
    "surprise": "#5bc0de",
    "neutral": "#6c757d",
    "anticipation": "#e83e8c",
    "trust": "#20c997",
}

INTERSECTING_6 = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
TRANSFORMER_7 = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NRC_8 = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


# ==========================================
# Cached Data Loading (No Recomputation)
# ==========================================
@st.cache_data
def load_title_summaries():
    lex_title_path = RESULTS_DIR / "lexicon_scores_by_title.csv"
    trans_title_path = RESULTS_DIR / "transformer_scores_by_title.csv"
    trans_test_path = RESULTS_DIR / "transformer_scores_by_title_test.csv"
    meta_path = DATA_DIR / "metadata_translators.csv"

    lex_df = pd.read_csv(lex_title_path) if lex_title_path.exists() else pd.DataFrame()
    
    if trans_title_path.exists():
        trans_df = pd.read_csv(trans_title_path)
    elif trans_test_path.exists():
        trans_df = pd.read_csv(trans_test_path)
    else:
        trans_df = pd.DataFrame()

    meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    return lex_df, trans_df, meta_df


@st.cache_data
def load_narrative_arcs():
    lex_arc_path = RESULTS_DIR / "lexicon_scores_narrative_arc.csv"
    trans_arc_path = RESULTS_DIR / "transformer_scores_narrative_arc.csv"
    trans_arc_test = RESULTS_DIR / "transformer_scores_narrative_arc_test.csv"

    lex_arc = pd.read_csv(lex_arc_path) if lex_arc_path.exists() else pd.DataFrame()
    if trans_arc_path.exists():
        trans_arc = pd.read_csv(trans_arc_path)
    elif trans_arc_test.exists():
        trans_arc = pd.read_csv(trans_arc_test)
    else:
        trans_arc = pd.DataFrame()

    return lex_arc, trans_arc


@st.cache_data
def load_comparative_metrics():
    comp_file = RESULTS_DIR / "comparative_analysis_metrics.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_sample_sentences(sample_n=500):
    sent_path = RESULTS_DIR / "transformer_scores_sentences.csv"
    if not sent_path.exists():
        sent_path = RESULTS_DIR / "transformer_scores_crime_and_punishment_roberta.csv"
    if sent_path.exists():
        df = pd.read_csv(sent_path, nrows=sample_n)
        return df
    return pd.DataFrame()


# Load core data
lex_titles, trans_titles, meta_translators = load_title_summaries()
lex_arcs, trans_arcs = load_narrative_arcs()
comparative_metrics = load_comparative_metrics()

# Available titles
available_titles = sorted(lex_titles["title"].unique()) if not lex_titles.empty else []


# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("📖 Dostoevsky NLP")
st.sidebar.markdown(
    "**Analisis Sentimen & Distribusi Emosi**\n\n"
    "*Leksikon (NRC/VADER) vs. Transformer (RoBERTa/BERT)*"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    [
        "📈 1. Emotional Arc per Judul",
        "📊 2. Distribusi Emosi Antar-Judul",
        "🔬 3. Leksikon vs. Transformer",
        "🔍 4. Penjelajah Kalimat & Anotasi",
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Metadata Korpus")
st.sidebar.write(f"- **Total Karya:** {len(available_titles)} novel/kumpulan cerita")
st.sidebar.write("- **Bahasa:** Terjemahan Inggris (Gutenberg)")
st.sidebar.write("- **Total Kalimat:** 66.410 kalimat")
st.sidebar.write("- **Penerjemah:** Constance Garnett, Eva Martin, C.J. Hogarth, Frederick Whishaw")


# ==============================================================================
# HALAMAN 1: EMOTIONAL ARC PER JUDUL
# ==============================================================================
if page == "📈 1. Emotional Arc per Judul":
    st.title("📈 Alur Emosi Dinamis (Emotional Arc) per Karya Fiksi")
    st.markdown(
        "Halaman ini memvisualisasikan bagaimana dinamika emosi berkembang dari awal kisah (0%) hingga akhir/epilog (100%) "
        "menggunakan 100 *narrative bins*. Anda dapat membandingkan kurva probabilitas Transformer kontekstual dengan kurva leksikon NRC."
    )

    if not available_titles:
        st.warning("Data alur narasi belum dimuat. Pastikan file di folder `results/` sudah lengkap.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
        with col_ctrl1:
            selected_title = st.selectbox("Pilih Judul Novel:", available_titles, index=0 if "Crime and Punishment" in available_titles else 0)
        with col_ctrl2:
            method_view = st.selectbox("Pendekatan Analisis:", ["Keduanya (Side-by-Side)", "Transformer (RoBERTa)", "Leksikon (NRC)"])
        with col_ctrl3:
            smoothing = st.slider("Penghalusan Kurva (Rolling Window):", min_value=1, max_value=15, value=5, help="Ukuran jendela rolling mean untuk menangkap tren makro alur narasi.")

        # Filter emotions to plot
        st.markdown("##### Pilih Emosi yang Ditampilkan:")
        default_emos = ["fear", "sadness", "joy", "anger"]
        cols_emo = st.columns(6)
        active_emotions = []
        for i, emo in enumerate(INTERSECTING_6):
            with cols_emo[i]:
                if st.checkbox(emo.capitalize(), value=(emo in default_emos), key=f"chk_{emo}"):
                    active_emotions.append(emo)

        show_picca = False
        if selected_title == "Crime and Punishment":
            show_picca = st.checkbox("📌 Tampilkan Titik Krisis Narasi Acuan Picca dkk. (2024)", value=True)

        st.divider()

        # Data extraction for selected title
        rob_sub = trans_arcs[trans_arcs["title"] == selected_title].sort_values("narrative_progress_pct") if not trans_arcs.empty else pd.DataFrame()
        lex_sub = lex_arcs[lex_arcs["title"] == selected_title].sort_values("narrative_progress_pct") if not lex_arcs.empty else pd.DataFrame()

        # Checkpoint markers for Crime and Punishment
        picca_checkpoints = [
            {"x": 17, "label": "1. Pembunuhan Alyona (Part I)", "color": EMOTION_COLORS["fear"]},
            {"x": 32, "label": "2. Kematian Marmeladov (Part II)", "color": EMOTION_COLORS["sadness"]},
            {"x": 67, "label": "3. Duel Interogasi Porfiry (Part IV-V)", "color": EMOTION_COLORS["surprise"]},
            {"x": 88, "label": "4. Bunuh Diri Svidrigailov (Part VI)", "color": EMOTION_COLORS["anger"]},
            {"x": 98, "label": "5. Penebusan di Siberia (Epilog)", "color": EMOTION_COLORS["joy"]}
        ]

        if method_view == "Keduanya (Side-by-Side)":
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                subplot_titles=(
                    f"Transformer (Fine-tuned RoBERTa) - Probabilitas Kontekstual ({selected_title})",
                    f"Leksikon (NRC Emotion Lexicon) - Kerapatan Kata Emosi ({selected_title})"
                )
            )

            # 1. RoBERTa
            if not rob_sub.empty:
                for emo in active_emotions:
                    y_val = rob_sub[f"roberta_prob_{emo}"].rolling(smoothing, min_periods=1).mean() * 100
                    fig.add_trace(
                        go.Scatter(
                            x=rob_sub["narrative_progress_pct"], y=y_val,
                            mode="lines", name=f"{emo.capitalize()} (RoBERTa)",
                            line=dict(color=EMOTION_COLORS[emo], width=2.5)
                        ),
                        row=1, col=1
                    )

            # 2. NRC
            if not lex_sub.empty:
                for emo in active_emotions:
                    y_val = lex_sub[f"nrc_prop_{emo}"].rolling(smoothing, min_periods=1).mean() * 100
                    fig.add_trace(
                        go.Scatter(
                            x=lex_sub["narrative_progress_pct"], y=y_val,
                            mode="lines", name=f"{emo.capitalize()} (NRC)",
                            line=dict(color=EMOTION_COLORS[emo], width=2.2, dash="dash")
                        ),
                        row=2, col=1
                    )

            if show_picca and selected_title == "Crime and Punishment":
                for cp in picca_checkpoints:
                    for row_idx in [1, 2]:
                        fig.add_vline(x=cp["x"], line_width=1.5, line_dash="dot", line_color=cp["color"], row=row_idx, col=1)
                        if row_idx == 1:
                            fig.add_annotation(
                                x=cp["x"], y=32, text=cp["label"], showarrow=False,
                                textangle=-90, font=dict(size=9, color=cp["color"]), row=row_idx, col=1
                            )

            fig.update_layout(height=700, hovermode="x unified", template="plotly_white")
            fig.update_xaxes(title_text="Progres Narasi Novel (%) [0 = Bab Awal, 100 = Akhir Kisah]", row=2, col=1)
            fig.update_yaxes(title_text="Probabilitas (%)", row=1, col=1)
            fig.update_yaxes(title_text="Kerapatan Kata (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        elif method_view == "Transformer (RoBERTa)":
            fig = go.Figure()
            if not rob_sub.empty:
                for emo in active_emotions:
                    y_val = rob_sub[f"roberta_prob_{emo}"].rolling(smoothing, min_periods=1).mean() * 100
                    fig.add_trace(
                        go.Scatter(
                            x=rob_sub["narrative_progress_pct"], y=y_val,
                            mode="lines", name=emo.capitalize(),
                            line=dict(color=EMOTION_COLORS[emo], width=2.5)
                        )
                    )

            if show_picca and selected_title == "Crime and Punishment":
                for cp in picca_checkpoints:
                    fig.add_vline(x=cp["x"], line_width=1.5, line_dash="dot", line_color=cp["color"])
                    fig.add_annotation(x=cp["x"], y=32, text=cp["label"], showarrow=False, textangle=-90, font=dict(size=10, color=cp["color"]))

            fig.update_layout(
                title=f"Dinamika Alur Emosi RoBERTa: {selected_title}",
                xaxis_title="Progres Narasi Novel (%)",
                yaxis_title="Probabilitas Emosi (%)",
                height=500, template="plotly_white", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        else: # Leksikon NRC
            fig = go.Figure()
            if not lex_sub.empty:
                for emo in active_emotions:
                    y_val = lex_sub[f"nrc_prop_{emo}"].rolling(smoothing, min_periods=1).mean() * 100
                    fig.add_trace(
                        go.Scatter(
                            x=lex_sub["narrative_progress_pct"], y=y_val,
                            mode="lines", name=emo.capitalize(),
                            line=dict(color=EMOTION_COLORS[emo], width=2.5)
                        )
                    )

            fig.update_layout(
                title=f"Dinamika Alur Emosi Leksikon NRC: {selected_title}",
                xaxis_title="Progres Narasi Novel (%)",
                yaxis_title="Kerapatan Kata Emosi (%)",
                height=500, template="plotly_white", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# HALAMAN 2: PERBANDINGAN DISTRIBUSI EMOSI ANTAR-JUDUL
# ==============================================================================
elif page == "📊 2. Distribusi Emosi Antar-Judul":
    st.title("📊 Perbandingan Distribusi Emosi Antar-Judul")
    st.markdown(
        "Halaman ini membandingkan sidik emosi (*affective fingerprint*) dari ke-12 karya Dostoevsky. "
        "Anda dapat memeriksa bagaimana novel filosofis akhir (*The Brothers Karamazov*) berbeda secara signifikan dengan karya epistolary awal (*Poor Folk*)."
    )

    if trans_titles.empty and lex_titles.empty:
        st.warning("Data agregasi judul belum tersedia.")
    else:
        col_tab1, col_tab2 = st.tabs(["📊 Stacked Bar Chart", "🕸️ Radar Chart Perbandingan"])

        with col_tab1:
            st.subheader("Distribusi Proporsi Emosi Komparatif (12 Judul)")
            model_choice = st.radio("Pilih Pendekatan Data:", ["Transformer (RoBERTa)", "Leksikon (NRC)"], horizontal=True)

            if model_choice == "Transformer (RoBERTa)" and not trans_titles.empty:
                # Merge translator metadata
                merged_t = pd.merge(trans_titles, meta_translators, on="title", how="left")
                fig = go.Figure()
                for emo in ["neutral", "surprise", "fear", "sadness", "anger", "joy", "disgust"]:
                    fig.add_trace(go.Bar(
                        y=merged_t["title"],
                        x=merged_t[f"roberta_pred_pct_{emo}"],
                        name=emo.capitalize(),
                        orientation="h",
                        marker=dict(color=EMOTION_COLORS[emo])
                    ))
                fig.update_layout(
                    barmode="stack", height=600,
                    xaxis_title="Proporsi Prediksi (%)",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif not lex_titles.empty:
                merged_l = pd.merge(lex_titles, meta_translators, on="title", how="left")
                fig = go.Figure()
                for emo in NRC_8:
                    fig.add_trace(go.Bar(
                        y=merged_l["title"],
                        x=merged_l[f"nrc_prop_{emo}"] * 100,
                        name=emo.capitalize(),
                        orientation="h",
                        marker=dict(color=EMOTION_COLORS.get(emo, "#333333"))
                    ))
                fig.update_layout(
                    barmode="stack", height=600,
                    xaxis_title="Kerapatan Kata Emosi (%)",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_tab2:
            st.subheader("Radar Chart Profil Emosi (Pilih Hingga 4 Judul)")
            selected_radar_titles = st.multiselect(
                "Pilih Novel untuk Dibandingkan:",
                available_titles,
                default=["Crime and Punishment", "The Brothers Karamazov", "Notes from the Underground"] if len(available_titles) >= 3 else available_titles[:2]
            )

            if selected_radar_titles:
                fig_radar = go.Figure()
                categories = [e.capitalize() for e in INTERSECTING_6]

                for t in selected_radar_titles:
                    sub = trans_titles[trans_titles["title"] == t]
                    if not sub.empty:
                        vals = [sub[f"roberta_pred_pct_{e}"].values[0] for e in INTERSECTING_6]
                        vals.append(vals[0]) # close radar loop
                        cats = categories + [categories[0]]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals, theta=cats, fill="toself", name=t
                        ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
                    showlegend=True, height=550, template="plotly_white"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()
        st.subheader("Tabel Statistik Ringkasan per Judul")
        st.dataframe(trans_titles if not trans_titles.empty else lex_titles, use_container_width=True)


# ==============================================================================
# HALAMAN 3: PERBANDINGAN LEKSIKON VS. TRANSFORMER
# ==============================================================================
elif page == "🔬 3. Leksikon vs. Transformer":
    st.title("🔬 Perbandingan Metodologis: Leksikon vs. Transformer")
    st.markdown(
        "Halaman ini menyajikan evaluasi komparatif matematis antara pendekatan leksikon berbasis kamus (NRC) "
        "dan pendekatan model bahasa probabilistik kontekstual (Fine-tuned RoBERTa)."
    )

    tab_eval1, tab_eval2, tab_eval3, tab_eval4 = st.tabs([
        "📐 Korelasi Spearman", 
        "🌐 Jensen-Shannon Divergence (JSD)", 
        "🎯 Evaluasi Benchmark Picca dkk. (2024)",
        "👥 Kesepakatan Anotasi (Cohen's Kappa)"
    ])

    with tab_eval1:
        st.subheader("Korelasi Peringkat Spearman pada 6 Emosi Beririsan")
        st.markdown(
            "Korelasi dihitung antara skor frekuensi NRC per kalimat / bin narasi dengan nilai probabilitas RoBERTa "
            "hanya pada 6 kelas emosi yang beririsan (*anger, disgust, fear, joy, sadness, surprise*)."
        )

        sent_corr_file = RESULTS_DIR / "spearman_correlation_sentences.csv"
        arc_corr_file = RESULTS_DIR / "spearman_correlation_narrative_arc.csv"

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Korelasi Tingkat Kalimat (66.410 Kalimat)")
            if sent_corr_file.exists():
                df_sc = pd.read_csv(sent_corr_file)
                st.dataframe(df_sc, use_container_width=True)
            else:
                st.info("Kalkulasi korelasi kalimat sedang diproses atau belum dijalankan.")

        with col_c2:
            st.markdown("##### Rata-rata Korelasi Alur Narasi (100 Bins)")
            if arc_corr_file.exists():
                df_ac = pd.read_csv(arc_corr_file)
                mean_arc_corr = df_ac.groupby("emotion")["spearman_rho"].mean().reset_index()
                mean_arc_corr.columns = ["Emosi", "Rata-rata Spearman rho"]
                st.dataframe(mean_arc_corr, use_container_width=True)
            else:
                st.info("Kalkulasi korelasi alur narasi sedang diproses.")

    with tab_eval2:
        st.subheader("Jensen-Shannon Divergence (JSD) Antar-Novel")
        st.markdown(
            "JSD mengukur jarak simetris antara dua distribusi probabilitas emosi ($0 \\le JSD \\le 1$). "
            "Semakin kecil nilai JSD, semakin mirip profil emosional kedua karya tersebut."
        )

        jsd_rob_file = RESULTS_DIR / "jsd_matrix_roberta.csv"
        jsd_cross_file = RESULTS_DIR / "jsd_cross_methods.csv"

        col_j1, col_j2 = st.columns([3, 2])
        with col_j1:
            if jsd_rob_file.exists():
                st.markdown("##### Heatmap Jarak JSD Antar-Novel (RoBERTa)")
                df_j = pd.read_csv(jsd_rob_file, index_col=0)
                fig_hm = px.imshow(
                    df_j, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis",
                    title="Matriks JSD Divergence Antar-Novel"
                )
                fig_hm.update_layout(height=500)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Matriks JSD sedang diproses.")

        with col_j2:
            if jsd_cross_file.exists():
                st.markdown("##### Divergensi Leksikon vs RoBERTa per Judul")
                df_cross = pd.read_csv(jsd_cross_file)
                st.dataframe(df_cross, use_container_width=True)
            else:
                st.info("Data cross-method JSD belum tersedia.")

    with tab_eval3:
        st.subheader("Validasi Benchmark terhadap Picca, Schnyder & Romele (2024)")
        st.markdown(
            "Studi acuan oleh **Picca, Schnyder & Romele (2024)** menganalisis alur sentimen dan emosi pada *Crime and Punishment*. "
            "Berikut perbandingan empiris deteksi titik krisis psikologis Raskolnikov antara model kita dan temuan literatur."
        )

        picca_file = RESULTS_DIR / "picca_comparison_crime_and_punishment.json"
        if picca_file.exists():
            with open(picca_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)

            st.write(f"**Referensi:** {p_data.get('reference_study')}")
            for cp in p_data.get("checkpoints_evaluated", []):
                with st.expander(f"📍 {cp['narrative_checkpoint']} (Progres {cp['novel_progress_pct']})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**Pola Ekspektasi Literatur:**\n{cp['picca_expected_pattern']}")
                    with c2:
                        st.write(f"**Prediksi RoBERTa (Dominan: {cp['dominant_roberta_emotion'].upper()}):**")
                        st.json(cp["empirical_roberta_probs_pct"])
                    with c3:
                        st.write(f"**Kerapatan NRC (Dominan: {cp['dominant_nrc_emotion'].upper()}):**")
                        st.json(cp["empirical_nrc_pct"])
        else:
            st.info("File evaluasi benchmark Picca dkk. sedang disiapkan.")

    with tab_eval4:
        st.subheader("Evaluasi Anotasi LLM Multi-Model & Cohen's Kappa")
        annot_info = comparative_metrics.get("annotation_and_agreement", {})

        st.markdown(
            """
            - **Desain Partisi Anotasi:** 2.000 kalimat disampel secara acak berlapis (*stratified random sampling*).
            - **Anotator 1:** **Google Gemini** (`gemini-3.8-flash-high`) — 1.200 kalimat.
            - **Anotator 2:** **Anthropic Claude** (`claude-sonnet-4-6`) — 1.200 kalimat.
            - **Subset Irisan Uji Reliabilitas:** 400 kalimat dikerjakan ganda oleh **Gemini** dan **Claude** secara independen (*double-blind*).
            - **Subset Mandiri:** 800 kalimat solo Gemini, 800 kalimat solo Claude.
            """
        )

        status = annot_info.get("human_annotation_status", "pending")
        if "completed" in status:
            kappa_val = annot_info.get("cohens_kappa_gemini_vs_claude", annot_info.get("cohens_kappa_annotator_1_vs_2"))
            acc_val = annot_info.get("raw_agreement_gemini_vs_claude", 0.70)
            st.success(f"✅ Anotasi LLM Multi-Model Selesai! Cohen's Kappa Gemini vs. Claude (400 kalimat overlap): **{kappa_val}** (Raw Agreement: **{acc_val * 100:.1f}%**)")
        else:
            st.warning("⏳ Anotasi sedang diproses.")

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("##### Evaluasi Model vs. Gold Standard Konsensus (2.000 Kalimat):")
            rob_gold = annot_info.get("roberta_vs_gold_consensus_2000", {})
            bert_gold = annot_info.get("bert_vs_gold_consensus_2000", {})
            if rob_gold and bert_gold:
                df_eval_gold = pd.DataFrame([
                    {"Model": "RoBERTa (Fine-tuned)", "Akurasi": f"{rob_gold.get('accuracy') * 100:.2f}%", "Macro F1": rob_gold.get('macro_f1'), "Cohen's Kappa": rob_gold.get('cohens_kappa')},
                    {"Model": "BERT (Fine-tuned)", "Akurasi": f"{bert_gold.get('accuracy') * 100:.2f}%", "Macro F1": bert_gold.get('macro_f1'), "Cohen's Kappa": bert_gold.get('cohens_kappa')},
                ])
                st.dataframe(df_eval_gold, use_container_width=True)

        with c_m2:
            st.markdown("##### Kesepakatan Antar-Model Transformer:")
            bert_rob = annot_info.get("bert_vs_roberta_agreement_2000", {})
            if bert_rob:
                st.write(f"- **Cohen's Kappa (BERT vs. RoBERTa):** `{bert_rob.get('cohens_kappa')}` (*Substantial Agreement*)")
                st.write(f"- **Raw Accuracy Agreement:** `{bert_rob.get('raw_agreement_accuracy') * 100:.2f}%`")


# ==============================================================================
# HALAMAN 4: PENJELAJAH KALIMAT & ANOTASI
# ==============================================================================
elif page == "🔍 4. Penjelajah Kalimat & Anotasi":
    st.title("🔍 Penjelajah Kalimat (Corpus & Prediction Explorer)")
    st.markdown("Eksplorasi kalimat-kalimat fiksi Dostoevsky beserta rincian probabilitas prediksi emosi 7-kelas dari model Transformer.")

    df_sent = load_sample_sentences(sample_n=1000)
    if not df_sent.empty:
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            search_query = st.text_input("Cari Kata / Frasa:", placeholder="misal: murder, blood, money, God, soul...")
        with col_f2:
            pred_filter = st.selectbox("Filter Emosi Prediksi:", ["Semua"] + TRANSFORMER_7)

        filtered = df_sent.copy()
        if search_query:
            filtered = filtered[filtered["text"].str.contains(search_query, case=False, na=False)]
        if pred_filter != "Semua" and "roberta_pred_emotion" in filtered.columns:
            filtered = filtered[filtered["roberta_pred_emotion"] == pred_filter]

        st.write(f"Menampilkan **{len(filtered)}** kalimat:")
        display_cols = ["sentence_id", "title", "text", "roberta_pred_emotion"] + [f"roberta_prob_{e}" for e in INTERSECTING_6]
        available_display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_display_cols].head(100), use_container_width=True)
    else:
        st.info("Dataset kalimat sedang diproses.")
