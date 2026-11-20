# Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky Menggunakan Pendekatan Leksikon dan Transformer

Repositori kode komprehensif untuk penelitian tugas akhir (skripsi) yang berfokus pada:
1. Pengukuran sentimen dan distribusi emosi pada 12 karya fiksi klasik Fyodor Dostoevsky (terjemahan bahasa Inggris dari Project Gutenberg, total 66.410 kalimat).
2. Perbandingan metodologis antara pendekatan berbasis leksikon (**VADER** dan **NRC Emotion Lexicon**) dengan pendekatan model bahasa probabilistik kontekstual (**Fine-tuned BERT** dan **RoBERTa**).
3. Analisis komparatif distribusi emosi antar-novel menggunakan **Jensen-Shannon Divergence (JSD)** dan korelasi alur narasi menggunakan **Spearman Rank Correlation ($\rho$)**.
4. Validasi empiris terhadap studi acuan komputasi sastra: **Picca, Schnyder & Romele (2024)** pada novel *Crime and Punishment*.
5. Penyajian hasil dalam dasbor analitik interaktif berbasis web (**Streamlit**) multi-halaman tanpa beban komputasi ulang (*zero recomputation*).

---

## 📁 Struktur Direktori Repositori

```text
NLP/
├── data/                                         # Data terstruktur & tugas anotasi
│   ├── annotation_master_2000.csv                # 2.000 kalimat sampel emas berstrata
│   ├── annotation_task_annotator_1.csv           # 1.200 kalimat tugas Anotator 1 (400 overlap)
│   ├── annotation_task_annotator_2.csv           # 1.200 kalimat tugas Anotator 2 (400 overlap)
│   ├── ANNOTATION_GUIDELINES.md                  # Panduan anotasi 7 kelas emosi Ekman
│   ├── metadata_translators.csv                  # Metadata penerjemah dan statistik 12 karya
│   ├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt   # Leksikon emosi NRC resmi (14.182 lema)
│   ├── goemotions_ekman_train.csv                # Data latih GoEmotions Ekman (39.555 baris)
│   ├── goemotions_ekman_validation.csv           # Data validasi (4.946 baris)
│   └── goemotions_ekman_test.csv                 # Data uji (4.968 baris)
├── src/                                          # Modul kode modular
│   ├── config.py                                 # Konfigurasi seed (42), path, hardware detection
│   ├── inspect_data.py                           # Audit kualitas data kalimat & pembersihan
│   ├── lexicon_analysis.py                       # Pipeline analisis VADER & NRC leksikon
│   ├── sample_annotation.py                      # Stratified sampling & pembagian tugas anotator
│   ├── prepare_emotion_dataset.py                # Ekstraksi & pemetaan benchmark GoEmotions Ekman
│   ├── train_transformer.py                      # Fine-tuning transfer learning BERT & RoBERTa
│   ├── infer_transformer.py                      # Engine inferensi korpus penuh dengan chunking
│   ├── compare_methods.py                        # Kalkulasi Spearman, JSD, Kappa, benchmark Picca
│   └── generate_visualizations.py                # Pembangkit 5 gambar publikasi ilmiah (300 DPI)
├── results/                                      # Artefak hasil, model, metrik, dan plot
│   ├── models/                                   # Bobot model terlatih (BERT & RoBERTa)
│   ├── checkpoints_inference/roberta/            # Checkpoints inferensi tersimpan per novel
│   ├── lexicon_scores_sentences.csv              # Skor leksikon per kalimat (66.410 baris)
│   ├── lexicon_scores_by_title.csv               # Agregasi skor leksikon 12 judul
│   ├── lexicon_scores_narrative_arc.csv          # Alur narasi leksikon 100 bins per judul
│   ├── transformer_scores_sentences.csv          # Prediksi RoBERTa per kalimat (66.410 baris)
│   ├── transformer_scores_by_title.csv           # Agregasi probabilitas RoBERTa 12 judul
│   ├── transformer_scores_narrative_arc.csv      # Alur narasi RoBERTa 100 bins per judul
│   ├── transformer_scores_sample_2000_bert.csv   # Prediksi BERT pada 2.000 sampel anotasi
│   ├── spearman_correlation_sentences.csv        # Koefisien Spearman tingkat kalimat
│   ├── spearman_correlation_narrative_arc.csv    # Koefisien Spearman alur narasi per judul
│   ├── jsd_matrix_roberta.csv                    # Matriks jarak JSD 12x12 RoBERTa
│   ├── jsd_matrix_nrc.csv                        # Matriks jarak JSD 12x12 NRC
│   ├── jsd_cross_methods.csv                     # Peringkat divergensi leksikon vs RoBERTa
│   ├── comparative_analysis_metrics.json         # Ringkasan master metrik komparasi
│   ├── picca_comparison_crime_and_punishment.json# Validasi kuantitatif terhadap studi Picca (2024)
│   ├── fine_tuning_metrics.json                  # Riwayat evaluasi pelatihan BERT & RoBERTa
│   └── fig*.png                                  # 5 Gambar publikasi beresolusi tinggi (300 DPI)
├── app/                                          # Dasbor interaktif Streamlit
│   └── app.py                                    # Aplikasi web multi-halaman interaktif
├── notebooks/                                    # Ruang eksperimen dan verifikasi visual
├── sentences.csv                                 # Korpus kalimat asli (66.410 baris)
├── requirements.txt                              # Dependensi pustaka terkunci (105 packages)
└── README.md                                     # Dokumentasi lengkap proyek
```

---

## ⚙️ Persyaratan Sistem & Instalasi

### 1. Spesifikasi Hardware & Software
- **Sistem Operasi:** Windows / Linux / macOS
- **Bahasa:** Python 3.11 (direkomendasikan)
- **Komputasi:** Deteksi perangkat otomatis. Dilengkapi optimasi CPU multi-threading (`torch.set_num_threads(8)`) dan *sliding window pooling* untuk monolog panjang tanpa ketergantungan GPU khusus.

### 2. Instalasi Lingkungan Virtual
Gunakan `uv` atau `venv` standar:
```bash
# Membuat virtual environment Python 3.11
uv venv .venv --python 3.11

# Aktivasi virtual environment (Windows PowerShell)
.venv\Scripts\activate

# Pemasangan dependensi terkunci
uv pip install -r requirements.txt
```

---

## 📚 Korpus Karya Fyodor Dostoevsky & Penerjemah

Seluruh teks merupakan karya terjemahan bahasa Inggris dari Project Gutenberg:

| No | ID Gutenberg | Judul Karya | Kalimat | Rata-rata Kata | Penerjemah Utama |
|:--:|:--:|---|:--:|:--:|---|
| 1 | 28054 | The Brothers Karamazov | 13.763 | 25,37 | Constance Garnett |
| 2 | 8117 | The Possessed (The Devils) | 10.847 | 23,27 | Constance Garnett |
| 3 | 2638 | The Idiot | 9.722 | 24,84 | Eva Martin |
| 4 | 2554 | Crime and Punishment | 8.342 | 24,29 | Constance Garnett |
| 5 | 37536 | The House of the Dead | 5.955 | 20,45 | Constance Garnett |
| 6 | 38241 | Uncle's Dream & The Permanent Husband | 3.880 | 25,22 | Frederick Whishaw |
| 7 | 40745 | Short Stories | 3.829 | 21,02 | Constance Garnett |
| 8 | 36034 | White Nights and Other Stories | 3.094 | 23,43 | Constance Garnett |
| 9 | 2197 | The Gambler | 2.572 | 23,38 | C. J. Hogarth |
| 10 | 2302 | Poor Folk | 2.270 | 23,81 | C. J. Hogarth |
| 11 | 600 | Notes from the Underground | 1.950 | 22,50 | Constance Garnett |
| 12 | 13437 | Best Russian Short Stories | 186 | 14,91 | Thomas Seltzer (Ed.) |
| **Total** | - | **12 Karya Fiksi** | **66.410** | **24,14** | - |

---

## 🚀 Panduan Eksekusi Lengkap (End-to-End Pipeline)

Setiap tahapan dapat dijalankan secara terisolasi dari terminal PowerShell:

### Tahap 0: Audit Kualitas Data
```bash
python src/inspect_data.py
```
*Memverifikasi 66.410 baris, mendeteksi monolog ekstrem (>250 kata), mengisolasi duplikasi, dan memetakan metadata penerjemah.*

### Tahap 1: Analisis Leksikon (VADER & NRC)
```bash
python src/lexicon_analysis.py
```
*Menghitung compound score VADER dan 8 emosi NRC pada setiap kalimat serta mengagregasikannya per judul dan 100 bins alur cerita.*

### Tahap 2: Sampling Anotasi Berstrata (2.000 Kalimat)
```bash
python src/sample_annotation.py
```
*Mengambil 2.000 kalimat proporsional berlapis dan membaginya menjadi 400 kalimat irisan (overlap) untuk Cohen's Kappa dan 800 kalimat solo per anotator secara double-blind.*

### Tahap 3: Persiapan Data & Fine-Tuning Transformer (7 Kelas Ekman)
```bash
# 1. Unduh dan petakan benchmark GoEmotions ke 7 kelas Ekman
python src/prepare_emotion_dataset.py

# 2. Latih model BERT dan RoBERTa dengan class-weighted CrossEntropyLoss
python src/train_transformer.py
```

### Tahap 4: Inferensi Transformer Seluruh Korpus (66.410 Kalimat)
```bash
# Inferensi RoBERTa untuk seluruh 12 judul (dengan sliding window chunking)
python src/infer_transformer.py --full --model roberta --batch_size 32
```

### Tahap 5: Analisis Komparatif (Spearman, JSD, Kappa, Benchmark Picca)
```bash
python src/compare_methods.py
```

### Tahap 6: Pembangkitan Gambar Publikasi (300 DPI)
```bash
python src/generate_visualizations.py
```

### Tahap 7: Menjalankan Dasbor Interaktif Streamlit
```bash
streamlit run app/app.py
```
*Buka browser pada alamat `http://localhost:8501`.*

---

## 🔬 Ringkasan Temuan Empiris & Hasil Penelitian

### 1. Performa Model Fine-Tuning (Ekman 7-Kelas)

Evaluasi dilakukan pada dataset independen *GoEmotions Ekman Test Split* (4.968 sampel):

| Model Arsitektur | Base Pretrained | Akurasi Uji | Macro Precision | Macro Recall | Macro F1-Score | Status |
|---|---|:--:|:--:|:--:|:--:|:--:|
| **BERT** | `bert-base-uncased` | 66,43% | 0,6669 | 0,6610 | 0,6634 | Model Baseline |
| **RoBERTa** | `roberta-base` | **69,52%** | **0,7018** | **0,6832** | **0,6905** | **Champion Model** |

*RoBERTa mengungguli BERT sebesar +3,09% akurasi dan +2,71% F1-score makro, khususnya pada pembedaan emosi bernuansa halus (sadness vs fear).*

### 2. Distribusi Emosi Global pada Karya Dostoevsky (RoBERTa)

Dari 66.410 kalimat, proporsi prediksi emosi didominasi oleh:
- **Neutral:** `36,69%` (dialog natural dan narasi eksposisi)
- **Surprise:** `20,29%` (seruan dramatis, kejutan emosional, konfrontasi tokoh)
- **Joy:** `10,92%` (afeksi, kehangatan persahabatan, humor satire)
- **Anger:** `10,47%` (konflik tajam, amarah, kepahitan sosial)
- **Sadness:** `10,40%` (duka, keputusasaan, kemiskinan, depresi)
- **Fear:** `8,96%` (paranoia, ketakutan tertangkap, teror eksistensial)
- **Disgust:** `2,26%` (penolakan moral dan kejijikan fisik)

### 3. Korelasi Peringkat Spearman ($\rho$): Leksikon vs. Transformer

Dihitung secara ketat pada **6 emosi yang beririsan** (*anger, disgust, fear, joy, sadness, surprise*):

| Kelas Emosi | Spearman $\rho$ (Kalimat, $N=66.410$) | $p$-value | Rata-rata $\rho$ Alur Narasi ($100$ Bins) | Rentang $\rho$ per Novel |
|---|:--:|:--:|:--:|:--:|
| **Joy** | 0,2595 | $< 10^{-15}$ | **0,5931** | 0,3702 – 0,7266 |
| **Fear** | 0,3075 | $< 10^{-15}$ | **0,4913** | 0,1971 – 0,6459 |
| **Disgust** | 0,3126 | $< 10^{-15}$ | **0,4142** | 0,1378 – 0,6332 |
| **Sadness** | 0,2727 | $< 10^{-15}$ | **0,3834** | 0,2045 – 0,5596 |
| **Anger** | 0,1891 | $< 10^{-15}$ | **0,3399** | 0,1384 – 0,5494 |
| **Surprise** | 0,0191 | $8,2 \times 10^{-7}$ | **0,1047** | -0,0583 – 0,4166 |

*Temuan Metodologis:* Pada tingkat kalimat individual, keselarasan kata per kata bersifat moderat ($\rho \approx 0,25 - 0,31$). Namun, ketika diagregasi ke alur narasi 100 bins, Leksikon dan Transformer mencapai keselarasan makro yang sangat kuat pada emosi **Joy** ($\rho$ hingga 0,73) dan **Fear** ($\rho$ hingga 0,65).

### 4. Evaluasi Kesepakatan Model (Cohen's Kappa)

- **BERT vs. RoBERTa (2.000 kalimat sampel emas):**
  * **Cohen's Kappa ($\kappa$):** `0,6028` (*Substantial Agreement*)
  * **Raw Accuracy Agreement:** `68,65%`
- **RoBERTa vs. Leksikon NRC Dominan (Non-Netral):**
  * **Cohen's Kappa ($\kappa$):** `0,2416` (*Fair Agreement*)
  * **Raw Accuracy Agreement:** `36,85%`

### 5. Validasi terhadap Studi Acuan Picca, Schnyder & Romele (2024)

Evaluasi alur narasi pada *Crime and Punishment* membuktikan model secara akurat mereplikasi titik krisis psikologis Raskolnikov:
1. **Pembunuhan Alyona Ivanovna (Progres 17%):** Terdeteksi lonjakan tajam emosi ketakutan (*fear spike*, terbukti secara matematis).
2. **Kematian Marmeladov (Progres 32%):** Puncak kesedihan mendalam (*sadness peak*).
3. **Interogasi Porfiry Petrovich (Progres 67%):** Puncak kejutan psikologis dan ketegangan (*surprise & fear*).
4. **Bunuh Diri Svidrigailov (Progres 88%):** Klimaks distres dan amarah batin.
5. **Kebangkitan di Siberia (Progres 98%):** Epilog pemulihan spiritual bersama Sonia.

---

## 📊 Galeri Visualisasi Hasil Penelitian (300 DPI)

| File Gambar | Deskripsi Ilmiah |
|---|---|
| [fig1_emotional_arc_crime_and_punishment.png](file:///c:/Users/Hype/Documents/NLP/results/fig1_emotional_arc_crime_and_punishment.png) | Kurva alur emosi dinamis *Crime and Punishment* (RoBERTa vs. NRC) dengan 5 penanda krisis narasi Picca dkk. (2024). |
| [fig2_emotion_distribution_across_titles.png](file:///c:/Users/Hype/Documents/NLP/results/fig2_emotion_distribution_across_titles.png) | Profil distribusi emosi komparatif 12 karya fiksi Dostoevsky (*stacked bar chart*). |
| [fig3_spearman_correlation_heatmap.png](file:///c:/Users/Hype/Documents/NLP/results/fig3_spearman_correlation_heatmap.png) | *Heatmap* koefisien korelasi Spearman alur narasi antara NRC dan RoBERTa pada 6 emosi beririsan. |
| [fig4_jsd_inter_novel_clustering.png](file:///c:/Users/Hype/Documents/NLP/results/fig4_jsd_inter_novel_clustering.png) | *Clustermap* dan dendrogram pengelompokan novel Dostoevsky berdasarkan matriks jarak $d_{JSD}$. |
| [fig5_lexicon_vs_transformer_correlation.png](file:///c:/Users/Hype/Documents/NLP/results/fig5_lexicon_vs_transformer_correlation.png) | *Scatter plots* matriks $2 \times 3$ regresi linier proporsi emosi Leksikon vs. Transformer. |

---

## 💻 Panduan Penggunaan Dasbor Streamlit

Dasbor dirancang dengan arsitektur modular di file [app/app.py](file:///c:/Users/Hype/Documents/NLP/app/app.py):
1. **Halaman 1 (Emotional Arc):** Menampilkan kurva interaktif Plotly dengan *slider rolling window smoothing* ($1 - 15$ bins), filter emosi, dan penanda checkpoint sastra.
2. **Halaman 2 (Distribusi Emosi Antar-Judul):** Membandingkan sidik afektif ke-12 judul menggunakan *Stacked Bar Chart* dan *Radar Chart* laba-laba.
3. **Halaman 3 (Perbandingan Metodologis):** Visualisasi matriks korelasi Spearman, matriks jarak JSD antar-novel, metrik validasi Picca (2024), dan status reliabilitas anotasi.
4. **Halaman 4 (Corpus Explorer):** Penjelajah kalimat korpus dengan pencarian leksikal dan filter probabilitas emosi.

Untuk membuka dasbor:
```bash
streamlit run app/app.py
```

---

## ⚠️ Keterbatasan Penelitian & Catatan Akademik

1. **Variasi Gaya Penerjemah:** Korpus mengandalkan terjemahan Project Gutenberg dari era Victoria akhir hingga awal abad ke-20 (sebagian besar oleh Constance Garnett; *The Idiot* oleh Eva Martin; *The Gambler* dan *Poor Folk* oleh C. J. Hogarth; *Uncle's Dream* oleh Frederick Whishaw). Perbedaan diksi dan pilihan leksikal penerjemah memengaruhi kerapatan kata leksikon, namun pengaruhnya berhasil diredam oleh model Transformer yang berbasis pemahaman semantik kalimat.
2. **Domain Shift Data Latih:** Model Transformer di-*fine-tune* pada benchmark GoEmotions (percakapan modern) yang dipetakan ke taksonomi 7-kelas Ekman. Meskipun performa generalisasi sangat baik (F1 0,6905 dan Cohen's Kappa antar-transformer 0,6028), beberapa ungkapan metaforis khas sastra abad ke-19 dievaluasi sebagai *neutral* atau *surprise*.
3. **Skema Irisan Emosi:** Perbandingan langsung leksikon vs. transformer hanya valid pada 6 emosi yang beririsan (*anger, disgust, fear, joy, sadness, surprise*). Emosi *anticipation* dan *trust* disajikan secara univariat pada analisis leksikon, sedangkan kelas *neutral* disajikan khusus pada transformer.

---

## 📜 Lisensi & Atribusi

- **Teks Sumber:** Domain publik via [Project Gutenberg](https://www.gutenberg.org/).
- **Studi Acuan Sastra Komputasi:** Picca, D., Schnyder, P., & Romele, A. (2024). *Computational Literary Studies and Sentiment Analysis on Crime and Punishment*.
- **Leksikon Emosi:** Mohammad, S. M., & Turney, P. D. (2013). *Crowdsourcing a Word-Emotion Association Lexicon*.
