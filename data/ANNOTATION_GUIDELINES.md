# Pedoman Anotasi Emosi Kalimat Fiksi Fyodor Dostoevsky

Dokumen ini merupakan panduan resmi bagi **Anotator 1** dan **Anotator 2** dalam memberikan label emosi pada sampel kalimat dari korpus novel Fyodor Dostoevsky.

---

## 🎯 Tujuan Anotasi
Mengidentifikasi emosi dominan yang diekspresikan, dirasakan oleh narator/tokoh, atau dibangkitkan oleh teks kalimat ke dalam **satu label tunggal** dari 7 kelas emosi yang telah distandardisasi.

---

## 🏷️ Skema 7 Kelas Emosi

| Label Emosi | Definisi & Karakteristik | Kata Kunci Indikatif | Contoh Kasus pada Dostoevsky |
|---|---|---|---|
| **anger** | Kemarahan, kebencian, kejengkelan, amarah, dendam, rasa terhina yang memicu kemarahan. | *furious, rage, indignation, irritated, hate, strike, curse* | *“I tell you that I won’t allow you to speak of her like that!”* / *He scowled fiercely.* |
| **disgust** | Rasa jijik, muak, penghinaan mendalam (*contempt*), penolakan moral/fisik terhadap sesuatu yang kotor/keji. | *loathsome, nauseating, vile, repulsive, sneer, contempt* | *“I felt at once sick at being here with you!”* / *A vile and repulsive thought entered his mind.* |
| **fear** | Ketakutan, kecemasan, panik, horor, teror, kegelisahan batin, perasaan bersalah yang menghantui. | *terror, dread, shudder, horrified, anxiety, trembling, alarm* | *And each time he passed, the young man had a sick, frightened feeling.* |
| **joy** | Kegembiraan, kebahagiaan, tawa, keceriaan, kepuasan batin, rasa cinta yang membahagiakan, harapan cerah. | *glad, delighted, laughed gaily, bliss, ecstasy, smile, triumph* | *“Splendid! splendid!” he cried with real delight.* |
| **sadness** | Kesedihan, keputusasaan, duka, tangisan, ratapan, depresi, penderitaan batin, rasa hampa. | *grief, weeping, sorrow, despair, crushed, tears, mournful* | *He was crushed by poverty and wept bitterly in the darkness.* |
| **surprise** | Keterkejutan, keheranan, ketidakterdugaan, takjub mendadak, kebingungan spontan. | *astonished, amazed, suddenly, started, bewildered, unexpected* | *“Good heavens! What on earth does this mean?”* |
| **neutral** | Netral, pernyataan fakta objektif, deskripsi latar/ruangan, transisi naratif tanpa muatan afektif emosional. | *walked, entered, room, table, said, clock, morning, day* | *He took the letter, put it on the table, and walked towards the window.* |

---

## 📋 Aturan & Prinsip Anotasi

1. **Prinsip Konteks Tokoh & Narator:**
   - Fokus pada emosi yang terpancar dari dialog tokoh atau deskripsi batin narator.
   - Jika kalimat berbentuk tuturan dialog (misal: *“Get out!”*), beri label emosi pembicara (*anger*), bukan pendengar.

2. **Resolusi Multilabel (Memilih Emosi Dominan):**
   - Jika satu kalimat mengandung campuran emosi (misal ketakutan dan kemarahan sekaligus), pilih emosi yang memiliki intensitas paling dominan.
   - Contoh: *“Trembling with rage, he brandished his fist.”* $\rightarrow$ Emosi pendorong aksi adalah **anger**.

3. **Skala Keyakinan (*Confidence Score*):**
   - `5`: Sangat yakin (makna emosi eksplisit, tidak ambigu).
   - `4`: Yakin (emosi tersirat dengan petunjuk kuat).
   - `3`: Cukup yakin (ada multitafsir antara 2 emosi, namun satu lebih menonjol).
   - `2`: Ragu-ragu (sangat ambigu atau konteks tidak cukup).
   - `1`: Tebakan murni (sangat sulit ditentukan).

4. **Kemandirian Anotasi (*Double-Blind*):**
   - Dilarang mendiskusikan atau mencocokkan hasil anotasi antar-anotator selama proses berlangsung.
   - Data overlap (400 kalimat) dimasukkan secara teracak untuk mengukur reliabilitas *Inter-Annotator Agreement* (Cohen's Kappa).

---

## 💾 Format Pengisian CSV
Buka file `annotation_task_annotator_X.csv` di Excel/Google Sheets, dan isi 3 kolom berikut:
- **`emotion_label`**: Tulis huruf kecil tepat salah satu dari: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, atau `neutral`.
- **`confidence_1_to_5`**: Angka bulat antara `1` sampai `5`.
- **`notes`** (opsional): Catatan singkat jika ada kalimat yang sangat ambigu atau memerlukan konteks bab.
