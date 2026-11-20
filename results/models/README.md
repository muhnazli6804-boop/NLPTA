# Model Weights (Bobot Model Transformer)

File bobot model berukuran besar (`model.safetensors`, ~438 MB untuk BERT dan ~498 MB untuk RoBERTa) melebihi batas unggah 100 MB GitHub.

Arsitektur model, konfigurasi tokenisasi, dan vocab tersimpan lengkap di subfolder ini:
- `bert_ekman/`: `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `roberta_ekman/`: `config.json`, `tokenizer.json`, `tokenizer_config.json`

### Cara Meregenerasi Bobot Model Secara Lokal
Jalankan skrip berikut di terminal:
```bash
# 1. Unduh dan siapkan dataset benchmark GoEmotions Ekman
python src/prepare_emotion_dataset.py

# 2. Latih model BERT & RoBERTa (bobot model.safetensors akan otomatis disimpan di sini)
python src/train_transformer.py
```
Metrik evaluasi lengkap tersimpan di `results/fine_tuning_metrics.json`.
