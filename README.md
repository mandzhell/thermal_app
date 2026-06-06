# Thermal Motor Diagnostic System

Aplikasi berbasis Streamlit untuk deteksi fault pada motor industri menggunakan citra termal dan model MobileNetV2.

## Struktur File

```
thermal_app/
├── app.py
├── requirements.txt
├── mobilenetv2_thermal_fault_model.keras   ← letakkan di sini
├── .streamlit/
│   └── config.toml
└── README.md
```

## Cara Deploy ke Streamlit Cloud

1. Upload seluruh isi folder ini ke repositori GitHub (pastikan file `.keras` ikut ter-upload — cek size limit GitHub 100 MB)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo → pilih branch → set **Main file path** ke `app.py`
4. Klik **Deploy**

> Catatan: Jika file model melebihi 100 MB, gunakan **Git LFS** atau hosting alternatif (Google Drive + gdown, Hugging Face Hub, dll.)

## Fitur Aplikasi

- Upload citra termal (JPG, PNG, BMP)
- Klasifikasi 4 kelas: Healthy, Stator Fault, Rotor Fault, Fan Fault
- Confidence score dengan 3 level kepercayaan (Tinggi / Sedang / Rendah)
- Probability bar untuk semua kelas
- Panel rekomendasi tindakan per fault
- Riwayat diagnosa dalam satu sesi

## Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
