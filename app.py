import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermal Motor Diagnostic",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ─── Global ─── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}

/* ─── Header ─── */
.app-header {
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-subtitle {
    font-size: 0.88rem;
    color: #6b7280;
    margin-top: 0.3rem;
    font-weight: 300;
}

/* ─── Status Badge ─── */
.status-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ─── Metric Card ─── */
.metric-card {
    background: #f8f9fc;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1;
}

/* ─── Confidence Bar ─── */
.conf-bar-wrap {
    background: #e5e7eb;
    border-radius: 2px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* ─── Recommendation Panel ─── */
.rec-panel {
    border-left: 4px solid;
    padding: 1rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin: 1rem 0;
}
.rec-panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}
.rec-panel-text {
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ─── Prob Bar ─── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
    gap: 0.8rem;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #374151;
    width: 130px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: #e5e7eb;
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
}
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── History Table ─── */
.hist-row {
    display: flex;
    align-items: center;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #f3f4f6;
    gap: 0.8rem;
    font-size: 0.85rem;
}
.hist-row:last-child { border-bottom: none; }
.hist-num {
    font-family: 'IBM Plex Mono', monospace;
    color: #9ca3af;
    width: 24px;
    flex-shrink: 0;
}
.hist-name { flex: 1; color: #374151; font-weight: 500; }
.hist-label { color: #6b7280; }
.hist-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
}
.hist-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ─── Sidebar ─── */
.sidebar-section {
    font-size: 0.78rem;
    color: #6b7280;
    line-height: 1.7;
}
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #9ca3af;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}

/* ─── Divider ─── */
.thin-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* ─── Upload zone ─── */
[data-testid="stFileUploader"] {
    border-radius: 6px;
}

/* ─── Button ─── */
.stButton > button {
    background-color: #1a1a2e !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    height: 2.8em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_NAMES = ["Healthy", "Stator Fault", "Rotor Fault", "Fan Fault"]
IMG_SIZE    = (224, 224)

FAULT_CONFIG = {
    "Healthy": {
        "status"  : "NORMAL",
        "severity": "LOW",
        "color"   : "#16a34a",
        "bg"      : "#f0fdf4",
        "action"  : "Tidak diperlukan tindakan khusus.",
        "schedule": "Lanjutkan jadwal maintenance rutin.",
        "detail"  : [
            "Motor beroperasi dalam kondisi normal.",
            "Suhu distribusi merata dan dalam batas aman.",
            "Lakukan inspeksi visual bulanan.",
            "Catat pembacaan suhu untuk analisis tren historis.",
        ],
    },
    "Stator Fault": {
        "status"  : "STATOR FAULT",
        "severity": "HIGH",
        "color"   : "#dc2626",
        "bg"      : "#fef2f2",
        "action"  : "Segera jadwalkan shutdown untuk inspeksi stator.",
        "schedule": "Inspeksi dalam 24-48 jam.",
        "detail"  : [
            "Terdeteksi anomali panas di area belitan stator.",
            "Kemungkinan penyebab: korsleting antar-belitan, isolasi rusak.",
            "Hentikan operasi jika suhu terus meningkat.",
            "Hubungi teknisi listrik untuk rewinding atau penggantian.",
            "Periksa suplai tegangan dan balance 3 phase.",
        ],
    },
    "Rotor Fault": {
        "status"  : "ROTOR FAULT",
        "severity": "HIGH",
        "color"   : "#ea580c",
        "bg"      : "#fff7ed",
        "action"  : "Jadwalkan shutdown untuk inspeksi rotor.",
        "schedule": "Inspeksi dalam 48-72 jam.",
        "detail"  : [
            "Terdeteksi anomali panas di area rotor.",
            "Kemungkinan penyebab: rotor bar rusak, bantalan aus, misalignment.",
            "Lakukan pengecekan vibrasi dan kebisingan abnormal.",
            "Periksa kondisi bearing dan lubrikasi.",
            "Cek alignment antara motor dan beban.",
        ],
    },
    "Fan Fault": {
        "status"  : "FAN FAULT",
        "severity": "MEDIUM",
        "color"   : "#2563eb",
        "bg"      : "#eff6ff",
        "action"  : "Inspeksi sistem pendingin segera.",
        "schedule": "Inspeksi dalam 72 jam.",
        "detail"  : [
            "Terdeteksi distribusi panas tidak merata akibat pendinginan buruk.",
            "Kemungkinan penyebab: kipas patah atau kotor, saluran udara tersumbat.",
            "Bersihkan sirip pendingin dan pastikan aliran udara lancar.",
            "Periksa kondisi fisik kipas pendingin.",
            "Jika kipas rusak, ganti sebelum suhu mencapai titik kritis.",
        ],
    },
}

SEVERITY_COLORS = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}

PROB_BAR_COLORS = {
    "Healthy"    : "#16a34a",
    "Stator Fault": "#dc2626",
    "Rotor Fault" : "#ea580c",
    "Fan Fault"   : "#2563eb",
}

# ── Session State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "mobilenetv2_thermal_fault_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Thermal Motor Diagnostic")
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        Arsitektur <strong>MobileNetV2</strong> yang di-<em>fine-tune</em> 
        pada dataset citra termal motor industri.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Kelas Fault</div>', unsafe_allow_html=True)
    for label, cfg in FAULT_CONFIG.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;'>"
            f"<div style='width:9px;height:9px;border-radius:50%;background:{cfg['color']};flex-shrink:0'></div>"
            f"<span style='font-size:0.82rem;color:#374151;'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-label">Format Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">JPG, PNG, BMP &mdash; resize otomatis ke 224×224 px</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Threshold Kepercayaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <span style="color:#16a34a;font-weight:600;">Tinggi</span> &ge; 80% &mdash; Hasil andal<br>
        <span style="color:#d97706;font-weight:600;">Sedang</span> 60–80% &mdash; Perlu verifikasi<br>
        <span style="color:#dc2626;font-weight:600;">Rendah</span> &lt; 60% &mdash; Inspeksi fisik
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()

# ── Main Header ────────────────────────────────────────────────────────────────
if model is None:
    st.error(
        "Model tidak ditemukan. Pastikan file `mobilenetv2_thermal_fault_model.keras` "
        "berada di folder yang sama dengan `app.py`."
    )
    st.stop()

st.markdown("""
<div class="app-header">
    <div class="app-title">Thermal Motor Diagnostic System</div>
    <div class="app-subtitle">
        Unggah citra termal motor untuk analisis kondisi dan rekomendasi pemeliharaan
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: Upload | Result ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("**Unggah Citra Termal**")
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        run_btn = st.button("Jalankan Diagnosa")
    else:
        st.markdown(
            "<div style='height:200px;border:1.5px dashed #d1d5db;border-radius:6px;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#9ca3af;font-size:0.85rem;'>Belum ada gambar diunggah</div>",
            unsafe_allow_html=True
        )
        run_btn = False

with col_right:
    if uploaded_file is not None and run_btn:
        with st.spinner("Menganalisis..."):
            # Preprocess
            img_resized = image.resize(IMG_SIZE)
            img_array   = np.array(img_resized, dtype=np.float32) / 255.0
            img_batch   = np.expand_dims(img_array, axis=0)

            # Predict
            preds      = model.predict(img_batch, verbose=0)[0]
            pred_idx   = int(np.argmax(preds))
            pred_label = LABEL_NAMES[pred_idx]
            confidence = float(preds[pred_idx])

        cfg = FAULT_CONFIG[pred_label]

        # Determine confidence level
        if confidence >= 0.80:
            conf_level = "Tinggi"
            conf_color = "#16a34a"
        elif confidence >= 0.60:
            conf_level = "Sedang"
            conf_color = "#d97706"
        else:
            conf_level = "Rendah — Inspeksi fisik direkomendasikan"
            conf_color = "#dc2626"

        # Metrics
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Kondisi Terdeteksi</div>
                <div class="metric-value" style="font-size:1.1rem;">{pred_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{confidence * 100:.1f}%</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill" style="width:{confidence*100:.1f}%;background:{conf_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Status Badge + confidence level info
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.8rem;margin:0.5rem 0;'>"
            f"<span class='status-badge' style='background:{cfg['bg']};color:{cfg['color']};'>"
            f"{cfg['status']}</span>"
            f"<span style='font-size:0.82rem;color:{conf_color};font-weight:500;'>"
            f"Kepercayaan: {conf_level}</span></div>",
            unsafe_allow_html=True
        )

        # Probability bars
        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#9ca3af;margin-bottom:0.7rem;'>Probabilitas Semua Kelas</div>", unsafe_allow_html=True)

        for lbl, prob in sorted(zip(LABEL_NAMES, preds), key=lambda x: -x[1]):
            bar_color = PROB_BAR_COLORS[lbl]
            is_pred = lbl == pred_label
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="font-weight:{'600' if is_pred else '400'};
                     color:{'#1a1a2e' if is_pred else '#6b7280'};">{lbl}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{prob*100:.1f}%;background:{bar_color};
                         opacity:{'1' if is_pred else '0.4'};"></div>
                </div>
                <div class="prob-pct">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            "filename" : uploaded_file.name,
            "label"    : pred_label,
            "confidence": confidence,
            "color"    : cfg["color"],
        })

    elif uploaded_file is None or not run_btn:
        st.markdown(
            "<div style='height:100%;display:flex;align-items:center;justify-content:center;"
            "color:#9ca3af;font-size:0.85rem;padding-top:3rem;text-align:center;'>"
            "Hasil diagnosa akan muncul<br>setelah gambar diproses.</div>",
            unsafe_allow_html=True
        )

# ── Recommendations (shown below when result exists) ──────────────────────────
if uploaded_file is not None and run_btn and 'pred_label' in dir():
    pass  # handled inline

# We track last result via session to show recommendations
if st.session_state.history:
    last = st.session_state.history[-1]
    last_label = last["label"]
    cfg = FAULT_CONFIG[last_label]

    if uploaded_file is not None and run_btn:
        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
        st.markdown("**Rekomendasi Tindakan**")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            sev_color = SEVERITY_COLORS[cfg["severity"]]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Severity</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:{sev_color};">
                    {cfg['severity']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with rcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Jadwal Inspeksi</div>
                <div style="font-size:0.92rem;font-weight:500;color:#1a1a2e;">{cfg['schedule']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-panel" style="border-color:{cfg['color']};background:{cfg['bg']};">
            <div class="rec-panel-title" style="color:{cfg['color']};">Tindakan Segera</div>
            <div class="rec-panel-text">{cfg['action']}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Lihat Detail Rekomendasi", expanded=True):
            for i, item in enumerate(cfg["detail"], 1):
                st.markdown(
                    f"<div style='padding:0.4rem 0;border-bottom:1px solid #f3f4f6;"
                    f"font-size:0.9rem;color:#374151;'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;color:#9ca3af;margin-right:0.6rem;'>{i:02d}</span>"
                    f"{item}</div>",
                    unsafe_allow_html=True
                )

# ── History ────────────────────────────────────────────────────────────────────
if len(st.session_state.history) > 1:
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("**Riwayat Diagnosa Sesi Ini**")
    st.markdown(
        "<div style='border:1px solid #e5e7eb;border-radius:6px;overflow:hidden;'>",
        unsafe_allow_html=True
    )
    for i, rec in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"""
        <div class="hist-row">
            <div class="hist-num">{len(st.session_state.history) - i + 1:02d}</div>
            <div class="hist-dot" style="background:{rec['color']};"></div>
            <div class="hist-name">{rec['filename']}</div>
            <div class="hist-label">{rec['label']}</div>
            <div class="hist-conf">{rec['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
