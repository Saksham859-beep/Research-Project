import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
from scipy.stats import skew
import matplotlib.pyplot as plt
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PCG Heart Sound Analysis",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 PCG Heart Sound Analysis System")
st.markdown("### ML vs CNN vs CNN-LSTM (Research Project Demo)")

# ===============================
# LOAD MODELS (SAFE PATH)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

gb_model = joblib.load(os.path.join(BASE_DIR, "gradient_boosting_pcg_model.joblib"))
cnn_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "cnn_pcg_physionet.h5"))
cnn_lstm_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "cnn_lstm_pcg_physionet.h5"))

# ===============================
# PARAMETERS
# ===============================
SR = 2000
N_MFCC = 40
N_FFT = 2048
HOP = 256
SEGMENT_SEC = 2.5
TIME_STEPS = 7

# ===============================
# FEATURE FUNCTIONS
# ===============================
def extract_mfcc(signal, sr):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP
    )
    return (mfcc - mfcc.mean()) / mfcc.std()

def extract_gb_features(mfcc):
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    sk = skew(mfcc, axis=1)
    return np.concatenate([mean, std, sk])

def split_signal(signal, sr):
    seg_len = int(SEGMENT_SEC * sr)
    return [signal[i:i+seg_len] for i in range(0, len(signal)-seg_len, seg_len)]

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📤 Upload PCG Audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # ===============================
    # LOAD AUDIO
    # ===============================
    signal, sr = librosa.load(uploaded_file, sr=SR)
    segments = split_signal(signal, sr)

    if len(segments) == 0:
        st.error("Audio file too short to process.")
        st.stop()

    st.success(f"Loaded PCG signal with {len(segments)} segments")

    # ===============================
    # WAVEFORM DISPLAY
    # ===============================
    st.subheader("📈 PCG Waveform")
    fig, ax = plt.subplots()
    ax.plot(signal, linewidth=0.8)
    ax.set_title("Heart Sound Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # ===============================
    # GRADIENT BOOSTING
    # ===============================
    gb_probs = []
    for seg in segments:
        mfcc = extract_mfcc(seg, sr)
        feat = extract_gb_features(mfcc).reshape(1, -1)
        gb_probs.append(gb_model.predict_proba(feat)[0][1])

    gb_prob = float(np.mean(gb_probs))
    gb_pred = "Abnormal" if gb_prob > 0.5 else "Normal"

    # ===============================
    # CNN
    # ===============================
    cnn_probs = []
    for seg in segments:
        mfcc = extract_mfcc(seg, sr)[..., np.newaxis]
        mfcc = np.expand_dims(mfcc, axis=0)
        cnn_probs.append(cnn_model.predict(mfcc)[0][0])

    cnn_prob = float(np.mean(cnn_probs))
    cnn_pred = "Abnormal" if cnn_prob > 0.5 else "Normal"

    # ===============================
    # CNN-LSTM (FIXED INPUT SHAPE)
    # ===============================
    lstm_segments = []

    for seg in segments:
        mfcc = extract_mfcc(seg, sr)

        # ensure MFCC shape (40, 20)
        if mfcc.shape[1] < 20:
            mfcc = np.pad(mfcc, ((0, 0), (0, 20 - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :20]

        mfcc = mfcc[..., np.newaxis]
        lstm_segments.append(mfcc)

    if len(lstm_segments) < TIME_STEPS:
        pad = [np.zeros((40, 20, 1))] * (TIME_STEPS - len(lstm_segments))
        lstm_segments.extend(pad)
    else:
        lstm_segments = lstm_segments[:TIME_STEPS]

    lstm_input = np.expand_dims(np.array(lstm_segments), axis=0)
    lstm_prob = float(cnn_lstm_model.predict(lstm_input)[0][0])
    lstm_pred = "Abnormal" if lstm_prob > 0.5 else "Normal"

    # ===============================
    # RESULTS DISPLAY
    # ===============================
    st.subheader("🔍 Model Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧠 Gradient Boosting", gb_pred, f"{gb_prob:.2f}")

    with col2:
        st.metric("🤖 CNN", cnn_pred, f"{cnn_prob:.2f}")

    with col3:
        st.metric("🔗 CNN-LSTM", lstm_pred, f"{lstm_prob:.2f}")

    st.markdown("---")
    st.info(
        "⚠️ This system is for **research and screening purposes only** "
        "and does not replace professional medical diagnosis."
    )

else:
    st.info("⬆️ Upload a PCG (.wav) file to start analysis.")
