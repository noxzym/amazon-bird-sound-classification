import io
import mimetypes
import tempfile

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import streamlit as st
import tensorflow as tf
from audiorecorder import audiorecorder

# === Load Model & Define Class Names ===
model = tf.keras.models.load_model("kicau_model.h5", compile=False)
class_names = [
    "Megarynchus pitangua (bobfly1)",
    "Myiarchus tuberculifer (ducfly)",
    "Attila spadiceus (brratt1)",
    "Thamnophilus doliatus (barant1)",
    "Piaya cayana (squcuc1)",
    "Taraba major (greant1)",
    "Cantorchilus leucotis (bubwre1)",
    "Sittasomus griseicapillus (oliwoo1)",
    "Glaucidium brasilianum (fepowl)",
    "Xiphorhynchus guttatus (butwoo1)",
]


# === Audio Preprocessing Functions ===
def denoise_audio(audio_array, sampling_rate):
    return nr.reduce_noise(y=audio_array, sr=sampling_rate)


def pad_or_trim(audio_array, sampling_rate, max_len):
    desired_len = int(max_len * sampling_rate)
    if len(audio_array) < desired_len:
        audio_array = np.pad(audio_array, (0, desired_len - len(audio_array)))
    else:
        audio_array = audio_array[:desired_len]
    return audio_array


def to_melspectrogram(audio_array, sampling_rate=16000, n_mels=40):
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.astype(np.float32)[..., np.newaxis]
    return mel_spec_db


def preprocess_mp3(file_path, sampling_rate=16000, max_len=5):
    try:
        audio_array, sr = sf.read(file_path)
        if sr != sampling_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
            sr = sampling_rate
    except RuntimeError:
        # fallback ke librosa untuk file yang tidak didukung soundfile (misal mp3)
        audio_array, sr = librosa.load(file_path, sr=sampling_rate)
    audio_array = denoise_audio(audio_array, sampling_rate)
    audio_array = pad_or_trim(audio_array, sampling_rate, max_len)
    mel_spec_db = to_melspectrogram(audio_array, sampling_rate)
    return mel_spec_db, sampling_rate


def inference(file_path, class_names):
    mel_spec_db, _ = preprocess_mp3(file_path)
    mel_spec_resized = tf.image.resize(mel_spec_db, model.input_shape[1:3]).numpy()
    mel_spec_resized = np.expand_dims(mel_spec_resized, axis=0)
    pred = model.predict(mel_spec_resized)[0]  # ambil array hasil prediksi

    df = (
        pd.DataFrame({"Nama Kelas": class_names, "Persentase Prediksi (%)": (pred * 100).round(2)})
        .sort_values("Persentase Prediksi (%)", ascending=False)
        .reset_index(drop=True)
    )

    return df


# === Streamlit UI ===
st.markdown(
    """
    <style>
    .block-container {
        max-width: 700px;
        margin: auto;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Streamlit UI ===
st.title("🎶 Klasifikasi Suara Burung")

tabs = st.tabs(["📁 Upload File", "🎤 Rekam Langsung", "🔗 Audio dari URL"])

file_path = None

# === Tab 1: Upload File ===
with tabs[0]:
    st.subheader("Unggah file audio (MP3/WAV/OGG)")
    uploaded_file = st.file_uploader("Pilih file audio", type=["mp3", "wav", "ogg"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/mp3")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

# === Tab 2: Rekam Langsung ===
with tabs[1]:
    st.subheader("Rekam langsung dari mikrofon")
    audio = audiorecorder("🎙️ Start Recording", "⏹ Stop Recording")
    if len(audio) > 0:
        st.audio(audio.export().read(), format="audio/wav")
        audio_array, sr = sf.read(io.BytesIO(audio.export().read()))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_array, sr)
            file_path = tmp.name

# === Tab 3: Audio dari URL ===
with tabs[2]:
    st.subheader("Masukkan link file audio (mp3/wav/ogg)")
    url = st.text_input("URL audio:")
    fetch_button = st.button("📥 Ambil Audio dari URL")

    if fetch_button and url:
        try:
            response = requests.get(url)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            ext = mimetypes.guess_extension(content_type)

            # fallback jika tidak dikenali
            if not ext or not ext.startswith("."):
                ext = ".mp3"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(response.content)
                file_path = tmp.name

            st.audio(
                file_path, format=content_type if content_type.startswith("audio") else "audio/mp3"
            )
        except Exception as e:
            st.error(f"Gagal mengambil audio: {e}")

# === Inference ===
if file_path:
    with st.spinner("🔍 Memproses suara burung..."):
        try:
            df_result = inference(file_path, class_names)
            st.success("✨ Hasil Prediksi Model:")
            st.dataframe(df_result, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal mengklasifikasi: {e}")
else:
    st.info("Silakan pilih salah satu input untuk memulai klasifikasi ya kakak~")
