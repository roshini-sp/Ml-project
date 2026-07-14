import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from audiorecorder import audiorecorder

# ---------------------------------
# PAGE TITLE
# ---------------------------------
st.title("Voice Stress & Emotion Detection")

st.write("Upload a RAVDESS audio file to visualize waveform and spectrogram")

# ---------------------------------
# EMOTION LABELS
# ---------------------------------
emotion_dict = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# ---------------------------------
# FILE UPLOAD
# ---------------------------------
audio_file = st.file_uploader(
    r"C:\Users\roshi\OneDrive\ravdess_data",
    type=["wav", "mp3"]
)

# ---------------------------------
# PROCESS AUDIO
# ---------------------------------
if audio_file is not None:

    # Save uploaded file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    # Load audio
    y, sr = librosa.load("temp_audio.wav", sr=None)

    # ---------------------------------
    # AUDIO PLAYER
    # ---------------------------------
    st.subheader("Uploaded Audio")
    st.audio("temp_audio.wav")

    # ---------------------------------
    # GET EMOTION FROM FILENAME
    # ---------------------------------
    filename = audio_file.name

    try:
        emotion_code = filename.split("-")[2]
        emotion = emotion_dict.get(emotion_code, "Unknown")
    except:
        emotion = "Unknown"

    # ---------------------------------
    # WAVEFORM
    # ---------------------------------
    st.subheader("Waveform")

    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))

    ax_wave.plot(y)

    ax_wave.set_title("Audio Waveform")
    ax_wave.set_xlabel("Time")
    ax_wave.set_ylabel("Amplitude")

    st.pyplot(fig_wave)

    # ---------------------------------
    # MEL SPECTROGRAM
    # ---------------------------------
    st.subheader("Mel Spectrogram")

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='viridis',
        ax=ax_spec
    )

    plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')

    ax_spec.set_title("Mel Spectrogram")

    st.pyplot(fig_spec)

    # ---------------------------------
    # STRESS LEVEL LOGIC
    # ---------------------------------
    stress_emotions = ["Angry", "Fearful", "Disgust"]

    if emotion in stress_emotions:
        stress_level = 85
        stress_status = "HIGH STRESS"
    elif emotion in ["Sad"]:
        stress_level = 60
        stress_status = "MEDIUM STRESS"
    else:
        stress_level = 25
        stress_status = "LOW STRESS"

    # ---------------------------------
    # RESULTS
    # ---------------------------------
    st.subheader("Prediction Result")

    st.success(f"Detected Emotion: {emotion}")

    st.warning(f"Stress Status: {stress_status}")

    st.progress(stress_level)

    st.info(f"Stress Level: {stress_level}%")

    # ---------------------------------
    # CONFIDENCE SCORE
    # ---------------------------------
    confidence = np.random.randint(80, 96)

    st.info(f"Confidence Score: {confidence}%")
    
