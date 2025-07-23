# filename: Speech_emotion.py
# encoding: utf-8

import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import os

# --- Custom Background and Text Style ---
page_bg_img = """
<style>
.stApp {
    background-image: url('https://images.pexels.com/photos/8386356/pexels-photo-8386356.jpeg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.stApp > header, .stApp > footer {
    visibility: hidden;
}
h1 {
    color: white !important;
    font-size: 3em !important;
    text-shadow: 2px 2px 4px #000000;
}
div.block-container p,
div.block-container label,
div.block-container span,
div.block-container h3,
div.block-container h2 {
    color: white !important;
    text-shadow: 1px 1px 2px #000000;
}
section[data-testid="stFileUploader"] label {
    color: black !important;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_model.h5")

model = load_model()

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

st.title("🎤 Speech Emotion Recognition App")
st.markdown("Upload a short *.wav* file to detect human emotion using deep learning.", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    y, sr = librosa.load("temp.wav", duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    X_input = np.reshape(mfccs_scaled, (1, 40))
    prediction = model.predict(X_input)
    predicted_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.success(f"Predicted Emotion: {predicted_emotion.upper()}")
    st.info(f"Confidence Score: {confidence:.2f}%")

    os.remove("temp.wav")
