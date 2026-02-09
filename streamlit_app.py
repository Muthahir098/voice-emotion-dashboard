import streamlit as st
import librosa
import pandas as pd
import plotly.express as px
from emotion_model import predict_emotion

st.set_page_config(page_title="Voice Emotion Dashboard", layout="wide")

st.title("🎙️ Voice Emotion & Sentiment Dashboard")
st.write("Upload an audio file to analyze emotion changes over time.")

audio_file = st.file_uploader("Upload WAV audio file", type=["wav"])

if audio_file:
    audio, sr = librosa.load(audio_file, sr=None)

    segment_duration = 2  # seconds
    times = []
    emotions = []

    for i in range(0, len(audio), segment_duration * sr):
        segment = audio[i:i + segment_duration * sr]
        if len(segment) < segment_duration * sr:
            continue

        emotion = predict_emotion(segment)

        time_sec = i / sr
        minute = int(time_sec // 60)
        second = int(time_sec % 60)

        times.append(f"{minute:02d}:{second:02d}")
        emotions.append(emotion)

    df = pd.DataFrame({
        "Time": times,
        "Emotion": emotions
    })

    # Detect emotion change points
    df["Emotion_Change"] = df["Emotion"].ne(df["Emotion"].shift())
    change_df = df[df["Emotion_Change"] == True]

    # Timeline chart
    df["Seconds"] = range(0, len(df) * segment_duration, segment_duration)

    fig = px.line(
        df,
        x="Seconds",
        y="Emotion",
        markers=True,
        title="Emotion Changes Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Complete Emotion Log (Minute : Second)")
    st.dataframe(df[["Time", "Emotion"]])

    st.subheader("🔄 Emotion Change Points")
    st.dataframe(change_df[["Time", "Emotion"]])