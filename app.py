from flask import Flask, render_template, request
import librosa
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os
from emotion_model import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    chart = None
    table = None
    change_table = None

    if request.method == "POST":
        file = request.files["audio"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Load audio
        audio, sr = librosa.load(path, sr=None)

        segment_duration = 2  # seconds
        times = []
        emotions = []

        # Segment audio
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

        # Create DataFrame
        df = pd.DataFrame({
            "Time": times,
            "Emotion": emotions
        })

        # Detect emotion changes
        df["Emotion_Change"] = df["Emotion"].ne(df["Emotion"].shift())

        change_df = df[df["Emotion_Change"] == True]

        # Emotion timeline chart
        df["Seconds"] = range(0, len(df) * segment_duration, segment_duration)

        fig = px.line(
            df,
            x="Seconds",
            y="Emotion",
            markers=True,
            title="Emotion Changes Over Time"
        )

        chart = pio.to_html(fig, full_html=False)

        # Tables
        table = df[["Time", "Emotion"]].to_html(index=False)
        change_table = change_df[["Time", "Emotion"]].to_html(index=False)

    return render_template(
        "index.html",
        chart=chart,
        table=table,
        change_table=change_table
    )

if __name__ == "__main__":
    app.run(debug=True)