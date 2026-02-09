import numpy as np

def predict_emotion(audio_segment):
    energy = np.mean(np.abs(audio_segment))

    if energy < 0.01:
        return "Neutral"
    elif energy < 0.03:
        return "Sad"
    elif energy < 0.06:
        return "Happy"
    elif energy < 0.1:
        return "Fear"
    else:
        return "Angry"