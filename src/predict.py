import joblib
import librosa
import numpy as np


model = joblib.load("music_genre_classifier.pkl")

def extract_features(file_path):
    """Extracts audio features (MFCCs, Chroma, Spectral Contrast, ZCR, RMSE) from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Load audio
    except Exception as e:
        print(f"⚠️ Warning: Could not load {file_path} - {e}")
        return None

    # Extract Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)

    # Aggregate features (mean & std deviation)
    features = np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        np.mean(zcr), np.std(zcr),
        np.mean(rmse), np.std(rmse)
    ])
    return features.reshape(1, -1)  # Reshape for prediction


def main():
    file_path = "/Users/xenios/Documents/MusicGenreClassification/musicTest/bonJovi.wav"

    features = extract_features(file_path)
    if  features is not None:
        prediction = model.predict(features)
        print(f"🎵 Predicted Genre: {prediction[0]}")


if __name__ == "__main__":
    main()

