#We will Extract MFCCs, Chroma, Spectral Contrast, and More

import os  # For file and directory management
import pandas as pd  # For data handling and saving to CSV
import librosa  # For audio processing
import librosa.display  # Required if you plan to visualize waveforms or spectrograms
import numpy as np  # For numerical computations


def extract_features(file_path):
    """Extracts MFCCs, Chroma, Spectral Contrast, ZCR, and RMSE from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Load the audio file
    except Exception as e:
        print(f"⚠️ Warning: Could not load {file_path} - {e}")
        return None  # Return None if file can't be processed
    # Extract Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCs
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral contrast
    zcr = librosa.feature.zero_crossing_rate(y)  # Zero-Crossing Rate
    rmse = librosa.feature.rms(y=y)  # Root Mean Square Energy

    # Aggregate features (mean & std deviation for each feature)
    features = np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        np.mean(zcr), np.std(zcr),
        np.mean(rmse), np.std(rmse)
    ])
    return features


def main():
    dataset_path = "/Users/xenios/Documents/MusicGenreClassification/dataset/genres_original"
    data = []

    # Iterate through genres (folders)
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)

        if os.path.isdir(genre_path):  # Ensure it's a folder
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)

                if file_path.endswith(".wav"):  # Ensure it's an audio file
                    features = extract_features(file_path)
                    
                    # Check if features is None (error in extraction)
                    if features is None:
                        print(f"Skipping {file_path} due to feature extraction failure.")
                        continue  # Skip this file and move to the next one

                    data.append([genre] + list(features))  # Store genre and features

    # Convert to DataFrame
    columns = ["genre"] + [f"mfcc_mean_{i}" for i in range(13)] + \
            [f"mfcc_std_{i}" for i in range(13)] + \
            [f"chroma_mean_{i}" for i in range(12)] + \
            [f"chroma_std_{i}" for i in range(12)] + \
            [f"spectral_contrast_mean_{i}" for i in range(7)] + \
            [f"spectral_contrast_std_{i}" for i in range(7)] + \
            ["zcr_mean", "zcr_std", "rmse_mean", "rmse_std"]

    # Ensure there is data before creating a DataFrame
    if data:
        df = pd.DataFrame(data, columns=columns)  # Convert to DataFrame
        df.to_csv("music_features.csv", index=False)  # Save features to a CSV file
        print("✅ Feature extraction completed and saved to 'music_features.csv'.")
    else:
        print("⚠️ No valid data extracted. Check for errors in file loading.")

if __name__ == "__main__":
    main()
