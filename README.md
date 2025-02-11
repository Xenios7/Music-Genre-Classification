# 🎵 Music Genre Classification

This project focuses on **classifying music genres** using **Machine Learning / Deep Learning** techniques. It analyzes audio features to predict the genre of a given music track.

## 🚀 Features
- Extracts **audio features** from music files.
- Uses **ML/DL models** for classification.
- Supports multiple music genres (e.g., Rock, Jazz, Classical, Pop, Hip-Hop).
- Easy-to-use pipeline for **training and prediction**.

## 🛠️ Technologies Used
- **Python** (Main Programming Language)
- **Librosa** (Feature Extraction)
- **Scikit-learn** / **TensorFlow** (Model Training)
- **NumPy & Pandas** (Data Processing)
- **Matplotlib & Seaborn** (Visualization)

## 📚 References & Inspirations
- GTZAN Dataset: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
- Feature Extraction with Librosa: [https://librosa.org/doc/main/feature.html](https://librosa.org/doc/main/feature.html)
- Music Classification Research Paper: [https://arxiv.org/pdf/1612.01017.pdf](https://arxiv.org/pdf/1612.01017.pdf)
- Scikit-learn Model Training: [https://scikit-learn.org/stable/supervised_learning.html](https://scikit-learn.org/stable/supervised_learning.html)


## 🎯 How to Use
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt

2️⃣ Train the Model
python train.py

3️⃣ Classify a Song
python classify.py path/to/music.wav

📊 Dataset
The project uses a dataset containing labeled music tracks from various genres. You can use:

GTZAN Dataset (Public Dataset)
Your own dataset (MP3/WAV files)
📈 Future Improvements
Improve model accuracy with CNN/RNN architectures.
Add real-time classification from microphone input.
Deploy as a Web API.
📝 License
This project is open-source and available under the MIT License.
