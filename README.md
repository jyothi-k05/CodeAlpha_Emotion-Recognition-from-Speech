# 🎙️ Emotion Recognition from Speech

This project detects emotions from audio speech using the RAVDESS dataset. It extracts MFCC features and trains an LSTM-based neural network.

## 📁 Files

- `speech_emotion.py` – Main script to train the model and plot results
- `requirements.txt` – Python dependencies
- `emotion_model.h5` – Trained Keras model (will be added after training)

## 📦 Dataset

Download the RAVDESS dataset here:  
[Download Audio_Speech_Actors_01-24.zip](https://drive.google.com/file/d/1syaa6NbhN1V-VLXuRi0NNZJlIK6tTrVc/view?usp=sharing)

After downloading, extract it like this:

project_folder/
├── ravdess_audio/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...

## 🚀 How to Run

```bash
pip install -r requirements.txt
python speech_emotion.py
