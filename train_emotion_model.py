import os
import zipfile
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
import matplotlib.pyplot as plt

# Paths
zip_path = "Audio_Speech_Actors_01-24.zip"  # Ensure it's in same folder
extract_path = "ravdess_audio"
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Emotion labels
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Feature extraction
features, labels = [], []
for actor in os.listdir(extract_path):
    actor_path = os.path.join(extract_path, actor)
    if not os.path.isdir(actor_path): continue
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            path = os.path.join(actor_path, file)
            emotion_code = file.split("-")[2]
            label = emotion_map.get(emotion_code)
            y, sr = librosa.load(path, duration=3, offset=0.5)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features.append(mfccs_scaled)
            labels.append(label)

# Prepare dataset
X = np.array(features)
y = np.array(labels)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# Model
model = Sequential([
    Reshape((40, 1), input_shape=(40,)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(8, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))



# Save model
model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")

# Accuracy plot
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Predict labels
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title("Confusion Matrix")
plt.show()
