# 🌭 Emotion-Based Music Recommendation System

An AI-powered application that detects a user's facial and hand landmarks to recognize their current **emotion**, then recommends mood-based songs via **Spotify** and **YouTube**.

---

## 🚀 Features

* 🎥 Real-time emotion detection using webcam
* 🧠 Deep Learning model trained with TensorFlow & PCA
* 📊 Emotion classification: Happy, Sad, Angry, Anxious, Surprise
* 🎵 Music recommendations based on emotion + user preferences
* 🌐 Streamlit-based intuitive web UI
* 🖼️ Custom background support

---

## 📁 Project Structure

```
📆 Emotion-Music-Recommender
🔹 collected_data/           # Saved .npy datasets (by emotion)
🔹 model.h5                  # Trained emotion classifier (Keras)
🔹 pca.pkl                   # PCA transformer
🔹 labels.npy                # Emotion labels
🔹 Music-app.py              # Streamlit UI application
🔹 data_collection.py        # Webcam-based data capture
🔹 data_training.py          # Training pipeline (with PCA + MLP)
🔹 background.jpg            # Optional background image
🔹 confusion_matrix.png      # Evaluation result
🔹 README.md                 # You're here!
```

---

## 🧠 Model Overview

* Input: 1020-dim vectors (face + hand landmarks)
* Preprocessed with PCA → 100 dimensions
* MLP Classifier: 512 → 256 → Softmax
* Accuracy: \~98% on test set (after fixing PCA leakage)

---

## 🛠️ How to Run Locally

### 1. 📦 Install Dependencies

```bash
pip install streamlit tensorflow mediapipe scikit-learn opencv-python joblib
```

---

### 2. ▶️ Run the App

```bash
streamlit run Music-app.py
```

---

### 3. 📸 Collect Your Own Data (Optional)

```bash
python data_collection.py
```

* Follow prompts
* Save 100 samples per emotion

---

### 4. 🧠 Train the Model (Optional)

```bash
python data_training.py
```

* Automatically loads `.npy` files from `collected_data/`
* Applies PCA and trains the model
* Saves `model.h5`, `pca.pkl`, `labels.npy`

---

## 🌈 Customization

### 📷 Add Your Own Background

Place your image as `background.jpg` in the project root. You can control opacity and theme via CSS inside `Music-app.py`.

---

## 📊 Evaluation

To evaluate model performance:

```bash
# Open Jupyter notebook and run:
Analysis.ipynb
```

Includes:

* Accuracy
* Confusion Matrix
* Precision/Recall/F1-score

---

## 🙇‍♂️ Author

**Your Name**
GitHub: [@joshianushree](https://github.com/joshianushree)

---

## 📜 License

MIT License. Use freely, modify responsibly, and give credit! 🎉
