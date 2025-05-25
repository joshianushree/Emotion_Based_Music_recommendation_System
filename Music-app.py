import cv2
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st
import time
import webbrowser
import pandas as pd
from collections import defaultdict
import mediapipe as mp
import base64
import os

# ----------------------------
# Background Image Function (Updated)
# ----------------------------
def add_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        opacity: 0.25;
        z-index: -1;
    }}

    .stApp, .block-container, .css-18e3th9, .main {{
        background-color: transparent !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Emotion Based Music Recommendation System",
    page_icon="🎵",
    layout="centered"
)
add_background("background.jpg")

# ----------------------------
# Caching heavy resources
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_pca():
    return joblib.load("pca.pkl")

@st.cache_data
def load_labels():
    return np.load("labels.npy")

@st.cache_resource
def load_holistic():
    return mp.solutions.holistic.Holistic()

emotion_labels = load_labels()
pca = load_pca()
model = load_model()
holistic = load_holistic()
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Base64 YouTube Logo (optional)
# ----------------------------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded
    return None

youtube_logo_path = r"D:\\youtube-logo-2431.png"
youtube_logo_base64 = get_base64_image(youtube_logo_path)

# ----------------------------
# Landmark Feature Extraction
# ----------------------------
def extract_landmark_features(res):
    features = []
    if res.face_landmarks:
        ref_face = res.face_landmarks.landmark[1]
        for lm in res.face_landmarks.landmark:
            features.append(lm.x - ref_face.x)
            features.append(lm.y - ref_face.y)
    else:
        return None

    if res.left_hand_landmarks:
        ref_left = res.left_hand_landmarks.landmark[8]
        for lm in res.left_hand_landmarks.landmark:
            features.append(lm.x - ref_left.x)
            features.append(lm.y - ref_left.y)
    else:
        features.extend([0.0] * 42)

    if res.right_hand_landmarks:
        ref_right = res.right_hand_landmarks.landmark[8]
        for lm in res.right_hand_landmarks.landmark:
            features.append(lm.x - ref_right.x)
            features.append(lm.y - ref_right.y)
    else:
        features.extend([0.0] * 42)

    return np.array(features)

# ----------------------------
# Emotion Prediction
# ----------------------------
def predict_emotion_from_landmarks(frame):
    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    features = extract_landmark_features(res)
    if features is None:
        return None
    features = features.reshape(1, -1)
    features_pca = pca.transform(features)
    emotion_prob = model.predict(features_pca)[0]
    emotion_confidences = dict(zip(emotion_labels, emotion_prob))
    return emotion_confidences

# ----------------------------
# Main App Logic (unchanged from your latest version)
# ----------------------------
def main():
    st.title("🎭 Emotion-Based Music Recommendation System")

    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'detected_emotions' not in st.session_state:
        st.session_state.detected_emotions = defaultdict(list)
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'language' not in st.session_state:
        st.session_state.language = ""
    if 'singer' not in st.session_state:
        st.session_state.singer = ""
    if 'emotion' not in st.session_state:
        st.session_state.emotion = ""
    if 'show_input_error' not in st.session_state:
        st.session_state.show_input_error = False
    if 'show_recommend_options' not in st.session_state:
        st.session_state.show_recommend_options = False

    st.session_state.language = st.text_input("Preferred Language 🎵", value=st.session_state.language, key="lang_input")
    st.session_state.singer = st.text_input("Favorite Singer 🎤", value=st.session_state.singer, key="singer_input")

    if st.button("📷 Capture Emotion"):
        if st.session_state.language and st.session_state.singer and not st.session_state.camera_active:
            st.session_state.camera_active = True
            st.session_state.start_time = time.time()
            st.session_state.detected_emotions = defaultdict(list)
            st.session_state.show_results = False
            st.session_state.show_recommend_options = False
            st.session_state.show_input_error = False
        else:
            st.session_state.show_input_error = True

    if st.session_state.show_input_error:
        st.error("Please enter both language and singer to start the camera.")

    if st.session_state.camera_active:
        camera_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while st.session_state.camera_active:
            elapsed = time.time() - start_time
            if elapsed >= 10:
                st.session_state.camera_active = False
                st.session_state.show_results = True
                cap.release()
                st.experimental_rerun()

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            frame_copy = frame.copy()
            emotion_confidences = predict_emotion_from_landmarks(frame)
            if emotion_confidences:
                for emotion, confidence in emotion_confidences.items():
                    st.session_state.detected_emotions[emotion].append(confidence)
                top_emotion = max(emotion_confidences.items(), key=lambda x: x[1])[0]
                cv2.putText(frame_copy, top_emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame_copy, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            time.sleep(0.05)

    if st.session_state.show_results:
        if st.session_state.detected_emotions:
            st.success("✅ Analysis complete!")
            emotion_averages = {
                emotion: sum(confs) / len(confs)
                for emotion, confs in st.session_state.detected_emotions.items()
            }
            dominant_emotion = max(emotion_averages.items(), key=lambda x: x[1])
            st.session_state.emotion = dominant_emotion[0]
            st.subheader("Emotion Detected")
            st.markdown(f"""
            <div style="text-align: center;">
                <h2 style="color: #4CAF50;">{dominant_emotion[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("😕 No faces detected during the session.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎧 Recommend me songs"):
                st.session_state.show_recommend_options = True
        with col2:
            if st.button("🔄 Reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()

        if st.session_state.show_recommend_options:
            query = f"{st.session_state.singer} {st.session_state.language} {st.session_state.emotion} songs single solo"
            spotify_url = f"https://open.spotify.com/search/{query}"
            youtube_url = f"https://www.youtube.com/results?search_query={query}"

            youtube_img_src = f"data:image/png;base64,{youtube_logo_base64}" if youtube_logo_base64 else "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg"

            html_content = f"""
            <div style="display: flex; flex-direction: column; align-items: flex-start; margin-top: 20px;">
              <div style="display: flex; gap: 20px;">
                <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                  <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" 
                       alt="Spotify" style="width:60px;">
                </a>
                <a href="{youtube_url}" target="_blank" style="text-decoration: none;">
                  <img src="{youtube_img_src}" 
                       alt="YouTube" style="width:80px; margin-top: -11px;">
                </a>
              </div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
