import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from speed1 import SpeedEstimator

# Konfigurasi halaman
st.set_page_config(page_title="🚗 Deteksi Kendaraan", layout="wide")
st.title("📁 Deteksi & Estimasi Kecepatan Kendaraan dari Video Upload")

# Inisialisasi model dan kelas kendaraan target
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck

# Fungsi untuk proses video upload
def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    speed_estimator = SpeedEstimator(names=model.names)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model.track(frame, persist=True, classes=VEHICLE_CLASSES)

        if results and results[0].boxes.id is not None:
            frame = speed_estimator.estimate_speed(frame, results)

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Proses video selesai.")

# Upload video dari user
uploaded_video = st.file_uploader("📤 Unggah Video Kendaraan Anda", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    process_uploaded_video(temp_video.name)
