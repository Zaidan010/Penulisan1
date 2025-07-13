import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from speed1 import SpeedEstimator
import tempfile

# Konfigurasi halaman
st.set_page_config(page_title="🚗 Deteksi Kendaraan", layout="wide")
st.title("🚗 Deteksi & Estimasi Kecepatan Kendaraan")

# Inisialisasi model dan kelas target
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck

# Pilih mode input
mode = st.radio("Pilih Mode Input:", ["📷 Kamera Langsung", "📁 Upload Video"])

# Fungsi estimasi kecepatan dari video statis
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

# Mode Kamera Real-Time
if mode == "📷 Kamera Langsung":

    class YOLOWithSpeed(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.speed_estimator = SpeedEstimator(names=model.names)

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            image_resized = cv2.resize(image, (640, 360))

            results = self.model.track(image_resized, persist=True, classes=VEHICLE_CLASSES)

            if results and results[0].boxes.id is not None:
                image_resized = self.speed_estimator.estimate_speed(image_resized, results)

            return av.VideoFrame.from_ndarray(image_resized, format="bgr24")

    st.info("🎥 Izinkan akses kamera saat diminta di browser.")
    webrtc_streamer(
        key="yolo-speed-live",
        video_transformer_factory=YOLOWithSpeed,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

# Mode Upload Video
elif mode == "📁 Upload Video":
    uploaded_video = st.file_uploader("Unggah Video Kendaraan", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        process_uploaded_video(temp_video.name)
