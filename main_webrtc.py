import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from speed1 import SpeedEstimator  # <- Tambahkan ini

# Load YOLOv8 once
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck

st.set_page_config(page_title="🚗 Real-time Vehicle Speed Detection", layout="wide")
st.title("🚗 Deteksi & Estimasi Kecepatan Kendaraan (Live Kamera HP/Laptop)")

class YOLOWithSpeed(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.speed_estimator = SpeedEstimator(names=model.names)  # Gunakan speed1.py

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_resized = cv2.resize(image, (640, 360))

        # Deteksi kendaraan
        results = self.model.track(image_resized, persist=True, classes=VEHICLE_CLASSES)

        # Estimasi kecepatan
        if results and results[0].boxes.id is not None:
            image_resized = self.speed_estimator.estimate_speed(image_resized, results)

        return av.VideoFrame.from_ndarray(image_resized, format="bgr24")

# Streaming kamera dari browser
webrtc_streamer(
    key="yolo-speed-live",
    video_transformer_factory=YOLOWithSpeed,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
