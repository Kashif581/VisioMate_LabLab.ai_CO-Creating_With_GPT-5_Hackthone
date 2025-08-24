import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import sys
import os
import time

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from YOLO_Detections.Yolo_detector import VideoProcessor
from YOLO_Detections.speak_detections import DetectionLog

st.set_page_config(page_title="VisioMate Navigator", layout="wide")
st.title("👁️ Spoken Detection Logs")

# Initialize DetectionLog
if "detection_log" not in st.session_state:
   st.session_state.detection_log = DetectionLog(repeat_interval=2.0)

# Initialize log history
if "log_history" not in st.session_state:
    st.session_state.log_history = []

# Layout: 2 columns
log_col, video_col = st.columns([1, 2])
log_placeholder = log_col.empty()
video_placeholder = video_col.empty()

# CSS for scrollable log box
st.markdown("""
<style>
.scrollable-log { height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9; font-family: monospace; }
.detection-entry { margin-bottom: 12px; }
.explanation { color: #2c7be5; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# Initialize VideoProcessor
processor = VideoProcessor(detection_log=st.session_state.detection_log)

# User selects mode
mode = st.radio("Select input mode:", ["Local Webcam", "Browser Camera (Cloud)"])

# Run detection flag
if "running" not in st.session_state:
    st.session_state.running = False

if st.button("Start Detection"):
    st.session_state.running = True
if st.button("Stop Detection"):
    st.session_state.running = False

# --- Local Webcam Mode ---
if mode == "Local Webcam" and st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        frame = processor.process_frame(frame)

        # Log new detections
        for obj in st.session_state.detection_log.last_spoken.keys():
            result = st.session_state.detection_log.log(obj)
            if result:
                detection, explanation = result
                timestamp = datetime.now().strftime("%H:%M:%S")
                entry_html = f"""
                <div class="detection-entry">
                    <b>[{timestamp}] {detection}</b><br>
                    <span class="explanation">➡ {explanation}</span>
                </div>
                """
                st.session_state.log_history.insert(0, entry_html)
                st.session_state.log_history = st.session_state.log_history[:100]

        # Render log
        log_html = "".join(st.session_state.log_history)
        log_placeholder.markdown(f'<div class="scrollable-log">{log_html}</div>', unsafe_allow_html=True)

        # Show video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        time.sleep(0.03)  # ~30 FPS

    cap.release()

# --- Browser Camera Mode (Cloud Compatible) ---
elif mode == "Browser Camera (Cloud)":
    st.info("Use your browser to capture frames. Click multiple times to simulate real-time detection.")
    frame_file = st.camera_input("Take a picture")
    if frame_file is not None:
        # Convert to OpenCV frame
        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = processor.process_frame(frame)

        # Log new detections
        for obj in st.session_state.detection_log.last_spoken.keys():
            result = st.session_state.detection_log.log(obj)
            if result:
                detection, explanation = result
                timestamp = datetime.now().strftime("%H:%M:%S")
                entry_html = f"""
                <div class="detection-entry">
                    <b>[{timestamp}] {detection}</b><br>
                    <span class="explanation">➡ {explanation}</span>
                </div>
                """
                st.session_state.log_history.insert(0, entry_html)
                st.session_state.log_history = st.session_state.log_history[:100]

        # Render log
        log_html = "".join(st.session_state.log_history)
        log_placeholder.markdown(f'<div class="scrollable-log">{log_html}</div>', unsafe_allow_html=True)

        # Show video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
