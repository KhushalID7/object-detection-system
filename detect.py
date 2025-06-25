import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Webcam Stream", layout="centered")
st.title("📸 Live Webcam Feed with OpenCV + Streamlit")

st.text_input("Class of object that you want to detect")

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# Initialize session state for camera
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Toggle camera state on button click
if st.button('Start/Stop Camera'):
    st.session_state.camera_on = not st.session_state.camera_on

FRAME_WINDOW = st.image([])

if st.session_state.camera_on:
    camera = cv2.VideoCapture(0)
    while st.session_state.camera_on:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        FRAME_WINDOW.image(frame)
        # Add a small delay to avoid high CPU usage
        if not st.session_state.camera_on:
            break
    camera.release()
else:
    FRAME_WINDOW.image([])  # Clear the image when camera is off