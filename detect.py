import streamlit as st
import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import numpy as np

st.set_page_config(page_title="Webcam Stream", layout="centered")
st.title("📸 Live Webcam Feed with OpenCV + Streamlit")

st.text_input("Class of object that you want to detect")

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

model = YOLOWorld(model_id="yolo_world/l")
classes = ["person","hand","Phone","Headphone","bottle",]



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
        
        results = model.infer(frame, text=classes, confidence=0.03)
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [classes[class_id] for class_id in detections.class_id]

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        FRAME_WINDOW.image(frame)
        # Add a small delay to avoid high CPU usage
        if not st.session_state.camera_on:
            break
    camera.release()
else:
    FRAME_WINDOW.image([])  # Clear the image when camera is off