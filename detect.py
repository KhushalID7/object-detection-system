import streamlit as st
import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import numpy as np

def get_classes_from_input():
    cl = st.text_input("Class of object that you want to detect")
    return cl.split()

def load_model():
    return YOLOWorld(model_id="yolo_world/l")

def camera_stream(model, classes):
    FRAME_WINDOW = st.image([])
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

        FRAME_WINDOW.image(annotated_image)
        if not st.session_state.camera_on:
            break
    camera.release()

def main():
    st.set_page_config(page_title="Webcam Stream", layout="centered")
    st.title("📸 Live Webcam Feed with OpenCV + Streamlit")

    classes = get_classes_from_input()
    model = load_model()

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    if st.button('Start/Stop Camera'):
        st.session_state.camera_on = not st.session_state.camera_on

    if st.session_state.camera_on:
        camera_stream(model, classes)
    else:
        st.image([])  # Clear the image when camera is off

if __name__ == "__main__":
    main()