import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLOWorld
import numpy as np

def get_classes_from_input():
    cl = st.text_input("Class of object that you want to detect")
    return cl.split()

def load_model():
    return YOLOWorld("yolov8x-worldv2.pt")

def camera_stream(model, classes):
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    

    model.set_classes(classes)  # Set your custom zero-shot class list

    while st.session_state.camera_on:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)   #bkc not related to line
        
        results = model.predict(frame_rgb, conf=0.5)  # ✅ correct

        detections = sv.Detections.from_ultralytics(results[0])  # ✅ fix

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [classes[class_id] for class_id in detections.class_id]

        annotated_image = bounding_box_annotator.annotate(
            scene=frame_rgb, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        FRAME_WINDOW.image(annotated_image)

    camera.release()


def main():
    st.set_page_config(page_title="Webcam Stream", layout="centered")
    st.title("Welcome to object detection app")

    classes = get_classes_from_input()
    model = load_model()
    #conn, cur = connect_to_db()

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