import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLOWorld
import numpy as np
import psycopg2

def connect_to_db():
    conn = psycopg2.connect(dbname="DEMOimg", user="postgres", password="1234")
    cur = conn.cursor()
    return conn, cur
    

def get_classes_from_input():
    cl = st.text_input("Class of object that you want to detect")
    return cl.split()

def load_model():
    return YOLOWorld("yolov8x-worldv2.pt")

def camera_stream(model, classes):
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    conn, cur = connect_to_db()

    model.set_classes(classes)  # Set your custom zero-shot class list

    while st.session_state.camera_on:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)   #bkc not related to line
        
        results = model.predict(frame_rgb, conf=0.3)  # ✅ correct

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
        str1 = results[0].verbose()
        str_contain_person = "person" in str1.lower()
        if str_contain_person:
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cur.execute("insert into only_person_detection(classes, image_data) values (%s, %s)", (str1, buffer.tobytes()))
            conn.commit()

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