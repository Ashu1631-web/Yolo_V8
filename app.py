import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from tracker import CentroidTracker
from utils import calculate_speed

st.set_page_config(page_title="AI Traffic Monitoring", layout="wide")

st.title("🚦 AI Traffic Monitoring System")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

model = YOLO("best.pt")
tracker = CentroidTracker()

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_video:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    line_position = 300
    vehicle_count = 0
    vehicle_data = []

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence)
        boxes = []

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

        objects = tracker.update(boxes)

        for objectID, centroid in objects.items():
            cX, cY = centroid
            cv2.circle(frame, (cX,cY), 4, (0,0,255), -1)

            if cY > line_position:
                vehicle_count += 1

        cv2.line(frame, (0,line_position), (frame.shape[1],line_position), (255,0,0),2)

        stframe.image(frame, channels="BGR")

    cap.release()

    st.success(f"Total Vehicles Detected: {vehicle_count}")

    df = pd.DataFrame({"Total Vehicles":[vehicle_count]})
    fig = px.bar(df, y="Total Vehicles", title="Vehicle Count Summary")
    st.plotly_chart(fig)
