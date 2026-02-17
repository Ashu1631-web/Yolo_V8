import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from tracker import CentroidTracker
from heatmap import generate_heatmap

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Traffic Monitoring System",
    layout="wide"
)

st.title("🚦 AI Traffic Monitoring & Analytics Dashboard")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5
)

# -------------------------------
# Load YOLO Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

tracker = CentroidTracker()

# -------------------------------
# Upload Video Only (Cloud Friendly)
# -------------------------------
uploaded_video = st.file_uploader(
    "📌 Upload Traffic Video",
    type=["mp4", "mov", "avi"]
)

if uploaded_video:

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Multi-Lane Lines
    lane_positions = [200, 350, 500]
    lane_counts = {1: 0, 2: 0, 3: 0}

    # Vehicle Counter
    counted_ids = set()
    total_count = 0

    stframe = st.empty()

    # Dashboard Tabs
    tab1, tab2 = st.tabs(["🎥 Live Monitoring", "📊 Admin Dashboard"])

    with tab1:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)

            boxes = []

            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

            objects = tracker.update(boxes)

            # Draw Lane Lines
            for lane in lane_positions:
                cv2.line(frame, (0, lane),
                         (frame.shape[1], lane),
                         (255, 0, 0), 2)

            # Vehicle Counting Logic
            for objectID, centroid in objects.items():
                cX, cY = centroid

                # Count only once per object
                if objectID not in counted_ids:

                    if cY > lane_positions[0] and cY < lane_positions[1]:
                        lane_counts[1] += 1
                        counted_ids.add(objectID)

                    elif cY > lane_positions[1] and cY < lane_positions[2]:
                        lane_counts[2] += 1
                        counted_ids.add(objectID)

                    elif cY > lane_positions[2]:
                        lane_counts[3] += 1
                        counted_ids.add(objectID)

            total_count = sum(lane_counts.values())

            # Heatmap Overlay
            heatmap = generate_heatmap(frame, boxes)
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

            stframe.image(frame, channels="BGR")

        cap.release()

    # -------------------------------
    # Admin Dashboard Analytics
    # -------------------------------
    with tab2:

        st.subheader("📌 Traffic Analytics Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("🚗 Total Vehicles", total_count)
        col2.metric("Lane 1 Count", lane_counts[1])
        col3.metric("Lane 2 + Lane 3", lane_counts[2] + lane_counts[3])

        # Lane Chart
        df_chart = pd.DataFrame({
            "Lane": ["Lane 1", "Lane 2", "Lane 3"],
            "Count": [lane_counts[1], lane_counts[2], lane_counts[3]]
        })

        fig = px.bar(
            df_chart,
            x="Lane",
            y="Count",
            title="🚦 Lane-wise Vehicle Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Excel Export
        report = pd.DataFrame([{
            "Lane 1": lane_counts[1],
            "Lane 2": lane_counts[2],
            "Lane 3": lane_counts[3],
            "Total Vehicles": total_count
        }])

        report.to_excel("traffic_report.xlsx", index=False)

        with open("traffic_report.xlsx", "rb") as f:
            st.download_button(
                "📥 Download Excel Traffic Report",
                f,
                "traffic_report.xlsx"
            )
