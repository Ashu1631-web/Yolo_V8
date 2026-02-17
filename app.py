import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from tracker import CentroidTracker
from heatmap import generate_heatmap
from modules.heatmap import generate_heatmap
import sys
st.write("Python Version:", sys.version)


# ----------------------------------------
# Streamlit Page Setup
# ----------------------------------------
st.set_page_config(
    page_title="AI Traffic Monitoring System",
    layout="wide"
)

st.title("🚦 AI Traffic Monitoring & Analytics Platform")

# ----------------------------------------
# Sidebar Controls
# ----------------------------------------
st.sidebar.header("⚙ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5
)

st.sidebar.header("🚘 Vehicle Class Filter")

selected_classes = st.sidebar.multiselect(
    "Select Vehicle Types",
    ["car", "bus", "truck", "motorcycle"],
    default=["car", "bus", "truck"]
)

# ----------------------------------------
# Load YOLO Model (Cached)
# ----------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

tracker = CentroidTracker()

# ----------------------------------------
# Upload Video (Cloud Friendly)
# ----------------------------------------
uploaded_video = st.file_uploader(
    "📌 Upload Traffic Video",
    type=["mp4", "mov", "avi"]
)

if uploaded_video:

    # Temporary file save
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Multi-Lane Lines
    lane_positions = [200, 350, 500]
    lane_counts = {1: 0, 2: 0, 3: 0}

    # Vehicle Counter Storage
    counted_ids = set()

    # Vehicle Type Counter
    vehicle_types = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}

    stframe = st.empty()

    # Tabs for UI
    tab1, tab2 = st.tabs(["🎥 Live Monitoring", "📊 Admin Dashboard"])

    # ----------------------------------------
    # LIVE MONITORING TAB
    # ----------------------------------------
    with tab1:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Detection
            results = model(frame, conf=confidence)

            boxes = []

            # Extract detections
            for det in results[0].boxes:
                cls_id = int(det.cls[0])
                label = model.names[cls_id]

                if label in selected_classes:

                    # Vehicle Type Count
                    vehicle_types[label] += 1

                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

                    # Draw bounding box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame, label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2)

            # Tracking objects
            objects = tracker.update(boxes)

            # Draw Lane Lines
            for lane in lane_positions:
                cv2.line(frame, (0, lane),
                         (frame.shape[1], lane),
                         (255, 0, 0), 2)

            # Lane Counting Logic (Count only once per object)
            for objectID, centroid in objects.items():
                cX, cY = centroid

                if objectID not in counted_ids:

                    if lane_positions[0] < cY < lane_positions[1]:
                        lane_counts[1] += 1
                        counted_ids.add(objectID)

                    elif lane_positions[1] < cY < lane_positions[2]:
                        lane_counts[2] += 1
                        counted_ids.add(objectID)

                    elif cY > lane_positions[2]:
                        lane_counts[3] += 1
                        counted_ids.add(objectID)

            # Total Vehicles
            total_count = sum(lane_counts.values())

            # Traffic Density Status
            if total_count < 10:
                density = "🟢 Low Traffic"
            elif total_count < 25:
                density = "🟡 Medium Traffic"
            else:
                density = "🔴 High Traffic"

            # Heatmap Overlay
            heatmap = generate_heatmap(frame, boxes)
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

            # Show Video Frame
            stframe.image(frame, channels="BGR")

        cap.release()

    # ----------------------------------------
    # ADMIN DASHBOARD TAB
    # ----------------------------------------
    with tab2:

        st.subheader("📌 Traffic Analytics Summary")

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🚗 Total Vehicles", total_count)
        col2.metric("Lane 1 Count", lane_counts[1])
        col3.metric("Lane 2 Count", lane_counts[2])
        col4.metric("Lane 3 Count", lane_counts[3])

        st.metric("🚦 Traffic Density Status", density)

        # Vehicle Type Metrics
        st.subheader("🚘 Vehicle Type Distribution")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Cars", vehicle_types["car"])
        c2.metric("Buses", vehicle_types["bus"])
        c3.metric("Trucks", vehicle_types["truck"])
        c4.metric("Motorcycles", vehicle_types["motorcycle"])

        # Lane Chart
        df_lane = pd.DataFrame({
            "Lane": ["Lane 1", "Lane 2", "Lane 3"],
            "Count": [lane_counts[1], lane_counts[2], lane_counts[3]]
        })

        fig1 = px.bar(
            df_lane,
            x="Lane",
            y="Count",
            title="🚦 Lane-wise Vehicle Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Vehicle Type Pie Chart
        df_vehicle = pd.DataFrame({
            "Vehicle Type": list(vehicle_types.keys()),
            "Count": list(vehicle_types.values())
        })

        fig2 = px.pie(
            df_vehicle,
            names="Vehicle Type",
            values="Count",
            title="🚘 Vehicle Type Breakdown"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ----------------------------------------
        # Excel Export Report
        # ----------------------------------------
        report = pd.DataFrame([{
            "Lane 1": lane_counts[1],
            "Lane 2": lane_counts[2],
            "Lane 3": lane_counts[3],
            "Total Vehicles": total_count,
            "Cars": vehicle_types["car"],
            "Buses": vehicle_types["bus"],
            "Trucks": vehicle_types["truck"],
            "Motorcycles": vehicle_types["motorcycle"],
            "Traffic Density": density
        }])

        report.to_excel("traffic_report.xlsx", index=False)

        with open("traffic_report.xlsx", "rb") as f:
            st.download_button(
                "📥 Download Full Traffic Excel Report",
                f,
                "traffic_report.xlsx"
            )
