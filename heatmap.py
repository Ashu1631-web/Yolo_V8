import numpy as np
import cv2

def generate_heatmap(frame, detections):
    """
    Creates a heatmap overlay based on detected vehicle bounding boxes.
    """

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x1, y1, x2, y2) in detections:
        heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = np.uint8(heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap_color
