"""
overlay.py — OpenCV drawing utilities for detection visualization.

This module draws bounding boxes, labels, and HUD information on frames.
Participants are encouraged to customize colors, styles, and added annotations.
"""

import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Color palette for different classes (BGR format for OpenCV)
# Index maps to COCO class ID % len(COLORS)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (0, 128, 128),  # teal
]


# COCO 17-keypoint skeleton: pairs of keypoint indices to connect with a line.
# Index order: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
# 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def get_color_for_label(label_id: int) -> tuple:
    """Return a consistent BGR color for a given class ID."""
    return COLORS[label_id % len(COLORS)]


def draw_keypoints(frame, keypoints: list, kp_threshold: float = 0.5) -> None:
    """
    Draw pose keypoints and the COCO skeleton on the frame (in-place).

    Args:
        frame: BGR image — modified in-place
        keypoints: list of 17 [x, y, visibility] entries in frame pixels
        kp_threshold: minimum visibility to draw a keypoint/limb
    """
    for a, b in COCO_SKELETON:
        if keypoints[a][2] > kp_threshold and keypoints[b][2] > kp_threshold:
            cv2.line(frame,
                     (keypoints[a][0], keypoints[a][1]),
                     (keypoints[b][0], keypoints[b][1]),
                     (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, v in keypoints:
        if v > kp_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def draw_detection(frame, detection: dict, label: str, config: dict) -> None:
    """
    Draw a single detection bounding box and label on the frame (in-place).

    Args:
        frame: BGR image (numpy array) — modified in-place
        detection: Dict with 'bbox', 'label_id', 'confidence'
        label: Human-readable class name string
        config: Full application config dict
    """
    x1, y1, x2, y2 = detection["bbox"]
    confidence = detection["confidence"]
    label_id = detection["label_id"]

    thickness = config["overlay"].get("box_thickness", 2)
    font_scale = config["overlay"].get("font_scale", 0.6)
    show_confidence = config["overlay"].get("show_confidence", True)

    color = get_color_for_label(label_id)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build label text
    text = f"{label}: {confidence:.0%}" if show_confidence else label

    # Calculate text background dimensions
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(y1 - 5, text_h + 5)

    # Draw filled rectangle behind label text for readability
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline),
        (x1 + text_w, label_y + baseline),
        color,
        cv2.FILLED
    )

    # Draw label text (black for contrast)
    cv2.putText(
        frame, text,
        (x1, label_y - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )

    # Pose models attach keypoints; draw the skeleton on top of the box.
    if "keypoints" in detection:
        draw_keypoints(frame, detection["keypoints"])


def draw_all_detections(frame, detections: list[dict], labels: list[str],
                        config: dict) -> None:
    """
    Draw all detections on the frame (in-place).

    Args:
        frame: BGR image — modified in-place
        detections: List of detection dicts from inference.run_inference()
        labels: List of class name strings indexed by label_id
        config: Full application config dict
    """
    for det in detections:
        label_id = det["label_id"]
        label = labels[label_id] if label_id < len(labels) else str(label_id)
        draw_detection(frame, det, label, config)


def draw_hud(frame, fps: float, num_detections: int, config: dict,
             model_name: str = "", inference_ms: float = 0.0,
             invoke_ms: float = 0.0) -> None:
    """
    Draw a heads-up display (HUD) overlay with runtime stats.

    Args:
        frame: BGR image — modified in-place
        fps: Current frames per second
        num_detections: Number of detections in current frame
        config: Full application config dict
        model_name: Short name of the model being used
    """
    if not config["overlay"].get("show_fps", True):
        return

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (255, 255, 255)
    shadow = (0, 0, 0)
    thickness = 1

    lines = [
        f"FPS: {fps:.1f}",
        f"Invoke: {invoke_ms:.1f} ms  Post: {inference_ms:.1f} ms",
        f"Detections: {num_detections}",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(datetime.now().strftime("%H:%M:%S"))

    y = 20
    for line in lines:
        # Draw shadow for readability on any background
        cv2.putText(frame, line, (11, y + 1), font, font_scale, shadow, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += 22


def draw_roi_zone(frame, zone: tuple, color=(0, 200, 255), label: str = "Zone") -> None:
    """
    Draw a named region-of-interest (ROI) rectangle on the frame.

    Args:
        frame: BGR image — modified in-place
        zone: (x1, y1, x2, y2) pixel coordinates of the zone
        color: BGR color tuple for the zone border
        label: Text label to display at top-left of zone
    """
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def annotate_frame(frame, detections: list[dict], labels: list[str],
                   fps: float, config: dict, model_name: str = "",
                   inference_ms: float = 0.0, invoke_ms: float = 0.0) -> np.ndarray:
    """
    Full annotation pipeline: draw detections + HUD on a copy of the frame.

    Args:
        frame: Original BGR image (not modified)
        detections: List of detection dicts
        labels: Class label strings
        fps: Current FPS for HUD display
        config: Full application config dict
        model_name: Optional model name for HUD

    Returns:
        annotated: Annotated copy of the frame
    """
    annotated = frame.copy()
    draw_all_detections(annotated, detections, labels, config)
    return annotated
