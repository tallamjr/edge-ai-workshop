"""
actions.py — Extensible action hooks triggered by detections.

This is the primary extension point for workshop participants.
Each function is called from the main inference loop when detections occur.

Extension ideas for participants:
- Send an HTTP POST to a remote endpoint
- Write to a CSV log file
- Toggle a GPIO output
- Publish to MQTT
- Save a snapshot image
- Send an alert email
"""

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Internal state ---
_last_alert_time: dict[str, float] = {}   # label -> last alert timestamp
_detection_count: int = 0
_alert_cooldown_s: float = 5.0            # seconds between repeated alerts for same class


def on_frame(frame, detections: list[dict], labels: list[str], config: dict) -> None:
    """
    Called once per frame with all current detections.

    This is the top-level hook — it dispatches to more specific handlers.
    Participants can add logic here or in the helper functions below.

    Args:
        frame: Current BGR image (numpy array) — do not modify
        detections: List of detection dicts {bbox, label_id, confidence}
        labels: Class label strings indexed by label_id
        config: Full application config dict
    """
    global _detection_count
    _detection_count += 1

    if not detections:
        return

    # Resolve label names
    named = []
    for det in detections:
        label_id = det["label_id"]
        label = labels[label_id] if label_id < len(labels) else str(label_id)
        named.append({**det, "label": label})

    # Log to CSV
    if config["actions"].get("enable_logging", False):
        _log_detections(named, config)

    # Check alert classes
    alert_classes = config["actions"].get("alert_classes", [])
    min_conf = config["actions"].get("alert_min_confidence", 0.7)
    for det in named:
        if det["label"] in alert_classes and det["confidence"] >= min_conf:
            _trigger_alert(det, frame, config)


def on_detection(detection: dict, label: str, frame, config: dict) -> None:
    """
    Called for every individual detection above threshold.

    Args:
        detection: Single detection dict {bbox, label_id, confidence}
        label: Human-readable class name
        frame: Current BGR image
        config: Full application config dict
    """
    # Example: log to console
    logger.debug(f"Detected: {label} ({detection['confidence']:.0%}) at {detection['bbox']}")


def on_alert_class_detected(detection: dict, label: str, frame, config: dict) -> None:
    """
    Called when a detection matches one of the configured alert_classes.

    This function is intentionally simple — participants should extend it.

    Args:
        detection: Detection dict
        label: Class name
        frame: Current BGR frame (can be used to save a snapshot)
        config: Full application config dict
    """
    logger.info(f"[ALERT] {label} detected with confidence {detection['confidence']:.0%}")

    # --- Extension point: save snapshot ---
    # import cv2
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # snapshot_path = f"/tmp/alert_{label}_{timestamp}.jpg"
    # cv2.imwrite(snapshot_path, frame)
    # logger.info(f"Snapshot saved to {snapshot_path}")

    # --- Extension point: HTTP notification ---
    # import requests
    # requests.post("http://your-server/alert", json={
    #     "label": label,
    #     "confidence": detection["confidence"],
    #     "timestamp": datetime.now().isoformat()
    # }, timeout=1)

    # --- Extension point: GPIO output (e.g., LED) ---
    # import gpiod
    # chip = gpiod.Chip('gpiochip0')
    # line = chip.get_line(20)  # GPIO pin 20
    # line.request(consumer="alert", type=gpiod.LINE_REQ_DIR_OUT)
    # line.set_value(1)
    # time.sleep(0.5)
    # line.set_value(0)
    # line.release()

    # --- Extension point: MQTT publish ---
    # import paho.mqtt.publish as publish
    # publish.single("imx95/alerts", payload=label, hostname="192.168.7.1")


def get_stats() -> dict:
    """
    Return current runtime statistics as a JSON-serializable dict.
    Called by streamer.py for the /status endpoint.

    Returns:
        dict with frame count and last detection time
    """
    return {
        "total_frames_processed": _detection_count,
        "last_alert_per_class": {
            k: datetime.fromtimestamp(v).isoformat()
            for k, v in _last_alert_time.items()
        }
    }


# --- Private helpers ---

def _log_detections(named_detections: list[dict], config: dict) -> None:
    """Append detections to a CSV log file."""
    log_file = config["actions"].get("log_file", "/tmp/detections.csv")
    file_exists = Path(log_file).exists()

    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])
            ts = datetime.now().isoformat()
            for det in named_detections:
                x1, y1, x2, y2 = det["bbox"]
                writer.writerow([ts, det["label"], f"{det['confidence']:.4f}",
                                  x1, y1, x2, y2])
    except OSError as e:
        logger.warning(f"Could not write to log file {log_file}: {e}")


def _trigger_alert(detection: dict, frame, config: dict) -> None:
    """Trigger alert for a detection, respecting cooldown per class."""
    label = detection["label"]
    now = time.monotonic()

    last = _last_alert_time.get(label, 0.0)
    if now - last < _alert_cooldown_s:
        return  # Still in cooldown

    _last_alert_time[label] = now
    on_alert_class_detected(detection, label, frame, config)
