"""
streamer.py — Flask MJPEG HTTP streaming server.

Runs in a background thread alongside the inference loop.
The main loop pushes annotated frames into a queue; Flask serves them
as a multipart MJPEG stream readable by any browser.

Endpoints:
  GET /              → HTML viewer page (index.html)
  GET /stream        → MJPEG video stream
  GET /status        → JSON runtime stats {fps, model, detections, ...}
  GET /detections    → JSON list of last N detections
  POST /config       → Update runtime config values (threshold, etc.)
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

logger = logging.getLogger(__name__)

# Shared state — written by inference loop, read by Flask endpoints
_frame_queue: queue.Queue = queue.Queue(maxsize=2)
_latest_detections: list[dict] = []
_detection_history: list[dict] = []        # Rolling history for /detections
_HISTORY_MAX = 100
_fps: float = 0.0
_invoke_ms: float = 0.0
_inference_ms: float = 0.0
_stage_latency: list[dict] = []
_model_name: str = ""
_config_ref: dict = {}                     # Reference to live config dict
_config_lock = threading.Lock()
_active_camera: str = ""
_camera_list_fn = None
_camera_switch_fn = None

app = Flask(__name__, template_folder="templates")
# Suppress Flask's default request logs to keep the terminal clean
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Public API — called from main.py
# ---------------------------------------------------------------------------

def register_camera_callbacks(list_fn, switch_fn) -> None:
    """Register callables used by /cameras and /camera endpoints."""
    global _camera_list_fn, _camera_switch_fn
    _camera_list_fn = list_fn
    _camera_switch_fn = switch_fn


def set_active_camera(device: str) -> None:
    global _active_camera
    _active_camera = device


def init(config: dict) -> None:
    """
    Initialise the streamer with the application config.
    Must be called before start().

    Args:
        config: Full application config dict (from config.json)
    """
    global _config_ref, _model_name
    _config_ref = config
    _model_name = Path(config["model"]["path"]).stem


def push_frame(annotated_frame, detections: list[dict],
               labels: list[str], fps: float,
               invoke_ms: float = 0.0, inference_ms: float = 0.0,
               stage_latency: list[dict] | None = None) -> None:
    """
    Push an annotated frame and detection metadata into the streamer.

    Called from the main inference loop on every frame.
    Non-blocking: drops the frame if the queue is full (viewer is slow).

    Args:
        annotated_frame: BGR numpy array with overlay already drawn
        detections: Current frame detections [{bbox, label_id, confidence}]
        labels: Class label strings
        fps: Current inference FPS
        invoke_ms: Accumulated NPU invoke time in milliseconds
        inference_ms: Post-processing time in milliseconds
        stage_latency: Per-stage [{label, min_ms, max_ms}] timing stats
    """
    global _latest_detections, _fps, _invoke_ms, _inference_ms, _stage_latency

    import cv2

    _fps = fps
    _invoke_ms = invoke_ms
    _inference_ms = inference_ms
    if stage_latency is not None:
        _stage_latency = stage_latency

    # Resolve label names for the detection snapshot
    named = []
    for det in detections:
        lid = det["label_id"]
        label = labels[lid] if lid < len(labels) else str(lid)
        named.append({
            "label": label,
            "confidence": round(det["confidence"], 3),
            "bbox": det["bbox"],
            "timestamp": datetime.now().isoformat()
        })
    _latest_detections = named

    # Append to rolling history
    if named:
        _detection_history.extend(named)
        if len(_detection_history) > _HISTORY_MAX:
            del _detection_history[:-_HISTORY_MAX]

    # Encode frame as JPEG and push into queue (drop if full)
    quality = _config_ref.get("streaming", {}).get("jpeg_quality", 75)
    ok, jpeg_buf = cv2.imencode(".jpg", annotated_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return

    try:
        _frame_queue.put_nowait(jpeg_buf.tobytes())
    except queue.Full:
        pass  # Drop frame — viewer can't keep up, that's fine


def start(config: dict) -> threading.Thread:
    """
    Start the Flask server in a background daemon thread.

    Args:
        config: Full application config dict

    Returns:
        thread: The background thread (already started)
    """
    init(config)

    host = config["streaming"].get("host", "0.0.0.0")
    port = config["streaming"].get("port", 5000)

    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
        name="flask-streamer"
    )
    thread.start()
    logger.info(f"Streamer started at http://{host}:{port}")
    return thread


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the HTML viewer page."""
    port = _config_ref.get("streaming", {}).get("port", 5000)
    return render_template("index.html", port=port, model_name=_model_name)


@app.route("/stream")
def stream():
    """MJPEG stream endpoint — open in browser or <img src='/stream'>."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """Return JSON runtime status."""
    import actions  # imported here to avoid circular deps at module level
    stats = actions.get_stats()
    return jsonify({
        "fps": round(_fps, 1),
        "invoke_ms": round(_invoke_ms, 1),
        "inference_ms": round(_inference_ms, 1),
        "stage_latency": _stage_latency,
        "model": _model_name,
        "model_variant": _config_ref.get("model", {}).get("variant", ""),
        "active_camera": _active_camera,
        "num_detections": len(_latest_detections),
        "detections": _latest_detections,
        "timestamp": datetime.now().isoformat(),
        **stats
    })


@app.route("/detections")
def detections():
    """Return the last N detections as JSON."""
    n = request.args.get("n", 20, type=int)
    return jsonify(_detection_history[-n:])


@app.route("/config", methods=["GET"])
def get_config():
    """Return the current runtime config (read-only view)."""
    with _config_lock:
        return jsonify(_config_ref)


@app.route("/config", methods=["POST"])
def update_config():
    """
    Update specific config values at runtime.

    Example body:
        {"inference": {"confidence_threshold": 0.6}}

    Only existing keys are updated (deep merge, not replace).
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    with _config_lock:
        _deep_merge(_config_ref, data)

    logger.info(f"Config updated via POST /config: {data}")
    return jsonify({"status": "ok", "updated": data})


@app.route("/cameras")
def get_cameras():
    """Return a list of available V4L2 camera devices."""
    cameras = _camera_list_fn() if _camera_list_fn else []
    for cam in cameras:
        cam['active'] = (cam['device'] == _active_camera)
    return jsonify(cameras)


@app.route("/camera", methods=["POST"])
def switch_camera():
    """Switch to a different camera device."""
    data = request.get_json(silent=True)
    device = (data or {}).get('device', '').strip()
    if not device:
        return jsonify({"error": "device field required"}), 400
    if _camera_switch_fn is None:
        return jsonify({"error": "camera switching not available"}), 503
    # Run in a background thread so the HTTP response returns immediately.
    threading.Thread(target=_camera_switch_fn, args=(device,), daemon=True).start()
    return jsonify({"status": "switching", "device": device})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_mjpeg():
    """Generator that yields MJPEG frames from the queue."""
    while True:
        try:
            jpeg_bytes = _frame_queue.get(timeout=5.0)
        except queue.Empty:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes +
            b"\r\n"
        )


def _deep_merge(base: dict, update: dict) -> None:
    """Recursively merge update into base (in-place)."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base:
            base[key] = value
        else:
            logger.warning(f"Config key '{key}' not found — ignoring")
