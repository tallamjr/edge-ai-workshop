"""
webcam_streamer.py — Laptop-side webcam MJPEG server for remote inference.

Run this on your LAPTOP (Linux, macOS, or WSL on Windows) to use your laptop
webcam as the camera source for the FRDM-IMX95 inference pipeline.

Architecture:
    Laptop                              Board (192.168.7.2)
    ──────────────────────────────────────────────────────
    webcam capture                      cv2.VideoCapture(remote_url)
        ↓                                   ↓
    resize to 640×640                   inference (NPU)
        ↓                                   ↓
    JPEG encode                         annotate frame
        ↓                                   ↓
    Flask MJPEG → ────── Ethernet ──→  push_frame → /stream → browser

The Ethernet hop (640×640 JPEG ≈ 15–25 KB) takes ~0.1ms at 1Gbps direct link.
With concurrent pipelining, throughput is limited by NPU inference (~24ms for
YOLOv8n at 640×640), yielding ~40 FPS.

Usage:
    # 1. Start this streamer on the laptop:
    python host/webcam_streamer.py

    # 2. Configure the board to use the remote source (already default in config.json):
    #    "camera": { "source": "remote", "remote_url": "http://192.168.7.1:5001/stream" }

    # 3. Start the application on the board:
    python board/main.py

    # 4. Open browser: http://192.168.7.2:5000

Options:
    --device    Webcam device index or path (default: 0)
    --width     Capture width before resize (default: 1280)
    --height    Capture height before resize (default: 720)
    --imgsz     Output stream size, square (default: 640)
    --fps       Capture FPS (default: 30)
    --host      Host to bind the Flask server on (default: 0.0.0.0)
    --port      Port to serve the MJPEG stream on (default: 5001)
    --quality   JPEG quality 1–95 (default: 80)
"""

import argparse
import logging
import signal
import sys
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global frame queue: capture thread → Flask generator
# maxsize=2: drop oldest frame if board isn't consuming fast enough,
# ensuring the board always gets the most recent frame.
# ---------------------------------------------------------------------------
_frame_queue: Queue = Queue(maxsize=2)
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------

def capture_thread(device, width: int, height: int, imgsz: int,
                   fps: int, quality: int) -> None:
    """
    Continuously capture frames from the local webcam, resize to imgsz×imgsz,
    JPEG-encode them, and push encoded bytes into _frame_queue.

    Runs in a daemon thread — exits when _stop_event is set.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam device: {device}")
        _stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Webcam opened: {actual_w}×{actual_h} @ {actual_fps:.0f}fps  →  "
                f"resizing to {imgsz}×{imgsz} for stream")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    frame_count = 0
    t0 = time.time()

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Webcam read failed — retrying...")
            time.sleep(0.05)
            continue

        # Resize to square imgsz×imgsz (board's model input size)
        resized = cv2.resize(frame, (imgsz, imgsz),
                             interpolation=cv2.INTER_LINEAR)

        # Encode as JPEG
        ok, buf = cv2.imencode(".jpg", resized, encode_params)
        if not ok:
            continue

        jpeg_bytes = buf.tobytes()

        # Non-blocking put — drop oldest frame if queue is full
        if _frame_queue.full():
            try:
                _frame_queue.get_nowait()
            except Empty:
                pass
        _frame_queue.put_nowait(jpeg_bytes)

        frame_count += 1
        if frame_count % 150 == 0:
            elapsed = time.time() - t0
            logger.info(f"Streaming: {frame_count} frames  "
                        f"avg {frame_count/elapsed:.1f} fps  "
                        f"frame size {len(jpeg_bytes)//1024} KB")

    cap.release()
    logger.info("Capture thread stopped.")


# ---------------------------------------------------------------------------
# Flask MJPEG server
# ---------------------------------------------------------------------------

def create_app(host_bind: str, port: int):
    """Create and return the Flask app serving the MJPEG stream."""
    from flask import Flask, Response

    app = Flask(__name__)
    # Suppress Flask request logs — they are noisy at 30fps
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    def mjpeg_generator():
        """Yield MJPEG multipart frames from the queue."""
        while not _stop_event.is_set():
            try:
                jpeg_bytes = _frame_queue.get(timeout=1.0)
            except Empty:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

    @app.route("/stream")
    def stream():
        return Response(
            mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/")
    def index():
        return (
            "<html><body style='background:#111;color:#eee;font-family:monospace'>"
            "<h2>Webcam Streamer</h2>"
            f"<p>MJPEG stream: <a href='/stream' style='color:#4fc'>/stream</a></p>"
            f"<p>Board pulls from: <code>http://192.168.7.1:{port}/stream</code></p>"
            "<img src='/stream' style='max-width:640px;border:1px solid #444'>"
            "</body></html>"
        )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Laptop webcam → MJPEG server for FRDM-IMX95 remote inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", default=0,
                        help="Webcam device index (0, 1, ...) or path (/dev/video0)")
    parser.add_argument("--width",   type=int, default=1280, help="Capture width")
    parser.add_argument("--height",  type=int, default=720,  help="Capture height")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Output stream resolution (square), should match model imgsz")
    parser.add_argument("--fps",     type=int, default=30,   help="Capture FPS")
    parser.add_argument("--host",    default="0.0.0.0",      help="Flask bind address")
    parser.add_argument("--port",    type=int, default=5001,  help="Flask port")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality (1–95). Higher = better quality, more bandwidth")
    args = parser.parse_args()

    # Convert device to int if numeric
    try:
        device = int(args.device)
    except (ValueError, TypeError):
        device = args.device

    logger.info("=" * 55)
    logger.info("  NXP Edge AI Workshop — Laptop Webcam Streamer")
    logger.info("=" * 55)
    logger.info(f"  Device  : {device}")
    logger.info(f"  Capture : {args.width}×{args.height} @ {args.fps}fps")
    logger.info(f"  Stream  : {args.imgsz}×{args.imgsz} JPEG (quality={args.quality})")
    logger.info(f"  Serving : http://{args.host}:{args.port}/stream")
    logger.info(f"  Board pulls from: http://192.168.7.1:{args.port}/stream")
    logger.info("=" * 55)

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Shutting down...")
        _stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start capture thread
    t = threading.Thread(
        target=capture_thread,
        args=(device, args.width, args.height, args.imgsz, args.fps, args.quality),
        daemon=True,
        name="WebcamCapture"
    )
    t.start()

    # Start Flask server (blocking)
    app = create_app(args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        t.join(timeout=2.0)
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
