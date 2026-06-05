"""
viewer.py — Optional OpenCV-based live stream viewer for the laptop.

An alternative to opening the browser. Displays the MJPEG stream in an
OpenCV window with keyboard shortcuts for runtime control.

Usage:
    python3 viewer.py [--board 192.168.7.2] [--port 5000]

Keyboard shortcuts (window must be in focus):
    q / ESC  — quit
    s        — save current frame snapshot to disk
    +/-      — increase/decrease confidence threshold (±0.05)
    r        — reset threshold to 0.50
"""

import argparse
import sys
import time
import urllib.request
from datetime import datetime

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None  # threshold control won't work, but viewing still will


def stream_frames(stream_url: str):
    """
    Generator that yields decoded BGR frames from an MJPEG stream URL.

    Args:
        stream_url: Full URL to the MJPEG stream endpoint

    Yields:
        frame: BGR numpy array
    """
    stream = urllib.request.urlopen(stream_url, timeout=10)
    buf = b""

    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk

        # Find JPEG start/end markers
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:
            jpeg = buf[start:end + 2]
            buf = buf[end + 2:]

            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def set_threshold(base_url: str, value: float) -> bool:
    """POST a new confidence threshold to the board's /config endpoint."""
    if requests is None:
        print("requests not installed — cannot update threshold")
        return False
    try:
        r = requests.post(
            f"{base_url}/config",
            json={"inference": {"confidence_threshold": round(value, 2)}},
            timeout=2.0
        )
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV viewer for FRDM-IMX95 MJPEG stream"
    )
    parser.add_argument("--board", default="192.168.7.2",
                        help="Board IP address (default: 192.168.7.2)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Streaming server port (default: 5000)")
    args = parser.parse_args()

    base_url = f"http://{args.board}:{args.port}"
    stream_url = f"{base_url}/stream"

    print(f"Connecting to {stream_url}")
    print("Keyboard: [q/ESC] quit  [s] snapshot  [+/-] threshold  [r] reset\n")

    threshold = 0.50
    frame_count = 0
    t_start = time.monotonic()

    cv2.namedWindow("FRDM-IMX95 Edge AI Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FRDM-IMX95 Edge AI Vision", 1280, 720)

    try:
        for frame in stream_frames(stream_url):
            cv2.imshow("FRDM-IMX95 Edge AI Vision", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # q or ESC
                break

            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"snapshot_{ts}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Snapshot saved: {fname}")

            elif key == ord('+') or key == ord('='):
                threshold = min(0.95, threshold + 0.05)
                ok = set_threshold(base_url, threshold)
                status = "OK" if ok else "FAILED"
                print(f"Threshold → {threshold:.2f} [{status}]")

            elif key == ord('-'):
                threshold = max(0.10, threshold - 0.05)
                ok = set_threshold(base_url, threshold)
                status = "OK" if ok else "FAILED"
                print(f"Threshold → {threshold:.2f} [{status}]")

            elif key == ord('r'):
                threshold = 0.50
                ok = set_threshold(base_url, threshold)
                status = "OK" if ok else "FAILED"
                print(f"Threshold reset → {threshold:.2f} [{status}]")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nViewer stopped. {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS avg)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
