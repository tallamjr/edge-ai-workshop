"""
detection_logger.py — Poll the board's /detections endpoint and log to CSV.

Runs on the participant's laptop over the direct Ethernet connection.
Useful as a Lab 1 extension challenge or as a standalone tool.

Usage:
    python3 detection_logger.py [--board 192.168.7.2] [--port 5000] [--output detections.csv]
    python3 detection_logger.py --watch          # Print live detections to terminal
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core polling logic
# ---------------------------------------------------------------------------

def fetch_detections(base_url: str, n: int = 20) -> list[dict]:
    """
    Fetch the last N detections from the board's /detections endpoint.

    Args:
        base_url: Board base URL, e.g. 'http://192.168.7.2:5000'
        n: Number of recent detections to retrieve

    Returns:
        List of detection dicts [{label, confidence, bbox, timestamp}]
    """
    url = f"{base_url}/detections?n={n}"
    resp = requests.get(url, timeout=3.0)
    resp.raise_for_status()
    return resp.json()


def fetch_status(base_url: str) -> dict:
    """
    Fetch the current runtime status from the board's /status endpoint.

    Args:
        base_url: Board base URL

    Returns:
        Status dict {fps, model, num_detections, detections, ...}
    """
    resp = requests.get(f"{base_url}/status", timeout=3.0)
    resp.raise_for_status()
    return resp.json()


def log_to_csv(detections: list[dict], output_path: str,
               seen_timestamps: set) -> int:
    """
    Append new (unseen) detections to a CSV file.

    Args:
        detections: List of detection dicts from /detections
        output_path: Path to output CSV file
        seen_timestamps: Set of already-logged timestamps (updated in-place)

    Returns:
        Number of new rows written
    """
    path = Path(output_path)
    file_exists = path.exists()
    new_rows = 0

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence",
                             "x1", "y1", "x2", "y2"])

        for det in detections:
            key = f"{det.get('timestamp','')}_{det.get('label','')}_{det.get('confidence','')}"
            if key in seen_timestamps:
                continue
            seen_timestamps.add(key)

            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            writer.writerow([
                det.get("timestamp", ""),
                det.get("label", ""),
                f"{det.get('confidence', 0):.4f}",
                x1, y1, x2, y2
            ])
            new_rows += 1

    return new_rows


# ---------------------------------------------------------------------------
# Watch mode — live terminal output
# ---------------------------------------------------------------------------

def watch_mode(base_url: str, interval: float) -> None:
    """Print live detection summaries to the terminal."""
    print(f"Watching {base_url}/status  (Ctrl+C to stop)\n")
    last_num = 0

    while True:
        try:
            status = fetch_status(base_url)
            fps = status.get("fps", 0)
            dets = status.get("detections", [])
            num = status.get("num_detections", 0)
            ts = datetime.now().strftime("%H:%M:%S")

            det_str = ", ".join(
                f"{d['label']}({d['confidence']:.0%})" for d in dets
            ) if dets else "—"

            print(f"[{ts}] FPS: {fps:5.1f} | Detections: {num:2d} | {det_str}")

        except requests.RequestException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}")

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Logger mode — poll and save to CSV
# ---------------------------------------------------------------------------

def logger_mode(base_url: str, output: str, interval: float) -> None:
    """Continuously poll /detections and append new rows to CSV."""
    print(f"Logging detections from {base_url}")
    print(f"Output file: {output}")
    print("Press Ctrl+C to stop.\n")

    seen: set = set()
    total_written = 0

    while True:
        try:
            detections = fetch_detections(base_url, n=50)
            n = log_to_csv(detections, output, seen)
            if n > 0:
                total_written += n
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"+{n} new detections logged (total: {total_written})")
        except requests.RequestException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Connection error: {e} — retrying...")

        time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log or watch detections from the FRDM-IMX95 board"
    )
    parser.add_argument("--board", default="192.168.7.2",
                        help="Board IP address (default: 192.168.7.2)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Streaming server port (default: 5000)")
    parser.add_argument("--output", default="detections.csv",
                        help="Output CSV file path (default: detections.csv)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Poll interval in seconds (default: 1.0)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: print live detections to terminal (no CSV)")
    args = parser.parse_args()

    base_url = f"http://{args.board}:{args.port}"

    # Verify connectivity
    print(f"Connecting to {base_url}...")
    try:
        status = fetch_status(base_url)
        print(f"Connected. Model: {status.get('model', '?')} | "
              f"FPS: {status.get('fps', 0):.1f}\n")
    except requests.RequestException as e:
        print(f"Could not connect to board at {base_url}: {e}")
        print("Is the board powered on and running main.py?")
        sys.exit(1)

    try:
        if args.watch:
            watch_mode(base_url, args.interval)
        else:
            logger_mode(base_url, args.output, args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
