# FRDM-IMX95 Edge AI Vision — Claude Code Context

This file gives you everything you need to extend the Edge AI Vision application
running on the NXP FRDM-IMX95 development board.

---

## Hardware Platform

**Board**: NXP FRDM-IMX95  
**IP address**: `192.168.7.2` (direct Ethernet to your laptop)  
**Web viewer**: http://192.168.7.2:5000 (open in browser)

**Processor**: i.MX 95 — 6× Arm Cortex-A55 @ 1.8 GHz  
**NPU**: NXP eIQ® Neutron (2 TOPS) — hardware-accelerated ML inference  
**RAM**: 8 GB LPDDR4X  
**Camera**: MIPI-CSI at `/dev/video0` (OS08A20, up to 4K)  
**OS**: Embedded Linux (Yocto BSP), Python 3.11

---

## Project Structure

```
board/               ← runs ON the FRDM-IMX95
  main.py            ← entry point: capture → inference → annotate → stream
  inference.py       ← TFLite model loading and frame inference
  overlay.py         ← OpenCV drawing: boxes, labels, HUD
  actions.py         ← YOUR MAIN EXTENSION POINT: hooks triggered by detections
  streamer.py        ← Flask MJPEG server + REST API
  templates/
    index.html       ← browser viewer (HTML/CSS/JS)
  config.json        ← runtime configuration (model, thresholds, etc.)

host/                ← runs ON YOUR LAPTOP
  viewer.py          ← OpenCV viewer alternative to browser
  detection_logger.py ← polls /detections, saves to CSV
```

**To run on the board**:
```bash
ssh user@192.168.7.2
cd /home/user/workshop
python3 main.py
```

---

## Python Environment (on the board)

**Python**: `/usr/bin/python3` (3.11)

**Available packages** (pre-installed):
- `tflite_runtime` — TFLite inference engine
- `cv2` (opencv-python-headless) — image processing
- `flask` — web server
- `numpy` — numerical arrays
- `gpiod` — GPIO control (optional)
- `smbus2` — I2C sensor access (optional)
- `paho-mqtt` — MQTT publish/subscribe (optional)
- `requests` — HTTP client

---

## NPU Acceleration

Models compiled with `neutron-converter` are loaded via the Neutron NPU delegate.
The delegate path is always `/usr/lib/libvx_delegate.so`.
If the delegate is not found, the model falls back to CPU automatically (see `inference.py`).

```python
import tflite_runtime.interpreter as tflite

delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
interpreter = tflite.Interpreter(
    model_path='/opt/models/yolov8n_neutron.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()
```

> **Note:** The compiled `.tflite` file is still a standard TFLite file —
> the Neutron hardware kernels are embedded in it. Loading it without the
> delegate works (CPU fallback), but at reduced performance.

---

## Models

Models are **not pre-compiled binaries** — they are sourced from Ultralytics,
exported as quantized int8 TFLite, then compiled for the Neutron NPU using the
NXP `neutron-converter` tool (part of the eIQ Toolkit). The script
`scripts/prepare_models.sh` automates the full pipeline.

### Model on the board

| File | Source | Task | Input | Classes |
|------|--------|------|-------|---------|
| `yolov8n_neutron.tflite` | Ultralytics YOLOv8n → neutron-converter | Object detection | 320×320 int8 | 80 (COCO) |

Label file: `/opt/models/labels/coco_labels.txt` (80 COCO class names, YOLOv8 order)

### Model preparation pipeline (run on laptop/WSL before workshop)

```bash
# 1. Export quantized TFLite from Ultralytics
pip install ultralytics
yolo export model=yolov8n.pt format=tflite int8=True imgsz=320
# → yolov8n_int8.tflite

# 2. Compile for Neutron NPU (requires eIQ Toolkit — see scripts/prepare_models.sh)
neutron-converter --input yolov8n_int8.tflite --output yolov8n_neutron.tflite

# 3. Deploy to board
scp yolov8n_neutron.tflite user@192.168.7.2:/opt/models/

# OR: run the all-in-one script
./scripts/prepare_models.sh                  # default: yolov8n, imgsz=320
./scripts/prepare_models.sh MODEL=yolov8s    # swap to larger model
```

`neutron-converter` is available from the NXP eIQ Toolkit:
https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT

To switch to a different YOLO variant, run `prepare_models.sh MODEL=yolov8s`,
then update `config.json`:
```json
{
  "model": {
    "path": "/opt/models/yolov8s_neutron.tflite",
    "labels_path": "/opt/models/labels/coco_labels.txt"
  }
}
```

---

## Camera Capture

### Default: laptop webcam (remote source)

The default configuration (`config.json` → `camera.source = "remote"`) pulls frames
from `host/webcam_streamer.py` running on the laptop over Ethernet:

```json
"camera": {
  "source": "remote",
  "remote_url": "http://192.168.7.1:5001/stream"
}
```

Start the laptop-side streamer **before** starting `main.py` on the board:

```bash
# On the laptop (Linux / WSL / macOS):
pip install -r host/requirements.txt
python host/webcam_streamer.py          # streams 640×640 MJPEG at :5001

# On the board:
python board/main.py                    # connects to http://192.168.7.1:5001/stream
```

The webcam_streamer resizes frames to **640×640** before JPEG encoding so no
resize is needed on the board — frames arrive already at model input size.

### Alternative: board-attached camera (local source)

```json
"camera": {
  "source": "local",
  "device": "/dev/video0",
  "width": 640,
  "height": 640,
  "fps": 30
}
```

OpenCV pattern (same as used internally by `main.py`):

```python
import cv2

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame = cap.read()   # frame is BGR numpy array, shape (640, 640, 3)
```

In both cases `main.py` runs a **background capture thread** with a
`Queue(maxsize=2)` so capture and inference execute concurrently.

---

## Detection Format

`inference.run_inference()` returns a list of dicts:

```python
[
    {
        "bbox": [x1, y1, x2, y2],   # pixel coordinates in original frame
        "label_id": 0,               # integer class index (COCO, 0=person)
        "confidence": 0.87           # float 0.0–1.0
    },
    ...
]
```

The YOLOv8 int8 TFLite output tensor has shape `[1, 4+num_classes, num_anchors]`.
`inference.py` handles dequantization, transposition, confidence filtering, and
per-class NMS internally — you never need to touch the raw tensor format.

---

## Extension Points

### 1. `actions.py` — Trigger actions on detections (PRIMARY)

The `on_alert_class_detected()` function is called when a detection matches
a class in `config.json → actions.alert_classes` with sufficient confidence.

```python
# actions.py — on_alert_class_detected()
def on_alert_class_detected(detection, label, frame, config):
    # Save snapshot when person detected
    import cv2
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"/tmp/alert_{label}_{ts}.jpg", frame)

    # HTTP POST notification
    import requests
    requests.post("http://192.168.7.1:8080/alert", json={
        "label": label,
        "confidence": detection["confidence"]
    }, timeout=1)
```

Alert classes are configured in `config.json`:
```json
{"actions": {"alert_classes": ["person", "car"], "alert_min_confidence": 0.7}}
```

### 2. `overlay.py` — Customize visualization

```python
# Change box color based on detection class
def get_color_for_label(label_id):
    if label_id == 0:   # person → red
        return (0, 0, 255)
    return COLORS[label_id % len(COLORS)]
```

```python
# Add ROI zone to the frame
from overlay import draw_roi_zone
draw_roi_zone(frame, zone=(100, 100, 600, 500), label="Restricted Area")
```

### 3. `config.json` — Runtime parameters

```json
{
  "inference": {
    "confidence_threshold": 0.5   ← lower = more detections, higher = fewer
  }
}
```

Can also be updated at runtime via HTTP:
```bash
curl -X POST http://192.168.7.2:5000/config \
  -H "Content-Type: application/json" \
  -d '{"inference": {"confidence_threshold": 0.65}}'
```

---

## REST API (on the board at port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Browser viewer page |
| `/stream` | GET | MJPEG video stream |
| `/status` | GET | JSON: fps, model, current detections |
| `/detections` | GET | JSON: last N detections (add `?n=50`) |
| `/config` | GET | JSON: current config |
| `/config` | POST | Update config values (deep merge) |

**Example — fetch current detections from the laptop**:
```python
import requests
data = requests.get("http://192.168.7.2:5000/detections?n=20").json()
for d in data:
    print(d["label"], d["confidence"], d["timestamp"])
```

---

## GPIO Control (optional)

```python
import gpiod

chip = gpiod.Chip('gpiochip0')
line = chip.get_line(20)   # GPIO pin number from board header
line.request(consumer="workshop", type=gpiod.LINE_REQ_DIR_OUT)
line.set_value(1)   # HIGH
line.set_value(0)   # LOW
line.release()
```

GPIO pin numbers are on the 2×20 EXPI header — see `docs/workshop_guide.md`.

---

## I2C Sensors (optional)

```python
import smbus2

bus = smbus2.SMBus(1)   # /dev/i2c-1
value = bus.read_byte_data(device_addr, register)
```

---

## MQTT (optional)

```python
import paho.mqtt.publish as publish

publish.single(
    topic="imx95/detections",
    payload='{"label": "person", "confidence": 0.92}',
    hostname="192.168.7.1"   # laptop IP
)
```

---

## Common Patterns

**Filter detections to specific classes only:**
```python
persons = [d for d in detections if labels[d["label_id"]] == "person"]
```

**Check if a detection is inside an ROI zone:**
```python
def bbox_in_zone(bbox, zone):
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return zx1 < cx < zx2 and zy1 < cy < zy2
```

**Save a detection snapshot:**
```python
import cv2
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
cv2.imwrite(f"/tmp/snapshot_{ts}.jpg", frame)
```

**Count detections per class:**
```python
from collections import Counter
counts = Counter(labels[d["label_id"]] for d in detections)
# e.g. Counter({'person': 3, 'car': 1})
```

---

## Important Notes

- **Never call `cv2.imshow()`** on the board — there is no display. The stream
  goes through Flask → browser. Use `cv2.imwrite()` for debugging.
- **The board has no internet access** — do not try to `pip install` or call
  external APIs outside the local 192.168.7.x subnet.
- **Files are edited via VS Code Remote-SSH** — save the file, then restart
  `main.py` in the terminal to pick up changes.
- **Restart the app**: `Ctrl+C` in the terminal, then `python3 main.py`.
- **Logs appear in the terminal** where `main.py` is running.
