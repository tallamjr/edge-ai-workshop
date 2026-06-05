# Edge AI Vision Workshop — Facilitator & Participant Guide

**Half-Day Workshop | NXP FRDM-IMX95 Freedom Board**
**"Vibe-Code Your Way to Edge AI"**

---

## Table of Contents

1. [Workshop Overview](#workshop-overview)
2. [Learning Objectives](#learning-objectives)
3. [Prerequisites](#prerequisites)
4. [Schedule at a Glance](#schedule-at-a-glance)
5. [Part 0 — Environment Setup (30 min)](#part-0--environment-setup-30-min)
6. [Part 1 — Introduction & Live Demo (20 min)](#part-1--introduction--live-demo-20-min)
7. [Part 2 — Code Tour (30 min)](#part-2--code-tour-30-min)
8. [Part 3 — Lab 1: Guided Challenges (60 min)](#part-3--lab-1-guided-challenges-60-min)
9. [Part 4 — Lab 2: Open Exploration (40 min)](#part-4--lab-2-open-exploration-40-min)
10. [Part 5 — Show & Tell (20 min)](#part-5--show--tell-20-min)
11. [Facilitator Notes](#facilitator-notes)
12. [Troubleshooting Reference](#troubleshooting-reference)

---

## Workshop Overview

This workshop puts AI-assisted coding at the center of embedded engineering. Participants use **Claude Code** as a co-pilot to extend a running Edge AI object-detection pipeline on real NXP hardware — no deep ML expertise required.

The board runs a Python application that:
- Captures frames from a USB webcam via OpenCV
- Runs TFLite inference on the eIQ® Neutron NPU (2 TOPS) using a 3-stage pipelined architecture
- Overlays bounding boxes, inference timing, and HUD data onto frames
- Streams the annotated video over WiFi to a browser via Flask MJPEG
- Exposes a REST API for live configuration

Participants connect their laptops to the board via WiFi, open the live stream in a browser, and then **vibe-code enhancements** by describing what they want to Claude Code in natural language.

---

## Learning Objectives

By the end of this workshop, participants will be able to:

1. Connect to and work on an embedded Linux board remotely via SSH and VS Code
2. Understand the structure of a real-time Edge AI inference pipeline
3. Use AI-assisted coding (Claude Code) to extend an embedded application without starting from scratch
4. Swap and compare TFLite models running on a hardware NPU
5. Customize visual overlays, detection logic, and REST API behavior
6. Stream and visualize live AI inference results from an embedded device

---

## Prerequisites

### Participant prerequisites
- Laptop with Wi-Fi (for internet + Claude Code)
- Basic Python familiarity (can read and modify Python code)
- Claude Code installed and authenticated (`npm install -g @anthropic/claude-code` or via your organization's setup)

### Board prerequisites (pre-configured by organizers)
- FRDM-IMX95 booted into Linux
- USB webcam connected (Logitech C922 or equivalent)
- `tflite_runtime` pre-installed in the Yocto BSP (exposed via `--system-site-packages`)

### Board WiFi connectivity
Connect to the board through serial interface (via Putty or any other means) and execute the following commands to connect to the local network:

```wpa_passphrase "${SSID}" "${SSID_PASSWORD}" > /tmp/wpa.conf
wpa_supplicant -B -i mlan0 -c /tmp/wpa.conf
udhcpc -i mlan0
```

Check the IP address of the board via:
```ip addr
```

### Model preparation (facilitator-only, done before the workshop)

Models are exported from Ultralytics with full int8 quantization and compiled for the Neutron NPU using the NXP eIQ Toolkit. Run on a **Linux laptop or WSL** terminal:

```bash
# 1. Install the NXP eIQ Toolkit (contains neutron-converter)
#    Download from: https://www.nxp.com/design/design-center/software/
#    eiq-ai-development-environment/
#    eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT
#    Then:
make install-eiq ARCHIVE=/path/to/EIQ-NEUTRON-SDK-3.1.2-LIN.zip

# 2. Install laptop Python dependencies
make install-deps

# 3. Run the full model pipeline (board must be reachable)
make model BOARD_IP=${BOARD_IP}
```

The pipeline:
1. Exports `yolov8n.pt` → `yolov8n_full_integer_quant.tflite` (fully int8 quantized — input and output tensors are int8, not float32)
2. Compiles → `yolov8n_neutron.tflite` using `neutron-converter --target imx95`
3. SCPs the compiled model + COCO labels to `/opt/models/` on the board

> **Why full integer quantization?** The Neutron NPU requires int8 I/O end-to-end. The standard `int8=True` Ultralytics export keeps float32 I/O (weights quantized only). `full_integer_quant` is the correct variant for NPU acceleration.


### Board Python environment setup

If the venv is not yet created on the board:

```bash
make board-deps BOARD_IP=${BOARD_IP}
```

This SCPs `board/requirements.txt` to the board and runs:
```bash
python3 -m venv --system-site-packages /home/root/edge_ai_workshop/.venv
.venv/bin/pip install -r requirements.txt
```

### Board inference environment setup
To prepare the inference environment run:

```
make board-deploy-app BOARD_IP=${BOARD_IP}
```

### Inference
To start the application run:
```
make board-start BOARD_IP=172.20.10.6
```

This command will return an IP address, copy paste it to open it into a browser. You should now see the annotated video from your camera.

---

## Switching YOLOv8 Model Variants

The pipeline supports four YOLOv8 variants out of the box. Larger variants detect more accurately but run slower on the NPU.

| Variant | Key | Parameters | Typical NPU latency |
|---------|-----|-----------|---------------------|
| YOLOv8n | `n` | 3.2 M  | fastest |
| YOLOv8s | `s` | 11.2 M | fast    |
| YOLOv8m | `m` | 25.9 M | medium  |
| YOLOv8l | `l` | 43.7 M | slowest |

### How to switch

**Step 1 — Edit `board/config.json`** on your laptop and change the `variant` field:

```json
{
  "model": {
    "variant": "s",
    "variants": {
      "n": "yolov8n",
      "s": "yolov8s",
      "m": "yolov8m",
      "l": "yolov8l"
    },
    "models_dir": "/opt/models"
  }
}
```

Set `"variant"` to `"n"`, `"s"`, `"m"`, or `"l"`.

**Step 2 — Compile and deploy** the new model (laptop/WSL terminal).  
`make model` reads the variant automatically from `config.json`:

```bash
make model BOARD_IP=${BOARD_IP}
```

This will:
1. Download the selected model weights from Ultralytics
2. Export to fully int8 quantized TFLite
3. Compile for the Neutron NPU with `neutron-converter`
4. Deploy the compiled model and the updated `config.json` to the board

**Step 3 — Restart the app** on the board to load the new model:

```bash
make board-start BOARD_IP=${BOARD_IP}
```

The sidebar in the browser will show the active variant in the model badge (top-right corner) and the per-stage pipeline latencies will update to reflect the new model's timing.

---

## Improving Performance with the Split Pipeline

By default `make model` deploys the model as a single TFLite file. The inference
loop runs all layers sequentially: CPU pre-processing → NPU → CPU post-processing.

`make model-split-pipeline` splits the compiled model into three separate sub-models
and runs each in its own thread, overlapping NPU inference with CPU work:

```
 Thread A (CPU pre)  ──►  Thread B (NPU)  ──►  Thread C (CPU post)
       frame N+1               frame N               frame N-1
```

While the NPU is crunching frame N, the CPU is already pre-processing frame N+1
and post-processing frame N-1 — keeping all three stages busy in parallel.

### How to enable the split pipeline

Run the following **instead of** (or after) `make model`:

```bash
make model-split-pipeline BOARD_IP=${BOARD_IP}
```

This will:
1. Analyze the compiled model to find the NPU/CPU boundary
2. Split it into `pre.tflite` (CPU) / `npu.tflite` (NPU) / `post.tflite` (CPU)
3. Generate a `pipeline.json` manifest describing the three stages
4. Deploy all sub-models and the updated `config.json` to the board

**Restart the app** to activate the new pipeline:

```bash
make board-start BOARD_IP=${BOARD_IP}
```

The **Pipeline Stages** card in the browser sidebar will show three rows — one per
stage — with their individual average latencies, confirming pipelined execution is active.

> **Note:** the split pipeline requires `tflite-extractor` from the NXP eIQ Toolkit
> to be available on `PATH` alongside `neutron-converter`.