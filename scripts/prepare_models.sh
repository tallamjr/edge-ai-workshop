#!/usr/bin/env bash
# =============================================================================
# prepare_models.sh — Export, compile and deploy a YOLOv8 model to FRDM-IMX95
#
# Run this on your laptop (Linux or WSL on Windows) BEFORE the workshop.
#
# What it does:
#   1. Checks prerequisites (python3, ultralytics, neutron-converter)
#   2. Exports the YOLOv8n model as a quantized int8 TFLite file via Ultralytics
#   3. Compiles the TFLite file for the NXP Neutron NPU using neutron-converter
#   4. Copies the compiled model + COCO labels to the board via SCP
#
# Usage:
#   chmod +x scripts/prepare_models.sh
#   ./scripts/prepare_models.sh                       # defaults
#   ./scripts/prepare_models.sh MODEL=yolov8s         # larger model
#   ./scripts/prepare_models.sh BOARD_IP=192.168.7.3  # different board IP
#
# Configurable variables (override on command line or by editing below):
#   MODEL      — Ultralytics model name (default: yolov8n)
#   IMGSZ      — Input resolution for export, square (default: 320)
#   BOARD_IP   — Board IP address (default: 192.168.7.2)
#   BOARD_USER — SSH user on board (default: user)
#   BOARD_DIR  — Destination directory on board (default: /opt/models)
#   OUT_DIR    — Local output directory for generated files (default: ./models/deploy)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse key=value arguments from command line
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        MODEL=*)      MODEL="${arg#*=}"      ;;
        IMGSZ=*)      IMGSZ="${arg#*=}"      ;;
        BOARD_IP=*)   BOARD_IP="${arg#*=}"   ;;
        BOARD_USER=*) BOARD_USER="${arg#*=}" ;;
        BOARD_DIR=*)  BOARD_DIR="${arg#*=}"  ;;
        OUT_DIR=*)    OUT_DIR="${arg#*=}"     ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "Usage: $0 [MODEL=yolov8n] [IMGSZ=320] [BOARD_IP=192.168.7.2]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-yolov8n}"
IMGSZ="${IMGSZ:-640}"
BOARD_IP="${BOARD_IP:-192.168.7.2}"
BOARD_USER="${BOARD_USER:-user}"
BOARD_DIR="${BOARD_DIR:-/opt/models}"
OUT_DIR="${OUT_DIR:-./models/deploy}"

TFLITE_FILE="${MODEL}_full_integer_quant.tflite"
NEUTRON_FILE="${MODEL}_neutron.tflite"
COCO_LABELS_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

ok()    { echo "[OK]    $*"; }
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
fail()  { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
print_header "NXP FRDM-IMX95 Edge AI — Model Preparation Script"
info "Model     : ${MODEL}"
info "Input size: ${IMGSZ}x${IMGSZ}"
info "Board     : ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}"
info "Output dir: ${OUT_DIR}"
echo ""

# ===========================================================================
# STEP 1 — Check prerequisites
# ===========================================================================
print_header "Step 1 — Checking prerequisites"

# Python 3
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install Python 3.9+ before running this script."
fi
PY_VER=$(python3 --version 2>&1)
ok "Python: ${PY_VER}"

# pip
if ! python3 -m pip --version &>/dev/null; then
    fail "pip not found. Install pip: python3 -m ensurepip --upgrade"
fi
ok "pip available"

# ultralytics
if ! python3 -c "import ultralytics" &>/dev/null 2>&1; then
    info "ultralytics not installed. Installing now..."
    python3 -m pip install --quiet "ultralytics>=8.0" || \
        fail "Failed to install ultralytics. Try: pip install ultralytics"
fi
ULYT_VER=$(python3 -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null)
ok "ultralytics: ${ULYT_VER}"

# neutron-converter — search PATH, common SDK extraction paths, and sibling dirs
NEUTRON_BIN=""

# 1) Already on PATH
if command -v neutron-converter &>/dev/null; then
    NEUTRON_BIN="neutron-converter"
fi

# 2) Common relative extraction paths next to this script or CWD
if [[ -z "${NEUTRON_BIN}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "${SCRIPT_DIR}/../eiq-neutron-sdk/bin/neutron-converter" \
        "${SCRIPT_DIR}/../EIQ-NEUTRON-SDK-3.1.2-LIN/bin/neutron-converter" \
        "./eiq-neutron-sdk/bin/neutron-converter" \
        "./EIQ-NEUTRON-SDK-3.1.2-LIN/bin/neutron-converter" \
        "${HOME}/eiq-neutron-sdk/bin/neutron-converter"
    do
        if [[ -x "${candidate}" ]]; then
            NEUTRON_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${NEUTRON_BIN}" ]]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│              neutron-converter NOT FOUND                    │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│                                                             │"
    echo "│  The Neutron NPU compiler is part of the NXP eIQ Toolkit.  │"
    echo "│                                                             │"
    echo "│  Download it here (NXP account required):                  │"
    echo "│                                                             │"
    echo "│    https://www.nxp.com/design/design-center/software/      │"
    echo "│    eiq-ai-development-environment/                         │"
    echo "│    eiq-toolkit-for-end-to-end-model-development-and-       │"
    echo "│    deployment:EIQ-TOOLKIT                                  │"
    echo "│                                                             │"
    echo "│  After downloading, extract the archive and do ONE of:     │"
    echo "│                                                             │"
    echo "│  Option A — Add to PATH:                                   │"
    echo "│    export PATH=\"\$PATH:/path/to/eiq-neutron-sdk/bin\"        │"
    echo "│    then re-run this script.                                │"
    echo "│                                                             │"
    echo "│  Option B — Place next to this script:                     │"
    echo "│    mv eiq-neutron-sdk/ scripts/                            │"
    echo "│    then re-run this script.                                │"
    echo "│                                                             │"
    echo "│  Expected binary name: neutron-converter                   │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    fail "Cannot continue without neutron-converter."
fi

NEUTRON_VER=$("${NEUTRON_BIN}" --version 2>&1 | head -1 || echo "unknown")
ok "neutron-converter: ${NEUTRON_BIN} (${NEUTRON_VER})"

# ssh / scp — needed for board deployment
if ! command -v ssh &>/dev/null; then
    warn "ssh not found — board deployment step will be skipped."
    SKIP_DEPLOY=1
else
    SKIP_DEPLOY=0
    ok "ssh available"
fi

# ===========================================================================
# STEP 2 — Export YOLOv8 model as int8 TFLite
# ===========================================================================
print_header "Step 2 — Exporting ${MODEL} as int8 TFLite (imgsz=${IMGSZ})"

mkdir -p "${OUT_DIR}"

info "Running: yolo export model=${MODEL}.pt format=tflite int8=True imgsz=${IMGSZ}"
info "(640×640 input matches webcam_streamer.py output — no resize needed on the board)"
python3 -c "
from ultralytics import YOLO
import shutil, os

model = YOLO('${MODEL}.pt')
results = model.export(format='tflite', int8=True, imgsz=${IMGSZ})

# Ultralytics places the file in a subdirectory; find and copy it
import pathlib
matches = list(pathlib.Path('.').rglob('${MODEL}*full_integer_quant*.tflite'))
if not matches:
    raise FileNotFoundError('Could not locate full_integer_quant TFLite file')
src = matches[0]

dst = pathlib.Path('${OUT_DIR}/${TFLITE_FILE}')
shutil.copy2(src, dst)
print(f'Exported: {dst}  ({dst.stat().st_size // 1024} KB)')
" || fail "Ultralytics export failed. Check output above for details."

ok "Exported: ${OUT_DIR}/${TFLITE_FILE}"

# ===========================================================================
# STEP 3 — Compile for Neutron NPU
# ===========================================================================
print_header "Step 3 — Compiling for NXP Neutron NPU"

info "Running: neutron-converter --input ${OUT_DIR}/${TFLITE_FILE} --output ${OUT_DIR}/${NEUTRON_FILE} --target imx95"

"${NEUTRON_BIN}" \
    --input  "${OUT_DIR}/${TFLITE_FILE}" \
    --output "${OUT_DIR}/${NEUTRON_FILE}" \
    --target imx95 \
    || fail "neutron-converter failed. Check output above for details."

if [[ ! -f "${OUT_DIR}/${NEUTRON_FILE}" ]]; then
    fail "Compiled model not found at ${OUT_DIR}/${NEUTRON_FILE}"
fi

COMPILED_SIZE=$(du -k "${OUT_DIR}/${NEUTRON_FILE}" | cut -f1)
ok "Compiled: ${OUT_DIR}/${NEUTRON_FILE} (${COMPILED_SIZE} KB)"

# ===========================================================================
# STEP 4 — Generate COCO labels file
# ===========================================================================
print_header "Step 4 — Generating COCO labels"

mkdir -p "${OUT_DIR}/labels"
LABELS_FILE="${OUT_DIR}/labels/coco_labels.txt"

# Write the 80 COCO class names (YOLOv8 order) directly — no internet needed
cat > "${LABELS_FILE}" << 'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF

ok "Labels written: ${LABELS_FILE} (80 COCO classes)"

# ===========================================================================
# STEP 5 — Deploy to board
# ===========================================================================
print_header "Step 5 — Deploying to board (${BOARD_USER}@${BOARD_IP})"

if [[ "${SKIP_DEPLOY}" -eq 1 ]]; then
    warn "ssh not available — skipping board deployment."
    echo ""
    echo "  Deploy manually:"
    echo "    scp ${OUT_DIR}/${NEUTRON_FILE} ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/"
    echo "    ssh ${BOARD_USER}@${BOARD_IP} 'mkdir -p ${BOARD_DIR}/labels'"
    echo "    scp ${LABELS_FILE} ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/labels/"
    echo ""
else
    # Test SSH connectivity first
    info "Testing SSH connectivity to ${BOARD_IP}..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes \
             "${BOARD_USER}@${BOARD_IP}" "echo ok" &>/dev/null; then
        warn "Cannot reach board at ${BOARD_IP} (SSH refused or timeout)."
        echo ""
        echo "  Ensure:"
        echo "    1. Board is powered on and booted"
        echo "    2. Ethernet cable is connected"
        echo "    3. Your laptop IP is 192.168.7.1 (same subnet)"
        echo "    4. SSH key or password auth is configured"
        echo ""
        echo "  Deploy manually once the board is reachable:"
        echo "    scp ${OUT_DIR}/${NEUTRON_FILE} ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/"
        echo "    ssh ${BOARD_USER}@${BOARD_IP} 'mkdir -p ${BOARD_DIR}/labels'"
        echo "    scp ${LABELS_FILE} ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/labels/"
    else
        ok "SSH reachable"

        # Ensure destination directories exist on board
        ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p '${BOARD_DIR}/labels'"

        # Copy compiled model
        info "Copying compiled model..."
        scp "${OUT_DIR}/${NEUTRON_FILE}" "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/"
        ok "Copied: ${NEUTRON_FILE} → ${BOARD_DIR}/"

        # Copy labels
        info "Copying COCO labels..."
        scp "${LABELS_FILE}" "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/labels/"
        ok "Copied: coco_labels.txt → ${BOARD_DIR}/labels/"

        # Verify on board
        BOARD_SIZE=$(ssh "${BOARD_USER}@${BOARD_IP}" \
            "du -k '${BOARD_DIR}/${NEUTRON_FILE}' | cut -f1" 2>/dev/null || echo "?")
        ok "Verified on board: ${BOARD_DIR}/${NEUTRON_FILE} (${BOARD_SIZE} KB)"
    fi
fi

# ===========================================================================
# Summary
# ===========================================================================
print_header "Done"
echo ""
echo "  Model pipeline:"
echo "    ${MODEL}.pt"
echo "    → ${OUT_DIR}/${TFLITE_FILE}   (Ultralytics full integer quant TFLite)"
echo "    → ${OUT_DIR}/${NEUTRON_FILE}  (Neutron NPU compiled)"
echo "    → ${BOARD_DIR}/${NEUTRON_FILE} on board"
echo ""
echo "  config.json is already set to use this model:"
echo "    \"path\": \"/opt/models/${NEUTRON_FILE}\""
echo ""
echo "  Start the application on the board:"
echo "    ssh ${BOARD_USER}@${BOARD_IP}"
echo "    cd /home/${BOARD_USER}/edge_ai_workshop"
echo "    python board/main.py"
echo ""
echo "  Open browser: http://${BOARD_IP}:5000"
echo ""
