# =============================================================================
# Makefile — NXP Edge AI Workshop Setup (WSL / Linux)
#
# Usage:
#   make                  → show help
#   make setup            → full end-to-end setup (laptop deps + model pipeline)
#   make install-deps     → install Python dependencies on the laptop
#   make check            → check all prerequisites without doing anything
#   make model            → export + compile + deploy the YOLOv8n model
#   make model MODEL=yolov8s   → same but with a larger model
#   make deploy           → (re)deploy already-compiled model to the board
#   make board-deps       → create .venv on board and pip install board/requirements.txt
#   make board-start      → SSH into the board and start main.py
#   make board-shell      → open an SSH shell into the board
#   make clean            → remove generated model files
#   make logs             → tail detection log from the board
#
# Configurable variables (override on command line):
#   MODEL      Ultralytics model name          (default: yolov8n)
#   IMGSZ      Model input size, square px     (default: 640)
#   BOARD_IP   Board IP address                (default: 192.168.7.2)
#   BOARD_USER SSH user on the board           (default: root)
#   BOARD_DIR  Model destination on board      (default: /opt/models)
#   BOARD_APP  App directory on board          (default: /home/root/edge_ai_workshop)
#   BOARD_VENV Python venv on board            (default: $(BOARD_APP)/.venv)
#   OUT_DIR    Local compiled model output dir (default: ./models_out)
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration — override any of these on the command line
# ---------------------------------------------------------------------------
MODEL      ?= yolov8n
# If MODEL was not set on the command line, read variant from board/config.json.
# Explicit override still works: make model MODEL=yolov8s
ifneq ($(origin MODEL),command line)
  MODEL := $(or $(shell python3 -c "import json; c=json.load(open('board/config.json')); v=c.get('model',{}).get('variant','n'); print(c.get('model',{}).get('variants',{}).get(v,''))" 2>/dev/null),yolov8n)
endif
IMGSZ      ?= 640
BOARD_IP   ?= 192.168.7.2
BOARD_USER ?= root
BOARD_DIR  ?= /opt/models
BOARD_APP  ?= /home/root/edge_ai_workshop
BOARD_VENV ?= $(BOARD_APP)/.venv
OUT_DIR    ?= ./models_out

BOARD        := $(BOARD_USER)@$(BOARD_IP)
TFLITE_FILE  := $(OUT_DIR)/$(MODEL)_full_integer_quant.tflite
NEUTRON_FILE := $(OUT_DIR)/$(MODEL)_neutron.tflite
SSH_OPTS     := -o ConnectTimeout=5 -o StrictHostKeyChecking=no

# ---------------------------------------------------------------------------
# Local Python virtual environment
# All Python operations use .venv/ in the project root.
# Create with: make venv  (or it is created automatically by install-deps)
# Activate manually: source .venv/bin/activate
# ---------------------------------------------------------------------------
VENV     ?= .venv
VENV_PY  := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# ---------------------------------------------------------------------------
# ANSI colour helpers (work in WSL bash)
# ---------------------------------------------------------------------------
BOLD  := \033[1m
GREEN := \033[0;32m
CYAN  := \033[0;36m
RESET := \033[0m

# ---------------------------------------------------------------------------
# Default target — print help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "$(BOLD)NXP Edge AI Workshop — WSL Setup Makefile$(RESET)"
	@echo ""
	@echo "$(CYAN)Quick start:$(RESET)"
	@echo "  make setup              Full setup: deps + model pipeline + deploy"
	@echo ""
	@echo "$(CYAN)Individual targets:$(RESET)"
	@echo "  make check              Check all prerequisites (no changes)"
	@echo "  make install-deps       Install Python dependencies (laptop/WSL)"
	@echo "  make model              Export YOLOv8n, compile for NPU, deploy"
	@echo "  make deploy             Re-deploy already-compiled model to board"
	@echo "  make model-analyze           Show CPU vs NPU op distribution in the compiled model"
	@echo "  make model-split             Split model into pre/npu/post sub-models for pipelining"
	@echo "  make model-split-deploy      Deploy split sub-models to the board"
	@echo "  make model-split-config      Patch and deploy config.json pointing at split pipeline"
	@echo "  make model-split-pipeline    analyze + split + deploy + config update in one step"
	@echo "  make board-deps         Create .venv on board and install from board/requirements.txt"
	@echo "  make board-deploy-app   Sync board/ application files to the board"
	@echo "  make board-start        Start the inference app on the board (SSH)"
	@echo "  make board-shell        Open SSH shell into the board"
	@echo "  make logs               Tail detection log from the board"
	@echo "  make venv               Create local Python venv in $(VENV)/"
	@echo "  make clean-model        Remove compiled model files in $(OUT_DIR) only"
	@echo "  make clean              clean-model + remove Python cache"
	@echo "  make install-eiq ARCHIVE=./file.zip  Install neutron-converter from local archive"
	@echo "  make net-setup          Configure Ethernet interface for direct board connection"
	@echo "  make net-check          Test connectivity to the board"
	@echo ""
	@echo "$(CYAN)Board access:$(RESET)"
	@echo "  SSH:    ssh root@$(BOARD_IP)"
	@echo "  Serial: use PuTTY on Windows (115200 8N1) on the J1 debug USB port"
	@echo ""
	@echo "$(CYAN)Configuration (override with VAR=value):$(RESET)"
	@echo "  MODEL      = $(MODEL)"
	@echo "  IMGSZ      = $(IMGSZ)"
	@echo "  BOARD_IP   = $(BOARD_IP)"
	@echo "  BOARD_USER = $(BOARD_USER)"
	@echo "  BOARD_DIR  = $(BOARD_DIR)"
	@echo "  BOARD_APP  = $(BOARD_APP)"
	@echo "  BOARD_VENV = $(BOARD_VENV)"
	@echo "  OUT_DIR    = $(OUT_DIR)"
	@echo ""
	@echo "$(CYAN)Examples:$(RESET)"
	@echo "  make setup BOARD_IP=192.168.7.3"
	@echo "  make model MODEL=yolov8s"
	@echo ""

# ---------------------------------------------------------------------------
# Create / update the local virtual environment
# ---------------------------------------------------------------------------
.PHONY: venv
venv: $(VENV_PY)

$(VENV_PY):
	@echo "$(BOLD)Creating Python virtual environment in $(VENV)/ ...$(RESET)"
	python3 -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	@echo "  [OK] venv created: $(VENV)/"
	@echo "  Activate with: source $(VENV)/bin/activate"
	@echo ""

# ---------------------------------------------------------------------------
# Full setup (recommended first-time target)
# ---------------------------------------------------------------------------
.PHONY: setup
setup: check venv install-deps model board-deps
	@echo ""
	@echo "$(GREEN)$(BOLD)Setup complete!$(RESET)"
	@echo ""
	@echo "  Next steps:"
	@echo "  1. Start the app on the board: make board-start"
	@echo "  2. Open browser:               http://$(BOARD_IP):5000"
	@echo ""

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
.PHONY: check
check:
	@echo "$(BOLD)Checking prerequisites...$(RESET)"
	@echo ""

	@# Python 3
	@if command -v python3 >/dev/null 2>&1; then \
		echo "  [OK] python3: $$(python3 --version)"; \
	else \
		echo "  [FAIL] python3 not found. Install: sudo apt install python3 python3-venv"; \
		exit 1; \
	fi

	@# python3-venv module
	@if python3 -m venv --help >/dev/null 2>&1; then \
		echo "  [OK] python3-venv available"; \
	else \
		echo "  [WARN] python3-venv not found. Install: sudo apt install python3-venv"; \
	fi

	@# venv created
	@if [ -x "$(VENV_PY)" ]; then \
		echo "  [OK] venv: $(VENV)/ ($$($(VENV_PY) --version))"; \
	else \
		echo "  [WARN] venv not created yet (run: make venv)"; \
	fi

	@# ultralytics
	@if [ -x "$(VENV_PY)" ] && $(VENV_PY) -c "import ultralytics" >/dev/null 2>&1; then \
		echo "  [OK] ultralytics: $$($(VENV_PY) -c 'import ultralytics; print(ultralytics.__version__)')"; \
	else \
		echo "  [WARN] ultralytics not installed (run: make install-deps)"; \
	fi

	@# neutron-converter
	@if command -v neutron-converter >/dev/null 2>&1; then \
		echo "  [OK] neutron-converter: $$(command -v neutron-converter)"; \
	elif find . -name "neutron-converter" -executable 2>/dev/null | grep -q .; then \
		echo "  [OK] neutron-converter found in local path"; \
	else \
		echo "  [WARN] neutron-converter not found on PATH."; \
		echo "         Download the eIQ Toolkit from:"; \
		echo "         https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT"; \
		echo "         Then: export PATH=\$$PATH:/path/to/eiq-neutron-sdk/bin"; \
	fi

	@# ssh
	@if command -v ssh >/dev/null 2>&1; then \
		echo "  [OK] ssh available"; \
	else \
		echo "  [WARN] ssh not found. Install: sudo apt install openssh-client"; \
	fi

	@# Board reachability
	@echo ""
	@echo "  Checking board at $(BOARD_IP)..."
	@if ping -c 1 -W 2 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [OK] Board reachable at $(BOARD_IP)"; \
	else \
		echo "  [WARN] Board NOT reachable at $(BOARD_IP)."; \
		echo "         Ensure Ethernet is connected and laptop IP is 192.168.7.1/24."; \
		echo "         Linux: sudo ip addr add 192.168.7.1/24 dev eth0 && sudo ip link set eth0 up"; \
	fi

	@echo ""
	@echo "  Prerequisite check complete."
	@echo ""

# ---------------------------------------------------------------------------
# Install Python dependencies on the laptop/WSL
# ---------------------------------------------------------------------------
.PHONY: install-deps
install-deps: venv
	@echo "$(BOLD)Installing laptop/WSL Python dependencies into $(VENV)/...$(RESET)"
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r host/requirements.txt
	@echo ""
	@echo "  [OK] Host dependencies installed in $(VENV)/"
	@echo "  Activate with: source $(VENV)/bin/activate"
	@echo ""

# ---------------------------------------------------------------------------
# Full model pipeline: export → compile → deploy
# ---------------------------------------------------------------------------
.PHONY: model
model: $(NEUTRON_FILE) deploy
	@echo ""
	@echo "$(GREEN)Model pipeline complete: $(MODEL)_neutron.tflite deployed to board.$(RESET)"
	@echo ""

$(TFLITE_FILE):
	@echo "$(BOLD)Step 1/3 — Exporting $(MODEL) as int8 TFLite (imgsz=$(IMGSZ))...$(RESET)"
	@mkdir -p $(OUT_DIR)
	$(VENV_PY) -c "\
from ultralytics import YOLO; import shutil, pathlib; \
m = YOLO('$(MODEL).pt'); \
m.export(format='tflite', int8=True, imgsz=$(IMGSZ)); \
matches = list(pathlib.Path('.').rglob('$(MODEL)*full_integer_quant*.tflite')); \
assert matches, 'full_integer_quant TFLite not found'; \
shutil.copy2(matches[0], '$(TFLITE_FILE)'); \
print(f'Exported: $(TFLITE_FILE)')"

$(NEUTRON_FILE): $(TFLITE_FILE)
	@echo "$(BOLD)Step 2/3 — Compiling $(MODEL)_int8.tflite for Neutron NPU...$(RESET)"
	@if command -v neutron-converter >/dev/null 2>&1; then \
		neutron-converter --input $(TFLITE_FILE) --output $(NEUTRON_FILE) --target $(NEUTRON_TARGET); \
	else \
		FOUND=$$(find . -name "neutron-converter" -executable 2>/dev/null | head -1); \
		if [ -n "$$FOUND" ]; then \
			$$FOUND --input $(TFLITE_FILE) --output $(NEUTRON_FILE) --target $(NEUTRON_TARGET); \
		else \
			echo ""; \
			echo "  [ERROR] neutron-converter not found."; \
			echo "  Download the eIQ Toolkit from:"; \
			echo "  https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/"; \
			echo "  eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT"; \
			echo "  Then: export PATH=\$$PATH:/path/to/eiq-neutron-sdk/bin && make model"; \
			echo ""; \
			exit 1; \
		fi \
	fi
	@echo "  [OK] Compiled: $(NEUTRON_FILE)"

# ---------------------------------------------------------------------------
# Deploy compiled model + labels to the board
# ---------------------------------------------------------------------------
.PHONY: deploy
deploy:
	@echo "$(BOLD)Step 3/3 — Deploying to board ($(BOARD))...$(RESET)"
	@if ! ping -c 1 -W 3 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [ERROR] Cannot reach board at $(BOARD_IP). Check Ethernet connection."; \
		exit 1; \
	fi
	@# Create per-model directory and labels directory
	ssh $(SSH_OPTS) $(BOARD) "mkdir -p $(BOARD_DIR)/$(MODEL) $(BOARD_DIR)/labels"
	@# Copy model into per-model subdirectory
	scp $(NEUTRON_FILE) $(BOARD):$(BOARD_DIR)/$(MODEL)/
	@echo "  [OK] Copied $(MODEL)_neutron.tflite → $(BOARD_DIR)/$(MODEL)/"
	@# Generate single-stage pipeline.json and deploy it
	@$(VENV_PY) -c "\
import json, pathlib; \
p = {'pipeline': [{'label': 'model', 'file': '$(MODEL)_neutron.tflite', 'use_npu': True}]}; \
pathlib.Path('/tmp/$(MODEL)_pipeline.json').write_text(json.dumps(p, indent=2))"
	scp /tmp/$(MODEL)_pipeline.json $(BOARD):$(BOARD_DIR)/$(MODEL)/pipeline.json
	@echo "  [OK] Generated pipeline.json → $(BOARD_DIR)/$(MODEL)/pipeline.json"
	@# Generate and copy COCO labels if not present
	@mkdir -p $(OUT_DIR)/labels
	@$(VENV_PY) -c "\
labels=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',\
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog',\
'horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',\
'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',\
'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',\
'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',\
'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',\
'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',\
'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',\
'teddy bear','hair drier','toothbrush']; \
open('$(OUT_DIR)/labels/coco_labels.txt','w').write('\n'.join(labels))"
	scp $(OUT_DIR)/labels/coco_labels.txt $(BOARD):$(BOARD_DIR)/labels/
	@echo "  [OK] Copied coco_labels.txt → $(BOARD_DIR)/labels/"
	@SIZE=$$(ssh $(SSH_OPTS) $(BOARD) "du -k $(BOARD_DIR)/$(MODEL)/$(MODEL)_neutron.tflite | cut -f1" 2>/dev/null); \
	echo "  [OK] Verified: $(BOARD_DIR)/$(MODEL)/$(MODEL)_neutron.tflite ($${SIZE} KB)"
	@# Deploy config.json so the board app picks up the new variant
	scp board/config.json $(BOARD):$(BOARD_APP)/board/config.json
	@echo "  [OK] Deployed config.json → $(BOARD_APP)/board/config.json"

# ---------------------------------------------------------------------------
# Full split pipeline: analyze → split → deploy models → update config
# ---------------------------------------------------------------------------
.PHONY: model-split-pipeline
model-split-pipeline: model-analyze model-split model-split-deploy
	@echo "$(GREEN)$(BOLD)Split pipeline complete.$(RESET)"
	@echo "  Run: make board-start BOARD_IP=$(BOARD_IP)"
	@echo ""

# model-split-config removed: the app now derives the pipeline path from
# model.variant + models_dir at startup; no manual config patching needed.

# ---------------------------------------------------------------------------
# Analyze NPU vs CPU op distribution in a compiled TFLite model
# ---------------------------------------------------------------------------
.PHONY: model-analyze
model-analyze:
	@echo "$(BOLD)Analyzing model op distribution...$(RESET)"
	$(VENV_PY) scripts/split_model.py --model $(OUT_DIR)/$(MODEL)_neutron.tflite

# ---------------------------------------------------------------------------
# Split a compiled TFLite into CPU-pre / NPU / CPU-post sub-models for
# pipelined execution on the board.
#
# Outputs: $(OUT_DIR)/split/{pre,npu,post}.tflite + pipeline.json
# To activate on the board, point config.json → model.path to the JSON:
#   "path": "/opt/models/split/pipeline.json"
# Then deploy with: make board-deploy-app && make model-split-deploy
# ---------------------------------------------------------------------------
SPLIT_DIR ?= $(OUT_DIR)/$(MODEL)

.PHONY: model-split
model-split:
	@echo "$(BOLD)Splitting model for pipelined NPU/CPU execution...$(RESET)"
	$(VENV_PY) scripts/split_model.py \
		--model $(OUT_DIR)/$(MODEL)_neutron.tflite \
		--split \
		--output-dir $(SPLIT_DIR)
	@echo ""
	@echo "  Deploy split models to board:"
	@echo "    make model-split-deploy BOARD_IP=$(BOARD_IP)"

.PHONY: model-split-deploy
model-split-deploy:
	@echo "$(BOLD)Deploying split models to board...$(RESET)"
	@if ! ping -c 1 -W 3 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [ERROR] Cannot reach board at $(BOARD_IP)."; exit 1; \
	fi
	ssh $(SSH_OPTS) $(BOARD) "mkdir -p $(BOARD_DIR)/$(MODEL)"
	scp $(SPLIT_DIR)/*.tflite $(SPLIT_DIR)/pipeline.json $(BOARD):$(BOARD_DIR)/$(MODEL)/
	@echo "  [OK] Split models deployed to $(BOARD_DIR)/$(MODEL)/"
	scp board/config.json $(BOARD):$(BOARD_APP)/board/config.json
	@echo "  [OK] Deployed config.json → $(BOARD_APP)/board/config.json"

# ---------------------------------------------------------------------------
# Install Python dependencies on the board
# Creates a venv at BOARD_VENV, copies board/requirements.txt, and installs.
# Uses --system-site-packages so the BSP-installed tflite_runtime is visible.
# ---------------------------------------------------------------------------
.PHONY: board-deps
board-deps:
	@echo "$(BOLD)Setting up Python venv on board from board/requirements.txt...$(RESET)"
	@if ! ping -c 1 -W 3 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [ERROR] Cannot reach board at $(BOARD_IP)."; exit 1; \
	fi
	scp board/resolv.conf $(BOARD):/etc/
	ssh $(SSH_OPTS) $(BOARD) "mkdir -p $(BOARD_APP)"
	scp board/requirements.txt $(BOARD):$(BOARD_APP)/requirements.txt
	@echo "  Creating venv at $(BOARD_VENV)..."
	ssh $(SSH_OPTS) $(BOARD) "python3 -m venv --system-site-packages $(BOARD_VENV) && $(BOARD_VENV)/bin/pip install --upgrade pip --quiet && $(BOARD_VENV)/bin/pip install -r $(BOARD_APP)/requirements.txt --quiet"
	ssh $(SSH_OPTS) $(BOARD) "$(BOARD_VENV)/bin/python -c 'import flask, numpy, cv2; print(\"[OK] flask={} numpy={} cv2={}\".format(flask.__version__, numpy.__version__, cv2.__version__))'"
	@echo "  [OK] Board venv ready: $(BOARD_VENV)"
	@echo "  Run the app with: make board-start"
	@echo ""

# ---------------------------------------------------------------------------
# Deploy project files to the board (first-time or after changes)
# ---------------------------------------------------------------------------
.PHONY: board-deploy-app
board-deploy-app:
	@echo "$(BOLD)Syncing project files to board...$(RESET)"
	@if ! ping -c 1 -W 3 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [ERROR] Cannot reach board at $(BOARD_IP)."; exit 1; \
	fi
	ssh $(SSH_OPTS) $(BOARD) "mkdir -p $(BOARD_APP)/board/templates"
	tar czf - --exclude='*.pyc' --exclude='__pycache__' -C board . | \
		ssh $(SSH_OPTS) $(BOARD) "tar xzf - -C $(BOARD_APP)/board"
	@echo "  [OK] Project files copied to $(BOARD_APP)"

# ---------------------------------------------------------------------------
# Start the inference app on the board (via SSH)
# ---------------------------------------------------------------------------
.PHONY: board-start
board-start:
	@echo "$(BOLD)Starting inference app on board...$(RESET)"
	@if ! ping -c 1 -W 3 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [ERROR] Cannot reach board at $(BOARD_IP)."; exit 1; \
	fi
	@echo "  Connecting to $(BOARD)..."
	@echo "  (Press Ctrl+C to stop the app and exit SSH)"
	@echo ""
	ssh $(SSH_OPTS) -t $(BOARD) "cd $(BOARD_APP)/board && $(BOARD_VENV)/bin/python main.py"

# ---------------------------------------------------------------------------
# Open an interactive SSH shell into the board
# ---------------------------------------------------------------------------
.PHONY: board-shell
board-shell:
	@echo "$(BOLD)Opening SSH shell to $(BOARD)...$(RESET)"
	ssh $(SSH_OPTS) -t $(BOARD)


# ---------------------------------------------------------------------------
# Tail the detection log from the board
# ---------------------------------------------------------------------------
.PHONY: logs
logs:
	@echo "$(BOLD)Tailing detection log from board (Ctrl+C to stop)...$(RESET)"
	ssh $(SSH_OPTS) $(BOARD) "tail -f /tmp/detections.csv"

# ---------------------------------------------------------------------------
# Install the NXP eIQ Neutron SDK from a locally downloaded archive
# ---------------------------------------------------------------------------
# Download the archive manually from NXP (NXP account required):
#   https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/
#   eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT
#
# Then install with:
#   make install-eiq ARCHIVE=/path/to/EIQ-NEUTRON-SDK-3.1.2-LIN.zip
# ---------------------------------------------------------------------------
EIQ_COLCODE      := EIQ-NEUTRON-SDK-3.1.2-LIN
EIQ_INSTALL_DIR  ?= $(HOME)/eiq-neutron-sdk
NEUTRON_TARGET   ?= imx95

.PHONY: install-eiq
install-eiq:
	@# Install from a locally downloaded archive (no login needed)
	@if [ -z "$(ARCHIVE)" ]; then \
		echo "  [ERROR] Specify the archive path: make install-eiq ARCHIVE=/path/to/$(EIQ_COLCODE).zip"; \
		exit 1; \
	fi
	@if [ ! -f "$(ARCHIVE)" ]; then \
		echo "  [ERROR] File not found: $(ARCHIVE)"; \
		exit 1; \
	fi
	@echo "$(BOLD)Installing eIQ Neutron SDK from $(ARCHIVE)...$(RESET)"
	mkdir -p $(EIQ_INSTALL_DIR)
	@if echo "$(ARCHIVE)" | grep -q "\.zip$$"; then \
		unzip -q "$(ARCHIVE)" -d $(EIQ_INSTALL_DIR); \
	else \
		tar xf "$(ARCHIVE)" -C $(EIQ_INSTALL_DIR); \
	fi
	@CONVERTER=$$(find $(EIQ_INSTALL_DIR) -name "neutron-converter" -type f | head -1); \
	if [ -n "$$CONVERTER" ]; then \
		chmod +x $$CONVERTER; \
		CONVERTER_DIR=$$(dirname $$CONVERTER); \
		echo "  [OK] neutron-converter installed at $$CONVERTER"; \
		if [ -d "$(VENV)/bin" ]; then \
			ln -sf $$CONVERTER $(VENV)/bin/neutron-converter 2>/dev/null && \
				echo "  [OK] Symlinked into venv: $(VENV)/bin/neutron-converter"; \
		fi; \
		grep -q "$$CONVERTER_DIR" ~/.bashrc 2>/dev/null || \
			echo "export PATH=\"\$$PATH:$$CONVERTER_DIR\"" >> ~/.bashrc; \
		echo "  [OK] PATH updated in ~/.bashrc"; \
		echo "  Run: source ~/.bashrc  or open a new terminal"; \
	else \
		echo "  [WARN] neutron-converter not found. Archive contents:"; \
		ls $(EIQ_INSTALL_DIR); \
	fi
	@echo ""


# ---------------------------------------------------------------------------
# Network setup helpers
# ---------------------------------------------------------------------------
.PHONY: net-setup
net-setup:
	@echo "$(BOLD)Setting up Ethernet interface for direct board connection...$(RESET)"
	@# Find the right Ethernet interface (skip lo and wl*)
	@IFACE=$$(ip -o link show | awk -F': ' '!/lo|wl/{print $$2}' | head -1); \
	if [ -z "$$IFACE" ]; then \
		echo "  [ERROR] No wired Ethernet interface found."; \
		echo "  Please set up manually: sudo ip addr add 192.168.7.1/24 dev <iface>"; \
		exit 1; \
	fi; \
	echo "  Using interface: $$IFACE"; \
	sudo ip addr add 192.168.7.1/24 dev $$IFACE 2>/dev/null || \
		echo "  (Address may already be set)"; \
	sudo ip link set $$IFACE up; \
	echo "  [OK] Interface $$IFACE configured: 192.168.7.1/24"

.PHONY: net-check
net-check:
	@echo "$(BOLD)Testing connectivity to board at $(BOARD_IP)...$(RESET)"
	@if ping -c 3 -W 2 $(BOARD_IP) >/dev/null 2>&1; then \
		echo "  [OK] Board reachable at $(BOARD_IP)"; \
	else \
		echo "  [FAIL] Board NOT reachable at $(BOARD_IP)"; \
		echo "  Run: make net-setup"; \
	fi

# ---------------------------------------------------------------------------
# Clean generated files
# ---------------------------------------------------------------------------
.PHONY: clean-model
clean-model:
	@echo "$(BOLD)Removing compiled model files in $(OUT_DIR)/...$(RESET)"
	rm -rf $(OUT_DIR)
	@echo "  [OK] Model files removed. Run 'make model' to recompile."

.PHONY: clean
clean: clean-model
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "  [OK] Cleaned."

.PHONY: clean-venv
clean-venv:
	@echo "$(BOLD)Removing virtual environment $(VENV)/...$(RESET)"
	rm -rf $(VENV)
	@echo "  [OK] venv removed. Recreate with: make venv"

.PHONY: all
all: setup
