# scripts/

Helper scripts for building, validating and deploying the YOLOv8 models. Run
them from the **repo root** using the project venv (`.venv/bin/python`). They are
grouped by scope:

| Folder | Scope |
| --- | --- |
| `common/` | Model-agnostic tooling (works for any YOLOv8 model) |
| `detect/` | Object-detection specific |
| `pose/` | Pose / keypoints specific |

For the full pipeline and the reasoning behind it, see
`docs/model-preparation-notes.md`. The NPU compile step needs the x86 VM in
`docs/utm-vm-setup.md`.

---

## common/

### `quantize_to_int8.py`

Quantise a float YOLOv8 TF `saved_model` to a full-integer **int8** TFLite. Uses
the legacy quantiser (the MLIR one deadlocks on macOS arm64) and 0-1 calibration
(the input scale must be ~1/255 to match the board's `rgb-128` preprocessing).
Model-agnostic: used both for the detection model and for the pose backbone.

```bash
.venv/bin/python scripts/common/quantize_to_int8.py \
  --saved-model models/sources/yolov8s_saved_model \
  --calib-dir   models/calib/calib_coco128 \
  --output      models/work/yolov8s_full_integer_quant.tflite \
  --imgsz 640 --num-images 128
```

Useful flags:
- `--output-dtype float32`: keep outputs float (used for the pose split backbone
  so it hands clean feature maps to the float head).
- `--new-quantizer`: MLIR per-channel quantiser; **x86 Linux only** (deadlocks on
  macOS).
- `--float-fallback`: let ops that will not quantise cleanly stay float on CPU.

### `split_model.py`

Analyse a **neutron-compiled** `.tflite` and optionally split it into
`pre`/`npu`/`post` sub-models for pipelined CPU/NPU execution, writing the
`pipeline.json` manifest the board reads. Requires `tflite-extractor` (eIQ
Toolkit) on `PATH` for `--split`.

```bash
# analyse only (reports which ops the NPU took vs left on CPU)
.venv/bin/python scripts/common/split_model.py --model models/deploy/yolov8s_neutron.tflite

# split into sub-models + manifest
.venv/bin/python scripts/common/split_model.py \
  --model models/deploy/yolov8s_neutron.tflite \
  --split --output-dir models/work/split/
```

---

## detect/

### `prepare_models.sh`

End-to-end **detection** pipeline in one command: export a YOLOv8 int8 TFLite via
Ultralytics, compile it for the Neutron NPU, and copy the model plus COCO labels
to the board. This is the original single-output detection path; run it on an x86
Linux box (or the VM) where `neutron-converter` is available.

```bash
./scripts/detect/prepare_models.sh                              # yolov8n, imgsz 640
./scripts/detect/prepare_models.sh MODEL=yolov8s BOARD_IP=192.168.1.236
```

Override via `KEY=value` args: `MODEL`, `IMGSZ`, `BOARD_IP`, `BOARD_USER`,
`BOARD_DIR`, `OUT_DIR` (default `./models/deploy`).

### `verify_tflite_detections.py`

Sanity-check a detection TFLite before shipping: run it on one image with the
board's exact preprocessing, decode the `[1, 84, 8400]` output and print the most
common classes. A correctly quantised int8 model reports the same classes as the
float model; a mis-calibrated one returns garbage (this is how the 0-255 vs 0-1
calibration bug was caught).

```bash
.venv/bin/python scripts/detect/verify_tflite_detections.py \
  --image models/calib/calib_coco128/coco128/images/train2017/000000000036.jpg \
  --float models/sources/yolov8s_saved_model/yolov8s_float32.tflite \
  --int8  models/work/yolov8s_full_integer_quant.tflite
```

---

## pose/

### `validate_split_pose.py`

Numerically validate the pose **CPU/NPU split** on the Mac, no board required.
Runs three pipelines on one image and reports the person-confidence channel plus
decoded detections:

1. float32 baseline (reference)
2. int8 backbone (NPU-bound) -> float32 head (CPU): the split
3. whole-model int8: the broken one whose confidence head decodes to zero

Confirms the split restores confidence (e.g. 0.821 vs a 0.825 baseline) where
whole-model int8 gives 0.000.

```bash
.venv/bin/python scripts/pose/validate_split_pose.py \
  --image models/calib/calib_coco128/coco128/images/train2017/000000000036.jpg
```

Defaults point at `models/work/split_pose/` and the `models/work/` baselines;
override with `--baseline`, `--backbone`, `--head`, `--all-int8`.

### `test_stage_chaining.py`

Regression test guarding `board/inference.py:invoke_stage`. A multi-output NPU
stage's outputs do **not** positionally match the next stage's inputs (the
converter permutes them), so `invoke_stage` must route tensors by **shape**.
Uses the pose backbone/head as fixtures, so the pose split must be built first.
Exit code 0 means pass.

```bash
.venv/bin/python scripts/pose/test_stage_chaining.py
```

---

## How they fit together

**Detection** (single output): `prepare_models.sh` does export -> compile ->
deploy in one go. To do it by hand: `quantize_to_int8.py` -> `neutron-converter`
on the VM -> `verify_tflite_detections.py` -> deploy.

**Pose** (backbone on NPU, head on CPU): graph surgery -> `quantize_to_int8.py`
on the backbone with `--output-dtype float32` -> `validate_split_pose.py` ->
`neutron-converter` on the VM -> `test_stage_chaining.py` -> deploy. The full
recipe is in `docs/model-preparation-notes.md` (Pose section) and
`docs/deployment.md`.
