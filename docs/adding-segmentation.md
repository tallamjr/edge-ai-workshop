# Adding the segmentation model (YOLOv8s-seg)

Step-by-step guide for the next person extending this repo with instance
segmentation. It reuses the exact backbone-on-NPU / head-on-CPU split proven for
pose (see `docs/model-preparation-notes.md`, Pose section). Read that first; this
doc only covers what is **different** for seg and gives the commands to run.

## Why seg is "pose plus a second output"

The split spine is identical, because seg shares the same YOLOv8s backbone and
neck. The differences all flow from one fact: **seg has two output tensors**,
where detection and pose have one.

| | output0 | output1 |
| --- | --- | --- |
| shape | `[1, 116, 8400]` | `[1, 32, 160, 160]` |
| meaning | 4 box + 80 class + **32 mask coefficients** per anchor | 32 **prototype masks** at 160x160 |

The final mask for a detection is `sigmoid(coeffs @ protos)`, cropped to its box.
The 80-class head is crushed by int8 exactly like pose's confidence head, so it
must stay float on the CPU; the backbone goes int8 on the NPU as before.

> **Mac users: the NPU compile cannot run on your machine.** `neutron-converter`
> is an x86-64 Linux binary. Build the model on the Mac, but do step 5 (compile)
> on the x86 VM. See `docs/utm-vm-setup.md`. Everything else runs on the Mac.

---

## Prerequisites

- The Mac venv (`.venv`) and calibration data (`models/calib/calib_coco128`).
- The seg sources are already in the repo: `models/sources/yolov8s-seg.onnx`,
  `.pt`, and `yolov8s-seg_saved_model/`.
- The x86 VM set up per `docs/utm-vm-setup.md` (for step 5 only).

---

## Step 1: Graph surgery (Mac)

Cut at the same three neck outputs as pose (the seg model shares them). The head
half will have **two** outputs, not one.

```python
# .venv/bin/python
import onnx
from onnx.utils import extract_model

SRC  = "models/sources/yolov8s-seg.onnx"
NECK = ["/model.15/cv2/act/Mul_output_0",   # P3 [1,128,80,80]
        "/model.18/cv2/act/Mul_output_0",   # P4 [1,256,40,40]
        "/model.21/cv2/act/Mul_output_0"]   # P5 [1,512,20,20]

extract_model(SRC, "models/work/split_seg/backbone.onnx",
              input_names=["images"], output_names=NECK)
extract_model(SRC, "models/work/split_seg/head.onnx",
              input_names=NECK, output_names=["output0", "output1"])
```

Sanity-check the head has both `output0 [1,116,8400]` and `output1 [1,32,160,160]`.

## Step 2: Convert and quantise (Mac)

Identical to pose. Convert both halves to TF (float), then int8-quantise **only
the backbone** with float outputs so it hands clean feature maps to the float
head.

```bash
.venv/bin/onnx2tf -i models/work/split_seg/head.onnx     -o models/work/split_seg/head_tf     -osd
.venv/bin/onnx2tf -i models/work/split_seg/backbone.onnx -o models/work/split_seg/backbone_tf -osd

.venv/bin/python scripts/common/quantize_to_int8.py \
  --saved-model models/work/split_seg/backbone_tf \
  --calib-dir   models/calib/calib_coco128 \
  --output      models/work/split_seg/backbone_int8.tflite \
  --imgsz 640 --num-images 128 --output-dtype float32
```

The backbone is the same architecture as detection/pose, so it compiles to the
NPU at ~98.7% (step 5). No new risk here.

## Step 3: New board code (the actual work)

This is what does not exist yet. Four edits:

1. **`board/inference.py` -> `postprocess_seg(out0, out1, frame, config)`**
   New decoder. Steps:
   - From `output0 [1,116,8400]`: split into box `[:4]`, class scores `[4:84]`,
     mask coeffs `[84:116]`. Threshold on max class score, run per-class NMS
     (reuse `_nms`), keep top-N. This half mirrors `postprocess_detections`.
   - For each kept detection with its 32 coeffs `c`: `mask = sigmoid(c @ protos)`
     where `protos = output1[0]` reshaped to `[32, 160*160]`. Result is a
     160x160 mask; upsample to the frame, crop to the detection's box,
     threshold at 0.5.
   - Return detections carrying a `"mask"` (HxW bool array) alongside
     `bbox/label_id/confidence`.

2. **`board/overlay.py` -> mask rendering**
   Alpha-blend each detection's mask onto the frame (per-class colour), then draw
   the existing box/label on top.

3. **`board/main.py` two-output plumbing** (line ~543). It currently does:
   ```python
   raw = raw_list[0]  # final stage always produces a single detection tensor
   ```
   The seg head emits two tensors, so `raw_list[0]` silently drops the prototype
   masks. Branch on `task == "seg"` to pass **both** tensors to `postprocess_seg`.
   Match them **by shape, not index** (`[1,116,8400]` vs `[1,32,160,160]`):
   onnx2tf/neutron may permute output order, the same lesson as the
   `invoke_stage` shape routing fix.

4. **`board/config.seg.json`** (copy `config.pose.json`): set `task: "seg"`, add a
   `seg` entry to `model.variants` (`"seg": "yolov8s-seg"`).

## Step 4: Validate on the Mac (no board)

Clone `scripts/pose/validate_split_pose.py` to `scripts/seg/validate_split_seg.py`.
Run the three pipelines (float baseline, int8-backbone -> float-head, whole-model
int8) and confirm the **class head is restored** (whole-model int8 collapses it
to ~0, the split matches the float baseline) and the decoded masks line up with
the baseline. Reuse the whole-model int8 reference already at
`models/work/yolov8s-seg_full_integer_quant.tflite`.

## Step 5: NPU compile (VM)

Mac cannot run this; do it on the x86 VM (`docs/utm-vm-setup.md`).

```bash
scp models/work/split_seg/backbone_int8.tflite utm@192.168.105.5:/home/utm/
ssh utm@192.168.105.5 \
  '~/edge-ai-workshop/bin/eiq-neutron-sdk-linux-3.1.2/bin/neutron-converter \
     --input /home/utm/backbone_int8.tflite \
     --output /home/utm/backbone_neutron.tflite --target imx95'
scp utm@192.168.105.5:/home/utm/backbone_neutron.tflite models/work/split_seg/
```

Expect a conversion ratio near `0.98` and an NPU latency estimate. The compiled
model carries a custom `NeutronGraph` op, so it only runs on the board, not the
Mac (that is normal).

## Step 6: Deploy (Mac -> board)

Stage the deploy artifacts in the board layout, then push them.

```bash
mkdir -p models/deploy/yolov8s-seg
cp models/work/split_seg/backbone_neutron.tflite        models/deploy/yolov8s-seg/
cp models/work/split_seg/head_tf/head_float32.tflite    models/deploy/yolov8s-seg/
cat > models/deploy/yolov8s-seg/pipeline.json <<'JSON'
{
  "pipeline": [
    {"label": "backbone_npu", "file": "backbone_neutron.tflite", "use_npu": true},
    {"label": "head_cpu",     "file": "head_float32.tflite",     "use_npu": false}
  ]
}
JSON

make board-deploy-app BOARD_IP=192.168.1.236
ssh root@192.168.1.236 'mkdir -p /opt/models/yolov8s-seg'
scp models/deploy/yolov8s-seg/* root@192.168.1.236:/opt/models/yolov8s-seg/
scp board/config.seg.json root@192.168.1.236:/home/root/edge_ai_workshop/board/config.json
# restart main.py (see docs/deployment.md for the safe restart helper)
```

The `invoke_stage` shape routing already handles the multi-input backbone -> head
handoff; no change needed there.

---

## Reuse vs new at a glance

| Reused unchanged | Written new for seg |
| --- | --- |
| graph surgery technique, `quantize_to_int8.py`, `invoke_stage` shape routing, neutron compile, deploy flow, validator skeleton | `postprocess_seg`, mask overlay, two-output `main.py` branch, `config.seg.json`, `scripts/seg/` |

The hard question (can it run mostly on the NPU) is already answered by pose: it
is the same backbone. The effort is the mask decode, the overlay, and the
two-output plumbing, not uncertainty. Budget roughly a focused half-day to
numerical validation, plus the VM compile.
