# Model preparation notes (hackathon working log)

How the YOLOv8s NPU model was actually built for the FRDM-IMX95, the problems
hit, and the current state. The clean, reusable scripts referenced here live
alongside this file in `scripts/`.

## Why we could not just use `make model`

`make model` is designed to run the whole pipeline on ONE x86 Linux box:
Ultralytics export -> int8 quantize -> `neutron-converter` -> deploy. Our setup
(Apple Silicon Mac on the board's WiFi + an x86 Linux VM with no working NAT
internet) broke its assumptions, in three ways:

1. **`neutron-converter` is an x86-64 Linux ELF** and cannot run on the arm64
   Mac, so `make model` cannot complete on the Mac at all. We split the work:
   Mac exports + quantizes, the VM runs the converter.
2. **The int8 quantization step deadlocks on macOS arm64.** Ultralytics' int8
   export goes onnx2tf -> TensorFlow MLIR quantizer, which hangs at 0% CPU on
   Apple Silicon. So even the export+quantize half of `make model` never
   finishes on the Mac. We let the export produce the float `saved_model`, then
   ran the int8 conversion ourselves with the legacy quantizer
   (`scripts/common/quantize_to_int8.py`).
3. The VM _could_ have run `make model` end to end (x86 Linux, no deadlock),
   **but its NAT internet is broken**, so it cannot download the weights or
   pip-install Ultralytics. Hence the final arrangement: **Mac exports +
   quantizes, VM converts, Mac deploys.**

On top of that, the first manual quantization used the wrong calibration input
range (0-255 instead of 0-1), which produced garbage detections and forced a
re-quantization (see Gotchas). The `scripts/` here are the cleaned-up result of
working around all of the above.

## Environment reality

- **Mac (Apple Silicon, arm64)**: on the board's WiFi; the only machine that can
  reach the board (`192.168.1.236`) and the internet. Runs Ultralytics export
  and the int8 quantisation. Cannot run `neutron-converter` (it is an x86-64
  Linux ELF).
- **Linux VM (UTM, x86_64 Ubuntu 24.04, `192.168.105.5`)**: runs
  `neutron-converter` (and the rest of the eIQ SDK). Internet depends on the UTM
  network mode (Shared Network provides NAT internet); the converter needs none
  either way. How to build this VM: `docs/utm-vm-setup.md`.
- **Board (`192.168.1.236`, root, no password)**: runs the Python TFLite
  pipeline (`tflite_runtime` + `/usr/lib/libneutron_delegate.so`).

## Pipeline that works

1. **Export (Mac)** float model from Ultralytics. The int8 step deadlocks on
   macOS, so we let the export produce the `*_saved_model/` and stop there:
   ```bash
   .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='tflite', int8=True, imgsz=640)"
   # exports to models/sources/yolov8s_saved_model/ (float32/float16/dynamic-range tflite + calib npy)
   ```
2. **Quantise int8 (Mac)** from the saved_model with correct 0-1 calibration:
   ```bash
   .venv/bin/python scripts/common/quantize_to_int8.py \
     --saved-model models/sources/yolov8s_saved_model --calib-dir models/calib/calib_coco128 \
     --output models/work/yolov8s_full_integer_quant.tflite --imgsz 640
   ```
3. **Verify (Mac)** before shipping:
   ```bash
   .venv/bin/python scripts/detect/verify_tflite_detections.py --image models/calib/calib_coco128/<img>.jpg \
     --float models/sources/yolov8s_saved_model/yolov8s_float32.tflite \
     --int8 models/work/yolov8s_full_integer_quant.tflite
   # float and int8 should report the same classes
   ```
4. **NPU compile (VM)** (one-time VM setup: `docs/utm-vm-setup.md`):
   ```bash
   scp models/work/yolov8s_full_integer_quant.tflite utm@192.168.105.5:/home/utm/
   ssh utm@192.168.105.5 '~/edge-ai-workshop/bin/eiq-neutron-sdk-linux-3.1.2/bin/neutron-converter \
     --input /home/utm/yolov8s_full_integer_quant.tflite \
     --output /home/utm/yolov8s_neutron.tflite --target imx95'
   scp utm@192.168.105.5:/home/utm/yolov8s_neutron.tflite models/deploy/
   ```
5. **Deploy + run (Mac -> board)**:
   ```bash
   make deploy BOARD_IP=192.168.1.236
   ssh root@192.168.1.236 'cd /home/root/edge_ai_workshop/board; \
     nohup .venv/bin/python main.py >/tmp/app.log 2>&1 </dev/null & echo $!'
   # view: http://192.168.1.236:5000
   ```

## Gotchas

The biggest lesson from this work: **most of the friction was the toolchain and
the host environment, not the model.** The model maths was sound. What actually
cost time was which quantiser runs on which CPU architecture, what the NPU
compiler will and will not accept, and where each binary is even allowed to run.
The model-specific surprises (the int8 confidence-head crush) were real but
fewer. Capture these so the next person does not rediscover them the hard way.

### Toolchain and environment (the expensive ones)

- **The MLIR quantiser deadlocks on macOS arm64, but runs fine on x86 Linux.**
  Ultralytics' int8 export, and any `TFLiteConverter` with
  `experimental_new_quantizer = True`, hangs at 0% CPU on Apple Silicon partway
  through the full-integer stage. The float32/float16 stages finish first, so it
  looks like it almost worked, then never returns. Two ways out: on the Mac use
  the **legacy** quantiser (`experimental_new_quantizer = False`, which
  `scripts/common/quantize_to_int8.py` does); or run the MLIR per-channel
  quantiser on the **x86 VM**, where it does not deadlock. Detection is fine with
  the legacy per-tensor quantiser. The per-channel MLIR path only matters if you
  are chasing the pose/seg head, and even then it does not rescue it (see the
  int16 entry and the Pose section).

- **`neutron-converter` is an x86-64 Linux ELF.** It cannot run on an arm64 Mac
  at all. This is the entire reason the x86 VM exists (`docs/utm-vm-setup.md`):
  build and quantise on the Mac, compile on the VM, deploy from the Mac. Do not
  burn time trying to make the converter run under Rosetta or a wheel; it is a
  standalone binary, not a Python package.

- **`neutron-converter` is int8-only; it rejects int16.** int16 activations
  preserve the pose/seg confidence head (where int8 crushes it to zero), so it is
  the obvious thing to reach for. But the converter runs all the way through
  tiling and microcode generation and then fails with
  `ERROR: Tensor data type invalid!`. So "use int16 to save the head" is a dead
  end on this NPU, and that rejection is precisely what forced the backbone/head
  split instead. The failure is late and unhelpful, so recognise it early.

- **Calibration input range must be 0-1, not 0-255.** THE bug behind garbage
  detections (faces labelled "snowboard"/"cell phone" at 98%). The model expects
  0-1 input; calibrating with 0-255 gives an input quant scale of 1.0 instead of
  ~1/255, so the board's `(rgb-128)` preprocessing feeds values 255x too large.
  Calibrate at 0-1 to get input scale 1/255, zero-point -128.

### Runtime and board

- **`use_gstreamer` must be false** in `board/config.json`: the
  `imxvideoconvert_g2d` path fails (`g2d_open: Init Dpu Handle fail`) and crashes
  `main.py` instead of falling back. OpenCV opens `/dev/video4` (the C270) fine.
- **Do not `pkill -f main.py`** in the same shell that launches `main.py` (the
  launch command contains "main.py", so pkill kills its own shell). Launch in a
  separate step, or match a pattern that excludes the launcher (`[m]ain.py`).
- **Chain compiled stages by tensor shape, not output index.** A compiler is free
  to permute a multi-output stage's outputs, so handing them to the next stage by
  position feeds the wrong tensors. `board/inference.py:invoke_stage` routes by
  shape; this bit us on the pose backbone -> head handoff (see "Board wiring note"
  in the Pose section). Applies to any multi-stage NPU/CPU pipeline, not just pose.

## Pose / keypoints: mostly on the NPU via a backbone/head split

The earlier conclusion was "pose cannot run on the NPU". That was true _only for
whole-model conversion_, and it is now superseded. Pose runs **mostly on the NPU**
by splitting the graph: the heavy backbone/neck goes int8 on the Neutron NPU, and
only the small head (where int8 does the damage) stays float on the CPU.

### Why whole-model conversion fails (still true)

- **int8 crushes the confidence head to zero.** Verified across five methods (TF
  legacy per-tensor, TF MLIR per-channel, eIQ per-channel int8 and float output,
  int8-with-float-fallback). Box and keypoint channels quantise fine but output
  channel 4 (person confidence) decodes to exactly 0, so nothing is detected. The
  conf branch's small-magnitude logits are destroyed by int8 _activations_.
- **int16 activations preserve the conf head but `neutron-converter` cannot
  compile int16** (`ERROR: Tensor data type invalid!`). The Neutron microcode
  generator is int8-only for these ops.

The key realisation: the crush is a property of the _head_, not the network. The
backbone/neck quantise fine. So cut the graph and keep only the head in float.

### The split that works

Cut the ONNX graph at the three neck outputs that feed the head (the natural
P3/P4/P5 boundary, post-SiLU activations, a clean quantisation boundary):

    /model.15/cv2/act/Mul_output_0   [1,128,80,80]   (P3)
    /model.18/cv2/act/Mul_output_0   [1,256,40,40]   (P4)
    /model.21/cv2/act/Mul_output_0   [1,512,20,20]   (P5)

- **backbone** (input -> 3 feature maps): 175 nodes / 45 Conv -> int8 -> Neutron.
- **head** (3 feature maps -> [1,56,8400]): 100 nodes / 28 Conv -> float32 CPU.

Reusable pipeline (all on the Mac, except neutron-converter on the VM):

    # 1. graph surgery — split pose ONNX at the neck outputs
    #    (onnx.utils.extract_model; produces models/work/split_pose/backbone.onnx + head.onnx)
    # 2. onnx2tf both halves to TF (float). head_float32.tflite is the CPU stage.
    .venv/bin/onnx2tf -i models/work/split_pose/head.onnx     -o models/work/split_pose/head_tf     -osd
    .venv/bin/onnx2tf -i models/work/split_pose/backbone.onnx -o models/work/split_pose/backbone_tf -osd
    # 3. int8-quantise ONLY the backbone (legacy quantizer, 0-1 calib, float32 output
    #    so it hands clean feature maps to the float head)
    .venv/bin/python scripts/common/quantize_to_int8.py \
      --saved-model models/work/split_pose/backbone_tf \
      --calib-dir models/calib/calib_coco128 \
      --output models/work/split_pose/backbone_int8.tflite \
      --imgsz 640 --num-images 128 --output-dtype float32
    # 4. validate numerically on the Mac (no board needed)
    .venv/bin/python scripts/pose/validate_split_pose.py \
      --image models/calib/calib_coco128/coco128/images/train2017/000000000036.jpg
    # 5. NPU-compile the backbone on the VM
    scp models/work/split_pose/backbone_int8.tflite utm@192.168.105.5:/home/utm/
    ssh utm@192.168.105.5 '~/edge-ai-workshop/bin/eiq-neutron-sdk-linux-3.1.2/bin/neutron-converter \
      --input /home/utm/backbone_int8.tflite --output /home/utm/backbone_neutron.tflite --target imx95'
    scp utm@192.168.105.5:/home/utm/backbone_neutron.tflite models/work/split_pose/

### Results (validated)

- **Confidence head restored.** On a real person image: float baseline conf 0.825
  (1 detection), split int8-backbone -> float-head conf **0.821** (1 detection),
  whole-model int8 **0.000** (0 detections). The split tracks the float baseline
  across every image tested; whole-model int8 is zero everywhere.
- **NPU coverage.** `neutron-converter` compiles the backbone at **230/233 ops =
  98.7%** onto a single Neutron graph (the 3 unconverted ops are the trailing
  float dequants feeding the head, expected). NPU latency estimate **22 ms**.
  This is exactly the compile that whole-model int16 pose could not achieve.
- **Why the Mac validation is trustworthy (reusable lesson).** The split is
  validated on the Mac with the **CPU int8 backbone standing in for the NPU one**.
  That is faithful: `neutron-converter` compiles the _same_ int8 graph to NPU
  microcode without changing its numerics, so the CPU int8 model is a numerical
  proxy for the NPU model (which cannot run on the Mac at all). Only on-NPU latency
  and any delegate-level quirks remain to confirm on hardware. This proxy is what
  let us de-risk the whole approach before the slow surgery -> quantise ->
  VM-compile -> board loop. Validate the cheap proxy first; touch the board last.

### Board wiring note (bug found and fixed)

The backbone's output order ([40x40, 80x80, 20x20]) does NOT positionally match
the head's input order ([80x80, 40x40, 20x20]): the converter is free to permute
a multi-output stage. `board/inference.py:invoke_stage` therefore routes each
feature map to the input slot with the matching shape, not by position. Positional
chaining (the old behaviour) fed swapped feature maps. Guarded by
`scripts/pose/test_stage_chaining.py`.

## Current state

- **Detection: YOLOv8s int8 -> Neutron NPU**, ~19.6 FPS. Working, deploy-ready
  (`models/deploy/yolov8s_neutron.tflite`). See `docs/deployment.md` section 1.
- **Pose: YOLOv8s-pose split, backbone int8 on NPU + head float on CPU**,
  deploy-ready (`models/deploy/yolov8s-pose/`: `pipeline.json` +
  `backbone_neutron.tflite` + `head_float32.tflite`). Numerically validated on the
  Mac (conf restored to 0.821 vs float 0.825); on-board FPS pending board access.
  See `docs/deployment.md` section 2.
- **Segmentation / CenterNet:** same int8 conf/class-head crush. The pose split
  approach applies (cut at the neck, mask-proto + coeff head on CPU); not yet
  built. Step-by-step extension guide: `docs/adding-segmentation.md`.
- **IREE CPU-vs-NPU comparison:** see `docs/iree-workflow.md` (public IREE has no
  Neutron backend; CPU path only).

## Open items / not yet validated

- **Pose has not run on the board NPU yet.** It is numerically validated on the
  Mac and the backbone compiles cleanly (98.7%), but actual on-NPU execution, live
  correctness, and FPS are unconfirmed. Treat pose as "proven in principle, not
  yet proven on hardware" until someone runs `docs/deployment.md` section 2 and
  checks the stream and the two-stage log lines. Detection, by contrast, is
  measured on the board (~19.6 FPS).
- **Deploy binaries are tracked; the rest of `models/` is not.** `.gitignore`
  ignores `models/*` but re-includes `models/deploy` (via `!models/deploy`), so
  the board-ready artifacts (neutron models, manifests, labels) are
  version-controlled and survive without the build machine. The sources,
  calibration data and intermediates (`models/sources`, `models/calib`,
  `models/work`) stay ignored and are rebuilt by the scripts. Rebuilding the NPU
  models from scratch still needs the x86 VM (`docs/utm-vm-setup.md`). Note: the
  ignore rule must be `models/*`, not `models`, or the re-include cannot take
  effect (git will not descend into an excluded directory).
