# IREE on the i.MX 95: workflow, comparison, and findings

This document records an experiment running the **IREE** compiler stack against the
same YOLOv8 model used by the main eIQ TFLite pipeline on the FRDM-IMX95, so the two
approaches can be compared on workflow and performance.

It is split into:

1. Conceptual comparison (IREE AOT vs eIQ TFLite + delegate)
2. The intended three-step IREE workflow (Import, Compile, Run)
3. What is publicly available vs commercial (the Neutron backend)
4. What we actually tested, with results
5. Performance comparison

---

## 1. Conceptual comparison

| Aspect                            | eIQ TFLite + Neutron delegate (this repo)                                                                             | IREE                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Compilation                       | `neutron-converter` (offline, x86 only) embeds NPU microcode into a `.tflite`; runtime delegate offloads matching ops | Ahead-of-time `iree-compile` produces a self-contained `.vmfb` (VM FlatBuffer) with the full execution schedule |
| Runtime on board                  | `tflite_runtime` interpreter + `libneutron_delegate.so`                                                               | `iree-run-module` or a C/C++ app on the IREE HAL runtime                                                        |
| Op not supported by NPU           | converter leaves it for CPU within the TFLite graph                                                                   | compiler lowers it to the `llvm-cpu` backend automatically                                                      |
| Partitioning                      | implicit, done by the converter                                                                                       | explicit multi-backend target list; compiler partitions and manages CPU/NPU memory hand-off                     |
| Host toolchain OS                 | x86-64 Linux only (`neutron-converter` is an x86 ELF)                                                                 | `iree-compile` has macOS/arm64 and Linux wheels                                                                 |
| Heterogeneous (CPU+GPU+NPU async) | not in this pipeline                                                                                                  | a stated IREE/Roofline strength, aimed at LLMs and mixed FP/INT models                                          |

The eIQ path is the officially supported route for standard fully-quantized vision
models such as YOLOv8. IREE's advantages (heterogeneous async execution, models
above the NPU's local memory, dynamic shapes) matter most for LLMs and mixed
architectures, not a single quantized CNN.

---

## 2. Intended three-step IREE workflow

### Step 1 - Import the model to MLIR

IREE does not consume `.tflite`/`.onnx`/`.pt` directly; it imports them to MLIR
(TOSA or StableHLO dialect) first.

```bash
# TFLite -> MLIR (TOSA)
iree-import-tflite model.tflite -o model.mlir
# (PyTorch would use torch-mlir instead)
```

### Step 2 - Compile AOT for the i.MX 95

```bash
iree-compile \
  --iree-input-type=tosa \
  --iree-hal-target-backends=neutron,llvm-cpu \
  --iree-llvmcpu-target-cpu=cortex-a55 \
  model.mlir \
  -o imx95_model.vmfb
```

- `--iree-hal-target-backends=neutron,llvm-cpu`: split the graph, NPU-supported ops to `neutron`, the rest to the Cortex-A55.
- `--iree-llvmcpu-target-cpu=cortex-a55`: tune the CPU fallback for the i.MX 95 application cores.
- output `.vmfb`: the deployable artifact.

### Step 3 - Run on the board

```bash
# On the i.MX 95:
iree-run-module \
  --module=imx95_model.vmfb \
  --device=neutron \
  --function=main \
  --input="1x224x224x3xi8"
```

---

## 3. Public vs commercial backends

- `iree-import-tflite`, `iree-compile` (with `llvm-cpu`), and `iree-run-module`
  are **open source** and pip-installable (`iree-base-compiler`,
  `iree-base-runtime`, TFLite importer tooling).
- The **`neutron` HAL target backend is Roofline AI's commercial integration**
  with NXP. It is not part of the public `iree-base-compiler`. See the Roofline
  case study: https://www.roofline.ai/case-studies/nxp-neutron-llm-enablement
- Therefore, with open tooling we can do **Import + CPU (Cortex-A55) compile + run**,
  but `--iree-hal-target-backends=neutron` is expected to fail (unregistered
  backend). The NPU comparison point comes from the eIQ pipeline instead.

---

## 4. What we tested

Host tools installed into an isolated `.venv-iree` (Python 3.12) via
`uv pip install iree-base-compiler iree-base-runtime`.

Verified facts (IREE compiler `3.11.0rc20260316`, LLVM 23):

- **Registered target backends:** `llvm-cpu`, `metal-spirv`, `vmvx`,
  `vmvx-inline`, `vulkan-spirv`. There is **no `neutron` backend** in public
  IREE, so the workflow's `--iree-hal-target-backends=neutron` fails with an
  unregistered-backend error. This confirms the Neutron NPU path needs
  Roofline's commercial build.
- **No `iree-import-tflite`** ships with `iree-base-compiler` 3.11 (only
  `iree-import-onnx`). So the open-tooling route is **ONNX**, not TFLite:
  `yolo export format=onnx` -> `iree-import-onnx` -> `iree-compile`.

Adjusted open-tooling workflow actually used:

```bash
# 1. Export YOLOv8s to ONNX (Ultralytics, host venv)
yolo export model=yolov8s.pt format=onnx imgsz=640 opset=17

# 2. Import ONNX -> MLIR
.venv-iree/bin/iree-import-onnx yolov8s.onnx -o yolov8s.mlir

# 3. Cross-compile for the board's Cortex-A55 (aarch64-linux)
.venv-iree/bin/iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-linux-gnu \
  --iree-llvmcpu-target-cpu=cortex-a55 \
  yolov8s.mlir -o yolov8s_cpu_a55.vmfb

# 4. Benchmark on the board (aarch64 runtime required)
iree-benchmark-module --module=yolov8s_cpu_a55.vmfb --device=local-task
```

- [x] Install IREE host tools (`iree-base-compiler`, runtime)
- [ ] Export YOLOv8s to ONNX + import to MLIR
- [ ] Cross-compile for `llvm-cpu` / cortex-a55 (aarch64) to `.vmfb`
- [x] Confirm `neutron` backend is **not** available in public IREE
- [ ] Get an aarch64 IREE runtime onto the board and benchmark the CPU `.vmfb`

---

## 5. Performance comparison

(Filled in once both pipelines run.)

| Pipeline                      | Device        | Model        | Latency / FPS                  | Notes                  |
| ----------------------------- | ------------- | ------------ | ------------------------------ | ---------------------- |
| eIQ TFLite + Neutron delegate | NPU (Neutron) | YOLOv8s int8 | TBD                            | reference NPU number   |
| eIQ TFLite (no delegate)      | CPU (A55)     | YOLOv8s int8 | TBD                            | `--no-npu`             |
| IREE `.vmfb`                  | CPU (A55)     | YOLOv8s int8 | TBD                            | open-tooling path      |
| IREE `.vmfb`                  | NPU (Neutron) | -            | not testable with open tooling | needs Roofline backend |
