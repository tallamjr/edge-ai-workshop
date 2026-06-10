#!/usr/bin/env python3
"""Quantize a float YOLOv8 TF saved_model to a full-integer int8 TFLite model.

Why this exists
---------------
`make model` (Ultralytics export with int8=True) does the export + int8
quantization in one go via onnx2tf. On macOS arm64 the int8 step DEADLOCKS in
TensorFlow's MLIR quantizer (0% CPU hang at the full_integer_quant stage; the
float32/float16/dynamic-range stages complete fine and leave a saved_model on
disk). This script runs only the int8 conversion from that already-exported
saved_model, with two crucial settings:

  1. Calibration images are normalised to 0-1, because the Ultralytics/onnx2tf
     YOLOv8 model expects 0-1 input. Calibrating with 0-255 produces an input
     quant scale of 1.0 (instead of ~1/255) and the resulting int8 model emits
     garbage (every anchor fires, faces classified as "snowboard"). With 0-1
     data the input quant is scale=1/255, zero_point=-128, which is exactly what
     the board's inference.py preprocessing -- (rgb_uint8 - 128) cast to int8 --
     feeds the model.

  2. experimental_new_quantizer = False (legacy quantizer). The new MLIR
     quantizer is what deadlocks on macOS arm64; the legacy quantizer runs fine
     and, given correct 0-1 calibration, produces an accurate model.

After this, compile for the Neutron NPU with neutron-converter on an x86 Linux
host (the converter is an x86-64 ELF; it cannot run on Apple Silicon):

    neutron-converter --input <out>.tflite --output <out>_neutron.tflite --target imx95

Usage
-----
    python scripts/quantize_to_int8.py \
        --saved-model models/sources/yolov8s_saved_model \
        --calib-dir models/calib/calib_coco128 \
        --output models/work/yolov8s_full_integer_quant.tflite \
        --imgsz 640 --num-images 128

The calibration directory should contain representative JPEG/PNG images
(coco128 works well). Download coco128 with:
    python -c "import urllib.request,zipfile,io; \
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://ultralytics.com/assets/coco128.zip').read())).extractall('calib_coco128')"
"""
import argparse
import pathlib

import cv2
import numpy as np
import tensorflow as tf


def load_calibration(calib_dir: str, imgsz: int, num_images: int) -> np.ndarray:
    """Load images, resize to imgsz, RGB, normalised to 0-1 (model input range)."""
    paths = sorted(pathlib.Path(calib_dir).rglob("*.jpg"))[:num_images]
    if not paths:
        raise FileNotFoundError(f"No .jpg images found under {calib_dir}")
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (imgsz, imgsz))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(im)
    arr = np.stack(out)
    print(f"calibration: {arr.shape} range [{arr.min()}, {arr.max()}]")
    return arr


def quantize(saved_model: str, calib: np.ndarray, output: str,
             output_dtype: str = "int8", new_quantizer: bool = False,
             float_fallback: bool = False) -> None:
    def representative_dataset():
        for i in range(calib.shape[0]):
            yield [calib[i:i + 1]]

    out_t = tf.float32 if output_dtype == "float32" else tf.int8
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    # float_fallback adds TFLITE_BUILTINS so the converter may leave ops it
    # cannot cleanly quantize in float (CPU) while still int8-quantising the rest
    # (NPU). Used to try to rescue the pose/seg confidence head.
    if float_fallback:
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    else:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8     # int8 input for the NPU (board feeds rgb-128)
    conv.inference_output_type = out_t
    # The MLIR (new) quantizer does per-channel weight quantization, which the
    # detect head tolerates per-tensor but the pose/seg heads do NOT (per-tensor
    # legacy quantization zeroes their outputs). The MLIR quantizer DEADLOCKS on
    # macOS arm64, so use new_quantizer=True only on x86 Linux (e.g. the VM).
    # For pose/seg also use output_dtype="float32" so the mixed-range output
    # (0-1 confidences + large coords / mask coeffs) is not crushed by one int8 scale.
    conv.experimental_new_quantizer = new_quantizer

    label = "MLIR per-channel" if new_quantizer else "legacy per-tensor"
    print(f"converting ({label} quantizer, {output_dtype} output)...")
    data = conv.convert()
    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    print(f"WROTE {out_path} ({len(data)} bytes)")
    print(f"input quant (scale, zero_point) = {inp['quantization']} "
          f"(expect scale ~0.0039, zp=-128)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved-model", required=True,
                    help="Path to the float TF saved_model dir (from Ultralytics export)")
    ap.add_argument("--calib-dir", required=True,
                    help="Directory of representative calibration images")
    ap.add_argument("--output", required=True, help="Output int8 .tflite path")
    ap.add_argument("--imgsz", type=int, default=640, help="Model input size (default 640)")
    ap.add_argument("--num-images", type=int, default=128,
                    help="Number of calibration images to use (default 128)")
    ap.add_argument("--output-dtype", choices=["int8", "float32"], default="int8",
                    help="Output tensor dtype. Use float32 for pose/seg (mixed-range output).")
    ap.add_argument("--new-quantizer", action="store_true",
                    help="Use the MLIR per-channel quantizer (x86 Linux only; deadlocks on macOS). "
                         "Required for pose/seg heads.")
    ap.add_argument("--float-fallback", action="store_true",
                    help="Allow ops that don't quantize cleanly to stay float (CPU) while the "
                         "rest are int8 (NPU). Tries to rescue the pose/seg confidence head.")
    args = ap.parse_args()

    calib = load_calibration(args.calib_dir, args.imgsz, args.num_images)
    quantize(args.saved_model, calib, args.output,
             output_dtype=args.output_dtype, new_quantizer=args.new_quantizer,
             float_fallback=args.float_fallback)


if __name__ == "__main__":
    main()
