#!/usr/bin/env python3
"""Validate the pose CPU/NPU split numerically on the Mac (no board required).

Compares three pipelines on a real image:
  1. baseline  : float32 pose model (reference truth)
  2. split     : int8 backbone (NPU-bound) -> float32 head (CPU)
  3. all-int8  : the whole-model int8 pose (the broken one, conf head crushed)

The key metric is the person-confidence channel (row 4 of the [1,56,8400]
output). int8 across the whole model decodes it to ~0 (nothing detected);
the split must restore it to near the float baseline.
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import tensorflow as tf

# scripts/pose/<this file> -> repo root is three levels up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "board"))
import inference  # noqa: E402  (board decoder, reused as-is)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def load_image(path, imgsz=640):
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, (imgsz, imgsz))
    rgb01 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return im, rgb01  # bgr frame for decode scaling, and 0-1 rgb tensor


def interp(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def run_single(it, x):
    """Feed one input tensor, respecting its dtype/quant. Return first output."""
    d = it.get_input_details()[0]
    if d["dtype"] == np.int8:
        # board preprocessing: uint8 - 128
        u8 = np.round(x * 255).astype(np.int16)
        xin = (u8 - 128).astype(np.int8)
    elif d["dtype"] == np.float32:
        xin = x.astype(np.float32)
    else:
        raise TypeError(d["dtype"])
    it.set_tensor(d["index"], xin)
    it.invoke()
    od = it.get_output_details()[0]
    out = it.get_tensor(od["index"]).astype(np.float32)
    if od["dtype"] == np.int8:
        s, z = od["quantization"]
        out = (out - z) * s
    return out


def run_split(backbone, head, x):
    """int8 backbone -> 3 float feature maps -> float head -> [1,56,8400]."""
    bd = backbone.get_input_details()[0]
    u8 = np.round(x * 255).astype(np.int16)
    xin = (u8 - 128).astype(np.int8)
    backbone.set_tensor(bd["index"], xin)
    backbone.invoke()
    feats = {tuple(o["shape"]): backbone.get_tensor(o["index"]).astype(np.float32)
             for o in backbone.get_output_details()}
    for hi in head.get_input_details():
        head.set_tensor(hi["index"], feats[tuple(hi["shape"])])
    head.invoke()
    return head.get_tensor(head.get_output_details()[0]["index"]).astype(np.float32)


def conf_max(raw):
    return float(raw[0, 4, :].max())


def decode(raw, frame, conf=0.5):
    cfg = {"model": {"task": "pose"},
           "inference": {"confidence_threshold": conf, "iou_threshold": 0.45,
                         "max_detections": 10}}
    return inference.postprocess_pose(raw, frame, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--baseline", default="models/work/yolov8s-pose_float32.tflite")
    ap.add_argument("--backbone", default="models/work/split_pose/backbone_int8.tflite")
    ap.add_argument("--head", default="models/work/split_pose/head_tf/head_float32.tflite")
    ap.add_argument("--all-int8", default="models/work/yolov8s-pose_full_integer_quant.tflite")
    args = ap.parse_args()

    frame, x = load_image(ROOT / args.image)
    xb = x[None]  # [1,640,640,3]

    base_raw = run_single(interp(ROOT / args.baseline), xb)
    split_raw = run_split(interp(ROOT / args.backbone), interp(ROOT / args.head), xb)

    print(f"image: {args.image}")
    print(f"{'pipeline':<22}{'conf_max':>10}{'detections':>12}")
    print("-" * 44)
    bd = decode(base_raw, frame)
    sd = decode(split_raw, frame)
    print(f"{'1. baseline float':<22}{conf_max(base_raw):>10.3f}{len(bd):>12}")
    print(f"{'2. split int8->float':<22}{conf_max(split_raw):>10.3f}{len(sd):>12}")

    ai_path = ROOT / args.all_int8
    if ai_path.exists():
        ai_raw = run_single(interp(ai_path), xb)
        ad = decode(ai_raw, frame)
        print(f"{'3. all-int8 (broken)':<22}{conf_max(ai_raw):>10.3f}{len(ad):>12}")

    # numerical agreement between baseline and split on the full output tensor
    mae = float(np.abs(base_raw - split_raw).mean())
    cm = float(np.abs(base_raw[0, 4] - split_raw[0, 4]).max())
    print("-" * 44)
    print(f"baseline vs split  output MAE : {mae:.4f}")
    print(f"baseline vs split  conf max|d|: {cm:.4f}")
    if bd and sd:
        print(f"baseline top conf={bd[0]['confidence']:.3f}  split top conf={sd[0]['confidence']:.3f}")


if __name__ == "__main__":
    main()
