#!/usr/bin/env python3
"""Sanity-check a YOLOv8 detection TFLite model before deploying it to the board.

Runs the model on a single image using the SAME preprocessing the board uses
(board/inference.py): for an int8 model the input is (rgb_uint8 - 128) cast to
int8; for a float model the input is rgb / 255.0. It then decodes the YOLOv8
output [1, 84, 8400] and prints the most common detected classes.

This is how the int8 calibration bug was caught: a correctly-quantised model
returns the same classes as the float32 model (e.g. person, elephant), while a
mis-calibrated one returns thousands of garbage detections (snowboard, boat,
cell phone) at high confidence.

Usage
-----
    # Compare a float and an int8 model on the same image:
    python scripts/verify_tflite_detections.py \
        --image some.jpg \
        --float models/sources/yolov8s_saved_model/yolov8s_float32.tflite \
        --int8 models/work/yolov8s_full_integer_quant.tflite

    # Or check a single model:
    python scripts/verify_tflite_detections.py --image some.jpg --int8 model.tflite
"""
import argparse
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def load_rgb(path: str, imgsz: int) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, (imgsz, imgsz))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0..255


def run(model_path: str, rgb: np.ndarray, conf: float):
    it = tf.lite.Interpreter(model_path=model_path)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    if ind["dtype"] == np.int8:
        x = (rgb.astype(np.int16) - 128).astype(np.int8)[None]   # board preprocessing
    else:
        x = (rgb / 255.0).astype(np.float32)[None]               # float model expects 0..1
    it.set_tensor(ind["index"], x)
    it.invoke()
    raw = it.get_tensor(outd["index"]).astype(np.float32)
    if outd["dtype"] == np.int8:
        s, z = outd["quantization"]
        raw = (raw - z) * s
    pred = raw[0]                      # [84, 8400]
    scores = pred[4:, :]
    maxs, cls = scores.max(0), scores.argmax(0)
    keep = maxs >= conf
    return Counter(COCO[i] for i in cls[keep]).most_common(6), int(keep.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--float", dest="float_model")
    ap.add_argument("--int8", dest="int8_model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    rgb = load_rgb(args.image, args.imgsz)
    if args.float_model:
        print("FLOAT32:", run(args.float_model, rgb, args.conf))
    if args.int8_model:
        print("INT8   :", run(args.int8_model, rgb, args.conf))


if __name__ == "__main__":
    main()
