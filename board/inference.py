"""
inference.py — TFLite model runner with NXP NPU delegate support.

This module handles model loading and frame inference.
Participants can swap models or adjust pre/post-processing here.
"""

import json
import os
import time
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_tflite():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
        logger.warning("tflite_runtime not found, using tensorflow.lite (no NPU)")
    return tflite


def load_labels(labels_path: str) -> list[str]:
    """Load class labels from a text file (one label per line)."""
    path = Path(labels_path)
    if not path.exists():
        logger.warning(f"Labels file not found: {labels_path}. Using numeric IDs.")
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def _load_single(model_path: str, config: dict, use_npu: bool):
    """
    Load one TFLite model file and return (interpreter, input_details, output_details).
    """
    tflite        = _get_tflite()
    delegate_path = config["model"].get("npu_delegate_path", "")
    num_threads   = config["model"].get("num_threads", 6)

    delegates = []
    if use_npu and Path(delegate_path).exists():
        try:
            os.environ["NEUTRON_ENABLE_ZERO_COPY"] = "1"
            delegates = [tflite.load_delegate(delegate_path)]
        except Exception as e:
            logger.warning(f"Failed to load NPU delegate: {e}. Falling back to CPU.")
    elif use_npu:
        logger.warning(f"NPU delegate not found at {delegate_path}. Running on CPU.")

    interp = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=num_threads,
    )
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def load_pipeline(config: dict) -> list[dict]:
    """
    Load the inference pipeline from config.

    config["model"]["path"] can be either:
      - A .tflite file  → single-stage pipeline (identical behaviour to before)
      - A .json manifest → multi-stage pipelined pipeline

    Each stage is a dict:
        label          str   human-readable segment name
        interpreter    TFLite Interpreter (already allocated)
        input_details  list  from get_input_details()
        output_details list  from get_output_details()
        is_npu         bool  True when this stage uses the NPU delegate

    The manifest format written by scripts/split_model.py:
        {
          "pipeline": [
            {"label": "pre",  "file": "pre.tflite",  "use_npu": false},
            {"label": "npu",  "file": "npu.tflite",  "use_npu": true},
            {"label": "post", "file": "post.tflite", "use_npu": false}
          ]
        }
    """
    model_path = config["model"]["path"]

    if model_path.endswith(".json"):
        manifest_path = Path(model_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pipeline manifest not found: {model_path}")
        manifest  = json.loads(manifest_path.read_text())
        base_dir  = manifest_path.parent
        stages    = []
        for seg in manifest["pipeline"]:
            seg_file = Path(seg["file"])
            seg_path = str(base_dir / seg_file) if not seg_file.is_absolute() else str(seg_file)
            use_npu  = seg.get("use_npu", False)
            interp, inp, out = _load_single(seg_path, config, use_npu)
            stages.append({"label": seg["label"], "interpreter": interp,
                           "input_details": inp, "output_details": out,
                           "is_npu": use_npu})
            logger.info(
                f"  Stage '{seg['label']}': {Path(seg_path).name} | "
                f"Input: {inp[0]['shape']} | NPU: {use_npu}")
        return stages
    else:
        use_npu  = config["model"].get("use_npu", False)
        interp, inp, out = _load_single(model_path, config, use_npu)
        num_threads = config["model"].get("num_threads", 6)
        logger.info(
            f"Model loaded: {Path(model_path).name} | "
            f"Input shape: {inp[0]['shape']} | "
            f"NPU: {use_npu} | CPU threads: {num_threads}")
        return [{"label": "model", "interpreter": interp,
                 "input_details": inp, "output_details": out,
                 "is_npu": use_npu}]


def invoke_stage(stage: dict, input_tensors: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """
    Run a single pipeline stage. Returns (output_tensors, invoke_ms).

    input_tensors is a list — one array per model input (usually 1, but the
    post-CPU stage after a multi-output NeutronGraph has 6 inputs).
    output_tensors is a list of all output tensors, NOT dequantized.
    invoke_ms is wall time for NPU stages only; 0.0 for CPU stages.
    Each output array is copied so the caller owns its memory.
    """
    interp  = stage["interpreter"]
    inp     = stage["input_details"]
    out_det = stage["output_details"]

    for i, tensor in enumerate(input_tensors):
        interp.set_tensor(inp[i]["index"], tensor)
    t = time.monotonic()
    interp.invoke()
    elapsed_ms = (time.monotonic() - t) * 1000

    outputs = [interp.get_tensor(d["index"]).copy() for d in out_det]
    return outputs, elapsed_ms


def load_model(config: dict):
    """Backward-compatible single-model loader. Returns (interpreter, input_details, output_details)."""
    stages = load_pipeline(config)
    s = stages[0]
    return s["interpreter"], s["input_details"], s["output_details"]


def preprocess_frame(frame, input_details: list) -> np.ndarray:
    """
    Resize and normalize a frame to match the model's expected input.

    Args:
        frame: BGR image as numpy array (from OpenCV)
        input_details: TFLite input tensor details

    Returns:
        input_data: Preprocessed image ready for inference
    """
    import cv2

    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    dtype = input_details[0]['dtype']

    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
    elif dtype == np.int8:
        # Full integer quant model: shift uint8 [0,255] → int8 [-128,127]
        input_data = np.expand_dims(rgb.astype(np.int16) - 128, axis=0).astype(np.int8)
    else:
        input_data = np.expand_dims(rgb / 255.0, axis=0).astype(np.float32)

    return input_data


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Non-Maximum Suppression — pure numpy, no external dependencies.

    Args:
        boxes:         [N, 4] array of [x1, y1, x2, y2] pixel coordinates
        scores:        [N]    array of confidence scores
        iou_threshold: IoU threshold above which overlapping boxes are suppressed

    Returns:
        List of indices of kept boxes, ordered by descending score.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou <= iou_threshold]

    return keep


def run_invoke(interpreter, input_details: list, output_details: list,
               input_data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Feed a preprocessed tensor to the interpreter and return the raw output.

    Returns:
        raw: float32 output tensor [1, 4+num_classes, num_anchors]
        invoke_ms: NPU/CPU invoke wall-time in milliseconds
    """
    interpreter.set_tensor(input_details[0]['index'], input_data)
    t = time.monotonic()
    interpreter.invoke()
    invoke_ms = (time.monotonic() - t) * 1000

    raw = interpreter.get_tensor(output_details[0]['index'])  # [1, 4+C, A]

    out_detail = output_details[0]
    if raw.dtype == np.int8:
        scale, zero_point = out_detail['quantization']
        raw = (raw.astype(np.float32) - zero_point) * scale

    return raw, invoke_ms


# Reusable fixed-size buffers for the two large per-frame allocations.
# Sized on first call and reused every frame to avoid malloc/free jitter.
_buf_max_scores: np.ndarray | None = None
_buf_mask:       np.ndarray | None = None


def postprocess_detections(raw: np.ndarray, frame, config: dict) -> list[dict]:
    """
    Decode raw YOLOv8 output tensor into structured detections.

    Args:
        raw: float32 tensor [1, 4+num_classes, num_anchors]
        frame: original BGR frame (used only for pixel-coordinate scaling)
        config: Full application config dict

    Returns:
        detections: list of {bbox, label_id, confidence}
    """
    global _buf_max_scores, _buf_mask

    threshold     = config["inference"]["confidence_threshold"]
    iou_threshold = config["inference"].get("iou_threshold", 0.45)
    max_det       = config["inference"]["max_detections"]
    frame_h, frame_w = frame.shape[:2]

    # raw: [1, 84, 8400] — work in the original C-contiguous layout to avoid
    # the .T copy (which allocates ~2.8 MB and reshuffles every element).
    pred   = raw[0]       # [84, 8400]
    boxes  = pred[:4, :]  # [4, 8400]  cx, cy, w, h (normalised)
    scores = pred[4:, :]  # [80, 8400]
    n_anchors = scores.shape[1]

    # Preallocate fixed-size buffers once; reuse every frame to avoid the
    # malloc/free jitter that caused occasional latency spikes.
    if _buf_max_scores is None or _buf_max_scores.shape[0] != n_anchors:
        _buf_max_scores = np.empty(n_anchors, dtype=np.float32)
        _buf_mask       = np.empty(n_anchors, dtype=bool)

    np.max(scores, axis=0, out=_buf_max_scores)
    np.greater_equal(_buf_max_scores, threshold, out=_buf_mask)
    mask = _buf_mask
    if not mask.any():
        return []

    # Operate only on the N << 8400 anchors that survive the threshold.
    confidences = _buf_max_scores[mask]       # reuse — no extra indexing needed
    scores_f    = scores[:, mask]             # [80, N]
    label_ids   = scores_f.argmax(axis=0)    # [N]
    boxes_f     = boxes[:, mask]             # [4, N]

    # cx,cy,w,h → x1,y1,x2,y2 in pixel coordinates
    cx, cy = boxes_f[0] * frame_w, boxes_f[1] * frame_h
    hw, hh = boxes_f[2] * frame_w / 2, boxes_f[3] * frame_h / 2
    pixel_boxes = np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=1)

    keep: list[int] = []
    for cls in np.unique(label_ids):
        m   = label_ids == cls
        idx = np.where(m)[0]
        kept = _nms(pixel_boxes[m], confidences[m], iou_threshold)
        keep.extend(idx[kept].tolist())

    keep.sort(key=lambda i: -confidences[i])
    keep = keep[:max_det]

    x1, y1, x2, y2 = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
    return [
        {
            "bbox": [
                int(np.clip(x1[i], 0, frame_w)),
                int(np.clip(y1[i], 0, frame_h)),
                int(np.clip(x2[i], 0, frame_w)),
                int(np.clip(y2[i], 0, frame_h)),
            ],
            "label_id": int(label_ids[i]),
            "confidence": float(confidences[i]),
        }
        for i in keep
    ]


def run_inference(interpreter, input_details: list, output_details: list,
                  frame, config: dict) -> tuple[list[dict], float]:
    """Convenience wrapper: preprocess + invoke + postprocess in one call."""
    input_data = preprocess_frame(frame, input_details)
    raw, invoke_ms = run_invoke(interpreter, input_details, output_details, input_data)
    return postprocess_detections(raw, frame, config), invoke_ms
