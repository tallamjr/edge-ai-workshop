#!/usr/bin/env python3
"""Reproduce + guard the multi-output NPU -> multi-input CPU stage handoff.

The backbone (NPU stage) emits 3 feature maps whose ORDER is chosen by the
converter, not by us. The head (CPU stage) expects them in its own order.
inference.invoke_stage must route each tensor to the input slot with the
matching shape, NOT by position, or the head receives swapped feature maps.

This test fails if invoke_stage chains positionally (the pre-fix behaviour).
"""
import pathlib
import sys

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board"))
import inference  # noqa: E402

BACKBONE = ROOT / "models/work/split_pose/backbone_int8.tflite"
HEAD = ROOT / "models/work/split_pose/head_tf/head_float32.tflite"


def _stage(path, use_npu=False):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return {"label": path.stem, "interpreter": it, "is_npu": use_npu,
            "input_details": it.get_input_details(),
            "output_details": it.get_output_details()}


def _reference_shape_matched(backbone, head, x_int8):
    """Ground truth: route feature maps to head inputs by shape."""
    bi = backbone["input_details"][0]
    backbone["interpreter"].set_tensor(bi["index"], x_int8)
    backbone["interpreter"].invoke()
    feats = {tuple(o["shape"]): backbone["interpreter"].get_tensor(o["index"])
             for o in backbone["output_details"]}
    for hi in head["input_details"]:
        head["interpreter"].set_tensor(hi["index"], feats[tuple(hi["shape"])])
    head["interpreter"].invoke()
    return head["interpreter"].get_tensor(head["output_details"][0]["index"]).copy()


def main():
    backbone = _stage(BACKBONE, use_npu=True)
    head = _stage(HEAD, use_npu=False)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(1, 640, 640, 3), dtype=np.int16)
    x_int8 = (img - 128).astype(np.int8)

    ref = _reference_shape_matched(_stage(BACKBONE, True), _stage(HEAD), x_int8)

    # Drive the real board path: invoke_stage(backbone) -> invoke_stage(head).
    feats, _ = inference.invoke_stage(backbone, [x_int8])
    out, _ = inference.invoke_stage(head, feats)

    max_abs = float(np.abs(out - ref).max())
    print(f"backbone out order: {[tuple(o['shape']) for o in backbone['output_details']]}")
    print(f"head    in  order: {[tuple(i['shape']) for i in head['input_details']]}")
    print(f"max|invoke_stage - shape_matched_reference| = {max_abs:.6f}")
    if max_abs > 1e-3:
        print("FAIL: invoke_stage routed feature maps to the wrong head inputs.")
        sys.exit(1)
    print("PASS: invoke_stage routes multi-output -> multi-input by shape.")


if __name__ == "__main__":
    main()
