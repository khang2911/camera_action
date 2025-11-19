#!/usr/bin/env python3
"""
Minimal TensorRT inference script that mirrors the C++ detector pipeline.
Usage:
    python3 trt_infer.py --engine hand.engine --image sample.jpg \
                         --input-width 864 --input-height 864 \
                         --conf-threshold 0.2 --nms-threshold 0.55
"""

import argparse
import logging
import os
from typing import Tuple, List

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import pycuda.driver as cuda
import tensorrt as trt


LOGGER = logging.getLogger("trt_infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def preprocess_image(
    image_path: str,
    input_w: int,
    input_h: int,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize with padding (fill=114), convert BGR->RGB, normalize, CHW."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_h, img_w = img.shape[:2]

    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (input_w - new_w) // 2
    pad_h = (input_h - new_h) // 2

    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)[None, ...]  # shape: [1,3,H,W]
    return chw, scale, (pad_h, pad_w)


def allocate_buffers(engine: trt.ICudaEngine):
    """Allocate host/device buffers for all bindings."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        idx = engine.get_binding_index(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        volume = trt.volume(engine.get_binding_shape(idx))
        host_mem = cuda.pagelocked_empty(volume, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        io = {"host": host_mem, "device": device_mem, "name": binding, "index": idx}
        if engine.binding_is_input(binding):
            inputs.append(io)
        else:
            outputs.append(io)

    return inputs, outputs, bindings, stream


def do_inference(
    context: trt.IExecutionContext,
    bindings,
    inputs,
    outputs,
    stream,
):
    """Copy input to device, execute, copy output back."""
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()
    return [out["host"] for out in outputs]


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    """Simple NMS in xyxy format."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return keep


def postprocess(
    output: np.ndarray,
    conf_thresh: float,
    nms_thresh: float,
    scale: float,
    pad: Tuple[int, int],
    orig_size: Tuple[int, int],
):
    """
    Output shape: [1, channels, num_anchors] (from TensorRT engine).
    Convert to [num_anchors, channels], then decode xywh/conf.
    """
    batch, channels, num_preds = output.shape
    assert batch == 1

    preds = output[0].T  # [num_preds, channels]
    boxes = preds[:, :4]
    scores = preds[:, 4]

    mask = scores > conf_thresh
    boxes, scores = boxes[mask], scores[mask]
    if boxes.size == 0:
        return []

    # Convert xywh in padded space to xyxy in original image space
    pad_h, pad_w = pad
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

    keep = nms(boxes_xyxy, scores, nms_thresh)
    detections = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(scores[idx]),
            "class_id": 0,
        })
    return detections


def main():
    parser = argparse.ArgumentParser(description="TensorRT inference script matching C++ pipeline.")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--image", required=True, help="Path to input image (BGR)")
    parser.add_argument("--input-width", type=int, required=True, help="Model input width")
    parser.add_argument("--input-height", type=int, required=True, help="Model input height")
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.55, help="NMS IoU threshold")

    args = parser.parse_args()

    LOGGER.info("Loading TensorRT engine: %s", args.engine)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, 3, args.input_height, args.input_width))

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    tensor, scale, pad = preprocess_image(args.image, args.input_width, args.input_height)
    orig_img = cv2.imread(args.image)
    orig_h, orig_w = orig_img.shape[:2]

    np.copyto(inputs[0]["host"], tensor.ravel())
    raw_outputs = do_inference(context, bindings, inputs, outputs, stream)

    # Assume single output
    output = raw_outputs[0].reshape(1, engine.get_binding_shape(1)[1], engine.get_binding_shape(1)[2])
    detections = postprocess(
        output,
        conf_thresh=args.conf_threshold,
        nms_thresh=args.nms_threshold,
        scale=scale,
        pad=pad,
        orig_size=(orig_w, orig_h),
    )

    LOGGER.info("Detections: %s", detections if detections else "None")


if __name__ == "__main__":
    main()

