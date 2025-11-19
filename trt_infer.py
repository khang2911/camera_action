#!/usr/bin/env python3
"""
TensorRT inference script that mirrors the C++ detector pipeline.
Supports single-image or video inference with optional frame ranges.

Examples:
  Single image:
    python3 trt_infer.py --engine hand.engine --image sample.jpg \
                         --input-width 864 --input-height 864

  Video frames 100-200:
    python3 trt_infer.py --engine hand.engine --video sample.mp4 \
                         --input-width 864 --input-height 864 \
                         --start-frame 100 --end-frame 200
"""

import argparse
import logging
import os
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import pycuda.driver as cuda
import tensorrt as trt


LOGGER = logging.getLogger("trt_infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def draw_detections(frame: np.ndarray, detections: List[dict], color=(0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes and scores on a frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{score:.2f}", (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame
def preprocess_frame(
    frame_bgr: np.ndarray,
    input_w: int,
    input_h: int,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize with padding (fill=0), convert BGR->RGB, normalize, CHW.
    Matches the current C++ preprocessing pipeline.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise RuntimeError("Empty frame provided to preprocess_frame")

    img_h, img_w = frame_bgr.shape[:2]
    # Convert to RGB first, then letterbox resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (input_w - new_w) // 2
    pad_h = (input_h - new_h) // 2

    padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    normalized = padded.astype(np.float32) / 255.0
    chw = normalized.transpose(2, 0, 1)[None, ...]  # shape: [1,3,H,W]
    return chw, scale, (pad_h, pad_w)


def allocate_buffers(engine: trt.ICudaEngine, context: trt.IExecutionContext):
    """Allocate host/device buffers for all bindings."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        idx = engine.get_binding_index(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = context.get_binding_shape(idx)
        volume = trt.volume(shape)
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
    parser.add_argument("--image", help="Path to input image (BGR)")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (inclusive, -1 = till end)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Early stop after processing this many frames (per video)")
    parser.add_argument("--save-frames", help="Directory to save frames with detections drawn")
    parser.add_argument("--input-width", type=int, required=True, help="Model input width")
    parser.add_argument("--input-height", type=int, required=True, help="Model input height")
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.55, help="NMS IoU threshold")

    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        parser.error("Please specify exactly one of --image or --video.")

    LOGGER.info("Loading TensorRT engine: %s", args.engine)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, 3, args.input_height, args.input_width))

    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(args.image)
        frame = cv2.imread(args.image)
        tensor, scale, pad = preprocess_frame(frame, args.input_width, args.input_height)
        np.copyto(inputs[0]["host"], tensor.ravel())
        raw_outputs = do_inference(context, bindings, inputs, outputs, stream)
        out_shape = tuple(context.get_binding_shape(outputs[0]["index"]))
        output = raw_outputs[0].reshape(out_shape)
        detections = postprocess(
            output,
            conf_thresh=args.conf_threshold,
            nms_thresh=args.nms_threshold,
            scale=scale,
            pad=pad,
            orig_size=(frame.shape[1], frame.shape[0]),
        )
        LOGGER.info("Detections: %s", detections if detections else "None")
        if detections and args.save_frames:
            os.makedirs(args.save_frames, exist_ok=True)
            draw_frame = draw_detections(frame.copy(), detections)
            out_path = os.path.join(args.save_frames, "frame_image.jpg")
            cv2.imwrite(out_path, draw_frame)
            LOGGER.info("Saved visualization: %s", out_path)
        return

    # Video path
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    LOGGER.info("Processing video %s (%s frames)", args.video, total_frames if total_frames >= 0 else "unknown")

    # Seek to start frame if needed
    start_frame = max(0, args.start_frame)
    end_frame = args.end_frame if args.end_frame >= 0 else (total_frames - 1 if total_frames > 0 else None)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed = 0
    frame_outputs = []
    while True:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if end_frame is not None and frame_idx > end_frame:
            LOGGER.info("Reached end frame (%d)", end_frame)
            break
        if args.max_frames > 0 and processed >= args.max_frames:
            LOGGER.info("Reached max frames limit (%d)", args.max_frames)
            break

        ret, frame = cap.read()
        if not ret:
            LOGGER.info("End of video or failed to read frame")
            break

        tensor, scale, pad = preprocess_frame(frame, args.input_width, args.input_height)
        np.copyto(inputs[0]["host"], tensor.ravel())
        raw_outputs = do_inference(context, bindings, inputs, outputs, stream)
        out_shape = tuple(context.get_binding_shape(outputs[0]["index"]))
        output = raw_outputs[0].reshape(out_shape)
        detections = postprocess(
            output,
            conf_thresh=args.conf_threshold,
            nms_thresh=args.nms_threshold,
            scale=scale,
            pad=pad,
            orig_size=(frame.shape[1], frame.shape[0]),
        )
        LOGGER.info("Frame %d: %d detections", frame_idx, len(detections))
        for det in detections:
            LOGGER.info("  bbox=%s score=%.3f", det["bbox"], det["score"])

        processed += 1
        frame_outputs.append((frame_idx, frame, detections))
        if args.save_frames and detections:
            os.makedirs(args.save_frames, exist_ok=True)
            out_path = os.path.join(args.save_frames, f"frame_{frame_idx:06d}.jpg")
            draw_frame = draw_detections(frame.copy(), detections)
            cv2.imwrite(out_path, draw_frame)
            LOGGER.info("Saved visualization: %s", out_path)

    cap.release()


if __name__ == "__main__":
    main()

