#!/usr/bin/env python3
"""
TensorRT inference script that mirrors the C++ pipeline.
Supports single-image or video inference with batching, tensor dumps, and visualization.
"""

import argparse
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  # Initializes CUDA context
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


def dump_array(array: np.ndarray, out_dir: str, prefix: str, batch_idx: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_batch{batch_idx:06d}.bin")
    array.astype(np.float32).tofile(path)
    LOGGER.info("Dumped %s", path)
    return path


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

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb.shape[:2]

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


def preprocess_batch(
    frames_info: List[Tuple[Optional[int], np.ndarray]],
    input_w: int,
    input_h: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]], List[Tuple[int, int]], List[Optional[int]], int]:
    """
    Preprocess a batch of frames. frames_info is a list of (frame_idx, frame).
    Returns tensors [batch,3,H,W], scales, pads, orig_sizes, frame_indices, valid_count.
    """
    tensors = np.zeros((batch_size, 3, input_h, input_w), dtype=np.float32)
    scales: List[float] = []
    pads: List[Tuple[int, int]] = []
    orig_sizes: List[Tuple[int, int]] = []
    frame_indices: List[Optional[int]] = []
    valid = len(frames_info)

    for i, (frame_idx, frame) in enumerate(frames_info):
        tensor, scale, pad = preprocess_frame(frame, input_w, input_h)
        tensors[i] = tensor[0]
        scales.append(scale)
        pads.append(pad)
        orig_sizes.append((frame.shape[1], frame.shape[0]))
        frame_indices.append(frame_idx)

    while len(scales) < batch_size:
        scales.append(1.0)
        pads.append((0, 0))
        orig_sizes.append((0, 0))
        frame_indices.append(None)

    return tensors, scales, pads, orig_sizes, frame_indices, valid


def dims_to_tuple(dims: trt.Dims) -> Tuple[int, ...]:
    return tuple(dims.d[i] for i in range(dims.nbDims))


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
) -> List[dict]:
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
    parser = argparse.ArgumentParser(description="TensorRT inference matching the C++ pipeline.")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--image", help="Path to input image (BGR)")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (inclusive, -1 = till end)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Early stop after processing this many frames")
    parser.add_argument("--input-width", type=int, required=True, help="Model input width")
    parser.add_argument("--input-height", type=int, required=True, help="Model input height")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (<= engine max)")
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--save-frames", help="Directory to save frames with detections drawn")
    parser.add_argument("--dump-input-dir", help="Directory to dump input tensors per batch")
    parser.add_argument("--dump-output-dir", help="Directory to dump raw TensorRT outputs per batch")

    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        parser.error("Please specify exactly one of --image or --video.")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    batch_size = args.batch_size
    context.set_binding_shape(0, (batch_size, 3, args.input_height, args.input_width))

    inputs, outputs, bindings, stream = allocate_buffers(engine, context)
    output_index = outputs[0]["index"]
    output_dims = dims_to_tuple(context.get_binding_shape(output_index))
    if output_dims[0] in (-1, 0):
        output_dims = (batch_size,) + output_dims[1:]
    output_shape = output_dims

    os.makedirs(args.save_frames, exist_ok=True) if args.save_frames else None
    batch_counter = 0

    def process_batch(frames_info: List[Tuple[Optional[int], np.ndarray]], batch_idx: int) -> int:
        if not frames_info:
            return 0
        tensors, scales, pads, orig_sizes, frame_indices, valid = preprocess_batch(
            frames_info, args.input_width, args.input_height, batch_size)
        np.copyto(inputs[0]["host"], tensors.ravel())
        raw_outputs = do_inference(context, bindings, inputs, outputs, stream)
        output = raw_outputs[0].reshape(output_shape)
        if args.dump_input_dir:
            dump_array(tensors, args.dump_input_dir, "inputs", batch_idx)
        if args.dump_output_dir:
            dump_array(output, args.dump_output_dir, "outputs", batch_idx)

        for i in range(valid):
            frame_idx, frame = frames_info[i]
            detections = postprocess(
                output[i:i+1],
                conf_thresh=args.conf_threshold,
                nms_thresh=args.nms_threshold,
                scale=scales[i],
                pad=pads[i],
                orig_size=orig_sizes[i],
            )
            LOGGER.info("Frame %s (batch %d idx %d): %d detections",
                        frame_idx, batch_idx, i, len(detections))
            for det in detections:
                LOGGER.info("  bbox=%s score=%.3f", det["bbox"], det["score"])
            if args.save_frames and detections and frame_idx is not None:
                out_path = os.path.join(args.save_frames, f"frame_{frame_idx:06d}.jpg")
                draw = draw_detections(frame.copy(), detections)
                cv2.imwrite(out_path, draw)
                LOGGER.info("Saved visualization: %s", out_path)
        return valid

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(args.image)
        process_batch([(0, frame)], batch_counter)
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    LOGGER.info("Processing video %s (%s frames)", args.video, total_frames if total_frames >= 0 else "unknown")

    start_frame = max(0, args.start_frame)
    end_frame = args.end_frame if args.end_frame >= 0 else (total_frames - 1 if total_frames > 0 else None)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_processed = 0
    frames_batch: List[Tuple[int, np.ndarray]] = []

    while True:
        if args.max_frames > 0 and total_processed >= args.max_frames:
            LOGGER.info("Reached max frames limit (%d)", args.max_frames)
            break

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if end_frame is not None and current_pos > end_frame:
            LOGGER.info("Reached end frame (%d)", end_frame)
            break

        ret, frame = cap.read()
        if not ret:
            LOGGER.info("End of video or failed to read frame")
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            break

        frames_batch.append((frame_idx, frame))
        if len(frames_batch) == batch_size:
            total_processed += process_batch(frames_batch, batch_counter)
            frames_batch = []
            batch_counter += 1

        if args.max_frames > 0 and total_processed >= args.max_frames:
            break

    if frames_batch and (args.max_frames <= 0 or total_processed < args.max_frames):
        total_processed += process_batch(frames_batch, batch_counter)

    LOGGER.info("Total frames processed: %d", total_processed)
    cap.release()


if __name__ == "__main__":
    main()

