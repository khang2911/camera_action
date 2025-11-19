#!/usr/bin/env python3
"""
Verify YOLO post-processing by replaying raw TensorRT outputs (from dumps)
and comparing them with the detections saved by the C++ pipeline.

Typical workflow:
  1. Run `run_cross_check.py --compare-target outputs ...` to dump tensors
  2. Run this script, pointing to the dumped `outputs_batchXXXXXX.bin`
     files and (optionally) a C++ binary results file.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    frame: int
    confidence: float
    bbox_cxcywh: Tuple[float, float, float, float]
    bbox_xyxy: Tuple[float, float, float, float]


def cxcywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    half_w = w / 2.0
    half_h = h / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def xyxy_to_cxcywh(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def list_output_bins(path: str) -> List[str]:
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "outputs_batch*.bin")
        files = glob.glob(pattern, recursive=True)
    else:
        files = [path]
    files.sort()
    return files


# ---------------------------------------------------------------------------
# YOLO-style post-processing (single-class detection)
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(union)
        mask = union > 0
        iou[mask] = inter[mask] / union[mask]

        order = order[1:][iou <= threshold]

    return keep


def scale_to_original(
    xyxy: np.ndarray,
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
) -> np.ndarray:
    if orig_w <= 0 or orig_h <= 0:
        return xyxy.copy()

    scale = min(input_w / orig_w, input_h / orig_h)
    pad_w = (input_w - orig_w * scale) / 2.0
    pad_h = (input_h - orig_h * scale) / 2.0

    x1 = (xyxy[:, 0] - pad_w) / scale
    y1 = (xyxy[:, 1] - pad_h) / scale
    x2 = (xyxy[:, 2] - pad_w) / scale
    y2 = (xyxy[:, 3] - pad_h) / scale

    clamped = np.stack(
        [
            np.clip(x1, 0, orig_w),
            np.clip(y1, 0, orig_h),
            np.clip(x2, 0, orig_w),
            np.clip(y2, 0, orig_h),
        ],
        axis=1,
    )
    return clamped


def postprocess_frame(
    predictions: np.ndarray,
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
    conf_threshold: float,
    nms_threshold: float,
) -> List[DetectionRecord]:
    # predictions shape: [num_anchors, channels]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    if boxes.size == 0:
        return []

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

    keep = nms(xyxy, scores, nms_threshold)
    if not keep:
        return []

    xyxy_kept = xyxy[keep]
    scores_kept = scores[keep]

    xyxy_orig = scale_to_original(xyxy_kept, orig_w, orig_h, input_w, input_h)

    detections: List[DetectionRecord] = []
    for idx, bbox_xyxy in zip(keep, xyxy_orig):
        bbox_cxcywh = xyxy_to_cxcywh(bbox_xyxy)
        detections.append(
            DetectionRecord(
                frame=-1,  # filled later
                confidence=float(scores_kept[len(detections)]),
                bbox_cxcywh=bbox_cxcywh,
                bbox_xyxy=tuple(map(float, bbox_xyxy)),
            )
        )
    return detections


# ---------------------------------------------------------------------------
# C++ binary output parsing
# ---------------------------------------------------------------------------

def parse_cpp_binary(path: str) -> Tuple[int, Dict[int, List[DetectionRecord]]]:
    results: Dict[int, List[DetectionRecord]] = {}
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            raise RuntimeError(f"{path}: empty file")
        model_type = struct.unpack("<i", header)[0]

        while True:
            frame_bytes = f.read(4)
            if len(frame_bytes) < 4:
                break
            frame_idx = struct.unpack("<i", frame_bytes)[0]

            num_det_bytes = f.read(4)
            if len(num_det_bytes) < 4:
                break
            num_dets = struct.unpack("<i", num_det_bytes)[0]

            dets: List[DetectionRecord] = []
            for _ in range(num_dets):
                bbox = struct.unpack("<4f", f.read(16))
                conf = struct.unpack("<f", f.read(4))[0]
                class_id = struct.unpack("<i", f.read(4))[0]
                num_keypoints = struct.unpack("<i", f.read(4))[0]
                # Skip keypoints data if present (pose models)
                for _ in range(num_keypoints):
                    f.read(12)  # 3 floats

                det = DetectionRecord(
                    frame=frame_idx,
                    confidence=conf,
                    bbox_cxcywh=bbox,
                    bbox_xyxy=cxcywh_to_xyxy(bbox),
                )
                dets.append(det)
            results[frame_idx] = dets

    return model_type, results


# ---------------------------------------------------------------------------
# Matching / comparison helpers
# ---------------------------------------------------------------------------

def match_detections(
    left: List[DetectionRecord],
    right: List[DetectionRecord],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    matches: List[Tuple[int, int, float]] = []
    unmatched_left = list(range(len(left)))
    unmatched_right = list(range(len(right)))

    used_right: set[int] = set()
    for li, det_left in enumerate(left):
        best_j = -1
        best_iou = 0.0
        for rj, det_right in enumerate(right):
            if rj in used_right:
                continue
            iou = compute_iou(det_left.bbox_xyxy, det_right.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = rj

        if best_j != -1 and best_iou >= iou_threshold:
            matches.append((li, best_j, best_iou))
            used_right.add(best_j)

    matched_left = {li for li, _, _ in matches}
    matched_right = {rj for _, rj, _ in matches}

    unmatched_left = [i for i in range(len(left)) if i not in matched_left]
    unmatched_right = [i for i in range(len(right)) if i not in matched_right]

    return matches, unmatched_left, unmatched_right


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_raw_outputs(
    files: List[str],
    frames: int,
    batch_size: int,
    output_channels: int,
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
    conf_threshold: float,
    nms_threshold: float,
    start_frame: int,
) -> Dict[int, List[DetectionRecord]]:
    frame_results: Dict[int, List[DetectionRecord]] = {}
    frames_remaining = frames
    frame_idx = start_frame

    for path in files:
        if frames_remaining <= 0:
            break

        data = np.fromfile(path, dtype=np.float32)
        if data.size == 0:
            continue

        per_batch = data.size // batch_size
        if data.size % batch_size != 0:
            raise ValueError(
                f"{path}: data size {data.size} not divisible by batch_size {batch_size}"
            )
        if per_batch % output_channels != 0:
            raise ValueError(
                f"{path}: batch slice {per_batch} not divisible by channels {output_channels}"
            )
        num_anchors = per_batch // output_channels

        reshaped = data.reshape(batch_size, output_channels, num_anchors)
        frames_in_this_file = min(batch_size, frames_remaining)

        for b in range(frames_in_this_file):
            predictions = reshaped[b].T  # [num_anchors, channels]
            dets = postprocess_frame(
                predictions,
                orig_w,
                orig_h,
                input_w,
                input_h,
                conf_threshold,
                nms_threshold,
            )

            for det in dets:
                det.frame = frame_idx

            frame_results[frame_idx] = dets
            frame_idx += 1
            frames_remaining -= 1

    if frames_remaining > 0:
        print(
            f"[WARN] Requested {frames} frames but only recovered {frames - frames_remaining}"
        )

    return frame_results


def summarize_results(results: Dict[int, List[DetectionRecord]], label: str) -> None:
    total = sum(len(v) for v in results.values())
    print(f"[INFO] {label}: {len(results)} frames, {total} detections in total")
    preview_frames = sorted(results.keys())[:3]
    for frame in preview_frames:
        dets = results[frame]
        print(f"  Frame {frame}: {len(dets)} detections")
        for det in dets[:5]:
            cx, cy, w, h = det.bbox_cxcywh
            print(
                f"    conf={det.confidence:.4f}, "
                f"cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}"
            )


def compare_with_cpp(
    python_results: Dict[int, List[DetectionRecord]],
    cpp_results: Dict[int, List[DetectionRecord]],
    iou_threshold: float,
) -> None:
    frames = sorted(python_results.keys())
    total_matches = 0
    total_python = 0
    total_cpp = 0

    for frame in frames:
        py = python_results.get(frame, [])
        cpp = cpp_results.get(frame, [])
        total_python += len(py)
        total_cpp += len(cpp)

        if not cpp:
            print(f"[WARN] Frame {frame}: no C++ detections found")
            continue

        matches, unmatched_py, unmatched_cpp = match_detections(
            py, cpp, iou_threshold
        )
        total_matches += len(matches)

        print(
            f"[Frame {frame}] python={len(py)} dets, cpp={len(cpp)} dets, "
            f"matches={len(matches)}, "
            f"py_only={len(unmatched_py)}, cpp_only={len(unmatched_cpp)}"
        )

        for li, rj, iou in matches[:3]:
            det_py = py[li]
            det_cpp = cpp[rj]
            conf_diff = abs(det_py.confidence - det_cpp.confidence)
            print(
                f"    match iou={iou:.3f}, "
                f"conf_py={det_py.confidence:.3f}, "
                f"conf_cpp={det_cpp.confidence:.3f}, "
                f"|Δconf|={conf_diff:.3f}"
            )

        if unmatched_py:
            print(
                f"    python-only detections (first 2): "
                f"{len(unmatched_py)} -> "
                + ", ".join(
                    f"conf={py[idx].confidence:.3f}" for idx in unmatched_py[:2]
                )
            )
        if unmatched_cpp:
            print(
                f"    cpp-only detections (first 2): "
                f"{len(unmatched_cpp)} -> "
                + ", ".join(
                    f"conf={cpp[idx].confidence:.3f}" for idx in unmatched_cpp[:2]
                )
            )

    if total_python == 0 and total_cpp == 0:
        print("[INFO] No detections to compare")
        return

    precision = total_matches / total_cpp if total_cpp else 0.0
    recall = total_matches / total_python if total_python else 0.0

    print("\n=== Summary ===")
    print(f"Total python detections : {total_python}")
    print(f"Total C++ detections    : {total_cpp}")
    print(f"Matched detections      : {total_matches}")
    print(f"Match precision (CPP)   : {precision:.3f}")
    print(f"Match recall   (Python) : {recall:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify YOLO postprocess using dumped raw outputs"
    )
    parser.add_argument(
        "--raw-output",
        required=True,
        help="Path to outputs_batch*.bin (file or directory)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Number of real frames contained in the dumps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size used during dumping",
    )
    parser.add_argument(
        "--output-channels",
        type=int,
        default=5,
        help="Number of output channels (5 for detection, 56 for pose)",
    )
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--orig-width", type=int, required=True)
    parser.add_argument("--orig-height", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument(
        "--cpp-bin",
        help="Optional path to the C++ binary detections file for comparison",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching python vs C++ detections",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = list_output_bins(args.raw_output)
    if not files:
        raise FileNotFoundError(
            f"No outputs_batch*.bin files found under {args.raw_output}"
        )

    python_results = process_raw_outputs(
        files=files,
        frames=args.frames,
        batch_size=args.batch_size,
        output_channels=args.output_channels,
        orig_w=args.orig_width,
        orig_h=args.orig_height,
        input_w=args.input_width,
        input_h=args.input_height,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        start_frame=args.start_frame,
    )
    summarize_results(python_results, label="Reconstructed (Python)")

    if args.cpp_bin:
        model_type, cpp_results = parse_cpp_binary(args.cpp_bin)
        print(f"[INFO] Parsed C++ binary ({args.cpp_bin}), model_type={model_type}")
        summarize_results(cpp_results, label="C++ binary")
        compare_with_cpp(python_results, cpp_results, args.match_iou)


if __name__ == "__main__":
    main()

