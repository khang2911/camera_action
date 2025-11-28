#!/usr/bin/env python3
"""
Python class to read binary detection files and return structured frame/detection data.
"""

import struct
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


@dataclass
class Keypoints:
    """Represents keypoints for pose detection."""
    xy: List  # List of [x, y] pairs
    conf: List  # List of confidence values


@dataclass
class DetectionsOutput:
    """Represents a detection output (for detection models)."""
    bbox: List  # [x_center, y_center, width, height]
    conf: float
    class_id: int
    id: int = None


@dataclass
class PoseOutput:
    """Represents a pose output (for pose models)."""
    bbox: List  # [x_center, y_center, width, height]
    keypoints: Keypoints
    conf: float
    class_id: int
    id: int = None


@dataclass
class Frame:
    """Represents a single frame with its detections."""
    frame_number: int
    detections: List  # List of DetectionsOutput or PoseOutput
    video_index: int
    frame_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary."""
        detections_dict = []
        for det in self.detections:
            if isinstance(det, PoseOutput):
                detections_dict.append({
                    'bbox': det.bbox,
                    'keypoints': {
                        'xy': det.keypoints.xy,
                        'conf': det.keypoints.conf
                    },
                    'conf': det.conf,
                    'class_id': det.class_id,
                    'id': det.id
                })
            else:  # DetectionsOutput
                detections_dict.append({
                    'bbox': det.bbox,
                    'conf': det.conf,
                    'class_id': det.class_id,
                    'id': det.id
                })
        
        return {
            'frame_number': self.frame_number,
            'num_detections': len(self.detections),
            'detections': detections_dict
        }

def expand_bbox(x1, y1, x2, y2, scale=1.2):
    w = x2 - x1
    h = y2 - y1
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * scale
    new_h = h * scale

    nx1 = int(cx - new_w / 2)
    ny1 = int(cy - new_h / 2)
    nx2 = int(cx + new_w / 2)
    ny2 = int(cy + new_h / 2)

    return nx1, ny1, nx2, ny2

def pose_to_bbox(keypoints, score_thresh=0.2):
    xs, ys = [], []
    for (x, y) in keypoints:
        xs.append(x)
        ys.append(y)

    if len(xs) == 0:
        return None  # không có bbox

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return expand_bbox(int(x_min), int(y_min), int(x_max), int(y_max))




class BinFileReader:
    """Reads multiple binary detection files and returns structured frame/detection data."""
    
    def __init__(self, list_filepaths: List[str]):
        """
        Initialize the bin file reader.
        
        Args:
            list_filepaths: List of file paths to binary detection files
        """
        self.filepaths = list_filepaths
        # self.filepaths.reverse()
        # print(self.filepaths)
        self.model_type: Optional[str] = None
        self.frames: List[Frame] = []

        # Check files exist
        for path in self.filepaths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Bin file not found: {path}")

        self.frame_id = 0

    def xywh2xyxy(self, xywh):
        return [xywh[0] - xywh[2]/2, xywh[1] - xywh[3]/2, xywh[0] + xywh[2]/2, xywh[1] + xywh[3]/2]

    def _read_single_file(self, filepath: str, video_index : int):
        """Read a single bin file and append frames to self.frames."""
        with open(filepath, 'rb') as f:
            # Read header (only for first file)
            if self.model_type is None:
                model_type_data = f.read(4)
                if len(model_type_data) < 4:
                    raise ValueError("Invalid bin file: missing header")

                model_type_int = struct.unpack('i', model_type_data)[0]
                self.model_type = "POSE" if model_type_int == 1 else "DETECTION"
            else:
                # Skip header for next files
                f.seek(4)
            
            # Read frames
            while True:
                frame_data = f.read(4)
                if len(frame_data) < 4:
                    break

                frame_number = struct.unpack('i', frame_data)[0]

                num_detections_data = f.read(4)
                if len(num_detections_data) < 4:
                    break

                num_detections = struct.unpack('i', num_detections_data)[0]

                detections = []
                for _ in range(num_detections):
                    bbox_data = f.read(16)
                    if len(bbox_data) < 16:
                        break
                    bbox = struct.unpack('4f', bbox_data)

                    confidence = struct.unpack('f', f.read(4))[0]
                    class_id = struct.unpack('i', f.read(4))[0]

                    num_keypoints = struct.unpack('i', f.read(4))[0]

                    keypoint_xy = []
                    keypoint_conf = []

                    for _ in range(num_keypoints):
                        kpt_data = f.read(12)
                        if len(kpt_data) < 12:
                            break

                        x, y, c = struct.unpack('3f', kpt_data)
                        keypoint_xy.append([x, y])
                        keypoint_conf.append(c)

                    if num_keypoints > 0:
                        kp = Keypoints(xy=keypoint_xy, conf=keypoint_conf)
                        # if bbox[2]/bbox[3] < 1.5:
                        # if confidence > 0.6:
                        # bbox = pose_to_bbox(keypoint_xy)
                        detections.append(
                            PoseOutput(bbox=self.xywh2xyxy(bbox), keypoints=kp, conf=confidence, class_id=class_id, id=None)
                        )
                    else:
                        detections.append(
                            DetectionsOutput(bbox=self.xywh2xyxy(bbox), conf=confidence, class_id=class_id, id=None)
                        )
                # print(frame_number)
                self.frames.append(Frame(frame_number=frame_number, detections=detections, video_index=video_index))

    def read(self) -> List[Frame]:
        """Read all files in the list and return frames."""
        self.frames = []
        self.model_type = None

        for i, filepath in enumerate(self.filepaths):
            self._read_single_file(filepath, video_index=i)
        
        sorted_frame = sorted(self.frames, key=lambda x : (x.video_index, x.frame_number))

        frame_id = 0
        prev_frame = None
        prev_video = None

        for frame in sorted_frame:
            if prev_video is None:
                # frame đầu tiên
                frame.frame_index = frame_id
            else:
                if frame.video_index == prev_video:
                    # CÙNG VIDEO → giữ khoảng nhảy gốc
                    frame_id += frame.frame_number - prev_frame
                else:
                    # VIDEO MỚI → không dùng frame_number để tính
                    frame_id += 1   # chỉ cộng 1 frame
                frame.frame_index = frame_id

            prev_video = frame.video_index
            prev_frame = frame.frame_number

        return sorted_frame

    def get_frames(self) -> List[Frame]:
        if not self.frames:
            self.read()
        return self.frames

    def get_model_type(self) -> Optional[str]:
        if not self.model_type:
            self.read()
        return self.model_type

    def get_frame_by_number(self, frame_number: int) -> Optional[Frame]:
        if not self.frames:
            self.read()
        for frame in self.frames:
            if frame.frame_number == frame_number:
                return frame
        return None

    def get_all_detections(self) -> List:
        if not self.frames:
            self.read()
        all_det = []
        for frame in self.frames:
            all_det.extend(frame.detections)
        return all_det

    def to_dict(self) -> Dict[str, Any]:
        if not self.frames:
            self.read()
        return {
            "model_type": self.model_type,
            "num_frames": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames],
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) < 2:
    #     print("Usage: python3 bin_file_reader.py <binary_file.bin>")
    #     sys.exit(1)
    
    filepaths = [
    "/age_gender/detection/camera_action/camera_action/test/alcohol/25-11-25/c03o25010003774_6fba14bc-b693-49ed-90fd-8ba74430176a.bin",
    "/age_gender/detection/camera_action/camera_action/test/alcohol/25-11-25/c03o25010003774_6fba14bc-b693-49ed-90fd-8ba74430176a_v1.bin"
  ]
    
    try:
        # Create reader and read file
        reader = BinFileReader(filepaths)
        frames = reader.read()
        
        print(f"Model Type: {reader.get_model_type()}")
        print(f"Total Frames: {len(frames)}")
        print(f"Total Detections: {sum(len(frame.detections) for frame in frames)}")
        print("\n" + "=" * 60)
        
        # # Example: Access frames and detections
        prev_frame = None
        for frame in frames:
            print(f"Frame {frame.frame_index}: {len(frame.detections)} detections")
            # if prev_frame:
            #     if frame.frame_number - prev_frame > 2:
            #         print(frame.frame_number, prev_frame)
            
            # prev_frame = frame.frame_number
            # for i, detection in enumerate(frame.detections):
            #     print(f"  Detection {i+1}:")
            #     print(f"    BBox: x1={detection.bbox[0]:.2f}, y1={detection.bbox[1]:.2f}, "
            #           f"x2={detection.bbox[2]:.2f}, y2={detection.bbox[3]:.2f}")
            #     print(f"    Confidence: {detection.conf:.4f}")
            #     print(f"    Class ID: {detection.class_id}")
            #     if isinstance(detection, PoseOutput):
            #         print(f"    Keypoints: {len(detection.keypoints.xy)} keypoints")
            #         for k, (xy, conf) in enumerate(zip(detection.keypoints.xy, detection.keypoints.conf)):
            #             if k < len(KEYPOINT_NAMES):
            #                 print(f"      {KEYPOINT_NAMES[k]}: ({xy[0]:.2f}, {xy[1]:.2f}) conf={conf:.3f}")
        
        # Example: Get specific frame
        if frames:
            first_frame = frames[0]
            print(f"\nFirst frame ({first_frame.frame_number}) has {len(first_frame.detections)} detections")
            
            if first_frame.detections:
                first_detection = first_frame.detections[0]
                print(f"First detection bbox: {first_detection.bbox}")
                if isinstance(first_detection, PoseOutput):
                    print(f"First detection has {len(first_detection.keypoints.xy)} keypoints")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

