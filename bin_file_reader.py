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


class BinFileReader:
    """Reads binary detection files and returns structured frame/detection data."""
    
    def __init__(self, filepath: str):
        """
        Initialize the bin file reader.
        
        Args:
            filepath: Path to the binary detection file
        """
        self.filepath = filepath
        self.model_type: Optional[str] = None
        self.frames: List[Frame] = []
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bin file not found: {filepath}")
    
    def read(self) -> List[Frame]:
        """
        Read the bin file and return a list of Frame objects.
        
        Returns:
            List[Frame]: List of Frame objects, each containing a list of Detection objects
        """
        self.frames = []
        
        with open(self.filepath, 'rb') as f:
            # Read header (model type - written once)
            model_type_data = f.read(4)
            if len(model_type_data) < 4:
                raise ValueError("Invalid bin file: missing header")
            
            model_type_int = struct.unpack('i', model_type_data)[0]
            self.model_type = "POSE" if model_type_int == 1 else "DETECTION"
            
            # Read frames until end of file
            while True:
                # Try to read frame number
                frame_data = f.read(4)
                if len(frame_data) < 4:
                    break  # End of file
                
                frame_number = struct.unpack('i', frame_data)[0]
                
                # Read number of detections
                num_detections_data = f.read(4)
                if len(num_detections_data) < 4:
                    # CRITICAL: We've already read the frame number, so the file is in an inconsistent state
                    # Raising an exception prevents silent corruption and makes the error clear
                    raise ValueError(f"Incomplete frame header at frame {frame_number}: expected 4 bytes for num_detections, got {len(num_detections_data)}. File may be corrupted or truncated.")
                
                num_detections = struct.unpack('i', num_detections_data)[0]
                
                # Read all detections for this frame
                detections = []
                for i in range(num_detections):
                    # Read bbox (4 floats: x_center, y_center, width, height)
                    bbox_data = f.read(4 * 4)
                    if len(bbox_data) < 16:
                        raise ValueError(f"Incomplete bbox data at frame {frame_number}, detection {i}: expected 16 bytes, got {len(bbox_data)}")
                    bbox = struct.unpack('4f', bbox_data)
                    
                    # Read confidence and class_id
                    confidence_data = f.read(4)
                    class_id_data = f.read(4)
                    if len(confidence_data) < 4:
                        raise ValueError(f"Incomplete confidence data at frame {frame_number}, detection {i}: expected 4 bytes, got {len(confidence_data)}")
                    if len(class_id_data) < 4:
                        raise ValueError(f"Incomplete class_id data at frame {frame_number}, detection {i}: expected 4 bytes, got {len(class_id_data)}")
                    
                    confidence = struct.unpack('f', confidence_data)[0]
                    class_id = struct.unpack('i', class_id_data)[0]
                    
                    # Read number of keypoints
                    num_keypoints_data = f.read(4)
                    if len(num_keypoints_data) < 4:
                        raise ValueError(f"Incomplete num_keypoints data at frame {frame_number}, detection {i}: expected 4 bytes, got {len(num_keypoints_data)}")
                    num_keypoints = struct.unpack('i', num_keypoints_data)[0]
                    
                    # Read keypoints
                    keypoint_xy = []
                    keypoint_conf = []
                    for k in range(num_keypoints):
                        kpt_data = f.read(3 * 4)
                        if len(kpt_data) < 12:
                            raise ValueError(f"Incomplete keypoint data at frame {frame_number}, detection {i}, keypoint {k}: expected 12 bytes, got {len(kpt_data)}")
                        x, y, conf = struct.unpack('3f', kpt_data)
                        keypoint_xy.append([x, y])
                        keypoint_conf.append(conf)
                    
                    # Create detection object based on model type
                    if num_keypoints > 0:
                        # Pose model - use PoseOutput
                        keypoints_obj = Keypoints(
                            xy=keypoint_xy,
                            conf=keypoint_conf
                        )
                        detections.append(PoseOutput(
                            bbox=list(bbox),  # Convert tuple to list
                            keypoints=keypoints_obj,
                            conf=confidence,
                            class_id=class_id,
                            id=None  # ID not stored in bin file
                        ))
                    else:
                        # Detection model - use DetectionsOutput
                        detections.append(DetectionsOutput(
                            bbox=list(bbox),  # Convert tuple to list
                            conf=confidence,
                            class_id=class_id,
                            id=None  # ID not stored in bin file
                        ))
                
                # Create Frame object with all detections
                frame = Frame(
                    frame_number=frame_number,
                    detections=detections
                )
                self.frames.append(frame)
        
        return self.frames
    
    def get_frames(self) -> List[Frame]:
        """
        Get the list of frames (reads file if not already read).
        
        Returns:
            List[Frame]: List of Frame objects
        """
        if not self.frames:
            self.read()
        return self.frames
    
    def get_model_type(self) -> Optional[str]:
        """Get the model type (POSE or DETECTION)."""
        if not self.model_type:
            self.read()
        return self.model_type
    
    def get_frame_by_number(self, frame_number: int) -> Optional[Frame]:
        """
        Get a specific frame by frame number.
        
        Args:
            frame_number: The frame number to retrieve
            
        Returns:
            Frame object if found, None otherwise
        """
        if not self.frames:
            self.read()
        
        for frame in self.frames:
            if frame.frame_number == frame_number:
                return frame
        return None
    
    def get_all_detections(self) -> List:
        """
        Get all detections from all frames as a flat list.
        
        Returns:
            List: All detections (DetectionsOutput or PoseOutput) from all frames
        """
        if not self.frames:
            self.read()
        
        all_detections = []
        for frame in self.frames:
            all_detections.extend(frame.detections)
        return all_detections
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all frames to a dictionary representation.
        
        Returns:
            Dictionary with model_type and frames
        """
        if not self.frames:
            self.read()
        
        return {
            'model_type': self.model_type,
            'num_frames': len(self.frames),
            'frames': [frame.to_dict() for frame in self.frames]
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 bin_file_reader.py <binary_file.bin>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        # Create reader and read file
        reader = BinFileReader(filepath)
        frames = reader.read()
        
        print(f"Model Type: {reader.get_model_type()}")
        print(f"Total Frames: {len(frames)}")
        print(f"Total Detections: {sum(len(frame.detections) for frame in frames)}")
        print("\n" + "=" * 60)
        
        # Example: Access frames and detections
        for frame in frames:
            print(f"\nFrame {frame.frame_number}: {len(frame.detections)} detections")
            for i, detection in enumerate(frame.detections):
                print(f"  Detection {i+1}:")
                print(f"    BBox: x_center={detection.bbox[0]:.2f}, y_center={detection.bbox[1]:.2f}, "
                      f"width={detection.bbox[2]:.2f}, height={detection.bbox[3]:.2f}")
                print(f"    Confidence: {detection.conf:.4f}")
                print(f"    Class ID: {detection.class_id}")
                if isinstance(detection, PoseOutput):
                    print(f"    Keypoints: {len(detection.keypoints.xy)} keypoints")
                    for k, (xy, conf) in enumerate(zip(detection.keypoints.xy, detection.keypoints.conf)):
                        if k < len(KEYPOINT_NAMES):
                            print(f"      {KEYPOINT_NAMES[k]}: ({xy[0]:.2f}, {xy[1]:.2f}) conf={conf:.3f}")
        
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

