#!/usr/bin/env python3
"""
Utility script to read and display binary output files from YOLO detection/pose models
"""

import struct
import sys
import os

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def read_detection_file(filepath):
    """Read a binary detection file and return parsed detections"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        # Read header
        model_type_int = struct.unpack('i', f.read(4))[0]
        num_detections = struct.unpack('i', f.read(4))[0]
        
        model_type = "POSE" if model_type_int == 1 else "DETECTION"
        
        print(f"Model Type: {model_type}")
        print(f"Number of Detections: {num_detections}")
        print("-" * 60)
        
        detections = []
        for i in range(num_detections):
            # Read bbox (4 floats)
            bbox = struct.unpack('4f', f.read(4 * 4))
            
            # Read confidence and class_id
            confidence = struct.unpack('f', f.read(4))[0]
            class_id = struct.unpack('i', f.read(4))[0]
            
            # Read number of keypoints
            num_keypoints = struct.unpack('i', f.read(4))[0]
            
            keypoints = []
            if num_keypoints > 0:
                for k in range(num_keypoints):
                    x, y, conf = struct.unpack('3f', f.read(3 * 4))
                    keypoints.append({
                        'name': KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"kpt_{k}",
                        'x': x,
                        'y': y,
                        'confidence': conf
                    })
            
            detections.append({
                'bbox': bbox,
                'confidence': confidence,
                'class_id': class_id,
                'keypoints': keypoints
            })
            
            # Print detection info
            print(f"Detection {i+1}:")
            print(f"  BBox: x_center={bbox[0]:.2f}, y_center={bbox[1]:.2f}, "
                  f"width={bbox[2]:.2f}, height={bbox[3]:.2f}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Class ID: {class_id}")
            
            if keypoints:
                print(f"  Keypoints ({len(keypoints)}):")
                for kpt in keypoints:
                    if kpt['confidence'] > 0.1:  # Only show visible keypoints
                        print(f"    {kpt['name']}: ({kpt['x']:.2f}, {kpt['y']:.2f}) "
                              f"conf={kpt['confidence']:.3f}")
            print()
        
        return detections

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 read_output.py <binary_file.bin>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    read_detection_file(filepath)

