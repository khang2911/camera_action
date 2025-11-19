#!/usr/bin/env python3
"""
Debug script to inspect detection values and help diagnose parsing issues.
Shows raw values from binary files to understand the data format.
"""

import struct
import argparse
import os
import sys

def debug_detection_file(filepath, num_frames=5, show_raw=False):
    """Debug a binary detection file to see actual values."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING: {filepath}")
    print(f"{'='*70}\n")
    
    with open(filepath, 'rb') as f:
        # Read model type
        model_type_bytes = f.read(4)
        if len(model_type_bytes) < 4:
            print("Error: File too short")
            return
        model_type = struct.unpack('i', model_type_bytes)[0]
        print(f"Model type: {model_type} (0=DETECTION, 1=POSE)")
        
        frame_count = 0
        while frame_count < num_frames:
            # Read frame number
            frame_num_bytes = f.read(4)
            if len(frame_num_bytes) < 4:
                break
            frame_number = struct.unpack('i', frame_num_bytes)[0]
            
            # Read number of detections
            num_dets_bytes = f.read(4)
            if len(num_dets_bytes) < 4:
                break
            num_detections = struct.unpack('i', num_dets_bytes)[0]
            
            print(f"\n{'='*70}")
            print(f"Frame {frame_number}: {num_detections} detections")
            print(f"{'='*70}")
            
            if num_detections == 0:
                frame_count += 1
                continue
            
            for i in range(min(num_detections, 3)):  # Show first 3 detections
                # Read bbox
                bbox_bytes = f.read(4 * 4)
                if len(bbox_bytes) < 16:
                    break
                bbox = struct.unpack('4f', bbox_bytes)
                
                # Read confidence and class_id
                conf_bytes = f.read(4)
                class_bytes = f.read(4)
                if len(conf_bytes) < 4 or len(class_bytes) < 4:
                    break
                confidence = struct.unpack('f', conf_bytes)[0]
                class_id = struct.unpack('i', class_bytes)[0]
                
                # Read keypoints
                num_kpts_bytes = f.read(4)
                if len(num_kpts_bytes) < 4:
                    break
                num_keypoints = struct.unpack('i', num_kpts_bytes)[0]
                
                print(f"\nDetection {i+1}:")
                print(f"  BBox: x_center={bbox[0]:.6f}, y_center={bbox[1]:.6f}, "
                      f"width={bbox[2]:.6f}, height={bbox[3]:.6f}")
                print(f"  Confidence: {confidence:.6f}")
                print(f"  Class ID: {class_id}")
                
                # Check if bbox values are in expected range
                x_c, y_c, w, h = bbox
                print(f"  BBox Analysis:")
                print(f"    x_center in [0,1]: {0.0 <= x_c <= 1.0} (value: {x_c})")
                print(f"    y_center in [0,1]: {0.0 <= y_c <= 1.0} (value: {y_c})")
                print(f"    width in [0,1]: {0.0 <= w <= 1.0} (value: {w})")
                print(f"    height in [0,1]: {0.0 <= h <= 1.0} (value: {h})")
                
                # Check if values seem reasonable
                if x_c < 0 or x_c > 1 or y_c < 0 or y_c > 1:
                    print(f"    ⚠️  WARNING: Coordinates outside [0,1] range!")
                    print(f"       If these are pixel coordinates, they need different scaling!")
                if w < 0 or w > 1 or h < 0 or h > 1:
                    print(f"    ⚠️  WARNING: Dimensions outside [0,1] range!")
                    print(f"       If these are pixel coordinates, they need different scaling!")
                
                # Check if values look like pixel coordinates (large numbers)
                if x_c > 100 or y_c > 100 or w > 100 or h > 100:
                    print(f"    ⚠️  WARNING: Values look like pixel coordinates, not normalized!")
                    print(f"       Expected normalized [0,1] but got values > 100")
                    print(f"       This suggests bbox coordinates are NOT normalized!")
                
                if confidence < 0 or confidence > 1:
                    print(f"    ⚠️  WARNING: Confidence outside [0,1] range!")
                
                if class_id < 0:
                    print(f"    ⚠️  WARNING: Class ID is negative!")
                
                # Show what the bbox would be if interpreted as pixel coordinates
                if x_c > 1 or y_c > 1 or w > 1 or h > 1:
                    print(f"  If interpreted as pixel coordinates:")
                    print(f"    BBox: x1={x_c-w/2:.1f}, y1={y_c-h/2:.1f}, "
                          f"x2={x_c+w/2:.1f}, y2={y_c+h/2:.1f}")
                else:
                    # Show what it would be for a 1920x1080 image (common video resolution)
                    print(f"  If interpreted as normalized [0,1] for 1920x1080 image:")
                    print(f"    BBox: x1={(x_c-w/2)*1920:.1f}, y1={(y_c-h/2)*1080:.1f}, "
                          f"x2={(x_c+w/2)*1920:.1f}, y2={(y_c+h/2)*1080:.1f}")
                
                # Show keypoints if present
                if num_keypoints > 0:
                    print(f"  Keypoints: {num_keypoints}")
                    for k in range(min(num_keypoints, 3)):  # Show first 3
                        kpt_bytes = f.read(3 * 4)
                        if len(kpt_bytes) < 12:
                            break
                        x, y, conf = struct.unpack('3f', kpt_bytes)
                        print(f"    Kpt {k}: x={x:.6f}, y={y:.6f}, conf={conf:.6f}")
                    # Skip remaining keypoints
                    for k in range(3, num_keypoints):
                        f.read(3 * 4)
            
            # Skip remaining detections for this frame
            for i in range(min(num_detections, 3), num_detections):
                f.read(4 * 4)  # bbox
                f.read(4)  # confidence
                f.read(4)  # class_id
                num_kpts = struct.unpack('i', f.read(4))[0]
                for k in range(num_kpts):
                    f.read(3 * 4)  # keypoint
            
            frame_count += 1
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    print(f"{'='*70}")
    print("If bbox values are outside [0,1], they might need sigmoid activation.")
    print("If confidence is outside [0,1], it might need sigmoid activation.")
    print("If class IDs are inconsistent, check the class score parsing logic.")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Debug binary detection files to diagnose parsing issues'
    )
    parser.add_argument('detection_file', type=str, help='Path to binary detection file')
    parser.add_argument('--num-frames', type=int, default=5,
                       help='Number of frames to inspect (default: 5)')
    parser.add_argument('--show-raw', action='store_true',
                       help='Show raw byte values (not implemented)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.detection_file):
        print(f"Error: File not found: {args.detection_file}")
        sys.exit(1)
    
    debug_detection_file(args.detection_file, args.num_frames, args.show_raw)

if __name__ == '__main__':
    main()

