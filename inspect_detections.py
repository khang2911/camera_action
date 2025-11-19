#!/usr/bin/env python3
"""
Utility script to inspect binary detection files and find frame numbers.
Helps determine start_frame and end_frame for visualization.
"""

import struct
import argparse
import os
import sys

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

MODEL_TYPES = {
    0: "DETECTION",
    1: "POSE"
}

def inspect_detection_file(filepath, show_details=False, show_detections=False):
    """
    Inspect a binary detection file and show frame information.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*70}\n")
    
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    detections_by_frame = {}
    total_detections = 0
    frames_with_detections = 0
    frames_without_detections = 0
    
    with open(filepath, 'rb') as f:
        # Read model type (header)
        try:
            model_type_bytes = f.read(4)
            if len(model_type_bytes) < 4:
                print("Error: File too short")
                return None
            model_type = struct.unpack('i', model_type_bytes)[0]
            model_type_name = MODEL_TYPES.get(model_type, f"UNKNOWN({model_type})")
            print(f"Model type: {model_type_name}")
        except struct.error:
            print("Error: Failed to read model type")
            return None
        
        # Read frame detections
        frame_numbers = []
        while True:
            # Read frame number
            frame_num_bytes = f.read(4)
            if len(frame_num_bytes) < 4:
                break  # End of file
            frame_number = struct.unpack('i', frame_num_bytes)[0]
            frame_numbers.append(frame_number)
            
            # Read number of detections
            num_dets_bytes = f.read(4)
            if len(num_dets_bytes) < 4:
                break
            num_detections = struct.unpack('i', num_dets_bytes)[0]
            
            frame_detections = []
            for i in range(num_detections):
                # Read bbox (4 floats: x_center, y_center, width, height)
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
                
                # Read number of keypoints
                num_kpts_bytes = f.read(4)
                if len(num_kpts_bytes) < 4:
                    break
                num_keypoints = struct.unpack('i', num_kpts_bytes)[0]
                
                keypoints = []
                if num_keypoints > 0:
                    for k in range(num_keypoints):
                        kpt_bytes = f.read(3 * 4)
                        if len(kpt_bytes) < 12:
                            break
                        x, y, conf = struct.unpack('3f', kpt_bytes)
                        keypoints.append({
                            'x': x, 'y': y, 'confidence': conf
                        })
                
                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'keypoints': keypoints
                }
                frame_detections.append(detection)
                total_detections += 1
            
            if num_detections > 0:
                frames_with_detections += 1
                detections_by_frame[frame_number] = {
                    'num_detections': num_detections,
                    'detections': frame_detections
                }
            else:
                frames_without_detections += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames with data: {len(frame_numbers)}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Frames without detections: {frames_without_detections}")
    print(f"Total detections: {total_detections}")
    
    if frame_numbers:
        frame_numbers_sorted = sorted(frame_numbers)
        min_frame = frame_numbers_sorted[0]
        max_frame = frame_numbers_sorted[-1]
        print(f"\nFrame range:")
        print(f"  Minimum frame number: {min_frame}")
        print(f"  Maximum frame number: {max_frame}")
        print(f"  Frame range: {max_frame - min_frame + 1} frames")
        
        # Show frame numbers with detections
        frames_with_dets = sorted([f for f in frame_numbers if f in detections_by_frame])
        if frames_with_dets:
            print(f"\nFrames with detections: {len(frames_with_dets)}")
            if len(frames_with_dets) <= 20:
                print(f"  Frame numbers: {frames_with_dets}")
            else:
                print(f"  First 10: {frames_with_dets[:10]}")
                print(f"  Last 10: {frames_with_dets[-10:]}")
                print(f"  ... and {len(frames_with_dets) - 20} more")
        
        # Show detection statistics
        if detections_by_frame:
            confidences = []
            class_ids = []
            for frame_data in detections_by_frame.values():
                for det in frame_data['detections']:
                    confidences.append(det['confidence'])
                    class_ids.append(det['class_id'])
            
            if confidences:
                print(f"\nDetection statistics:")
                print(f"  Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
                print(f"  Average confidence: {sum(confidences) / len(confidences):.4f}")
                unique_classes = sorted(set(class_ids))
                print(f"  Classes detected: {unique_classes}")
                print(f"  Number of unique classes: {len(unique_classes)}")
        
        # Show details for first few frames
        if show_details and detections_by_frame:
            print(f"\n{'='*70}")
            print("DETAILS (first 5 frames with detections)")
            print(f"{'='*70}")
            count = 0
            for frame_num in sorted(detections_by_frame.keys()):
                if count >= 5:
                    break
                frame_data = detections_by_frame[frame_num]
                print(f"\nFrame {frame_num}: {frame_data['num_detections']} detections")
                if show_detections:
                    for i, det in enumerate(frame_data['detections'][:3]):  # Show first 3
                        bbox = det['bbox']
                        print(f"  Detection {i+1}:")
                        print(f"    BBox: x_center={bbox[0]:.2f}, y_center={bbox[1]:.2f}, "
                              f"width={bbox[2]:.2f}, height={bbox[3]:.2f}")
                        print(f"    Confidence: {det['confidence']:.4f}, Class: {det['class_id']}")
                        if det['keypoints']:
                            print(f"    Keypoints: {len(det['keypoints'])}")
                count += 1
        
        # Print command suggestion
        print(f"\n{'='*70}")
        print("VISUALIZATION COMMAND SUGGESTION")
        print(f"{'='*70}")
        print(f"To visualize all frames:")
        print(f"  python3 visualize_detections.py <video_path> {filepath} <output_video>")
        print(f"\nTo visualize specific range:")
        print(f"  python3 visualize_detections.py <video_path> {filepath} <output_video> \\")
        print(f"    --start_frame {min_frame} --end_frame {max_frame}")
        print(f"\nTo visualize frames with detections only:")
        if frames_with_dets:
            print(f"  python3 visualize_detections.py <video_path> {filepath} <output_video> \\")
            print(f"    --start_frame {frames_with_dets[0]} --end_frame {frames_with_dets[-1]}")
        print()
    
    return {
        'min_frame': min_frame if frame_numbers else None,
        'max_frame': max_frame if frame_numbers else None,
        'frames_with_detections': frames_with_dets if frame_numbers else [],
        'total_detections': total_detections
    }

def main():
    parser = argparse.ArgumentParser(
        description='Inspect binary detection files to find frame numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python3 inspect_detections.py output/video_0_alcohol.bin
  
  # Detailed inspection with detection details
  python3 inspect_detections.py output/video_0_alcohol.bin --details --show-detections
  
  # Inspect multiple files
  python3 inspect_detections.py output/*.bin
        """
    )
    parser.add_argument('detection_files', nargs='+', type=str, 
                       help='Path(s) to binary detection file(s)')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed information about frames')
    parser.add_argument('--show-detections', action='store_true',
                       help='Show detection details (requires --details)')
    
    args = parser.parse_args()
    
    results = []
    for filepath in args.detection_files:
        result = inspect_detection_file(filepath, args.details, args.show_detections)
        if result:
            results.append((filepath, result))
    
    # Summary across all files
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL FILES")
        print(f"{'='*70}")
        all_min_frames = [r[1]['min_frame'] for r in results if r[1]['min_frame'] is not None]
        all_max_frames = [r[1]['max_frame'] for r in results if r[1]['max_frame'] is not None]
        if all_min_frames and all_max_frames:
            print(f"Overall frame range: {min(all_min_frames)} - {max(all_max_frames)}")
        total_dets = sum(r[1]['total_detections'] for r in results)
        print(f"Total detections across all files: {total_dets}")
        print()

if __name__ == '__main__':
    main()

