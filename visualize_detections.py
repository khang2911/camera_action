#!/usr/bin/env python3
"""
Visualization script to draw detections and keypoints on video frames.
Reads binary output files and overlays bounding boxes and keypoints on the original video.
"""

import struct
import cv2
import argparse
import os
import sys
from pathlib import Path

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Keypoint connections for drawing skeleton (COCO format)
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Colors for different classes (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
]

def read_detection_file(filepath):
    """
    Read binary detection file.
    
    File format:
    - Header: model_type (int) - written once
    - For each frame: frame_number (int), num_detections (int), detections...
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    detections_by_frame = {}
    
    with open(filepath, 'rb') as f:
        # Read model type (header)
        try:
            model_type_bytes = f.read(4)
            if len(model_type_bytes) < 4:
                print(f"Error: File too short: {filepath}")
                return None
            model_type = struct.unpack('i', model_type_bytes)[0]
        except struct.error:
            print(f"Error: Failed to read model type from {filepath}")
            return None
        
        # Read frame detections
        while True:
            # Read frame number
            frame_num_bytes = f.read(4)
            if len(frame_num_bytes) < 4:
                break  # End of file
            frame_number = struct.unpack('i', frame_num_bytes)[0]
            
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
                            'name': KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"kpt_{k}",
                            'x': x,
                            'y': y,
                            'confidence': conf
                        })
                
                detection = {
                    'bbox': bbox,  # (x_center, y_center, width, height)
                    'confidence': confidence,
                    'class_id': class_id,
                    'keypoints': keypoints
                }
                frame_detections.append(detection)
            
            if frame_detections:
                detections_by_frame[frame_number] = {
                    'model_type': model_type,
                    'detections': frame_detections
                }
    
    return detections_by_frame

def draw_detection(frame, detection, color):
    """Draw a single detection on the frame."""
    x_center, y_center, width, height = detection['bbox']
    confidence = detection['confidence']
    class_id = detection['class_id']
    
    # Convert center-width-height to top-left corner format
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Clamp to image bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = f"Class {class_id}: {confidence:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1, label_size[1] + 10)
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                  (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw keypoints if present
    if detection['keypoints']:
        # Draw keypoint connections (skeleton)
        for conn in KEYPOINT_CONNECTIONS:
            if conn[0] < len(detection['keypoints']) and conn[1] < len(detection['keypoints']):
                kpt1 = detection['keypoints'][conn[0]]
                kpt2 = detection['keypoints'][conn[1]]
                if kpt1['confidence'] > 0.3 and kpt2['confidence'] > 0.3:
                    pt1 = (int(kpt1['x']), int(kpt1['y']))
                    pt2 = (int(kpt2['x']), int(kpt2['y']))
                    if 0 <= pt1[0] < w and 0 <= pt1[1] < h and \
                       0 <= pt2[0] < w and 0 <= pt2[1] < h:
                        cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw keypoints
        for kpt in detection['keypoints']:
            if kpt['confidence'] > 0.3:
                x, y = int(kpt['x']), int(kpt['y'])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 4, color, -1)
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

def visualize_video(video_path, detection_file, output_path, start_frame=0, end_frame=None):
    """Visualize detections on video."""
    print(f"Reading detections from: {detection_file}")
    detections_by_frame = read_detection_file(detection_file)
    
    if detections_by_frame is None:
        print("Failed to read detection file")
        return False
    
    print(f"Found detections for {len(detections_by_frame)} frames")
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video {output_path}")
        cap.release()
        return False
    
    frame_number = 0
    frames_with_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number < start_frame:
            frame_number += 1
            continue
        
        if frame_number >= end_frame:
            break
        
        # Get detections for this frame
        # Detection files use 1-indexed frame numbers (first frame is frame 1)
        # Video frames are 0-indexed (first frame is frame 0)
        # So we need to add 1 to match detection file frame numbers
        detection_frame_key = frame_number + 1
        detections = detections_by_frame.get(detection_frame_key, {}).get('detections', [])
        
        if detections:
            frames_with_detections += 1
            # Get color for this frame (cycle through colors)
            base_color = CLASS_COLORS[frame_number % len(CLASS_COLORS)]
            
            for det in detections:
                class_id = det['class_id']
                color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
                draw_detection(frame, det, color)
        
        out.write(frame)
        frame_number += 1
        
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames ({frames_with_detections} with detections)")
    
    cap.release()
    out.release()
    print(f"Visualization complete: {output_path}")
    print(f"Total frames processed: {frame_number}, Frames with detections: {frames_with_detections}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO detections on video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Frame numbers in detection files are 1-indexed (first frame is frame 1).
Use inspect_detections.py to find the frame range in your detection file:
  python3 inspect_detections.py <detection_file>

Examples:
  # Visualize all frames
  python3 visualize_detections.py video.mp4 detections.bin output.mp4
  
  # Visualize specific frame range
  python3 visualize_detections.py video.mp4 detections.bin output.mp4 --start_frame 100 --end_frame 200
        """
    )
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('detection_file', type=str, help='Path to binary detection file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--start_frame', type=int, default=0, 
                       help='Start frame number (0-indexed for video, default: 0). Note: Detection files use 1-indexed frames.')
    parser.add_argument('--end_frame', type=int, default=None, 
                       help='End frame number (0-indexed for video, default: all frames)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.detection_file):
        print(f"Error: Detection file not found: {args.detection_file}")
        sys.exit(1)
    
    success = visualize_video(args.video_path, args.detection_file, 
                             args.output_path, args.start_frame, args.end_frame)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

