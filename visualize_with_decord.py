#!/usr/bin/env python3
"""
Visualize detection results from bin file on video using decord for efficient frame reading.
Uses frame index from bin file to directly seek to the correct frame in the video.
"""

import struct
import sys
import os
import argparse
import cv2
import numpy as np
from decord import VideoReader, cpu

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Keypoint connections for skeleton drawing (COCO format)
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Colors for different body parts
KPT_COLORS = [
    (0, 255, 255),    # Yellow for head (BGR)
    (255, 0, 0),      # Blue for left arm
    (0, 0, 255),     # Red for right arm
    (0, 255, 0),     # Green for left leg
    (255, 0, 255),   # Magenta for right leg
]

# Detection box colors
DETECTION_COLORS = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
]


def read_detection_file(filepath):
    """Read a binary detection file and return detections grouped by frame number.
    
    Returns:
        dict: {frame_number: [list of detections]}
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    detections_by_frame = {}
    
    with open(filepath, 'rb') as f:
        # Read header (model type - written once)
        model_type_int = struct.unpack('i', f.read(4))[0]
        model_type = "POSE" if model_type_int == 1 else "DETECTION"
        
        print(f"Model Type: {model_type}")
        print(f"Reading detections from: {filepath}")
        
        # Read frames until end of file
        while True:
            # Try to read frame number
            frame_data = f.read(4)
            if len(frame_data) < 4:
                break  # End of file
            
            frame_number = struct.unpack('i', frame_data)[0]
            num_detections = struct.unpack('i', f.read(4))[0]
            
            frame_detections = []
            for i in range(num_detections):
                # Read bbox (4 floats: x_center, y_center, width, height)
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
            
            detections_by_frame[frame_number] = frame_detections
    
    print(f"Loaded {len(detections_by_frame)} frames with detections")
    return detections_by_frame, model_type


def draw_detection(frame, detection, color, is_pose=False):
    """Draw a single detection on the frame."""
    h, w = frame.shape[:2]
    
    # Parse bbox (x_center, y_center, width, height)
    x_center, y_center, width, height = detection['bbox']
    
    # Check if coordinates are normalized [0,1] or pixel-based
    if 0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0:
        # Normalized coordinates - convert to pixels
        x_center = x_center * w
        y_center = y_center * h
        width = width * w
        height = height * h
    
    # Convert center-width-height to top-left-bottom-right
    x1 = int(x_center - width * 0.5)
    y1 = int(y_center - height * 0.5)
    x2 = int(x_center + width * 0.5)
    y2 = int(y_center + height * 0.5)
    
    # Clamp to frame bounds
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label with confidence
    label = f"Class {detection['class_id']} {detection['confidence']*100:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw keypoints and skeleton for pose models
    if is_pose and detection['keypoints']:
        # Draw skeleton connections first (so keypoints appear on top)
        for conn in KEYPOINT_CONNECTIONS:
            if conn[0] < len(detection['keypoints']) and conn[1] < len(detection['keypoints']):
                kpt1 = detection['keypoints'][conn[0]]
                kpt2 = detection['keypoints'][conn[1]]
                
                if kpt1['confidence'] > 0.1 and kpt2['confidence'] > 0.1:
                    # Handle normalized vs pixel coordinates
                    if 0.0 <= kpt1['x'] <= 1.0 and 0.0 <= kpt1['y'] <= 1.0:
                        kpt1_x = int(kpt1['x'] * w)
                        kpt1_y = int(kpt1['y'] * h)
                    else:
                        kpt1_x = int(max(0, min(w - 1, kpt1['x'])))
                        kpt1_y = int(max(0, min(h - 1, kpt1['y'])))
                    
                    if 0.0 <= kpt2['x'] <= 1.0 and 0.0 <= kpt2['y'] <= 1.0:
                        kpt2_x = int(kpt2['x'] * w)
                        kpt2_y = int(kpt2['y'] * h)
                    else:
                        kpt2_x = int(max(0, min(w - 1, kpt2['x'])))
                        kpt2_y = int(max(0, min(h - 1, kpt2['y'])))
                    
                    # Use darker shade for skeleton
                    skeleton_color = tuple(int(c * 0.6) for c in color)
                    cv2.line(frame, (kpt1_x, kpt1_y), (kpt2_x, kpt2_y), skeleton_color, 2)
        
        # Draw keypoints
        for i, kpt in enumerate(detection['keypoints']):
            if kpt['confidence'] <= 0.1:
                continue
            
            # Handle normalized vs pixel coordinates
            if 0.0 <= kpt['x'] <= 1.0 and 0.0 <= kpt['y'] <= 1.0:
                kpt_x = int(kpt['x'] * w)
                kpt_y = int(kpt['y'] * h)
            else:
                kpt_x = int(max(0, min(w - 1, kpt['x'])))
                kpt_y = int(max(0, min(h - 1, kpt['y'])))
            
            # Determine keypoint color based on body part
            if i <= 4:
                kpt_color = KPT_COLORS[0]  # Yellow for head
            elif i == 5 or i == 7 or i == 9:
                kpt_color = KPT_COLORS[1]  # Blue for left arm
            elif i == 6 or i == 8 or i == 10:
                kpt_color = KPT_COLORS[2]  # Red for right arm
            elif i == 11 or i == 13 or i == 15:
                kpt_color = KPT_COLORS[3]  # Green for left leg
            else:
                kpt_color = KPT_COLORS[4]  # Magenta for right leg
            
            # Draw keypoint as a circle with border
            cv2.circle(frame, (kpt_x, kpt_y), 6, (0, 0, 0), -1)  # Black border
            cv2.circle(frame, (kpt_x, kpt_y), 5, kpt_color, -1)   # Filled circle


def visualize_video(video_path, bin_file, output_dir=None, max_frames=None):
    """Visualize detections on video using decord for efficient frame reading."""
    
    # Read detections from bin file
    print(f"Reading detections from: {bin_file}")
    detections_by_frame, model_type = read_detection_file(bin_file)
    if detections_by_frame is None:
        return
    
    is_pose = (model_type == "POSE")
    
    # Get all frame numbers that have detections
    frame_numbers = sorted(detections_by_frame.keys())
    if not frame_numbers:
        print("No detections found in bin file!")
        return
    
    print(f"Frames with detections: {len(frame_numbers)}")
    print(f"Frame range: {frame_numbers[0]} to {frame_numbers[-1]}")
    
    # Limit frames if requested
    if max_frames:
        frame_numbers = frame_numbers[:max_frames]
        print(f"Limiting to first {max_frames} frames")
    
    # Open video with decord
    print(f"Opening video: {video_path}")
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_video_frames = len(vr)
        print(f"Video has {total_video_frames} frames")
    except Exception as e:
        print(f"Error opening video with decord: {e}")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving visualized frames to: {output_dir}")
    
    # Process each frame with detections
    processed_count = 0
    skipped_count = 0
    
    for frame_idx, frame_number in enumerate(frame_numbers):
        # Check if frame number is valid for this video
        if frame_number < 0 or frame_number >= total_video_frames:
            print(f"Warning: Frame {frame_number} is out of range (0-{total_video_frames-1}), skipping")
            skipped_count += 1
            continue
        
        # Read frame using decord (0-indexed)
        try:
            frame = vr[frame_number].asnumpy()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error reading frame {frame_number}: {e}")
            skipped_count += 1
            continue
        
        # Get detections for this frame
        detections = detections_by_frame[frame_number]
        
        # Draw each detection
        for det_idx, detection in enumerate(detections):
            color = DETECTION_COLORS[detection['class_id'] % len(DETECTION_COLORS)]
            draw_detection(frame, detection, color, is_pose=is_pose)
        
        # Draw frame number on top-left corner
        cv2.putText(frame, f"Frame {frame_number} ({len(detections)} detections)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save or display frame
        if output_dir:
            output_path = os.path.join(output_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(output_path, frame)
            if (frame_idx + 1) % 10 == 0:
                print(f"Processed {frame_idx + 1}/{len(frame_numbers)} frames...")
        else:
            # Display frame
            cv2.imshow("Detection Visualization", frame)
            print(f"Frame {frame_number}: {len(detections)} detections (Press any key to continue, 'q' to quit)")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Stopped by user")
                break
        
        processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed_count} frames")
    print(f"  Skipped: {skipped_count} frames")
    if output_dir:
        print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize detection results from bin file on video using decord"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--bin-file", required=True, help="Path to detection bin file")
    parser.add_argument("--output-dir", help="Directory to save visualized frames (if not specified, displays frames)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.bin_file):
        print(f"Error: Bin file not found: {args.bin_file}")
        sys.exit(1)
    
    visualize_video(args.video, args.bin_file, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()

