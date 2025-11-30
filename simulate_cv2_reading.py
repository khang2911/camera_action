#!/usr/bin/env python3
"""
Simulate video reading using cv2 (OpenCV) without seeking to verify frame count
against expected frames based on time window.

This script reads a video sequentially using cv2.VideoCapture and filters frames
by time window, then compares the actual frame count with the expected count.
"""

import cv2
import sys
import argparse
import re
from pathlib import Path


def parse_log_for_video_info(log_file, video_key=None):
    """Parse log file to extract video information."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find video initialization and time window info
    video_path = None
    start_ts = None
    end_ts = None
    expected_frames = None
    moment_time = None
    
    for i, line in enumerate(lines):
        # Look for video initialization
        if '[VideoReader] Initialized reader for' in line and '[NVDEC]' in line:
            match = re.search(r'\[VideoReader\] Initialized reader for (.+?)\s+\[NVDEC\]', line)
            if match:
                video_path = match.group(1)
        
        # Look for time window info (new format)
        if '[VideoReader] Time window:' in line and 'start_ts=' in line:
            # Try new format with moment_time
            match = re.search(r'start_ts=([\d.]+).*?end_ts=([\d.]+).*?moment_time=([\d.]+)', line)
            if match:
                start_ts = float(match.group(1))
                end_ts = float(match.group(2))
                moment_time = float(match.group(3))
                # Look for expected_frames in same or next line
                if 'expected_frames~' in line:
                    ef_match = re.search(r'expected_frames~(\d+)', line)
                    if ef_match:
                        expected_frames = int(ef_match.group(1))
            else:
                # Try old format
                match = re.search(r'start_ts=([\d.]+), end_ts=([\d.]+), expected_frames~(\d+)', line)
                if match:
                    start_ts = float(match.group(1))
                    end_ts = float(match.group(2))
                    expected_frames = int(match.group(3))
    
    if not video_path or not start_ts or not end_ts:
        print("Failed to parse video info from log")
        return None
    
    return {
        'video_path': video_path,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'expected_frames': expected_frames,
        'moment_time': moment_time
    }


def get_video_metadata(video_path):
    """Get video metadata using cv2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Get video start time (we'll estimate from first frame timestamp if available)
    # For now, we'll use a placeholder
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }


def read_video_with_time_window(video_path, start_ts, end_ts, moment_time=None):
    """
    Read video using cv2 and filter frames by time window.
    
    Args:
        video_path: Path to video file
        start_ts: Start timestamp (absolute)
        end_ts: End timestamp (absolute)
        moment_time: Video start time (absolute timestamp of first frame)
    
    Returns:
        dict with frame counts and statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n=== Video Metadata ===")
    print(f"Video: {video_path}")
    print(f"Reported FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # If moment_time not provided, we need to estimate it
    # For cv2 reading without seeking, we'll read from the beginning
    # So moment_time should be the video's actual start time
    # We can estimate it as: moment_time = start_ts - (some offset)
    # But for accurate simulation, we should use the actual moment_time from metadata
    if moment_time is None:
        # Estimate: if we're seeking to a position, moment_time is before start_ts
        # For now, use a conservative estimate: assume video starts some time before start_ts
        # This is a simplification - in reality, moment_time should come from video metadata
        print(f"Warning: moment_time not provided")
        print(f"  For accurate simulation, we need the video's actual start timestamp (moment_time)")
        print(f"  Without it, we'll estimate it from the time window")
        # Estimate: if start_ts is far into the video, moment_time is earlier
        # We'll use start_ts as a fallback, but this may not be accurate
        moment_time = start_ts - (duration * 0.5)  # Estimate: video starts halfway through its duration before start_ts
        print(f"  Using estimated moment_time: {moment_time:.6f} (video may start earlier)")
    
    print(f"\n=== Time Window ===")
    print(f"Moment time (video start): {moment_time:.6f}")
    print(f"Start timestamp: {start_ts:.6f}")
    print(f"End timestamp: {end_ts:.6f}")
    print(f"Time window duration: {end_ts - start_ts:.6f} seconds")
    
    # Calculate expected frames
    expected_frames = (end_ts - start_ts) * fps
    print(f"Expected frames (based on reported FPS): {expected_frames:.0f}")
    
    # Read frames sequentially
    frames_in_window = 0
    frames_before_window = 0
    frames_after_window = 0
    total_read = 0
    frame_timestamps = []
    
    print(f"\n=== Reading Frames ===")
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_read += 1
        
        # Calculate timestamp for this frame
        # Frame timestamp = moment_time + (frame_number / fps)
        frame_ts = moment_time + (frame_number / fps)
        frame_timestamps.append(frame_ts)
        
        # Check if frame is in time window
        if frame_ts < start_ts:
            frames_before_window += 1
        elif frame_ts > end_ts:
            frames_after_window += 1
            # Can stop early if we're past end_ts
            # But continue to see all frames for analysis
        else:
            frames_in_window += 1
        
        frame_number += 1
        
        # Progress indicator
        if total_read % 1000 == 0:
            print(f"  Read {total_read} frames, in window: {frames_in_window}")
    
    cap.release()
    
    # Calculate actual FPS within time window
    time_window_duration = end_ts - start_ts
    actual_fps = frames_in_window / time_window_duration if time_window_duration > 0 else 0
    
    # Calculate FPS based on first and last frame timestamps
    if first_frame_in_window is not None and last_frame_in_window is not None:
        first_ts = moment_time + (first_frame_in_window / fps) if fps > 0 else moment_time
        last_ts = moment_time + (last_frame_in_window / fps) if fps > 0 else moment_time
        actual_time_span = last_ts - first_ts
        fps_from_span = (frames_in_window - 1) / actual_time_span if actual_time_span > 0 and frames_in_window > 1 else 0
    else:
        fps_from_span = 0
        first_ts = None
        last_ts = None
    
    # Calculate statistics
    result = {
        'total_frames_in_video': total_frames,
        'total_frames_read': total_read,
        'frames_in_window': frames_in_window,
        'frames_before_window': frames_before_window,
        'frames_after_window': frames_after_window,
        'expected_frames': expected_frames,
        'reported_fps': fps,
        'actual_fps': actual_fps,
        'time_window_duration': time_window_duration,
        'moment_time': moment_time,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'frame_timestamps': frame_timestamps,
        'first_frame_in_window': first_frame_in_window,
        'last_frame_in_window': last_frame_in_window,
        'fps_from_span': fps_from_span,
        'first_ts': first_ts,
        'last_ts': last_ts
    }
    
    return result


def print_analysis(result):
    """Print detailed analysis of results."""
    print(f"\n=== Analysis ===")
    print(f"Frames in time window: {result['frames_in_window']}")
    if result.get('expected_frames') is not None:
        expected = result['expected_frames']
        print(f"Expected frames: {expected:.0f}")
        print(f"Difference: {expected - result['frames_in_window']:.0f} frames")
        diff_percent = ((expected - result['frames_in_window']) / expected * 100) if expected > 0 else 0
        print(f"Difference: {diff_percent:.1f}%")
    else:
        # Calculate expected from time window and FPS
        expected = result['time_window_duration'] * result['reported_fps']
        print(f"Expected frames (calculated): {expected:.0f}")
        print(f"Difference: {expected - result['frames_in_window']:.0f} frames")
        diff_percent = ((expected - result['frames_in_window']) / expected * 100) if expected > 0 else 0
        print(f"Difference: {diff_percent:.1f}%")
    
    print(f"\nFPS Comparison:")
    print(f"  Reported FPS: {result['reported_fps']:.2f}")
    print(f"  Actual FPS (frames/time_window_duration): {result['actual_fps']:.2f}")
    if result['fps_from_span'] > 0:
        print(f"  Actual FPS (from first/last frame span): {result['fps_from_span']:.2f}")
    
    if result['reported_fps'] > 0:
        fps_diff_percent = ((result['reported_fps'] - result['actual_fps']) / result['reported_fps'] * 100)
        print(f"  FPS difference: {fps_diff_percent:.1f}%")
    
    if result['first_frame_in_window'] is not None:
        print(f"\nFrame Range in Time Window:")
        print(f"  First frame: #{result['first_frame_in_window']} (ts={result['first_ts']:.6f})")
        print(f"  Last frame: #{result['last_frame_in_window']} (ts={result['last_ts']:.6f})")
        print(f"  Frame span: {result['last_frame_in_window'] - result['first_frame_in_window'] + 1} frames")
    
    print(f"\nFrame Distribution:")
    print(f"  Before time window: {result['frames_before_window']}")
    print(f"  In time window: {result['frames_in_window']}")
    print(f"  After time window: {result['frames_after_window']}")
    
    # Check if we stopped early
    if result['frames_after_window'] > 0:
        print(f"\n⚠️  Note: Found {result['frames_after_window']} frames after end_ts")
        print(f"   This suggests the video continues beyond the time window")
    
    # Check timestamp range
    if result['frame_timestamps']:
        first_ts = result['frame_timestamps'][0]
        last_ts = result['frame_timestamps'][-1]
        print(f"\nTimestamp Range:")
        print(f"  First frame: {first_ts:.6f}")
        print(f"  Last frame: {last_ts:.6f}")
        print(f"  Video duration (from timestamps): {last_ts - first_ts:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Simulate cv2 video reading with time window filtering')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--log', type=str, help='Path to log file to parse')
    parser.add_argument('--start-ts', type=float, help='Start timestamp (absolute)')
    parser.add_argument('--end-ts', type=float, help='End timestamp (absolute)')
    parser.add_argument('--moment-time', type=float, help='Video start time (absolute timestamp)')
    
    args = parser.parse_args()
    
    # Parse from log if provided
    if args.log:
        info = parse_log_for_video_info(args.log)
        if info:
            video_path = info['video_path']
            start_ts = info['start_ts']
            end_ts = info['end_ts']
            expected_frames = info.get('expected_frames')
            moment_time = info.get('moment_time')
            
            print(f"Parsed from log:")
            print(f"  Video: {video_path}")
            print(f"  Start TS: {start_ts}")
            print(f"  End TS: {end_ts}")
            if expected_frames:
                print(f"  Expected frames: {expected_frames}")
            if moment_time:
                print(f"  Moment time: {moment_time}")
            
            # Override with command line args if provided
            if args.video:
                video_path = args.video
            if args.start_ts:
                start_ts = args.start_ts
            if args.end_ts:
                end_ts = args.end_ts
            if args.moment_time:
                moment_time = args.moment_time
        else:
            print("Failed to parse log file")
            return 1
    else:
        # Use command line arguments
        if not args.video or not args.start_ts or not args.end_ts:
            print("Error: Must provide --video, --start-ts, --end-ts or use --log")
            return 1
        video_path = args.video
        start_ts = args.start_ts
        end_ts = args.end_ts
        moment_time = args.moment_time
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Read video and filter by time window
    result = read_video_with_time_window(video_path, start_ts, end_ts, moment_time)
    
    if result is None:
        return 1
    
    # Print analysis
    print_analysis(result)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

