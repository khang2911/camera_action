#!/usr/bin/env python3
"""
Simulate the time window calculation and frame reading logic to verify correctness.

This script simulates:
1. Time window calculation in parseJsonToVideoClips
2. Seek position calculation in initializeMetadata
3. Frame timestamp calculation in readFrame
4. Frame filtering based on time window

Usage:
    python simulate_time_window.py --log <log_file>
    python simulate_time_window.py --manual
"""

import re
import sys
import argparse
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class VideoClip:
    """Simulate VideoClip structure"""
    moment_time: float
    duration_seconds: float
    start_timestamp: float
    end_timestamp: float
    has_time_window: bool
    path: str
    video_index: int


@dataclass
class FrameInfo:
    """Information about a frame"""
    frame_number: int  # Frame number in video (0-indexed from start of video)
    timestamp: float   # Absolute timestamp
    in_time_window: bool
    should_be_read: bool


def parse_video_reader_log(line: str) -> Optional[Dict]:
    """Parse VideoReader initialization log line."""
    pattern = r'Time window: start_ts=([\d.]+), end_ts=([\d.]+), moment_time=([\d.]+), duration=([\d.]+), video_range=\[([\d.]+), ([\d.]+)\], actual_range=\[([\d.]+), ([\d.]+)\], fps=([\d.]+), time_window_duration=([\d.]+), actual_available_duration=([\d.]+), total_expected_frames~(\d+), actual_expected_frames~(\d+), seek_to_frame=(\d+), seek_timestamp=([\d.]+), remaining_duration=([\d.]+), remaining_frames~(\d+)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        'start_ts': float(match.group(1)),
        'end_ts': float(match.group(2)),
        'moment_time': float(match.group(3)),
        'duration': float(match.group(4)),
        'video_start': float(match.group(5)),
        'video_end': float(match.group(6)),
        'actual_start': float(match.group(7)),
        'actual_end': float(match.group(8)),
        'fps': float(match.group(9)),
        'time_window_duration': float(match.group(10)),
        'actual_available_duration': float(match.group(11)),
        'total_expected_frames': int(match.group(12)),
        'actual_expected_frames': int(match.group(13)),
        'seek_to_frame': int(match.group(14)),
        'seek_timestamp': float(match.group(15)),
        'remaining_duration': float(match.group(16)),
        'remaining_frames': int(match.group(17)),
    }


def parse_reader_finished_log(line: str) -> Optional[Dict]:
    """Parse Reader finished log line."""
    pattern = r'Reader \d+ finished video ([^:]+): frames_read=(\d+), path=(.+)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        'video_key': match.group(1),
        'frames_read': int(match.group(2)),
        'path': match.group(3),
    }


def simulate_time_window_calculation(global_start_ts: float, global_end_ts: float,
                                     video_moment_time: float, video_duration: float,
                                     current_window_start: float) -> Tuple[float, float, float]:
    """
    Simulate the time window calculation logic from parseJsonToVideoClips.
    
    Returns: (start_timestamp, end_timestamp, next_window_start)
    """
    video_start = video_moment_time
    video_end = video_moment_time + video_duration
    
    # Sequential time window logic (from thread_pool.cpp lines 908-915)
    start_timestamp = max(current_window_start, video_start)
    end_timestamp = min(global_end_ts, video_end)
    next_window_start = end_timestamp
    
    return start_timestamp, end_timestamp, next_window_start


def simulate_seek_calculation(start_timestamp: float, moment_time: float, fps: float) -> int:
    """
    Simulate the seek position calculation from initializeMetadata.
    
    Returns: estimated_frame (total_frames_read_ after seek)
    """
    offset_seconds = start_timestamp - moment_time
    if offset_seconds > 0.0:
        estimated_frame = int(max(0.0, (offset_seconds * fps)))
    else:
        estimated_frame = 0
    
    return estimated_frame


def simulate_frame_timestamp_calculation(frame_number: int, moment_time: float, fps: float) -> float:
    """
    Simulate the frame timestamp calculation from readFrame.
    
    This is the CURRENT (BUGGY) implementation:
    current_ts = moment_time + (total_frames_read_ - 1) / fps
    
    Returns: calculated timestamp
    """
    # Note: frame_number is 0-indexed from start of video
    # In the code, total_frames_read_ is incremented BEFORE this calculation
    # So we use (frame_number) instead of (frame_number - 1) to match the logic
    current_ts = moment_time + (frame_number) / fps
    return current_ts


def simulate_frame_timestamp_using_pts(frame_number: int, moment_time: float, fps: float) -> float:
    """
    Simulate frame timestamp using actual frame PTS (PROPOSED FIX).
    
    This assumes frame PTS is accurate and represents the frame's position in the video.
    
    Returns: calculated timestamp using PTS
    """
    # Frame PTS would be: frame_number / fps (relative to video start)
    frame_pts_seconds = frame_number / fps
    current_ts = moment_time + frame_pts_seconds
    return current_ts


def simulate_frame_reading(video_info: Dict, use_actual_pts: bool = False) -> List[FrameInfo]:
    """
    Simulate reading frames and filtering based on time window.
    
    Returns: List of FrameInfo for frames that should be read
    """
    moment_time = video_info['moment_time']
    duration = video_info['duration']
    start_ts = video_info['start_ts']
    end_ts = video_info['end_ts']
    fps = video_info['fps']
    seek_to_frame = video_info['seek_to_frame']
    
    # Calculate total frames in video
    total_frames_in_video = int(duration * fps)
    
    frames = []
    frames_read = 0
    
    # CRITICAL: total_frames_read_ starts at seek_to_frame (estimated_frame) after seeking
    # Then it's incremented BEFORE timestamp calculation in readFrame()
    # So for the first frame after seek:
    #   total_frames_read_ = seek_to_frame (from initializeMetadata)
    #   ++total_frames_read_ → total_frames_read_ = seek_to_frame + 1
    #   current_ts = moment_time + (total_frames_read_ - 1) / fps = moment_time + seek_to_frame / fps
    
    # Start from seek position
    # frame_num represents the actual frame number in the video (0-indexed from start)
    # But we need to simulate total_frames_read_ which starts at seek_to_frame
    total_frames_read = seek_to_frame
    
    for frame_num in range(seek_to_frame, total_frames_in_video):
        # Increment total_frames_read_ FIRST (as in C++ code line 549)
        total_frames_read += 1
        
        # Calculate timestamp using current (buggy) method
        if use_actual_pts:
            # Using actual PTS: frame_num / fps gives the frame's position in video
            current_ts = moment_time + (frame_num / fps)
        else:
            # Current buggy method: use total_frames_read_ (which might not match frame_num!)
            # This is the KEY BUG: total_frames_read_ is estimated, but frame_num is actual
            current_ts = moment_time + (total_frames_read - 1) / fps
        
        # Check if frame is in time window
        in_time_window = start_ts <= current_ts <= end_ts
        
        # Frame should be read if in time window
        should_be_read = in_time_window
        
        if should_be_read:
            frames_read += 1
        
        frames.append(FrameInfo(
            frame_number=frame_num,
            timestamp=current_ts,
            in_time_window=in_time_window,
            should_be_read=should_be_read
        ))
        
        # Stop if past end timestamp (as in C++ code line 577)
        if current_ts > end_ts:
            break
    
    return frames


def analyze_simulation(video_info: Dict, reader_info: Dict, use_actual_pts: bool = False) -> Dict:
    """Analyze the simulation results."""
    # Simulate frame reading
    frames = simulate_frame_reading(video_info, use_actual_pts=use_actual_pts)
    
    # Count frames that should be read
    frames_should_read = sum(1 for f in frames if f.should_be_read)
    frames_actually_read = reader_info.get('frames_read', 0)
    expected_frames = video_info.get('actual_expected_frames', video_info.get('total_expected_frames', 0))
    
    # Find issues
    issues = []
    
    # Check if simulation matches expected
    if abs(frames_should_read - expected_frames) > 10:
        issues.append({
            'type': 'simulation_mismatch_expected',
            'description': f'Simulation says {frames_should_read} frames should be read, but expected is {expected_frames}',
            'difference': frames_should_read - expected_frames,
        })
    
    # Check if simulation matches actual
    if abs(frames_should_read - frames_actually_read) > 10:
        issues.append({
            'type': 'simulation_mismatch_actual',
            'description': f'Simulation says {frames_should_read} frames should be read, but actually read {frames_actually_read}',
            'difference': frames_should_read - frames_actually_read,
        })
    
    # Check first and last frame timestamps
    if frames:
        first_frame = next((f for f in frames if f.should_be_read), None)
        last_frame = next((f for f in reversed(frames) if f.should_be_read), None)
        
        if first_frame:
            if abs(first_frame.timestamp - video_info['start_ts']) > 0.1:
                issues.append({
                    'type': 'first_frame_timestamp_off',
                    'description': f'First frame timestamp {first_frame.timestamp:.3f} is off from start_ts {video_info["start_ts"]:.3f}',
                    'difference': first_frame.timestamp - video_info['start_ts'],
                })
        
        if last_frame:
            if abs(last_frame.timestamp - video_info['end_ts']) > 0.1:
                issues.append({
                    'type': 'last_frame_timestamp_off',
                    'description': f'Last frame timestamp {last_frame.timestamp:.3f} is off from end_ts {video_info["end_ts"]:.3f}',
                    'difference': last_frame.timestamp - video_info['end_ts'],
                })
    
    # Get first and last frames that should be read
    frames_in_window = [f for f in frames if f.should_be_read]
    
    return {
        'video_key': reader_info.get('video_key', 'unknown'),
        'frames_should_read': frames_should_read,
        'frames_actually_read': frames_actually_read,
        'expected_frames': expected_frames,
        'issues': issues,
        'first_frame': frames_in_window[0] if frames_in_window else (frames[0] if frames else None),
        'last_frame': frames_in_window[-1] if frames_in_window else (frames[-1] if frames else None),
        'first_actual_frame': frames[0] if frames else None,
        'last_actual_frame': frames[-1] if frames else None,
        'frames': frames[:10] + frames[-10:] if len(frames) > 20 else frames,  # First 10 and last 10
        'video_info': video_info,  # Include for debugging
    }


def print_simulation_results(results: Dict, use_actual_pts: bool = False):
    """Print simulation results."""
    method = "Using Actual PTS" if use_actual_pts else "Using Frame Number Calculation"
    
    print("\n" + "="*80)
    print(f"Simulation Results: {results['video_key']} ({method})")
    print("="*80)
    # Get time window info from video_info
    video_info = results.get('video_info', {})
    start_ts = video_info.get('start_ts', 0)
    end_ts = video_info.get('end_ts', 0)
    moment_time = video_info.get('moment_time', 0)
    seek_to_frame = video_info.get('seek_to_frame', 0)
    
    print(f"\nTime Window Info:")
    print(f"  start_timestamp: {start_ts:.6f}")
    print(f"  end_timestamp:   {end_ts:.6f}")
    print(f"  moment_time:     {moment_time:.6f}")
    print(f"  seek_to_frame:   {seek_to_frame}")
    
    print(f"\nFrame Counts:")
    print(f"  Frames Should Read (simulation): {results['frames_should_read']:,}")
    print(f"  Frames Actually Read (from log):  {results['frames_actually_read']:,}")
    print(f"  Expected Frames (from log):       {results['expected_frames']:,}")
    
    if results['first_actual_frame']:
        print(f"\nFirst Frame (from seek position):")
        print(f"  Frame Number: {results['first_actual_frame'].frame_number}")
        print(f"  Timestamp:    {results['first_actual_frame'].timestamp:.6f}")
        print(f"  In Window:    {results['first_actual_frame'].in_time_window}")
    
    if results['first_frame'] and results['first_frame'] != results['first_actual_frame']:
        print(f"\nFirst Frame (in time window):")
        print(f"  Frame Number: {results['first_frame'].frame_number}")
        print(f"  Timestamp:    {results['first_frame'].timestamp:.6f}")
        print(f"  In Window:    {results['first_frame'].in_time_window}")
    
    if results['last_frame']:
        print(f"\nLast Frame (in time window):")
        print(f"  Frame Number: {results['last_frame'].frame_number}")
        print(f"  Timestamp:    {results['last_frame'].timestamp:.6f}")
        print(f"  In Window:    {results['last_frame'].in_time_window}")
    
    if results['last_actual_frame']:
        print(f"\nLast Frame (simulated):")
        print(f"  Frame Number: {results['last_actual_frame'].frame_number}")
        print(f"  Timestamp:    {results['last_actual_frame'].timestamp:.6f}")
        print(f"  In Window:    {results['last_actual_frame'].in_time_window}")
    
    if results['issues']:
        print(f"\nIssues Found:")
        for i, issue in enumerate(results['issues'], 1):
            print(f"  {i}. {issue['type']}: {issue['description']}")
            if 'difference' in issue:
                print(f"     → Difference: {issue['difference']:,} frames")
    else:
        print(f"\n✓ No issues found - simulation matches expected!")
    
    print("="*80 + "\n")


def process_log_file(log_file: str, use_actual_pts: bool = False):
    """Process a log file and simulate frame reading."""
    video_info = None
    reader_info = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Try to parse VideoReader log
            if 'Time window:' in line and 'VideoReader' in line:
                video_info = parse_video_reader_log(line)
            
            # Try to parse Reader finished log
            if 'finished video' in line and 'frames_read=' in line:
                reader_info = parse_reader_finished_log(line)
                
                # If we have both, simulate
                if video_info and reader_info:
                    results = analyze_simulation(video_info, reader_info, use_actual_pts=use_actual_pts)
                    print_simulation_results(results, use_actual_pts=use_actual_pts)
                    
                    # Also test with actual PTS if we're using frame number calculation
                    if not use_actual_pts:
                        print("\n" + "="*80)
                        print("COMPARISON: Testing with Actual PTS (Proposed Fix)")
                        print("="*80)
                        results_pts = analyze_simulation(video_info, reader_info, use_actual_pts=True)
                        print_simulation_results(results_pts, use_actual_pts=True)
                    
                    # Reset for next video
                    video_info = None
                    reader_info = None


def manual_mode():
    """Interactive mode for manual input."""
    print("Manual Simulation Mode")
    print("="*80)
    print("Enter video information:")
    
    try:
        global_start_ts = float(input("Global start_timestamp: "))
        global_end_ts = float(input("Global end_timestamp: "))
        moment_time = float(input("Video moment_time: "))
        duration = float(input("Video duration (seconds): "))
        fps = float(input("Video FPS: "))
        
        # Simulate time window calculation
        start_ts, end_ts, next_start = simulate_time_window_calculation(
            global_start_ts, global_end_ts, moment_time, duration, global_start_ts
        )
        
        print(f"\nCalculated Time Window:")
        print(f"  start_timestamp: {start_ts:.6f}")
        print(f"  end_timestamp:   {end_ts:.6f}")
        print(f"  next_window_start: {next_start:.6f}")
        
        # Simulate seek
        seek_frame = simulate_seek_calculation(start_ts, moment_time, fps)
        print(f"\nSeek Calculation:")
        print(f"  seek_to_frame: {seek_frame}")
        
        # Simulate frame reading
        video_info = {
            'moment_time': moment_time,
            'duration': duration,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'fps': fps,
            'seek_to_frame': seek_frame,
        }
        
        frames = simulate_frame_reading(video_info, use_actual_pts=False)
        frames_read = sum(1 for f in frames if f.should_be_read)
        
        print(f"\nFrame Reading Simulation (Current Method):")
        print(f"  Frames that should be read: {frames_read}")
        if frames:
            print(f"  First frame: #{frames[0].frame_number}, ts={frames[0].timestamp:.6f}, in_window={frames[0].in_time_window}")
            last_read = next((f for f in reversed(frames) if f.should_be_read), None)
            if last_read:
                print(f"  Last frame read: #{last_read.frame_number}, ts={last_read.timestamp:.6f}")
        
        # Test with actual PTS
        frames_pts = simulate_frame_reading(video_info, use_actual_pts=True)
        frames_read_pts = sum(1 for f in frames_pts if f.should_be_read)
        
        print(f"\nFrame Reading Simulation (Using Actual PTS):")
        print(f"  Frames that should be read: {frames_read_pts}")
        if frames_pts:
            print(f"  First frame: #{frames_pts[0].frame_number}, ts={frames_pts[0].timestamp:.6f}, in_window={frames_pts[0].in_time_window}")
            last_read_pts = next((f for f in reversed(frames_pts) if f.should_be_read), None)
            if last_read_pts:
                print(f"  Last frame read: #{last_read_pts.frame_number}, ts={last_read_pts.timestamp:.6f}")
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Simulate time window calculation and frame reading logic'
    )
    parser.add_argument('log_file', nargs='?', help='Log file to analyze')
    parser.add_argument('--manual', action='store_true', help='Interactive manual mode')
    parser.add_argument('--use-pts', action='store_true', help='Use actual PTS instead of frame number calculation')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_mode()
    elif args.log_file:
        process_log_file(args.log_file, use_actual_pts=args.use_pts)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

