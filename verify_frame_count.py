#!/usr/bin/env python3
"""
Script to verify why frames_read doesn't match total_expected_frames.

Usage:
    python verify_frame_count.py <log_file>
    python verify_frame_count.py --input "log message here"
    python verify_frame_count.py --interactive
    python verify_frame_count.py --video <video_path> --log <log_file>
"""

import re
import sys
import argparse
import subprocess
import json
import os
from typing import Optional, Dict, Tuple


def parse_video_reader_log(line: str) -> Optional[Dict]:
    """Parse VideoReader initialization log line."""
    # Pattern: Time window: start_ts=X, end_ts=Y, moment_time=Z, duration=D, ...
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
    # Pattern: Reader X finished video Y: frames_read=Z, path=P
    pattern = r'Reader \d+ finished video ([^:]+): frames_read=(\d+), path=(.+)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        'video_key': match.group(1),
        'frames_read': int(match.group(2)),
        'path': match.group(3),
    }


def get_video_metadata(video_path: str) -> Optional[Dict]:
    """Get actual video metadata using ffprobe."""
    if not os.path.exists(video_path):
        return None
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # Get frame count and duration
        nb_frames = video_stream.get('nb_frames')
        if nb_frames:
            nb_frames = int(nb_frames)
        else:
            # Calculate from duration and fps
            duration = float(video_stream.get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den > 0 else 0
            else:
                fps = float(fps_str)
            if duration > 0 and fps > 0:
                nb_frames = int(duration * fps)
            else:
                nb_frames = None
        
        duration = float(video_stream.get('duration', 0))
        if duration == 0:
            duration = float(data.get('format', {}).get('duration', 0))
        
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 0
        else:
            fps = float(fps_str)
        
        return {
            'nb_frames': nb_frames,
            'duration': duration,
            'fps': fps,
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'codec': video_stream.get('codec_name'),
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_frame_count(video_info: Dict, reader_info: Dict, video_metadata: Optional[Dict] = None) -> Dict:
    """Analyze why frames_read doesn't match expected frames."""
    results = {
        'video_key': reader_info.get('video_key', 'unknown'),
        'path': reader_info.get('path', 'unknown'),
        'frames_read': reader_info.get('frames_read', 0),
        'total_expected_frames': video_info.get('total_expected_frames', 0),
        'actual_expected_frames': video_info.get('actual_expected_frames', 0),
        'remaining_frames': video_info.get('remaining_frames', 0),
        'discrepancy': 0,
        'discrepancy_percent': 0.0,
        'causes': [],
    }
    
    # Calculate discrepancy
    expected = video_info.get('actual_expected_frames', video_info.get('total_expected_frames', 0))
    actual = results['frames_read']
    results['discrepancy'] = expected - actual
    if expected > 0:
        results['discrepancy_percent'] = (results['discrepancy'] / expected) * 100.0
    
    # Check if video ends before end_ts
    video_end_time = video_info.get('video_end', 0)
    end_ts = video_info.get('end_ts', 0)
    if video_end_time < end_ts:
        time_diff = end_ts - video_end_time
        frames_missing = time_diff * video_info.get('fps', 0)
        results['causes'].append({
            'type': 'video_ends_early',
            'description': f'Video ends {time_diff:.2f}s before end_ts',
            'frames_missing': int(frames_missing),
        })
    
    # Check if seek position is after start_ts
    seek_timestamp = video_info.get('seek_timestamp', 0)
    start_ts = video_info.get('start_ts', 0)
    if seek_timestamp > start_ts:
        time_diff = seek_timestamp - start_ts
        frames_skipped = time_diff * video_info.get('fps', 0)
        results['causes'].append({
            'type': 'seek_after_start',
            'description': f'Seek position is {time_diff:.2f}s after start_ts',
            'frames_skipped': int(frames_skipped),
        })
    
    # Check if remaining_frames calculation matches
    remaining_frames = video_info.get('remaining_frames', 0)
    if abs(remaining_frames - actual) > 10:  # Allow 10 frame tolerance
        results['causes'].append({
            'type': 'remaining_frames_mismatch',
            'description': f'remaining_frames ({remaining_frames}) doesn\'t match frames_read ({actual})',
            'difference': remaining_frames - actual,
        })
    
    # Use video metadata if available
    if video_metadata and 'error' not in video_metadata:
        video_nb_frames = video_metadata.get('nb_frames')
        video_duration = video_metadata.get('duration', 0)
        video_fps = video_metadata.get('fps', 0)
        
        # Calculate frames from seek position to end of video
        seek_frame = video_info.get('seek_to_frame', 0)
        if video_nb_frames and seek_frame < video_nb_frames:
            frames_available_in_video = video_nb_frames - seek_frame
            results['video_metadata'] = {
                'total_frames': video_nb_frames,
                'duration': video_duration,
                'fps': video_fps,
                'frames_available_from_seek': frames_available_in_video,
            }
            
            # Check if video has enough frames
            if frames_available_in_video < expected:
                frames_missing = expected - frames_available_in_video
                results['causes'].append({
                    'type': 'video_insufficient_frames',
                    'description': f'Video only has {frames_available_in_video:,} frames from seek position, need {expected:,}',
                    'frames_missing': frames_missing,
                })
            
            # Check if video ends before end_ts
            video_end_time = video_info.get('video_end', 0)
            end_ts = video_info.get('end_ts', 0)
            if video_end_time < end_ts:
                time_diff = end_ts - video_end_time
                frames_missing = time_diff * video_info.get('fps', 0)
                results['causes'].append({
                    'type': 'video_ends_early',
                    'description': f'Video ends {time_diff:.2f}s before end_ts (video_end={video_end_time:.2f}, end_ts={end_ts:.2f})',
                    'frames_missing': int(frames_missing),
                })
            
            # Check if actual frames read matches video's available frames
            if abs(actual - frames_available_in_video) > 10:
                diff = actual - frames_available_in_video
                if diff > 0:
                    results['causes'].append({
                        'type': 'read_more_than_available',
                        'description': f'Read {actual:,} frames but video only has {frames_available_in_video:,} from seek position',
                        'extra_frames': diff,
                    })
                else:
                    results['causes'].append({
                        'type': 'read_less_than_available',
                        'description': f'Read {actual:,} frames but video has {frames_available_in_video:,} available from seek position',
                        'missing_frames': -diff,
                    })
        elif video_metadata.get('error'):
            results['causes'].append({
                'type': 'metadata_error',
                'description': f'Could not read video metadata: {video_metadata["error"]}',
            })
    else:
        # Check if discrepancy is significant
        if abs(results['discrepancy']) > 50:  # More than 50 frames difference
            if not results['causes']:
                results['causes'].append({
                    'type': 'unknown',
                    'description': f'Large discrepancy ({results["discrepancy"]} frames) with no obvious cause. Consider using --video to get actual video metadata.',
                })
    
    return results


def print_analysis(results: Dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*80)
    print(f"Frame Count Analysis: {results['video_key']}")
    print("="*80)
    print(f"Video Path: {results['path']}")
    
    # Show video metadata if available
    if 'video_metadata' in results:
        vm = results['video_metadata']
        print(f"\nVideo Metadata (from file):")
        if vm.get('total_frames') is not None:
            print(f"  Total Frames:             {vm.get('total_frames'):,}")
        else:
            print(f"  Total Frames:             N/A (could not determine)")
        print(f"  Duration:                 {vm.get('duration', 0):.2f}s")
        print(f"  FPS:                      {vm.get('fps', 0):.3f}")
        if vm.get('frames_available_from_seek') is not None:
            print(f"  Frames Available (from seek): {vm.get('frames_available_from_seek'):,}")
        if vm.get('width') and vm.get('height'):
            print(f"  Resolution:               {vm.get('width')}x{vm.get('height')}")
        if vm.get('codec'):
            print(f"  Codec:                    {vm.get('codec')}")
    
    print(f"\nFrame Counts:")
    print(f"  Frames Read (actual):     {results['frames_read']:,}")
    print(f"  Total Expected:           {results['total_expected_frames']:,}")
    print(f"  Actual Expected:         {results['actual_expected_frames']:,}")
    print(f"  Remaining Frames:         {results['remaining_frames']:,}")
    print(f"\nDiscrepancy:")
    print(f"  Difference:               {results['discrepancy']:,} frames")
    print(f"  Percentage:                {results['discrepancy_percent']:.2f}%")
    
    if results['causes']:
        print(f"\nPossible Causes:")
        for i, cause in enumerate(results['causes'], 1):
            print(f"  {i}. {cause['type']}: {cause['description']}")
            if 'frames_missing' in cause:
                print(f"     → Missing ~{cause['frames_missing']:,} frames")
            if 'frames_skipped' in cause:
                print(f"     → Skipped ~{cause['frames_skipped']:,} frames")
            if 'difference' in cause:
                print(f"     → Difference: {cause['difference']:,} frames")
            if 'extra_frames' in cause:
                print(f"     → Extra frames read: {cause['extra_frames']:,}")
            if 'missing_frames' in cause:
                print(f"     → Missing frames: {cause['missing_frames']:,}")
    else:
        print(f"\n✓ No obvious causes found - frame count matches expected!")
    
    print("="*80 + "\n")


def process_log_file(log_file: str, use_video_metadata: bool = False):
    """Process a log file and analyze frame counts."""
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
                
                # If we have both, analyze
                if video_info and reader_info:
                    video_metadata = None
                    if use_video_metadata and reader_info.get('path'):
                        video_path = reader_info['path']
                        print(f"\nReading video metadata from: {video_path}...")
                        video_metadata = get_video_metadata(video_path)
                        if video_metadata and 'error' not in video_metadata:
                            print(f"  ✓ Video has {video_metadata.get('nb_frames', 'unknown'):,} frames, duration {video_metadata.get('duration', 0):.2f}s, fps {video_metadata.get('fps', 0):.3f}")
                        elif video_metadata:
                            print(f"  ✗ Error: {video_metadata.get('error')}")
                        else:
                            print(f"  ✗ Could not read video metadata (file not found or ffprobe error)")
                    
                    results = analyze_frame_count(video_info, reader_info, video_metadata)
                    print_analysis(results)
                    # Reset for next video
                    video_info = None
                    reader_info = None


def process_input(input_text: str, use_video_metadata: bool = False):
    """Process input text directly."""
    lines = input_text.split('\n')
    video_info = None
    reader_info = None
    
    for line in lines:
        if 'Time window:' in line and 'VideoReader' in line:
            video_info = parse_video_reader_log(line)
        
        if 'finished video' in line and 'frames_read=' in line:
            reader_info = parse_reader_finished_log(line)
            
            if video_info and reader_info:
                video_metadata = None
                if use_video_metadata and reader_info.get('path'):
                    video_path = reader_info['path']
                    print(f"\nReading video metadata from: {video_path}...")
                    video_metadata = get_video_metadata(video_path)
                    if video_metadata and 'error' not in video_metadata:
                        print(f"  ✓ Video has {video_metadata.get('nb_frames', 'unknown'):,} frames, duration {video_metadata.get('duration', 0):.2f}s, fps {video_metadata.get('fps', 0):.3f}")
                    elif video_metadata:
                        print(f"  ✗ Error: {video_metadata.get('error')}")
                    else:
                        print(f"  ✗ Could not read video metadata (file not found or ffprobe error)")
                
                results = analyze_frame_count(video_info, reader_info, video_metadata)
                print_analysis(results)
                video_info = None
                reader_info = None


def interactive_mode():
    """Interactive mode to paste log lines."""
    print("Interactive Mode - Paste log lines (Ctrl+D or empty line to finish):")
    print("="*80)
    
    lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        except EOFError:
            break
    
    process_input('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Verify why frames_read doesn\'t match total_expected_frames'
    )
    parser.add_argument('log_file', nargs='?', help='Log file to analyze')
    parser.add_argument('--input', '-i', help='Input log text directly')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--video', '-v', action='store_true', 
                       help='Read actual video metadata using ffprobe (requires ffprobe and video file access)')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.input:
        process_input(args.input, use_video_metadata=args.video)
    elif args.log_file:
        process_log_file(args.log_file, use_video_metadata=args.video)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

