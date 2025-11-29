#!/usr/bin/env python3
"""
Script to verify why frames_read doesn't match total_expected_frames.

Usage:
    python verify_frame_count.py <log_file>
    python verify_frame_count.py --input "log message here"
    python verify_frame_count.py --interactive
"""

import re
import sys
import argparse
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


def analyze_frame_count(video_info: Dict, reader_info: Dict) -> Dict:
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
    
    # Check if discrepancy is significant
    if abs(results['discrepancy']) > 50:  # More than 50 frames difference
        if not results['causes']:
            results['causes'].append({
                'type': 'unknown',
                'description': f'Large discrepancy ({results["discrepancy"]} frames) with no obvious cause',
            })
    
    return results


def print_analysis(results: Dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*80)
    print(f"Frame Count Analysis: {results['video_key']}")
    print("="*80)
    print(f"Video Path: {results['path']}")
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
    else:
        print(f"\n✓ No obvious causes found - frame count matches expected!")
    
    print("="*80 + "\n")


def process_log_file(log_file: str):
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
                    results = analyze_frame_count(video_info, reader_info)
                    print_analysis(results)
                    # Reset for next video
                    video_info = None
                    reader_info = None


def process_input(input_text: str):
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
                results = analyze_frame_count(video_info, reader_info)
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
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.input:
        process_input(args.input)
    elif args.log_file:
        process_log_file(args.log_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

