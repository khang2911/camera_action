# Frame Count Verification Script

## Purpose

This script analyzes log output to determine why `frames_read` doesn't match `total_expected_frames`.

## Usage

### 1. From Log File (without video metadata)
```bash
python verify_frame_count.py <log_file>
```

### 2. From Log File (with video metadata - RECOMMENDED)
```bash
python verify_frame_count.py --video <log_file>
```
This will read actual video files using `ffprobe` to get real frame counts and duration.

### 3. Direct Input
```bash
python verify_frame_count.py --input "paste log lines here"
python verify_frame_count.py --video --input "paste log lines here"  # With video metadata
```

### 4. Interactive Mode
```bash
python verify_frame_count.py --interactive
python verify_frame_count.py --video --interactive  # With video metadata
# Then paste log lines, press Enter twice or Ctrl+D to finish
```

## Requirements

- Python 3.6+
- `ffprobe` (from FFmpeg) - required for `--video` option
  - Install: `apt-get install ffmpeg` or `brew install ffmpeg`

## What It Analyzes

The script extracts:
- Time window information (start_ts, end_ts, moment_time, duration)
- Video range vs actual range
- Expected frames (total and actual)
- Seek position and remaining frames
- Actual frames read

**With `--video` option, it also:**
- Reads actual video file using `ffprobe`
- Gets real frame count, duration, and FPS from video file
- Compares expected frames with video's actual available frames
- Identifies if video file doesn't have enough frames

## Output

The script shows:
- Frame counts (read, expected, remaining)
- Discrepancy (difference and percentage)
- Possible causes:
  - Video ends early (before end_ts)
  - Seek position after start_ts (frames skipped)
  - Remaining frames mismatch
  - Unknown causes for large discrepancies

## Example

```bash
# From your log file
python verify_frame_count.py main.log

# Or paste log lines directly
python verify_frame_count.py --input "
[2025-11-29 18:56:07.264] [INFO ] [VideoReader] Time window: start_ts=1763344286.061826, end_ts=1763344771.840247, ...
[2025-11-29 18:57:30.123] [INFO ] [Reader] Reader 2 finished video c03o25010001960_0a4277b4-07e4-476e-9b5a-9858abc9e494_v0: frames_read=2445, path=...
"
```

## Expected Output Format

### Without --video option:
```
================================================================================
Frame Count Analysis: c03o25010001960_0a4277b4-07e4-476e-9b5a-9858abc9e494_v0
================================================================================
Video Path: /shared_storage/action_videos/...

Frame Counts:
  Frames Read (actual):     2,445
  Total Expected:           3,662
  Actual Expected:          3,662
  Remaining Frames:         3,662

Discrepancy:
  Difference:               1,217 frames
  Percentage:               33.23%

Possible Causes:
  1. remaining_frames_mismatch: remaining_frames (3662) doesn't match frames_read (2445)
     → Difference: 1,217 frames
================================================================================
```

### With --video option (RECOMMENDED):
```
Reading video metadata from: /shared_storage/action_videos/.../video.mp4...
  ✓ Video has 11,700 frames, duration 900.14s, fps 12.997

================================================================================
Frame Count Analysis: c03o25010001960_0a4277b4-07e4-476e-9b5a-9858abc9e494_v0
================================================================================
Video Path: /shared_storage/action_videos/...

Video Metadata (from file):
  Total Frames:             11,700
  Duration:                 900.14s
  FPS:                      12.997
  Frames Available (from seek): 3,664

Frame Counts:
  Frames Read (actual):     2,445
  Total Expected:           3,662
  Actual Expected:          3,662
  Remaining Frames:         3,662

Discrepancy:
  Difference:               1,217 frames
  Percentage:               33.23%

Possible Causes:
  1. read_less_than_available: Read 2,445 frames but video has 3,664 available from seek position
     → Missing frames: 1,219
  2. video_ends_early: Video ends X.XXs before end_ts (video_end=..., end_ts=...)
     → Missing ~1,217 frames
================================================================================
```

## Why Use --video Option?

The `--video` option provides **much better diagnosis** by:
1. **Reading actual video file** to get real frame count and duration
2. **Comparing expected vs available frames** from the seek position
3. **Identifying root causes** like:
   - Video file doesn't have enough frames
   - Video ends before end_ts
   - Frames are being read correctly but video is shorter than expected

