# Frame Count Verification Script

## Purpose

This script analyzes log output to determine why `frames_read` doesn't match `total_expected_frames`.

## Usage

### 1. From Log File
```bash
python verify_frame_count.py <log_file>
```

### 2. Direct Input
```bash
python verify_frame_count.py --input "paste log lines here"
```

### 3. Interactive Mode
```bash
python verify_frame_count.py --interactive
# Then paste log lines, press Enter twice or Ctrl+D to finish
```

## What It Analyzes

The script extracts:
- Time window information (start_ts, end_ts, moment_time, duration)
- Video range vs actual range
- Expected frames (total and actual)
- Seek position and remaining frames
- Actual frames read

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
  1. video_ends_early: Video ends X.XXs before end_ts
     → Missing ~1,217 frames
================================================================================
```

