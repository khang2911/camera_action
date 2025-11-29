# Time Window Simulation Script

## Purpose

This script simulates the time window calculation and frame reading logic to identify bugs without modifying the C++ code.

## What It Simulates

1. **Time Window Calculation** (`parseJsonToVideoClips`):
   - Sequential time window logic across videos
   - `start_timestamp = max(current_window_start, video_start)`
   - `end_timestamp = min(global_end_ts, video_end)`

2. **Seek Position Calculation** (`initializeMetadata`):
   - `estimated_frame = floor((start_timestamp - moment_time) * fps)`
   - Sets `total_frames_read_ = estimated_frame`

3. **Frame Timestamp Calculation** (`readFrame`):
   - **Current (Buggy) Method**: `current_ts = moment_time + (total_frames_read_ - 1) / fps`
   - **Proposed Fix**: Use actual frame PTS from FFmpeg

4. **Frame Filtering**:
   - Frames are filtered if `current_ts < start_timestamp` or `current_ts > end_timestamp`

## Usage

### 1. From Log File
```bash
python simulate_time_window.py <log_file>
```

This will:
- Parse VideoReader logs to extract time window info
- Simulate frame reading using current (buggy) method
- Compare with actual frames read
- Also test with proposed fix (using actual PTS)

### 2. Manual Mode
```bash
python simulate_time_window.py --manual
```

Interactive mode to test specific scenarios:
- Enter global time window
- Enter video information
- See calculated time window and frame counts

### 3. Test Proposed Fix Only
```bash
python simulate_time_window.py <log_file> --use-pts
```

## Output

The script shows:
- **Frame Counts**: How many frames should be read (simulation) vs actually read vs expected
- **First/Last Frame**: Frame numbers and timestamps of first and last frames
- **Issues**: Any mismatches or problems found
- **Comparison**: Results using current method vs proposed fix (using actual PTS)

## Example Output

```
================================================================================
Simulation Results: c03o24120002258_f863ec86-42f2-437f-84b1-d54e3ab01b42_v0 (Using Frame Number Calculation)
================================================================================

Frame Counts:
  Frames Should Read (simulation): 2,866
  Frames Actually Read (from log):  582
  Expected Frames (from log):       2,866

First Frame:
  Frame Number: 4652
  Timestamp:    1763344286.061826
  In Window:    True

Last Frame:
  Frame Number: 5233
  Timestamp:    1763344286.061826
  In Window:    True

Issues Found:
  1. simulation_mismatch_actual: Simulation says 2,866 frames should be read, but actually read 582
     → Difference: 2,284 frames

================================================================================
COMPARISON: Testing with Actual PTS (Proposed Fix)
================================================================================

Simulation Results: c03o24120002258_f863ec86-42f2-437f-84b1-d54e3ab01b42_v0 (Using Actual PTS)
================================================================================

Frame Counts:
  Frames Should Read (simulation): 2,866
  Frames Actually Read (from log):  582
  Expected Frames (from log):       2,866

✓ No issues found - simulation matches expected!
```

## What to Look For

1. **Simulation Mismatch**: If simulation says different number of frames than expected, there's a bug in the calculation logic
2. **Timestamp Off**: If first/last frame timestamps don't match start_ts/end_ts, the timestamp calculation is wrong
3. **Comparison Results**: If using actual PTS gives correct results but frame number calculation doesn't, that confirms Bug #2

## Next Steps

After running the simulation:
1. If simulation matches expected but actual doesn't → Bug is in C++ implementation (not logic)
2. If simulation doesn't match expected → Bug is in calculation logic
3. If using actual PTS fixes it → Confirms we should use actual frame PTS instead of calculating from frame number

