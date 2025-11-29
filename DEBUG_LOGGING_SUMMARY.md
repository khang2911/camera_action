# Debug Logging Summary

## Comprehensive Logging Added

I've added extensive debug logging throughout the video reading pipeline to identify exactly where and why frames are being lost.

## Logging Points Added

### 1. Time Window Initialization (`VideoReader::initialize`)
- **Location**: `src/video_reader.cu` line ~250
- **What it logs**:
  - All time window parameters (start_ts, end_ts, moment_time, etc.)
  - Expected frames vs actual available frames
  - Seek position and remaining frames
  - **NEW**: Whether video ends before end_ts (critical for diagnosis)

### 2. Seek Position Calculation (`VideoReader::initializeMetadata`)
- **Location**: `src/video_reader.cu` line ~960
- **What it logs**:
  - Offset seconds calculation
  - Estimated frame number after seek
  - FPS and timestamp values used

### 3. Frame-by-Frame Processing (`VideoReader::readFrame`)
- **Location**: `src/video_reader.cu` line ~547-585
- **What it logs**:
  - **First 10 frames**: Detailed info for every frame
  - **Every 1000th frame**: Periodic status updates
  - **For each frame**:
    - `total_frames_read_` (all frames, including skipped)
    - `actual_frame_position_` (frames in time window)
    - Calculated timestamp (`current_ts`)
    - **NEW**: Actual frame PTS from FFmpeg (if available)
    - **NEW**: Timestamp difference (calculated vs actual PTS)
    - Time window boundaries (start_ts, end_ts)
  - **When skipping frames**: Logs frames skipped before start_ts
  - **When stopping**: Detailed stop reason with all relevant values

### 4. End of Stream Detection
- **Location**: `src/video_reader.cu` lines ~614, ~674, ~716
- **What it logs**:
  - When `end_of_stream_` is set to true
  - Frame position when EOF is detected
  - Remaining time window if applicable
  - Expected frames remaining

### 5. Reader Loop (`ThreadPool::processVideo`)
- **Location**: `src/thread_pool.cpp` line ~1001
- **What it logs**:
  - **NEW**: When `readFrame()` first returns false
  - **NEW**: Consecutive failure count
  - **NEW**: Final frame position when stopping
  - Video completion with actual vs expected frames

## Key Diagnostic Information

The logs will now show:

1. **Timestamp Calculation Accuracy**:
   - Calculated timestamp vs actual frame PTS
   - Difference between them (reveals if calculation is wrong)

2. **Stop Conditions**:
   - Did it stop because `current_ts > end_timestamp`?
   - Did it stop because `end_of_stream_` is true?
   - What was the last frame's timestamp?

3. **Frame Counting**:
   - `total_frames_read_` (all frames from video)
   - `actual_frame_position_` (frames in time window)
   - Expected vs actual counts

4. **Video File Issues**:
   - Does video end before `end_timestamp`?
   - How many frames remain when video ends?

## What to Look For in Logs

When you run the code, look for these patterns:

1. **"*** STOPPING: Reached end of time window ***"**:
   - This is expected - check if `frames_in_window` matches expected
   - Check if `current_ts` is close to `end_ts`

2. **"*** STOPPING: Reached end of video file ***"**:
   - This indicates video ended early
   - Check `remaining` time and `expected_frames_remaining`
   - This is likely the bug!

3. **"*** EOF DETECTED ***"**:
   - Shows when FFmpeg detects end of file
   - Compare with time window to see if it's premature

4. **Frame timestamp differences**:
   - Large `ts_diff` values indicate timestamp calculation is wrong
   - This could cause premature stopping

5. **"readFrame() returned false"**:
   - Shows when the reader loop detects failure
   - Check consecutive failures to see if it's persistent

## Expected Output

With these logs, you should see exactly:
- Why the reader stops after 582 frames instead of 2,867
- Whether it's due to end of stream or timestamp calculation
- What the actual vs calculated timestamps are
- How many frames remain when stopping

This will definitively identify the root cause of the bug.

