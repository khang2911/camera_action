# Time Window Calculation Bug Analysis

## Critical Bug #1: Incorrect Timestamp Calculation After Seek

### Location
`src/video_reader.cu` lines 951-952 and 556

### The Problem

**In `initializeMetadata()` (line 951-952):**
```cpp
long long estimated_frame = static_cast<long long>(std::max(0.0, std::floor(offset_seconds * fps_)));
total_frames_read_ = estimated_frame;
```

**In `readFrame()` (line 549, 556):**
```cpp
++total_frames_read_;  // Incremented BEFORE timestamp check
double current_ts = clip_.moment_time + static_cast<double>(total_frames_read_ - 1) / effective_fps;
```

### Why This Is Wrong

The timestamp calculation assumes `total_frames_read_` represents the frame number from the start of the video (`moment_time`). However, after seeking:

1. `estimated_frame = floor((start_timestamp - moment_time) * fps_)`
2. `total_frames_read_ = estimated_frame` (set in `initializeMetadata`)
3. First frame after seek: `total_frames_read_ = estimated_frame + 1`
4. `current_ts = moment_time + (estimated_frame + 1 - 1) / fps_ = moment_time + estimated_frame / fps_`

**Example:**
- `moment_time = 1000.0`
- `start_timestamp = 1005.0`
- `fps = 100.0`
- `offset_seconds = 5.0`
- `estimated_frame = floor(5.0 * 100.0) = 500`
- First frame: `current_ts = 1000.0 + 500 / 100.0 = 1005.0` ✓ (correct!)

**But if there's a mismatch:**
- If actual seek position is frame 450 (due to `AVSEEK_FLAG_BACKWARD`), but we estimate 500
- First frame: `current_ts = 1000.0 + 500 / 100.0 = 1005.0`
- But actual frame timestamp might be `1000.0 + 450 / 100.0 = 1004.5`
- This causes frames to be incorrectly filtered!

### Impact

- **Frames read less than expected**: If `estimated_frame` is too high, `current_ts` will be ahead of actual frame timestamps, causing early stopping
- **Frames read more than expected**: If `estimated_frame` is too low, `current_ts` will lag behind, causing frames to pass the `end_timestamp` check

## Critical Bug #2: Time Window End Calculation

### Location
`src/thread_pool.cpp` line 911

### The Problem

```cpp
clip.end_timestamp = std::min(global_end_ts, video_end);
```

This sets `end_timestamp` to the **earlier** of:
- `global_end_ts` (the global time window end)
- `video_end` (the video's actual end time: `moment_time + duration`)

### Why This Can Cause Issues

**Scenario 1: Video ends before global_end_ts**
- `video_end = 1100.0`, `global_end_ts = 1200.0`
- `clip.end_timestamp = min(1200.0, 1100.0) = 1100.0` ✓ (correct - video ends at 1100)

**Scenario 2: Video extends beyond global_end_ts**
- `video_end = 1200.0`, `global_end_ts = 1100.0`
- `clip.end_timestamp = min(1100.0, 1200.0) = 1100.0` ✓ (correct - stop at global end)

**BUT**: The sequential logic updates `current_window_start = clip.end_timestamp` (line 915)

**Problem Scenario:**
- Video 0: `video_end = 1100.0`, `global_end_ts = 1200.0`
  - `clip0.end_timestamp = 1100.0` (stops at video end)
  - `current_window_start = 1100.0`
- Video 1: `video1_start = 1105.0`, `video1_end = 1200.0`, `global_end_ts = 1200.0`
  - `clip1.start_timestamp = max(1100.0, 1105.0) = 1105.0` ✓
  - `clip1.end_timestamp = min(1200.0, 1200.0) = 1200.0` ✓
  - This should work correctly!

**However**, if Video 1's `video1_start < 1100.0`:
- `clip1.start_timestamp = max(1100.0, 1050.0) = 1100.0` (starts from where video0 ended)
- But Video 1's actual start is 1050.0, so frames from 1050.0 to 1100.0 are skipped!

## Critical Bug #3: FPS Mismatch

### Location
`src/video_reader.cu` line 936, 553

### The Problem

```cpp
fps_ = computeFps(duration_hint);  // Uses video metadata
double effective_fps = (fps_ > 0.0) ? fps_ : 30.0;  // Used for timestamp calculation
```

**From verification results:**
- Video metadata shows `fps = 100.000`
- But actual video might have different frame rate
- If FPS is wrong, timestamp calculation will be wrong

### Impact

If `fps_` is incorrect:
- `estimated_frame = floor(offset_seconds * fps_)` will be wrong
- `current_ts = moment_time + (total_frames_read_ - 1) / fps_` will be wrong
- Frames will be filtered incorrectly

## Root Cause Summary

1. **Timestamp calculation after seek**: `estimated_frame` might not match actual seek position
2. **FPS accuracy**: If FPS is wrong, all timestamp calculations are wrong
3. **Time window boundaries**: Sequential logic might skip frames between videos

## Recommended Fixes

### Fix #1: Use Actual Frame Timestamps

Instead of calculating `current_ts` from frame number, use the actual frame timestamp from FFmpeg:

```cpp
// In readFrame(), after receiving frame:
AVFrame* frame = ...;  // Get from decoder
double frame_pts = frame->pts * av_q2d(video_stream_->time_base);
double current_ts = clip_.moment_time + frame_pts;  // Use actual frame timestamp
```

### Fix #2: Verify FPS

Add validation to ensure FPS is correct:
```cpp
// Compare computed FPS with video metadata
// Log warning if mismatch is significant
```

### Fix #3: Better Seek Position Tracking

After seeking, read a few frames and verify their timestamps match expectations:
```cpp
// After seek, read first frame and verify its timestamp
// Adjust total_frames_read_ if needed
```

