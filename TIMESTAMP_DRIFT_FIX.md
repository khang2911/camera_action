# Timestamp Drift Fix

## Problem Identified

From the log analysis, we discovered a critical bug in the time window filtering logic:

1. **Timestamp Calculation Drift**: The code was using `total_frames_read_` to calculate timestamps after seeking, but this value is only an estimate based on `offset_seconds * fps_`. The actual frame position after seeking can differ significantly due to:
   - FPS inaccuracy
   - Seek precision (AVSEEK_FLAG_BACKWARD seeks to nearest keyframe before target)
   - Frame rate variations in the video

2. **Evidence from Logs**:
   - Line 1248: `calc_ts_diff=-162.217003` (162 seconds drift!)
   - Line 1052: `calc_ts_diff=-30.542778` (30 seconds drift)
   - Videos were stopping prematurely because `calc_ts > end_ts` even though the actual PTS-based timestamp was still within the time window

3. **Example from Log**:
   ```
   [2025-11-29 20:11:14.872] [INFO ] [VideoReader] *** STOPPING: Reached end of time window ***
     current_ts=1763345626.042997, end_ts=1763345625.967567
   [2025-11-29 20:11:14.872] [INFO ] [VideoReader]   Frame PTS comparison:
     frame_pts=699.260000, actual_ts_from_pts=1763345788.260000, calc_ts_diff=-162.217003
   ```
   The calculated timestamp (`current_ts`) exceeded `end_ts`, but the actual PTS-based timestamp (`actual_ts_from_pts`) was still 162 seconds away from the calculated value, meaning the frame was actually still within the time window.

## Solution

Modified `src/video_reader.cu` to **use the actual PTS (Presentation Timestamp) from the frame** when available, instead of relying solely on the calculated timestamp from `total_frames_read_`.

### Changes Made

1. **Prioritize PTS over calculated timestamp**:
   - When `frame_pts_seconds >= 0.0`, use `actual_ts_from_pts = clip_.moment_time + frame_pts_seconds` as `current_ts`
   - Fall back to calculated timestamp only when PTS is unavailable

2. **Enhanced logging**:
   - Log both calculated and PTS-based timestamps
   - Show the difference between them for diagnosis
   - Indicate when PTS is not available

### Code Changes

**Before**:
```cpp
double current_ts = clip_.moment_time + static_cast<double>(total_frames_read_ - 1) / effective_fps;
```

**After**:
```cpp
double current_ts;
double actual_ts_from_pts = -1.0;
double calc_ts_from_frame_count = clip_.moment_time + static_cast<double>(total_frames_read_ - 1) / effective_fps;

if (frame_pts_seconds >= 0.0) {
    // Use actual PTS from frame (most accurate)
    actual_ts_from_pts = clip_.moment_time + frame_pts_seconds;
    current_ts = actual_ts_from_pts;
} else {
    // Fall back to calculated timestamp (less accurate, but better than nothing)
    current_ts = calc_ts_from_frame_count;
}
```

## Expected Results

1. **Accurate time window filtering**: Frames will be filtered based on their actual timestamps from the video file, not estimates
2. **No premature stopping**: Videos will continue reading until the actual end of the time window or end of file
3. **Better frame count accuracy**: `frames_read` should now match `total_expected_frames` more closely (accounting for video file duration limits)

## Testing

After rebuilding, check the logs for:
- Reduced or eliminated `calc_ts_diff` values (should be close to 0 when PTS is available)
- `frames_read` matching `total_expected_frames` more closely
- No premature "Reached end of time window" messages when frames are still available

