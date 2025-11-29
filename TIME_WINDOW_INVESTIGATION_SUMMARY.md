# Time Window Investigation Summary

## Bugs Found

### Bug #1: Timestamp Calculation After Seek (CRITICAL)

**Location**: `src/video_reader.cu` lines 951-952, 549, 556

**Problem**:
1. After seeking, `total_frames_read_` is set to `estimated_frame = floor((start_timestamp - moment_time) * fps_)`
2. But `AVSEEK_FLAG_BACKWARD` might seek to an earlier frame than estimated
3. The timestamp calculation `current_ts = moment_time + (total_frames_read_ - 1) / fps_` assumes `total_frames_read_` matches the actual frame position
4. If there's a mismatch, timestamps will be wrong, causing incorrect filtering

**Impact**: 
- **Videos reading less**: If estimated position is ahead of actual, timestamps are too high → stops early
- **Videos reading more**: If estimated position is behind actual, timestamps are too low → continues past end

**Evidence from verification**:
- Video 1: Read 582 (expected 2866) - likely stopped early due to wrong timestamp
- Video 5: Read 4398 (expected 2970) - likely continued too long due to wrong timestamp

### Bug #2: Not Using Actual Frame PTS

**Location**: `src/video_reader.cu` line 732

**Problem**:
- The code has access to `frame_->pts` (actual frame presentation timestamp from FFmpeg)
- But it calculates timestamp from frame number instead: `current_ts = moment_time + (total_frames_read_ - 1) / fps_`
- This ignores the actual frame timestamp, which is more accurate

**Solution**: Use actual frame PTS:
```cpp
// In receiveFrame() or readFrame(), after getting frame:
double frame_pts_seconds = frame_->pts * av_q2d(video_stream_->time_base);
double current_ts = clip_.moment_time + frame_pts_seconds;
```

### Bug #3: FPS Accuracy

**Location**: `src/video_reader.cu` line 936

**Problem**:
- FPS is computed from video metadata, but might not match actual frame rate
- Verification shows `fps = 100.000` which seems high
- If FPS is wrong, all timestamp calculations are wrong

**Impact**: All timestamp-based filtering will be incorrect

### Bug #4: Sequential Time Window Logic

**Location**: `src/thread_pool.cpp` lines 908, 911, 915

**Problem**:
- `clip.start_timestamp = std::max(current_window_start, video_start)` might skip frames
- If `video_start < current_window_start`, frames from `video_start` to `current_window_start` are skipped
- This might be intentional (sequential processing), but could cause issues if videos overlap

## Root Cause

The main issue is **Bug #1**: The timestamp calculation after seek doesn't account for the actual seek position. The code estimates the frame number but doesn't verify it matches the actual frame position after seeking.

## Recommended Fix Priority

1. **HIGH**: Use actual frame PTS instead of calculating from frame number
2. **HIGH**: Verify and correct `total_frames_read_` after seeking by checking first frame's PTS
3. **MEDIUM**: Add validation for FPS accuracy
4. **LOW**: Review sequential time window logic for edge cases

## Next Steps

Since we found the bugs, we should:
1. Fix the timestamp calculation to use actual frame PTS
2. Add verification after seek to ensure `total_frames_read_` matches actual position
3. Test with the verification script to confirm fixes

If fixes don't work, we'll use Python simulation as backup plan.

