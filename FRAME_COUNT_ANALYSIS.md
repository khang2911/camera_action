# Frame Count Verification Analysis

## Summary of Issues

The verification results reveal **critical problems** with the time window filtering logic:

### Pattern 1: Videos Reading LESS Than Expected (Videos 1-3)

**Video 1 (c03o24120002258):**
- Expected: 2,866 frames
- Actually read: 582 frames (79.69% missing!)
- Video has: 7,048 frames available from seek position
- **Problem**: Reader stopped after only 582 frames, but video has 7,048 available

**Video 2 (c03o24090015483):**
- Expected: 3,398 frames
- Actually read: 2,864 frames (15.72% missing)
- Video has: 6,806 frames available from seek position
- **Problem**: Reader stopped early, missing 534 frames

**Video 3 (c03o24100009841):**
- Expected: 4,462 frames
- Actually read: 2,867 frames (35.75% missing)
- Video has: 7,271 frames available from seek position
- **Problem**: Reader stopped early, missing 1,595 frames

### Pattern 2: Videos Reading MORE Than Expected (Videos 4-5)

**Video 4 (c03o24100009594):**
- Expected: 2,653 frames
- Actually read: 3,398 frames (28.08% MORE than expected!)
- Video has: 7,524 frames available from seek position
- **Problem**: Reader continued past the time window boundary

**Video 5 (c03o25010000906):**
- Expected: 2,970 frames
- Actually read: 4,398 frames (48.08% MORE than expected!)
- Video has: 5,642 frames available from seek position
- **Problem**: Reader continued past the time window boundary

## Root Cause Analysis

### Issue 1: Time Window Calculation May Be Incorrect

The sequential time window logic in `parseJsonToVideoClips()` sets:
```cpp
clip.end_timestamp = std::min(global_end_ts, video_end);
current_window_start = clip.end_timestamp;  // For next video
```

**Problem**: If a video's `video_end` (moment_time + duration) is **before** `global_end_ts`, then:
- The video's `end_timestamp` is set to `video_end`
- But the reader might stop when it reaches `video_end` even though there are more frames in the time window
- OR the next video's `start_timestamp` might be incorrectly set

### Issue 2: Timestamp Calculation in readFrame()

The reader calculates frame timestamp as:
```cpp
double current_ts = clip_.moment_time + static_cast<double>(total_frames_read_ - 1) / effective_fps;
```

**Potential Problems**:
1. If `fps_` is incorrect (e.g., video metadata says 100 FPS but actual is different), timestamps will be wrong
2. If `total_frames_read_` starts from a seek position, the calculation might be off
3. The `-1` adjustment might cause off-by-one errors

### Issue 3: Reader Stops Too Early or Continues Too Long

The reader stops when:
```cpp
if (current_ts > clip_.end_timestamp) {
    return false;  // Past end time, done with this clip
}
```

**Problems**:
- If `current_ts` calculation is wrong, it might stop too early (Pattern 1)
- If `clip_.end_timestamp` is wrong, it might continue too long (Pattern 2)
- If the time window spans multiple videos, the logic might not account for sequential processing correctly

## Specific Issues Per Video

### Video 1: Only 582 frames read (expected 2,866)
- **Hypothesis**: The reader's `end_timestamp` is set incorrectly, causing it to stop at `video_end` instead of the actual time window end
- **Evidence**: Video has 7,048 frames available, but only 582 were read

### Video 2: 2,864 frames read (expected 3,398)
- **Hypothesis**: Similar to Video 1, but closer to expected (missing 534 frames)
- **Evidence**: Video has 6,806 frames available

### Video 3: 2,867 frames read (expected 4,462)
- **Hypothesis**: Reader stopped early, possibly due to incorrect `end_timestamp` or timestamp calculation
- **Evidence**: Video has 7,271 frames available

### Video 4: 3,398 frames read (expected 2,653) - **READ TOO MUCH**
- **Hypothesis**: The `end_timestamp` is set incorrectly, allowing the reader to continue past the time window
- **Evidence**: Read 745 frames MORE than expected

### Video 5: 4,398 frames read (expected 2,970) - **READ TOO MUCH**
- **Hypothesis**: Similar to Video 4, `end_timestamp` is wrong
- **Evidence**: Read 1,428 frames MORE than expected

## Recommendations

1. **Add detailed logging** to track:
   - `clip_.start_timestamp` and `clip_.end_timestamp` for each video
   - `current_ts` for each frame (at least first and last few)
   - When and why the reader stops (`current_ts > end_timestamp`)

2. **Verify time window calculation**:
   - Check if `current_window_start` is being updated correctly
   - Verify that `clip.end_timestamp` is set correctly for each video
   - Ensure sequential time windows don't overlap or have gaps

3. **Verify timestamp calculation**:
   - Check if `fps_` is correct (100 FPS seems high - verify with ffprobe)
   - Verify `total_frames_read_` starts from correct position after seek
   - Check if the `-1` adjustment in timestamp calculation is correct

4. **Check for edge cases**:
   - What happens when `video_end < global_end_ts`?
   - What happens when `video_end > global_end_ts`?
   - What happens when videos are processed sequentially?

## Next Steps

1. Add debug logging to see actual `start_timestamp`, `end_timestamp`, and `current_ts` values
2. Verify the sequential time window logic is working correctly
3. Check if FPS calculation is accurate (100 FPS seems suspicious)
4. Investigate why some videos read less and others read more than expected

