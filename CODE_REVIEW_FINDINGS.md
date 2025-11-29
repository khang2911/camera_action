# C++ Code Review Findings

## Code Logic Review

After carefully reviewing the C++ code, I found **one potential bug** and several areas that need debug logging.

## Potential Bug Found

### Bug: End of Stream Handling

**Location**: `src/video_reader.cu` lines 612-615, 672-676

**Issue**: The reader stops when `end_of_stream_` is true, but this happens when the **video file ends**, not when the time window ends.

**Code**:
```cpp
// Line 612-615
if (!sent_any && end_of_stream_) {
    return false;  // Stops when video ends
}

// Line 672-676
if (ret == AVERROR_EOF) {
    end_of_stream_ = true;
    avcodec_send_packet(codec_ctx_, nullptr);
    return true;
}
```

**Problem**: If the video file ends before `clip_.end_timestamp` is reached, the reader will stop early. This could explain why videos are reading fewer frames than expected.

**Example**:
- `clip_.end_timestamp = 1763344633.588997` (from simulation)
- Video file ends at frame 7520 (timestamp ~1763344633.622217)
- But if the video file actually ends earlier (e.g., at frame 5234), the reader would stop there

**However**: This might be correct behavior if the video file is shorter than expected. We need to verify if the video file actually ends before `end_timestamp`.

## Logic That Appears Correct

### 1. Time Window Calculation (parseJsonToVideoClips)
```cpp
clip.start_timestamp = std::max(current_window_start, video_start);
clip.end_timestamp = std::min(global_end_ts, video_end);
current_window_start = clip.end_timestamp;
```
This logic is correct for sequential time windows.

### 2. Seek Position Calculation (initializeMetadata)
```cpp
double offset_seconds = clip_.start_timestamp - clip_.moment_time;
long long estimated_frame = floor(offset_seconds * fps_);
total_frames_read_ = estimated_frame;
```
This is correct, but `AVSEEK_FLAG_BACKWARD` might seek to an earlier frame, causing timestamp mismatch.

### 3. Frame Timestamp Calculation (readFrame)
```cpp
++total_frames_read_;
double current_ts = clip_.moment_time + (total_frames_read_ - 1) / effective_fps;
```
This is mathematically correct IF `total_frames_read_` matches the actual frame position. The issue is that after seeking, `total_frames_read_` is estimated and might not match the actual frame position.

### 4. Frame Filtering (readFrame)
```cpp
if (current_ts < clip_.start_timestamp) continue;  // Skip frames before start
if (current_ts > clip_.end_timestamp) return false;  // Stop if past end
```
This logic is correct.

## Areas Needing Debug Logging

Since the logic appears mostly correct, we need debug logging to identify where the reader is actually stopping:

1. **When `readFrame()` returns false**:
   - Is it because `current_ts > end_timestamp`?
   - Is it because `end_of_stream_` is true?
   - Is it because of some other error?

2. **Timestamp calculation verification**:
   - What is the actual `current_ts` when stopping?
   - What is `clip_.end_timestamp`?
   - What is `total_frames_read_` when stopping?

3. **Seek position verification**:
   - What is the actual frame position after seeking?
   - Does it match `estimated_frame`?

4. **End of stream detection**:
   - When does `end_of_stream_` become true?
   - Does it happen before or after `end_timestamp`?

## Recommended Debug Logging

Add logging to track:
1. When `readFrame()` returns false and why
2. The last frame's timestamp and position before stopping
3. Whether `end_of_stream_` is true when stopping
4. The actual frame position after seeking vs estimated

