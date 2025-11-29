# Debug Logging Cleanup Summary

## Removed Debug Logging

1. **Detailed time window initialization logging**:
   - Removed: Full time window details (start_ts, end_ts, moment_time, duration, video_range, actual_range, fps, time_window_duration, actual_available_duration, total_expected_frames, actual_expected_frames, seek_to_frame, seek_timestamp, remaining_duration, remaining_frames)
   - Kept: Simple warning if video ends before end_ts

2. **Frame-by-frame detailed logging**:
   - Removed: Logging for first 10 frames and every 1000th frame with total_read, actual_pos, current_ts, start_ts, end_ts, frame_pts, actual_ts_from_pts, calc_ts, ts_diff, moment_time, fps
   - Removed: Frame format logging (format, expected format, match status, hw_frames_ctx)

3. **Skip frame logging**:
   - Removed: "Skipping frame before start_ts" debug messages

4. **PTS comparison logging**:
   - Removed: "Frame PTS comparison" messages with frame_pts, actual_ts_from_pts, calc_ts, calc_ts_diff

5. **EOF detailed logging**:
   - Removed: Detailed EOF messages with total_read, actual_pos, current_ts, end_ts, remaining time, expected_frames_remaining
   - Removed: "EOF DETECTED" messages from sendNextPacket

6. **Seek calculation logging**:
   - Removed: "Seek calculation" debug messages with offset_seconds, estimated_frame, start_ts, moment_time, fps

7. **GPU/CPU conversion stats**:
   - Removed: "GPU conversion stats" and "CPU conversion stats" messages logged every 1000 frames

8. **Duplicate PTS logging**:
   - Removed: "Frame with duplicate PTS detected" debug messages

## Kept Essential Status Messages

1. **"Reached end of time window: frames_read=X"**
   - Logged when video reaches the end of the time window
   - Shows the number of frames read within the time window

2. **"Reached end of video file: frames_read=X"**
   - Logged when video file ends before reaching end_ts
   - Shows the number of frames read before video ended

3. **"Time window: video ends Xs before end_ts"** (warning)
   - Logged during initialization if video is shorter than time window
   - Useful for understanding why fewer frames were read than expected

## Other Kept Logging

- Essential decoder initialization messages (hardware decoder found, codec opened, etc.)
- Error messages (GPU conversion failed, etc.)
- Warnings (format mismatches, unused decoder options, etc.)

These are necessary for troubleshooting and are not verbose debug logs.

## Result

The log output is now clean and focused on:
- Essential initialization information
- Status messages about video completion (end of time window or end of file)
- Warnings and errors only

No more verbose frame-by-frame or detailed timestamp debugging information.

