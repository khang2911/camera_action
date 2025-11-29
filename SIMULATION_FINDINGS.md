# Simulation Findings - Root Cause Identified

## Key Discovery

The simulation logic is **CORRECT** and matches expected frames, but the **C++ implementation is stopping early**.

## Evidence

### Video 1 (c03o24120002258)
- **Simulation says**: 2,867 frames should be read ✓
- **Expected**: 2,866 frames ✓
- **Actually read**: 582 frames ✗
- **Difference**: Missing 2,285 frames (79.7%)

### Video 2 (c03o24090015483)
- **Simulation says**: 3,398 frames should be read ✓
- **Expected**: 3,398 frames ✓
- **Actually read**: 2,864 frames ✗
- **Difference**: Missing 534 frames (15.7%)

### Video 3 (c03o24100009841)
- **Simulation says**: 4,462 frames should be read ✓
- **Expected**: 4,462 frames ✓
- **Actually read**: 2,867 frames ✗
- **Difference**: Missing 1,595 frames (35.8%)

## Critical Pattern

**The first frame from seek position is OUTSIDE the time window:**
- Frame 4652: Timestamp 1763344412.945553, **In Window: False**
- Frame 4653: Timestamp 1763344413.022497, **In Window: True** ✓

**The C++ code behavior:**
1. Seeks to frame 4652 (estimated position)
2. Reads frame 4652 → timestamp is BEFORE `start_ts` → **skips it** (correct)
3. Reads frame 4653 → timestamp is in window → **should read it**
4. But then **stops early** after reading only 582 frames instead of 2,867

## Root Cause Hypothesis

The C++ code is **stopping prematurely** for one of these reasons:

### Hypothesis 1: Early Stop Condition
The reader might be hitting an early stop condition that's not in the simulation:
- Video ends before `end_ts`?
- Some other condition causing early return?

### Hypothesis 2: Seek Position Mismatch
The `AVSEEK_FLAG_BACKWARD` might seek to a position that's **further back** than estimated:
- Estimated: frame 4652
- Actual seek: might be frame 4000 or earlier
- This would cause timestamps to be wrong, leading to early stopping

### Hypothesis 3: FPS Mismatch
The FPS (100.000) might be incorrect:
- If actual FPS is different, timestamps will be wrong
- This could cause frames to be filtered incorrectly

### Hypothesis 4: Time Window Boundary Issue
The `end_timestamp` might be set incorrectly:
- If `end_timestamp` is set to `video_end` instead of `global_end_ts`, it would stop early
- This matches the pattern where videos stop before reading all expected frames

## Most Likely Cause

**Hypothesis 4** is most likely: The `end_timestamp` is being set to `video_end` instead of `global_end_ts` when `video_end < global_end_ts`.

Looking at the code:
```cpp
clip.end_timestamp = std::min(global_end_ts, video_end);
```

If `video_end < global_end_ts`, then `end_timestamp = video_end`, which would cause the reader to stop when the video ends, even if there's more time window to cover.

But wait - the sequential logic should handle this. If video 0 ends at `video_end`, video 1 should continue from there.

**Unless...** the videos are being processed independently, and each video's `end_timestamp` is being clipped to its own `video_end`, causing early stopping.

## Next Steps

1. **Check the actual `end_timestamp` value** in the C++ code for each video
2. **Verify if `video_end < global_end_ts`** is causing early stopping
3. **Check if the sequential time window logic is working correctly**
4. **Add logging to see when and why the reader stops**

The simulation proves the logic is correct - the bug is in the C++ implementation stopping early.

