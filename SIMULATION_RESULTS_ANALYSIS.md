# Simulation Results Analysis

## Critical Finding

The simulation results reveal a **critical issue**: The first and last frames shown have `In Window: False`, but the simulation still says frames should be read. This indicates:

1. **The timestamp calculation is producing timestamps OUTSIDE the time window**
2. **But frames are still being counted as "should be read"**

This suggests the simulation logic has a bug, OR there's a fundamental issue with how timestamps are being calculated.

## Key Observations

### Pattern 1: Videos Reading Less Than Expected

**Video 1 (c03o24120002258):**
- Simulation says: 2,867 frames should be read
- Actually read: 582 frames
- **First Frame timestamp: 1763344412.945553, In Window: False**
- **Last Frame timestamp: 1763344633.622217, In Window: False**

**Video 2 (c03o24090015483):**
- Simulation says: 3,398 frames should be read
- Actually read: 2,864 frames
- **First Frame timestamp: 1763344439.566108, In Window: False**
- **Last Frame timestamp: 1763344701.100272, In Window: False**

**Video 3 (c03o24100009841):**
- Simulation says: 4,462 frames should be read
- Actually read: 2,867 frames
- **First Frame timestamp: 1763344408.892936, In Window: False**
- **Last Frame timestamp: 1763344752.402797, In Window: False**

### Pattern 2: Videos Reading More Than Expected

**Video 4 (c03o24100009594):**
- Simulation says: 2,654 frames should be read
- Actually read: 3,398 frames
- **First Frame timestamp: 1763344450.316437, In Window: False**
- **Last Frame timestamp: 1763344654.601672, In Window: False**

**Video 5 (c03o25010000906):**
- Simulation says: 2,971 frames should be read
- Actually read: 4,398 frames
- **First Frame timestamp: 1763344616.550782, In Window: False**
- **Last Frame timestamp: 1763344855.248195, In Window: False**

## Root Cause Hypothesis

The fact that **BOTH methods** (frame number calculation and actual PTS) give the **same results** suggests:

1. **The problem is NOT in the timestamp calculation method**
2. **The problem is likely in:**
   - The seek position (`seek_to_frame`) being incorrect
   - The time window boundaries (`start_ts`, `end_ts`) being wrong
   - The FPS being incorrect (100 FPS seems high)
   - The `moment_time` not matching the actual video start

## Next Steps

1. **Fix the simulation** to properly show which frames are in the window
2. **Add debug output** to show:
   - Actual `start_ts` and `end_ts` from the log
   - Calculated timestamps for first few frames
   - Whether frames are being filtered correctly
3. **Compare timestamps** to see if they're consistently off by a certain amount
4. **Check if FPS is correct** - 100 FPS seems suspiciously high

## Questions to Answer

1. What are the actual `start_ts` and `end_ts` values from the log?
2. Are the calculated timestamps consistently off by a fixed amount?
3. Is the FPS (100.000) correct, or is it causing timestamp calculation errors?
4. Is the `seek_to_frame` position correct after seeking?

The simulation needs to be fixed to show the actual time window boundaries and verify if frames are being correctly identified as in/out of the window.

