# Performance Diagnosis - Still Low FPS

## Current Stats Analysis

### Detector Performance:
- **FPS**: 3-5 FPS (target: ~200 FPS) ❌
- **AvgTime**: 193-239ms/frame (target: ~5ms/frame) ❌
- **Preprocessor FPS**: 13-15 FPS
- **Preprocessor AvgTime**: 50-51ms/frame

### Key Insight:
**AvgTime = 193-239ms/frame suggests we're processing mostly 1-frame batches!**

If batch_size = 16 and we process full batches:
- Inference time: ~200ms per batch
- AvgTime = 200ms / 16 = 12.5ms/frame ✓

But we're seeing 193-239ms/frame, which means:
- **We're processing mostly 1-frame batches!**
- 1 frame batch: 200ms inference / 1 frame = 200ms/frame ✓ (matches stats)

## Root Cause Analysis

### Problem 1: Batch Collection Not Working
The batch collection loop should collect 16 frames, but it's only collecting 1 frame per batch.

**Possible causes:**
1. **Video switching too frequent**: Every frame is from a different video, so we can't collect full batches
2. **Queue pop timeout too short**: We're not waiting long enough to collect full batches
3. **Logic error**: The batch collection loop is breaking too early

### Problem 2: Preprocessor Bottleneck
- Preprocessor FPS: 13-15 FPS
- Preprocessor AvgTime: 50-51ms/frame
- This limits detector throughput to ~13-15 FPS per engine

But with 3 engines, each should get ~5 FPS, which matches the stats.

## Solutions

### Solution 1: Fix Batch Collection
We need to ensure we're actually collecting full batches (16 frames) before processing.

**Check:**
- Are frames from the same video?
- Is the batch collection loop working correctly?
- Are we breaking the loop too early?

### Solution 2: Optimize Preprocessor
- Preprocessor is taking 50ms/frame, which is 10x the target of 5ms/frame
- This is the bottleneck limiting overall throughput

### Solution 3: Add Batch Size Logging
We need to log the actual batch size being processed to confirm we're processing partial batches.

