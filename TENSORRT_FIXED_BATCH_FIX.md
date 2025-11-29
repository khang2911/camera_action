# TensorRT Fixed Batch Size - Critical Performance Fix

## Problem

With TensorRT engines using **fixed batch_size=16**, processing partial batches is EXTREMELY inefficient:

- **1 frame**: Takes ~200ms GPU time (same as 16 frames) = **5 FPS** ✗
- **8 frames**: Takes ~200ms GPU time (same as 16 frames) = **40 FPS** ✗  
- **16 frames**: Takes ~200ms GPU time = **80 FPS** (still not 200 FPS, but much better) ✓

**The GPU processes batch_size=16 regardless of how many real frames we send!**

## Root Cause

The timeout logic was processing partial batches too early:
- Large partial (12-15 frames): 10ms timeout
- Medium partial (8-11 frames): 20ms timeout
- Small partial (< 8 frames): 100ms timeout

But queues are **429-500/500 full**, meaning frames ARE available! We just weren't waiting long enough to collect full batches.

## Fix Applied

### Aggressive Timeout Strategy:
1. **Full batch (16 frames)**: Process immediately (0ms)
2. **Large partial (12-15 frames)**: Wait up to **500ms** for more frames
3. **Medium partial (8-11 frames)**: Wait up to **1000ms**
4. **Small partial (< 8 frames)**: Wait up to **2000ms** (avoid wasting GPU)

### Key Changes:
- Increased timeouts dramatically to prioritize full batches
- Reduced queue pop timeout to 1ms (frames should be available immediately)
- Only process partial batches if queue is truly empty (no frames available)

## Why This Works

With queues 429-500/500 full:
- Frames are available immediately
- We should be able to collect 16 frames quickly
- The longer timeouts only matter if queue is actually empty
- This ensures we maximize GPU utilization

## Expected Results

### Before:
- **FPS**: 3-6 FPS
- **AvgTime**: 197-283ms/frame
- **Batch Efficiency**: ~5-10% (mostly 1-2 frame batches!)

### After:
- **FPS**: 50-80 FPS (limited by actual inference time, not batch collection)
- **AvgTime**: 12-20ms/frame (200ms batch / 16 frames)
- **Batch Efficiency**: 80-95% (mostly 16-frame batches)

## Note on 200 FPS Target

To achieve **200 FPS**, the actual inference time per batch must be:
- **200 FPS = 5ms per frame**
- **Batch of 16 = 80ms total**

But current AvgTime shows **197-283ms per batch**, which suggests:
- Actual inference time is ~200ms per batch (not 80ms)
- This gives **80 FPS** (200ms / 16 frames = 12.5ms per frame = 80 FPS)

**To reach 200 FPS, the TensorRT engine inference itself must be faster (~80ms per batch instead of 200ms).**

This fix ensures we're not wasting GPU time on partial batches, but the engine itself may need optimization or a faster GPU.

## Monitoring

After fix, check:
- **Batch sizes**: Should see mostly 16-frame batches
- **AvgTime**: Should drop to 12-20ms/frame (200ms batch / 16)
- **FPS**: Should increase to 50-80 FPS (limited by inference time)
- **Queue levels**: Should remain high (frames available)

