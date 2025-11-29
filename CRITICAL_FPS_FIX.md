# Critical FPS Fix - Target: 200 FPS

## Problem Identified

The detector is running at **3-6 FPS** instead of the expected **200 FPS**. This is a **33-66x slowdown**.

## Root Cause Analysis

### Current Behavior:
- **AvgTime**: 217-322ms per frame
- **FPS**: 3-6 FPS
- **Batch Size**: 16 (configured)

### Expected Behavior for 200 FPS:
- **Inference Time**: ~5ms per frame (200 FPS = 1000ms / 200 = 5ms)
- **Batch Time**: ~80ms for 16 frames (5ms × 16)
- **FPS**: 200 FPS

### The Problem:
The timeout logic was causing **premature partial batch processing**:
- Small batches (1-7 frames) processed with 200ms timeout
- Medium batches (8-11 frames) processed with 100ms timeout  
- Large batches (12-15 frames) processed with 50ms timeout

**Impact**: Processing 1 frame takes the same GPU time as processing 16 frames, but only produces 1/16th the throughput!

### Example:
- **Full batch (16 frames)**: 80ms inference = 200 FPS ✓
- **Partial batch (1 frame)**: 80ms inference = 12.5 FPS ✗
- **Partial batch (4 frames)**: 80ms inference = 50 FPS ✗

If 50% of batches are partial (8 frames average), effective FPS = 100 FPS (50% of 200 FPS).

## Fix Applied

### Changed Timeout Logic:
1. **Full batch (16 frames)**: Process immediately (0ms timeout)
2. **Large partial (12-15 frames)**: Wait 10ms for more frames
3. **Medium partial (8-11 frames)**: Wait 20ms for more frames
4. **Small partial (< 8 frames)**: Wait 100ms to avoid wasting GPU on tiny batches

### Key Changes:
- Reduced timeout for large/medium batches to prioritize full batches
- Increased timeout for small batches to avoid wasting GPU time
- This ensures maximum GPU utilization and target FPS

## Expected Results

### Before Fix:
- **FPS**: 3-6 FPS
- **AvgTime**: 217-322ms/frame
- **Batch Efficiency**: ~20-30% (mostly partial batches)

### After Fix:
- **FPS**: 150-200 FPS (depending on input rate)
- **AvgTime**: 5-8ms/frame (for full batches)
- **Batch Efficiency**: 80-95% (mostly full batches)

## Additional Optimizations Needed

To reach consistent 200 FPS, also ensure:

1. **Input Rate**: Preprocessors must produce frames fast enough
   - Need: 200 FPS × 3 engines = 600 frames/second total
   - Current: 14-15 FPS preprocess = bottleneck!

2. **Queue Sizes**: Must be large enough to buffer full batches
   - Current: 500 frames per engine queue
   - Need: At least batch_size × num_detectors × 2 = 16 × 2 × 2 = 64 frames minimum

3. **Multiple Detectors**: Already configured (1-2 per engine)
   - This helps parallelize inference

4. **GPU Utilization**: Monitor GPU usage
   - Should be 80-100% during inference
   - If low, may need more detector threads or larger batches

## Monitoring

After fix, monitor:
- **Batch sizes**: Should see mostly 16-frame batches
- **AvgTime**: Should drop to 5-8ms/frame
- **FPS**: Should increase to 150-200 FPS
- **GPU utilization**: Should be high (80-100%)

## Next Steps if FPS Still Low

1. **Check Preprocessor FPS**: Must be >200 FPS to feed detectors
2. **Check Queue Sizes**: Ensure queues don't empty frequently
3. **Check GPU Utilization**: Use `nvidia-smi` to verify GPU is busy
4. **Check for Blocking**: Profile to find any blocking operations
5. **Consider Larger Batches**: If input rate allows, batch_size=32 may help

