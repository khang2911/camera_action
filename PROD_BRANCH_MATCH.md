# Matched Prod Branch Batch Collection Logic

## Changes Applied

Updated batch collection to match **prod branch** exactly, which works correctly with TensorRT fixed batch_size.

## Key Differences from Previous Implementation

### Previous (Not Working):
- Complex timeout logic for partial batches (10ms, 20ms, 100ms, 500ms, 1000ms, 2000ms)
- Processed partial batches after timeout
- This caused low FPS because partial batches waste GPU time

### Prod Branch (Working):
- **Simple 100ms timeout** for queue pop
- **NO partial batch processing** - only processes when `batch_tensors.size() == batch_size`
- Just continues waiting if queue is empty
- Logs waiting status every 5 seconds

## Implementation Details

### Batch Collection Loop:
```cpp
while (batch_tensors.size() < batch_size && !stop_flag_) {
    if (!engine_group->frame_queue->pop(frame_data, 100)) {
        if (stop_flag_) break;
        // Log waiting status periodically (every 5 seconds)
        continue;  // Just continue waiting - no partial batch processing!
    }
    // ... add frame to batch ...
}

// Only process if exactly batch_size frames
if (batch_tensors.size() == batch_size) {
    // Process batch
}
```

### Why This Works:

1. **TensorRT Fixed Batch Size**: With fixed batch_size=16, TensorRT always processes 16 frames
   - 1 frame = ~200ms GPU time (same as 16 frames)
   - 16 frames = ~200ms GPU time
   - So partial batches waste GPU time!

2. **Prod Branch Strategy**: 
   - Only processes full batches (16 frames)
   - Waits indefinitely for full batches
   - No partial batch processing = maximum GPU efficiency

3. **With Queues 429-500/500 Full**:
   - Frames are available immediately
   - Should collect 16 frames quickly
   - No need for partial batch processing

## Expected Results

### Before (Complex Timeout Logic):
- **FPS**: 3-6 FPS
- **AvgTime**: 197-283ms/frame
- **Issue**: Processing too many partial batches

### After (Prod Branch Logic):
- **FPS**: Should match prod branch performance
- **AvgTime**: Should match prod branch (likely 12-20ms/frame for full batches)
- **Batch Efficiency**: 100% (only full batches)

## Additional Features Preserved

- **Video switching**: Still handles frames from different videos correctly
- **Path caching**: Still uses cached output paths
- **Video key reuse**: Still uses pre-computed video_key
- **Frame sorting**: Still sorts batches if needed

## Testing

After this change, performance should match prod branch. Monitor:
- **Batch sizes**: Should see only 16-frame batches
- **FPS**: Should match prod branch
- **AvgTime**: Should match prod branch
- **Queue levels**: Should remain high (frames available)

