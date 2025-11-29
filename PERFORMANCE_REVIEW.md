# Performance Review & Optimization Recommendations

## Current Architecture Overview

**Pipeline Flow:**
1. **Reader Threads** → Read frames from video → Push to `raw_frame_queue_`
2. **Preprocessor Threads** → Pop from `raw_frame_queue_` → Preprocess → Push to engine-specific queues
3. **Detector Threads** → Pop from engine queues → Run TensorRT inference → Write to bin files

## Identified Performance Bottlenecks

### 🔴 Critical Issues

#### 1. **Unnecessary Frame Cloning** (High Impact)
**Location:** `src/thread_pool.cpp:720`
```cpp
FrameData frame_data(frame.clone(), video_id, actual_frame_position, clip.path, ...);
```

**Problem:** Every frame is deep-copied when creating `FrameData`, even though the frame is only read by preprocessors.

**Impact:** 
- Memory bandwidth bottleneck
- CPU time wasted on copying large frames (1920x1080 = ~6MB per frame)
- At 30 FPS, this is ~180MB/s of unnecessary copying

**Recommendation:**
- Use move semantics or shared pointers for frames
- Only clone if multiple threads need to modify the frame simultaneously
- Consider using `cv::Mat` reference counting (already implemented, but clone() breaks it)

#### 2. **Synchronous CUDA Operations** (High Impact)
**Location:** `src/yolo_detector.cpp:342, 897, 1031`
```cpp
cudaMemcpyAsync(..., stream_);
cudaStreamSynchronize(stream_);  // Blocks CPU thread
```

**Problem:** CPU thread blocks waiting for GPU operations to complete, preventing overlap of computation and data transfer.

**Impact:**
- GPU idle time while CPU waits
- CPU idle time while GPU processes
- No pipelining between batches

**Recommendation:**
- Remove `cudaStreamSynchronize` after input copy (only sync before reading output)
- Use CUDA events to track completion instead of blocking
- Implement async pipeline: while batch N is inferencing, copy batch N+1 to GPU

#### 3. **File I/O Synchronization** (Medium Impact)
**Location:** `src/yolo_detector.cpp:799`
```cpp
std::lock_guard<std::mutex> file_lock(*file_mutex);
// Synchronous file write
out_file.write(...);
```

**Problem:** Each frame's detections are written synchronously, blocking the detector thread.

**Impact:**
- Detector threads blocked on disk I/O
- Multiple threads contending for file mutex
- Disk I/O not batched

**Recommendation:**
- Use async file I/O with a dedicated writer thread
- Batch multiple frames' detections before writing
- Use memory-mapped files for faster writes
- Consider buffering writes and flushing periodically

### 🟡 Medium Priority Issues

#### 4. **Preprocessing Redundancy** (Medium Impact)
**Location:** `src/thread_pool.cpp:976, 1125`
```cpp
if (!tensor) {
    tensor = engine_group->acquireBuffer();
    engine_group->preprocessor->preprocessToFloat(frame_data.frame, *tensor);
}
```

**Problem:** If preprocessor didn't create `preprocessed_data`, detector does it again, duplicating work.

**Impact:**
- Wasted CPU cycles
- Potential queue imbalance if preprocessors are slow

**Recommendation:**
- Ensure preprocessors always create `preprocessed_data`
- Add monitoring to detect when this fallback occurs
- Consider increasing preprocessor threads if this happens frequently

#### 5. **Queue Size Limitations** (Medium Impact)
**Location:** `src/thread_pool.cpp:89`
```cpp
raw_frame_queue_ = std::make_unique<FrameQueue>(500);
```

**Problem:** Fixed queue size of 500 may cause blocking if readers are faster than preprocessors.

**Impact:**
- Reader threads block when queue is full
- Reduced parallelism

**Recommendation:**
- Make queue size configurable
- Monitor queue sizes and adjust thread counts dynamically
- Consider unbounded queues with memory limits

#### 6. **Video Reading with OpenCV** (Medium Impact)
**Location:** `src/video_reader.cpp`

**Problem:** OpenCV's `VideoCapture` may not be optimal for high-throughput scenarios.

**Impact:**
- Slower frame reading compared to specialized libraries
- No hardware acceleration
- Seeking may be slow

**Recommendation:**
- Consider using FFmpeg directly (you tried this but reverted due to performance issues)
- Use hardware-accelerated decoding (NVDEC) if available
- Pre-decode frames in batches
- Consider using decord (Python) or similar for C++

#### 7. **Buffer Pool Contention** (Low-Medium Impact)
**Location:** `src/thread_pool.cpp:26-52`

**Problem:** Buffer pool uses mutex for every acquire/release operation.

**Impact:**
- Lock contention when many threads access buffer pool
- Potential serialization bottleneck

**Recommendation:**
- Use lock-free data structures (e.g., `boost::lockfree::stack`)
- Per-thread buffer pools to reduce contention
- Pre-allocate buffers at startup

### 🟢 Low Priority / Optimization Opportunities

#### 8. **Statistics Lock Contention** (Low Impact)
**Location:** Multiple locations in `src/thread_pool.cpp`

**Problem:** Frequent locking of `stats_.stats_mutex` for atomic operations.

**Impact:**
- Minor overhead, but accumulates with many threads

**Recommendation:**
- Use atomic variables instead of mutex-protected variables
- Batch statistics updates
- Use lock-free counters

#### 9. **String Operations in Hot Path** (Low Impact)
**Location:** `src/thread_pool.cpp:967-971` (generateOutputPath called frequently)

**Problem:** String concatenation and path generation happens for every frame.

**Impact:**
- Minor CPU overhead
- Memory allocations

**Recommendation:**
- Cache output paths per video/engine combination
- Use string views where possible
- Pre-compute paths at video start

#### 10. **NMS Performance** (Low Impact)
**Location:** `src/yolo_detector.cpp:applyNMS`

**Problem:** CPU-based NMS may be slow for many detections.

**Impact:**
- Post-processing bottleneck if many detections per frame

**Recommendation:**
- Use GPU-accelerated NMS (TensorRT plugin)
- Optimize NMS algorithm (currently O(n²))
- Consider using `std::nth_element` for faster sorting

## Recommended Optimization Priority

### Phase 1: Quick Wins (High Impact, Low Risk)
1. ✅ Remove unnecessary `frame.clone()` - use move semantics
2. ✅ Remove `cudaStreamSynchronize` after input copy
3. ✅ Cache output file paths per video/engine

### Phase 2: Medium Effort (High Impact, Medium Risk)
4. ✅ Implement async file I/O with writer thread
5. ✅ Use CUDA events for async pipeline
6. ✅ Make queue sizes configurable and monitor them

### Phase 3: Advanced (Medium-High Impact, Higher Risk)
7. ✅ Implement GPU-accelerated NMS
8. ✅ Use hardware-accelerated video decoding (NVDEC)
9. ✅ Implement lock-free buffer pools

## Performance Monitoring Recommendations

Add metrics to track:
- Queue sizes (raw_frame_queue, per-engine queues)
- GPU utilization
- CPU utilization per thread type
- Memory bandwidth usage
- File I/O wait times
- Frame drop rate (if queues are full)
- Preprocessing fallback frequency

## Expected Performance Gains

| Optimization | Expected Speedup | Effort |
|-------------|------------------|--------|
| Remove frame.clone() | 10-20% | Low |
| Async CUDA operations | 15-30% | Medium |
| Async file I/O | 5-15% | Medium |
| GPU NMS | 5-10% | High |
| Hardware video decode | 10-20% | High |
| **Combined** | **50-100%** | - |

## Implementation Notes

- Test each optimization independently
- Use profiling tools (nvprof, perf, Intel VTune) to measure impact
- Monitor for regressions in correctness
- Consider A/B testing in production

