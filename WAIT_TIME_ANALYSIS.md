# Wait Time Analysis - Target: 5ms/frame

## Wait Times > 5ms Found in Pipeline

### 1. Preprocessor Dispatcher - Pop from Raw Queue
- **Location**: Line 1124
- **Current**: `raw_frame_queue_->pop(raw_frame, 50)` - **50ms** ⚠️
- **Impact**: High - This is the entry point for preprocessing
- **Fix**: Reduce to 1ms when queue has frames

### 2. Preprocessor Worker - Pop from Preprocess Group Queue
- **Location**: Line 1195-1196
- **Current**: `timeout = (attempt == 0) ? 100 : 5` - **100ms on first attempt** ⚠️
- **Impact**: High - Workers wait up to 100ms per frame
- **Fix**: Reduce first attempt to 1ms, keep retry at 5ms

### 3. Preprocessor Dispatcher - Push to Preprocess Groups
- **Location**: Line 1153-1155
- **Current**: 10ms, 50ms, 200ms progressive timeouts ⚠️
- **Impact**: Medium - Only affects when queues are full
- **Fix**: Reduce to 1ms, 5ms, 10ms

### 4. Reader - Push to Raw Queue
- **Location**: Line 1057-1059
- **Current**: 10ms, 50ms, 200ms progressive timeouts ⚠️
- **Impact**: Medium - Only affects when raw queue is full
- **Fix**: Reduce to 1ms, 5ms, 10ms

### 5. Engine Preprocessor - Push to Engine Queues
- **Location**: Line 1290-1292
- **Current**: 10ms, 50ms progressive timeouts ⚠️
- **Impact**: Medium - Only affects when engine queues are full
- **Fix**: Reduce to 1ms, 5ms

### 6. Detector - Pop from Engine Queue
- **Location**: Line 1426
- **Current**: `pop_timeout = (queue->size() > 100) ? 1 : 100` - **100ms when queue <= 100** ⚠️
- **Impact**: High - Detectors wait up to 100ms per frame when queue is small
- **Fix**: Reduce to 1ms when queue has any frames, 5ms when empty

## Summary

**Total wait times per frame (worst case)**:
- Preprocessor dispatcher pop: 50ms
- Preprocessor worker pop: 100ms
- Detector pop: 100ms
- **Total: 250ms** (50x the target of 5ms!)

**With queues full (best case)**:
- All operations should be immediate (< 1ms)
- But current timeouts still cause delays even when frames are available

## Fix Strategy

1. **Reduce all timeouts to <= 5ms when queues have frames**
2. **Use 1ms timeout when queue size > threshold**
3. **Only use longer timeouts (5-10ms) when queues are truly empty**
4. **Eliminate 50ms+ timeouts entirely**

