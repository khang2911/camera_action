# Final Performance Diagnosis

## The Real Problem

**AvgTime = 193-239ms/frame means we're processing mostly 1-frame batches!**

### Math:
- If batch_size = 16 and we process full batches:
  - Inference: ~200ms per batch
  - AvgTime = 200ms / 16 = 12.5ms/frame ✓
  
- But we're seeing 193-239ms/frame, which means:
  - We're processing mostly 1-frame batches
  - 1 frame batch: 200ms inference / 1 frame = 200ms/frame ✓ (matches stats!)

## Root Cause

The batch collection is breaking too early, processing partial batches instead of waiting for full batches.

### Why This Happens:
1. **Video switching detection is too aggressive**: We break after 10 consecutive different video frames
2. **We're processing partial batches when videos finish**: This is inefficient
3. **The logic allows partial batch processing**: We should ONLY process full batches

## Solution

**Match prod branch exactly: NEVER process partial batches, only full batches!**

Even when videos finish, we should:
- Drop the partial batch (frames will be lost, but processing continues)
- OR wait indefinitely for full batches (prod branch strategy)

The current logic processes partial batches when videos finish, which causes the low FPS.

