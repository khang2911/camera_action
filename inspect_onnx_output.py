#!/usr/bin/env python3
"""
Script to inspect ONNX model output and compare with C++ TensorRT output.
This helps diagnose if confidence values are actually low or if there's a parsing issue.
"""

import numpy as np
import onnxruntime as ort
import sys
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available, will use dummy input")

def inspect_onnx_output(onnx_path, image_path=None):
    """Inspect ONNX model output for a given input."""
    
    print("=" * 80)
    print("ONNX Model Output Inspection")
    print("=" * 80)
    
    # Load ONNX model
    print(f"\nLoading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path)
    
    # Get input/output info
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]
    
    print(f"\nInput:")
    print(f"  Name: {input_info.name}")
    print(f"  Shape: {input_info.shape}")
    print(f"  Type: {input_info.type}")
    
    print(f"\nOutput:")
    print(f"  Name: {output_info.name}")
    print(f"  Shape: {output_info.shape}")
    print(f"  Type: {output_info.type}")
    
    # Prepare input
    batch_size, channels, height, width = input_info.shape
    print(f"\nPreparing input: batch_size={batch_size}, channels={channels}, height={height}, width={width}")
    
    if image_path and os.path.exists(image_path) and HAS_CV2:
        print(f"Loading image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}, using dummy input")
            img_input = np.zeros((batch_size, channels, height, width), dtype=np.float32)
        else:
            # Resize to model input size
            img_resized = cv2.resize(img, (width, height))
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] and convert to float32
            img_normalized = img_rgb.astype(np.float32) / 255.0
            # Add batch dimension and transpose to CHW format
            img_input = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)
            # Pad batch if needed
            if batch_size > 1:
                img_input = np.repeat(img_input, batch_size, axis=0)
                print(f"  Padded batch to {batch_size} (repeating first image)")
    else:
        if image_path and not HAS_CV2:
            print("Warning: cv2 not available, cannot load image, using dummy input")
        elif image_path:
            print(f"Warning: Image path provided but file not found: {image_path}, using dummy input")
        else:
            print("No image provided, using dummy input (zeros)")
        img_input = np.zeros((batch_size, channels, height, width), dtype=np.float32)
    
    print(f"Input shape: {img_input.shape}")
    print(f"Input range: [{img_input.min():.6f}, {img_input.max():.6f}]")
    
    # Run inference
    print("\nRunning inference...")
    output_name = output_info.name
    outputs = sess.run([output_name], {input_info.name: img_input})
    output = outputs[0]
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Analyze output format
    # TensorRT/C++ format: [batch, channels, num_anchors] = [batch, 5, 15309]
    # Python postprocess does: output[0].T to get [num_anchors, channels] = [15309, 5]
    batch_size_out, num_channels, num_anchors = output.shape
    print(f"\nOutput dimensions: batch={batch_size_out}, channels={num_channels}, num_anchors={num_anchors}")
    
    # Extract batch 0
    batch_0 = output[0]  # Shape: [channels, num_anchors] = [5, 15309]
    print(f"\nBatch 0 shape: {batch_0.shape}")
    
    # Transpose to match Python postprocess: predictions = output[0].T
    predictions = batch_0.T  # Shape: [num_anchors, channels] = [15309, 5]
    print(f"After transpose (predictions = output[0].T): {predictions.shape}")
    
    # Extract confidence channel (channel 4, index 4)
    confidences = predictions[:, 4]  # Shape: [num_anchors]
    print(f"\nConfidence values (predictions[:, 4]):")
    print(f"  Shape: {confidences.shape}")
    print(f"  Min: {confidences.min():.10f}")
    print(f"  Max: {confidences.max():.10f}")
    print(f"  Mean: {confidences.mean():.10f}")
    print(f"  Std: {confidences.std():.10f}")
    print(f"  Non-zero count: {np.count_nonzero(confidences)}")
    print(f"  Values > 0.01: {np.sum(confidences > 0.01)}")
    print(f"  Values > 0.1: {np.sum(confidences > 0.1)}")
    print(f"  Values > 0.2: {np.sum(confidences > 0.2)}")
    
    # Show top confidence values
    top_k = 20
    top_indices = np.argsort(confidences)[::-1][:top_k]
    print(f"\nTop {top_k} confidence values:")
    for i, idx in enumerate(top_indices):
        conf = confidences[idx]
        # Get corresponding bbox values
        cx = predictions[idx, 0]
        cy = predictions[idx, 1]
        w = predictions[idx, 2]
        h = predictions[idx, 3]
        print(f"  Rank {i+1:2d}: Anchor {idx:5d}, conf={conf:.10f}, "
              f"bbox=[cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}]")
    
    # Show first few anchors
    print(f"\nFirst 10 anchors (after transpose):")
    for i in range(min(10, num_anchors)):
        cx = predictions[i, 0]
        cy = predictions[i, 1]
        w = predictions[i, 2]
        h = predictions[i, 3]
        conf = predictions[i, 4]
        print(f"  Anchor {i:5d}: cx={cx:8.2f}, cy={cy:8.2f}, w={w:8.2f}, h={h:8.2f}, conf={conf:.10f}")
    
    # Compare with C++ debug output format
    print(f"\n" + "=" * 80)
    print("Comparison with C++ TensorRT output format:")
    print("=" * 80)
    print(f"\nC++ reads as [num_anchors, channels] after transpose:")
    print(f"  Anchor 0, Channel 0 (cx): index=0 -> value={predictions[0, 0]:.5f}")
    print(f"  Anchor 0, Channel 1 (cy): index=1 -> value={predictions[0, 1]:.5f}")
    print(f"  Anchor 0, Channel 4 (conf): index=4 -> value={predictions[0, 4]:.10f}")
    print(f"\nC++ raw_output_debug.txt shows (for alcohol model, 8400 anchors):")
    print(f"  Anchor 0: cx=4.80859, cy=9.3125, conf=1.49012e-07")
    print(f"  (Note: This is a different model with different anchor count)")
    
    # Check if values need sigmoid
    print(f"\n" + "=" * 80)
    print("Sigmoid analysis:")
    print("=" * 80)
    raw_conf_sample = confidences[:100]  # Sample first 100
    sigmoid_conf_sample = 1.0 / (1.0 + np.exp(-raw_conf_sample))
    print(f"Sample of first 100 confidence values:")
    print(f"  Raw min: {raw_conf_sample.min():.10f}, max: {raw_conf_sample.max():.10f}")
    print(f"  After sigmoid min: {sigmoid_conf_sample.min():.10f}, max: {sigmoid_conf_sample.max():.10f}")
    print(f"  Raw mean: {raw_conf_sample.mean():.10f}")
    print(f"  After sigmoid mean: {sigmoid_conf_sample.mean():.10f}")
    
    top_raw = confidences[top_indices[0]]
    top_sigmoid = 1.0 / (1.0 + np.exp(-top_raw))
    print(f"\nTop confidence value:")
    print(f"  Raw: {top_raw:.10f}")
    print(f"  After sigmoid: {top_sigmoid:.10f}")
    print(f"  (If raw is near 0, sigmoid gives ~0.5, which matches C++ output)")
    
    return output, predictions, confidences


if __name__ == "__main__":
    onnx_path = "hand_yolo11.onnx"
    image_path = None
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        sys.exit(1)
    
    inspect_onnx_output(onnx_path, image_path)

