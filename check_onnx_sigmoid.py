#!/usr/bin/env python3
"""Check which parts of the ONNX output come from sigmoid."""

import onnx
import numpy as np

model = onnx.load('hand_yolo11.onnx')
nodes = list(model.graph.node)

# Find the output concat node
out_node = None
for node in nodes:
    if node.name == '/model.23/Concat_5':
        out_node = node
        break

print("Output Concat node:")
print(f"  Name: {out_node.name}")
print(f"  Inputs: {out_node.input}")
print(f"  Outputs: {out_node.output}")

# Find the sigmoid node
sigmoid_node = None
for node in nodes:
    if node.name == '/model.23/Sigmoid':
        sigmoid_node = node
        break

print(f"\nSigmoid node:")
print(f"  Name: {sigmoid_node.name}")
print(f"  Inputs: {sigmoid_node.input}")
print(f"  Outputs: {sigmoid_node.output}")

# Find Mul_2 node
mul_node = None
for node in nodes:
    if node.name == '/model.23/Mul_2':
        mul_node = node
        break

print(f"\nMul_2 node:")
print(f"  Name: {mul_node.name}")
print(f"  Inputs: {mul_node.input}")
print(f"  Outputs: {mul_node.output}")

# Check the shape of each input to the concat
print(f"\nChecking input shapes to concat...")
# We need to trace back to find the shapes
# The output is [16, 5, 15309], so the concat might be along channel dimension

# Let's check if we can infer from the model structure
print("\nModel output shape: [16, 5, 15309]")
print("If sigmoid is applied to confidence only (1 channel), then:")
print("  - Mul_2 output: [16, 4, 15309] (bbox: x, y, w, h)")
print("  - Sigmoid output: [16, 1, 15309] (confidence)")
print("  - Concat along channel: [16, 5, 15309]")

print("\nIf sigmoid is NOT applied to confidence, then confidence values")
print("should be in [0, 1] range if they're probabilities, or need sigmoid if they're logits.")
print("\nFrom our inspection, confidence values are ~0.00013, which suggests:")
print("  1. They're logits (need sigmoid) - but sigmoid node exists!")
print("  2. They're probabilities but very low")
print("  3. The sigmoid is applied to something else, not confidence")

