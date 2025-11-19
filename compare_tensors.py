#!/usr/bin/env python3
"""
Utility to compare two binary tensors (e.g., dumps from C++ and Python pipelines).

Examples:
    python3 compare_tensors.py --tensor-a tensor_cpp.bin --tensor-b tensor_py.bin \
        --dtype float32 --shape 1x3x640x640 --max-print 5
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np


def parse_shape(shape_str: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not shape_str:
        return None
    parts = shape_str.lower().replace("x", " ").replace(",", " ").split()
    return tuple(int(p) for p in parts)


def load_tensor(path: str, dtype: np.dtype, shape: Optional[Tuple[int, ...]]):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.fromfile(path, dtype=dtype)
    if shape:
        expected = int(np.prod(shape))
        if data.size != expected:
            raise ValueError(
                f"{path}: expected {expected} elements for shape {shape}, "
                f"found {data.size}"
            )
        data = data.reshape(shape)
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare two binary tensors.")
    parser.add_argument("--tensor-a", required=True, help="Path to first tensor file")
    parser.add_argument("--tensor-b", required=True, help="Path to second tensor file")
    parser.add_argument("--dtype", default="float32", help="Element dtype (default float32)")
    parser.add_argument(
        "--shape",
        help="Optional tensor shape, e.g. '16x3x640x640'. "
             "If omitted, tensors are treated as 1D vectors."
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for mismatch count")
    parser.add_argument("--max-print", type=int, default=0,
                        help="Print the top-N largest differences (default 0)")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    shape = parse_shape(args.shape)

    tensor_a = load_tensor(args.tensor_a, dtype, shape)
    tensor_b = load_tensor(args.tensor_b, dtype, shape)

    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Tensor shapes differ: {tensor_a.shape} vs {tensor_b.shape}")

    diff = tensor_a - tensor_b
    abs_diff = np.abs(diff)

    print("=== Tensor Comparison ===")
    print(f"Shape            : {tensor_a.shape}")
    print(f"Dtype            : {dtype}")
    print(f"Total elements   : {tensor_a.size}")
    print(f"Max abs diff     : {abs_diff.max():.10f}")
    print(f"Mean abs diff    : {abs_diff.mean():.10f}")
    print(f"RMS diff         : {np.sqrt(np.mean(diff ** 2)):.10f}")

    tol = args.tol
    mismatches = int(np.sum(abs_diff > tol))
    print(f"Elements > {tol:.2e}: {mismatches} ({mismatches / tensor_a.size:.4%})")

    if args.max_print > 0:
        flat_indices = np.argpartition(abs_diff.ravel(), -args.max_print)[-args.max_print:]
        ordered = flat_indices[np.argsort(abs_diff.ravel()[flat_indices])[::-1]]
        print(f"\nTop {args.max_print} differences:")
        for idx in ordered:
            coord = np.unravel_index(idx, tensor_a.shape)
            print(
                f"  idx={coord}: A={tensor_a[coord]:.10f}, "
                f"B={tensor_b[coord]:.10f}, diff={diff[coord]:.10f}"
            )


if __name__ == "__main__":
    main()

