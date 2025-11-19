#!/usr/bin/env python3
"""
Orchestrate the cross-check workflow:
  1. Run the Python TensorRT script (trt_infer.py) to dump tensors.
  2. Run the C++ binary (AICameraSolution) to dump tensors/logs.
  3. Compare the resulting dumps with compare_tensors.py.

NOTE:
  - The C++ binary must support dumping tensors to directories specified via the
    environment variables `CPP_INPUT_DUMP_DIR` and `CPP_OUTPUT_DUMP_DIR`
    (or whatever mechanism you have wired up). Adjust this script as needed.
  - The Python script already supports `--dump-input-dir` / `--dump-output-dir`.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from typing import List


LOGGER = logging.getLogger("cross_check")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_command(cmd: List[str], env=None, cwd=None, label: str = ""):
    label = f" ({label})" if label else ""
    LOGGER.info("Running%s: %s", label, " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def sorted_bin_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, "*.bin")
    files = sorted(glob.glob(pattern))
    return files


def main():
    parser = argparse.ArgumentParser(description="Run Python + C++ cross-check workflow.")
    parser.add_argument("--engine", required=True, help="TensorRT engine path for Python script")
    parser.add_argument("--video", required=True, help="Video path to process")
    parser.add_argument("--input-width", type=int, required=True, help="Model input width")
    parser.add_argument("--input-height", type=int, required=True, help="Model input height")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (matches C++)")
    parser.add_argument("--python-script", default="trt_infer.py", help="Path to trt_infer.py")
    parser.add_argument("--cpp-binary", default="./build/AICameraSolution", help="Path to C++ executable")
    parser.add_argument("--cpp-config", default="config.yaml", help="Config file for C++ executable")
    parser.add_argument("--cpp-extra-args", nargs=argparse.REMAINDER,
                        help="Extra args passed to the C++ binary (append after '--')")
    parser.add_argument("--dump-dir", default="cross_check", help="Base directory to store dumps")
    parser.add_argument("--compare-shape", help="Shape for compare_tensors (e.g. '16x3x864x864')")
    parser.add_argument("--compare-dtype", default="float32", help="Element dtype for compare_tensors")
    parser.add_argument("--compare-target", choices=["inputs", "outputs"], default="outputs",
                        help="Which tensors to compare (default outputs)")
    parser.add_argument("--max-frames", type=int, default=16,
                        help="Max frames to process for quick checks (applies to both runs)")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.dump_dir)
    py_input_dir = os.path.join(base_dir, "python_inputs")
    py_output_dir = os.path.join(base_dir, "python_outputs")
    py_frames_dir = os.path.join(base_dir, "python_frames")
    cpp_input_dir = os.path.join(base_dir, "cpp_inputs")
    cpp_output_dir = os.path.join(base_dir, "cpp_outputs")

    for folder in [py_input_dir, py_output_dir, py_frames_dir, cpp_input_dir, cpp_output_dir]:
        os.makedirs(folder, exist_ok=True)

    # 1) Run Python TensorRT inference with dumps
    python_cmd = [
        sys.executable, args.python_script,
        "--engine", args.engine,
        "--video", args.video,
        "--input-width", str(args.input_width),
        "--input-height", str(args.input_height),
        "--batch-size", str(args.batch_size),
        "--max-frames", str(args.max_frames),
        "--dump-input-dir", py_input_dir,
        "--dump-output-dir", py_output_dir,
        "--save-frames", py_frames_dir,
    ]
    run_command(python_cmd, label="python")

    # 2) Run C++ binary (expects it to honor env vars to dump tensors)
    cpp_cmd = [os.path.abspath(args.cpp_binary), "--config", args.cpp_config]
    if args.cpp_extra_args:
        cpp_cmd.extend(args.cpp_extra_args)

    cpp_env = os.environ.copy()
    cpp_env["CPP_INPUT_DUMP_DIR"] = cpp_input_dir
    cpp_env["CPP_OUTPUT_DUMP_DIR"] = cpp_output_dir
    cpp_env["CPP_MAX_FRAMES"] = str(args.max_frames)
    run_command(cpp_cmd, env=cpp_env, label="cpp")

    # 3) Compare tensors
    target = args.compare_target
    py_dir = py_input_dir if target == "inputs" else py_output_dir
    cpp_dir = cpp_input_dir if target == "inputs" else cpp_output_dir

    py_files = sorted_bin_files(py_dir)
    cpp_files = sorted_bin_files(cpp_dir)

    if not py_files:
        LOGGER.warning("No Python %s dumps found in %s", target, py_dir)
        return
    if not cpp_files:
        LOGGER.warning("No C++ %s dumps found in %s", target, cpp_dir)
        return

    compare_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_tensors.py")
    if not os.path.exists(compare_script):
        LOGGER.error("compare_tensors.py not found at %s", compare_script)
        return

    count = min(len(py_files), len(cpp_files))
    LOGGER.info("Comparing %d %s tensors", count, target)
    for idx in range(count):
        LOGGER.info("Comparing batch %d: %s vs %s", idx, cpp_files[idx], py_files[idx])
        compare_cmd = [
            sys.executable, compare_script,
            "--tensor-a", cpp_files[idx],
            "--tensor-b", py_files[idx],
            "--dtype", args.compare_dtype,
        ]
        if args.compare_shape:
            compare_cmd.extend(["--shape", args.compare_shape])
        run_command(compare_cmd, label=f"compare batch {idx}")


if __name__ == "__main__":
    main()

