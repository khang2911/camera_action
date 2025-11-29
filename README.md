# AI Camera Solution - C++ Project

A high-performance C++ application for parallel video processing with YOLO object detection using TensorRT.

## Overview

This project implements a multi-threaded video processing pipeline with two types of parallel processes:

1. **Process 1 (Reader/Preprocessor)**: Reads video frames and preprocesses them (CPU only)
   - Padding and resizing to 640x640
   - Normalization (divide all pixels by 255)

2. **Process 2 (YOLO Detector)**: Runs YOLO detection using TensorRT (GPU)
   - Processes preprocessed frames
   - Outputs detection results as binary files

The number of each process type is configurable, allowing dynamic scaling based on your hardware capabilities.

## Features

- **Dynamic Worker Pools**: Configure any number of reader and detector threads
- **Thread-Safe Queue**: Efficient frame passing between processes
- **GPU Acceleration**: TensorRT-based YOLO inference
- **Parallel Processing**: Multiple videos processed simultaneously
- **Binary Output**: Detection results saved as binary files

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended) or macOS
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: Sufficient space for video files and output

### Software Dependencies
- **C++17 compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.15 or higher
- **OpenCV** 4.x (with video codec support)
- **CUDA Toolkit** (compatible with your GPU, typically 10.0+)
- **TensorRT** (compatible with your CUDA version)
- **yaml-cpp** (for configuration file parsing)
- **cpp_redis** (for Redis queue integration)

## Installation

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install OpenCV
sudo apt-get install -y libopencv-dev

# Install yaml-cpp
sudo apt-get install -y libyaml-cpp-dev
```

#### macOS (using Homebrew)
```bash
# Install build tools
brew install cmake

# Install OpenCV
brew install opencv

# Install yaml-cpp
brew install yaml-cpp
```

### 1.5. Install cpp_redis (for Redis queue support)

#### Ubuntu/Debian
```bash
# Clone cpp_redis repository
git clone https://github.com/Cylix/cpp_redis.git
cd cpp_redis

# Initialize and update submodules (tacopie dependency)
git submodule init
git submodule update

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install

# Update library cache
sudo ldconfig
```

#### macOS (using Homebrew)
```bash
# cpp_redis is not available via Homebrew, so build from source:
git clone https://github.com/Cylix/cpp_redis.git
cd cpp_redis
git submodule init
git submodule update
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j4
sudo make install
```

**Note**: If you install cpp_redis to a custom location, set the `CPP_REDIS_ROOT` environment variable:
```bash
export CPP_REDIS_ROOT=/path/to/cpp_redis
```

### 2. Install CUDA Toolkit

Download and install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### 3. Install TensorRT

Download TensorRT from [NVIDIA Developer](https://developer.nvidia.com/tensorrt) and follow the installation guide.

Set environment variables (add to `~/.bashrc` or `~/.zshrc`):
```bash
export TENSORRT_ROOT=/path/to/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

### 4. Build the Project

```bash
# Clone or navigate to the project directory
cd /path/to/LC

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project (use -j$(nproc) for parallel build)
make -j$(nproc)

# Or on macOS with limited cores
make -j4
```

If CMake cannot find dependencies, you may need to specify paths:
```bash
cmake \
    -DOpenCV_DIR=/path/to/opencv \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DTensorRT_ROOT=/path/to/TensorRT \
    -DCPP_REDIS_ROOT=/path/to/cpp_redis \
    ..
```

### 5. Verify Build

After building, you should see the executable:
```bash
ls -lh AICameraSolution
```

## Configuration

### Using YAML Config File (Recommended)

1. Copy the example config file:
```bash
cp config.yaml my_config.yaml
cp video_list.txt my_video_list.txt
```

2. Create your video list file (`my_video_list.txt`):
```bash
# Video List File
# One video path per line
# Lines starting with # are treated as comments and ignored
# Empty lines are also ignored

/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.mp4
/path/to/video4.mp4
```

3. Edit `my_config.yaml`:
```yaml
# YOLO Model Configuration
# Multiple engines can be configured - each frame will go through ALL engines
models:
  - path: "/path/to/yolo_engine1.engine"  # Path to TensorRT engine file
    num_detectors: 4                      # Number of detector threads for this engine
    name: "model1"                        # Optional: name identifier for output files
  - path: "/path/to/yolo_engine2.engine"
    num_detectors: 4
    name: "model2"
  - path: "/path/to/yolo_engine3.engine"
    num_detectors: 6
    name: "model3"

# Video Sources
videos:
  list_file: "./my_video_list.txt"  # Path to text file containing video paths

# Thread Configuration
threads:
  num_readers: 10      # Number of reader/preprocessor threads (CPU)

# Output Configuration
output:
  dir: "./output"      # Output directory for binary result files
```

**Note**: Each engine runs in parallel with its own thread pool. Each frame is processed by all engines, producing separate output files for each engine.

**Note**: The video list file supports:
- One video path per line
- Comments (lines starting with `#`)
- Empty lines (automatically ignored)
- Absolute or relative paths

## Running

### Method 1: Using Config File (Recommended)

```bash
# Using default config.yaml in project root
./AICameraSolution --config config.yaml

# Or specify a custom config file
./AICameraSolution --config /path/to/my_config.yaml
```

### Method 2: Command Line Arguments

```bash
./AICameraSolution \
    --num-readers 10 \
    --num-detectors 4 \
    --engine /path/to/yolo.engine \
    --output-dir ./output \
    --videos video1.mp4 video2.mp4 video3.mp4
```

### Method 3: Mix Config File and Command Line Overrides

```bash
# Load config.yaml but override specific values
./AICameraSolution \
    --config config.yaml \
    --num-readers 20 \
    --output-dir ./custom_output
```

### Command Line Arguments

- `--config PATH`: Path to YAML config file (recommended)
- `--num-readers N`: Number of reader/preprocessor threads (overrides config)
- `--num-detectors N`: Number of YOLO detector threads (overrides config)
- `--engine PATH`: Path to TensorRT engine file (overrides config)
- `--output-dir PATH`: Output directory for bin files (overrides config)
- `--videos PATH1 PATH2 ...`: List of video file paths (overrides config)
- `--help` or `-h`: Show help message

### Example Usage Scenarios

#### Scenario 1: Single Video Processing
```bash
./AICameraSolution \
    --config config.yaml \
    --videos /path/to/single_video.mp4
```

#### Scenario 2: High-Performance Setup
```bash
# Use more threads for faster processing
./AICameraSolution \
    --config config.yaml \
    --num-readers 20 \
    --num-detectors 8
```

#### Scenario 3: Custom Output Location
```bash
./AICameraSolution \
    --config config.yaml \
    --output-dir /mnt/storage/detection_results
```

## Architecture

### Components

1. **VideoReader**: Reads frames from video files
2. **Preprocessor**: Handles padding, resizing, and normalization
3. **YOLODetector**: TensorRT-based YOLO inference engine
4. **FrameQueue**: Thread-safe queue for frame data
5. **ThreadPool**: Manages reader and detector worker threads

### Processing Flow

```
Video Files → [Reader Threads] → Frame Queues (one per engine) → [Detector Threads] → Binary Files
                (Preprocess)         ↓            ↓         ↓         (TensorRT GPU)
                                     ↓            ↓         ↓
                              [Engine 1]    [Engine 2]  [Engine 3]
                              Thread Pool   Thread Pool Thread Pool
```

**Key Features**:
- Each frame is preprocessed once and pushed to ALL engine queues
- Each engine has its own thread pool with configurable number of detector threads
- Engines run in parallel, independently processing frames
- Each engine produces its own output files

## Output Format

Detection results are saved as binary files with the naming convention:
```
output/video_XXXX_frame_XXXXXX_ENGINENAME_detector_X.bin
```

Where:
- `XXXX`: Zero-padded video ID
- `XXXXXX`: Zero-padded frame number
- `ENGINENAME`: Engine name (from config, e.g., "model1", "model2", "model3")
- `X`: Detector thread ID

**Example**: For a frame from video 0, frame 123, processed by engine "model1" with detector thread 2:
```
output/video_0000_frame_000123_model1_detector_2.bin
```

**Note**: Each frame produces one output file per engine. For example, with 3 engines, each frame generates 3 separate bin files (one per engine).

## Troubleshooting

### Build Issues

**Problem**: CMake cannot find OpenCV
```bash
# Solution: Specify OpenCV path
cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 ..
```

**Problem**: CMake cannot find TensorRT
```bash
# Solution: Set TensorRT path
export TENSORRT_ROOT=/path/to/TensorRT
cmake -DTensorRT_ROOT=$TENSORRT_ROOT ..
```

**Problem**: CUDA not found
```bash
# Solution: Set CUDA path
export CUDA_PATH=/usr/local/cuda
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH ..
```

**Problem**: yaml-cpp not found
```bash
# Ubuntu/Debian
sudo apt-get install libyaml-cpp-dev

# macOS
brew install yaml-cpp

# Or build from source
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build && cd build
cmake .. && make && sudo make install
```

**Problem**: cpp_redis library not found
```bash
# Quick install using the provided script
./install_cpp_redis.sh

# Or manual installation
git clone https://github.com/Cylix/cpp_redis.git
cd cpp_redis
git submodule init
git submodule update
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig  # Linux only

# Or specify custom installation path when running cmake
cmake -DCPP_REDIS_ROOT=/path/to/cpp_redis ..
```

### Runtime Issues

**Problem**: "Cannot open video" errors
- Check video file paths are correct and accessible
- Verify video codec is supported by OpenCV
- Ensure video files are not corrupted

**Problem**: "Failed to initialize detector"
- Verify TensorRT engine file path is correct
- Check engine file is compatible with your GPU
- Ensure CUDA and TensorRT versions are compatible

**Problem**: Out of memory errors
- Reduce number of threads (`--num-readers` and `--num-detectors`)
- Reduce frame queue size (modify `FrameQueue` constructor in code)
- Process fewer videos simultaneously

**Problem**: GPU not detected
```bash
# Check GPU is visible
nvidia-smi

# Verify CUDA installation
nvcc --version
```

## Performance Tuning

### Optimal Thread Configuration

- **CPU-bound preprocessing**: Set `num_readers` based on CPU cores (typically 2x CPU cores)
- **GPU-bound inference**: Set `num_detectors` based on GPU memory and batch size (typically 2-8)
- **Balanced setup**: Start with `num_readers = 10` and `num_detectors = 4`, then adjust based on utilization

### Monitoring Performance

```bash
# Monitor CPU usage
htop

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor disk I/O
iostat -x 1
```

## Project Structure

```
LC/
├── CMakeLists.txt          # Build configuration
├── config.yaml             # Example configuration file
├── video_list.txt          # Example video list file
├── README.md               # This file
├── .gitignore             # Git ignore rules
├── include/                # Header files
│   ├── config_parser.h     # YAML config parser
│   ├── frame_queue.h       # Thread-safe queue
│   ├── preprocessor.h      # Image preprocessing
│   ├── thread_pool.h       # Worker thread management
│   ├── video_reader.h      # Video frame reader
│   └── yolo_detector.h     # TensorRT YOLO inference
└── src/                    # Source files
    ├── main.cpp            # Main entry point
    ├── config_parser.cpp
    ├── frame_queue.cpp
    ├── preprocessor.cpp
    ├── thread_pool.cpp
    ├── video_reader.cpp
    └── yolo_detector.cpp
```

## Notes

- **Video Processing**: Each video is assigned to an available reader thread dynamically
- **Multi-Engine Architecture**: Each frame goes through ALL configured engines in parallel
- **Frame Distribution**: Each engine has its own queue; frames are pushed to all engine queues simultaneously
- **Parallel Execution**: Engines run independently with their own thread pools
- **Memory Management**: Each engine's frame queue size is limited (default: 100 frames) to prevent excessive memory usage
- **Preprocessing**: Done once per frame on CPU (padding, resize to 640x640, normalize by 255), then shared across all engines
- **Inference**: Done on GPU using TensorRT for maximum performance
- **Output Format**: Binary files contain raw TensorRT output (float arrays), one file per engine per frame
- **Thread Configuration**: Each engine can have a different number of detector threads based on its workload

## License

[Add your license here]

