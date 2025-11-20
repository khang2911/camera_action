cmake_minimum_required(VERSION 3.10)
project(AICameraSolution VERSION 1.0.0 LANGUAGES CXX CUDA)

# CUDA Configuration
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
find_package(CUDA REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options(-Wno-deprecated-declarations)

# Check platform for CUDA paths
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "ARM64 platform detected")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message(STATUS "x86_64 platform detected")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

# Include and Link Directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    PATHS
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
    ${TensorRT_ROOT}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

find_library(TensorRT_LIBRARY nvinfer
    PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    ${TensorRT_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# yaml-cpp (for configuration file parsing)
find_package(yaml-cpp REQUIRED)

# Redis (redis-plus-plus / lcpp_redis)
find_path(REDIS_PLUS_PLUS_INCLUDE_DIR sw/redis++/redis++.h
    PATHS
    /usr/include
    /usr/local/include
    ${REDIS_PLUS_PLUS_ROOT}/include
    ${REDIS_ROOT}/include
)

find_library(REDIS_PLUS_PLUS_LIBRARY redis++
    PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    ${REDIS_PLUS_PLUS_ROOT}/lib
    ${REDIS_ROOT}/lib
)

if(REDIS_PLUS_PLUS_INCLUDE_DIR AND REDIS_PLUS_PLUS_LIBRARY)
    message(STATUS "redis-plus-plus found")
    message(STATUS "  REDIS_PLUS_PLUS_INCLUDE_DIR: ${REDIS_PLUS_PLUS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "redis-plus-plus library not found. Please install redis-plus-plus development library.")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${yaml-cpp_INCLUDE_DIRS}
    ${REDIS_PLUS_PLUS_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/main.cpp
    src/video_reader.cpp
    src/preprocessor.cpp
    src/yolo_detector.cpp
    src/thread_pool.cpp
    src/frame_queue.cpp
    src/config_parser.cpp
    src/logger.cpp
    src/redis_queue.cpp
)

# Header files
set(HEADERS
    include/video_reader.h
    include/preprocessor.h
    include/yolo_detector.h
    include/thread_pool.h
    include/frame_queue.h
    include/video_clip.h
    include/config_parser.h
    include/logger.h
    include/redis_queue.h
)

# Libraries
set(ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARY}
    yaml-cpp
    nvinfer
    nvinfer_plugin
    cudart
    pthread
    ${OpenMP_CXX_LIBRARIES}
    ${REDIS_PLUS_PLUS_LIBRARY}
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${ALL_LIBS})
