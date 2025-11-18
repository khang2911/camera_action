cmake_minimum_required(VERSION 3.10)
project(AICameraSolution VERSION 1.0.0 LANGUAGES CXX CUDA)

# Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_compile_options(-Wno-deprecated-declarations)

# CUDA Configuration
find_package(CUDA REQUIRED)

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

# Find required packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Eigen3 REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

# GLib (via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)

# GStreamer (via pkg-config)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

# FFmpeg libraries (optional, for advanced video processing)
# These are typically found via pkg-config or direct linking
find_library(AVUTIL_LIB avutil PATHS /usr/lib /usr/local/lib)
find_library(AVFORMAT_LIB avformat PATHS /usr/lib /usr/local/lib)
find_library(AVCODEC_LIB avcodec PATHS /usr/lib /usr/local/lib)
find_library(SWSCALE_LIB swscale PATHS /usr/lib /usr/local/lib)

# Optional: TritonClient (if available)
# Uncomment and set paths if using Triton inference server
# set(TritonClient_DIR ${PROJECT_SOURCE_DIR}/libs/triton/lib/cmake/TritonClient)
# set(TritonCommon_DIR ${PROJECT_SOURCE_DIR}/libs/triton/lib/cmake/TritonCommon)
# set(TritonClientBuild_DIR ${PROJECT_SOURCE_DIR}/libs/triton)
# find_package(TritonCommon REQUIRED)
# find_package(TritonClient REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${yaml-cpp_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
    /usr/local/include/eigen3
    ${Eigen3_INCLUDE_DIRS}
)

# Link directories (if needed for custom library paths)
# link_directories(
#     ${PROJECT_SOURCE_DIR}/libs
# )

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
)

# Header files
set(HEADERS
    include/video_reader.h
    include/preprocessor.h
    include/yolo_detector.h
    include/thread_pool.h
    include/frame_queue.h
    include/config_parser.h
    include/logger.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# All libraries
set(ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARY}
    ${GLIB_LIBRARIES}
    ${GST_LIBRARIES}
    ${GST_APP_LIBRARIES}
    yaml-cpp
    CURL::libcurl
    ${Eigen3_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    pthread
)

# TensorRT and CUDA libraries
set(TENSORRT_LIBS
    nvinfer
    nvinfer_plugin
    cudart
)

# FFmpeg libraries (if found)
if(AVUTIL_LIB AND AVFORMAT_LIB AND AVCODEC_LIB AND SWSCALE_LIB)
    list(APPEND ALL_LIBS
        ${AVUTIL_LIB}
        ${AVFORMAT_LIB}
        ${AVCODEC_LIB}
        ${SWSCALE_LIB}
    )
    message(STATUS "FFmpeg libraries found")
else()
    # Fallback to direct linking if find_library fails
    list(APPEND ALL_LIBS
        -lavutil
        -lavformat
        -lavcodec
        -lswscale
    )
    message(STATUS "Using direct FFmpeg library linking")
endif()

# System libraries
list(APPEND ALL_LIBS
    -lz
    -lm
    -lglib-2.0
)

# Optional libraries (uncomment if needed)
# -lcpp_redis
# -ltacopie
# grpcclient
# httpclient
# TritonClient libraries (if using Triton)
# ${TritonClient_LIBRARIES}
# ${TritonCommon_LIBRARIES}

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${ALL_LIBS}
    ${TENSORRT_LIBS}
)

# Compiler flags (already set above, but can add more)
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wno-deprecated-declarations
)

