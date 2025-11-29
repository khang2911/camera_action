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

# FFmpeg (for NVDEC decoding)
find_path(FFMPEG_INCLUDE_DIR
    NAMES libavformat/avformat.h
    PATHS /usr/include /usr/local/include /opt/homebrew/include
)
find_library(AVCODEC_LIBRARY avcodec PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib)
find_library(AVFORMAT_LIBRARY avformat PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib)
find_library(AVUTIL_LIBRARY avutil PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib)
find_library(SWSCALE_LIBRARY swscale PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib)

if(NOT FFMPEG_INCLUDE_DIR OR NOT AVCODEC_LIBRARY OR NOT AVFORMAT_LIBRARY OR NOT AVUTIL_LIBRARY OR NOT SWSCALE_LIBRARY)
    message(FATAL_ERROR "FFmpeg libraries (avcodec, avformat, avutil, swscale) are required for NVDEC decoding.")
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

# yaml-cpp (for configuration file parsing)
find_package(yaml-cpp REQUIRED)

# Redis (cpp_redis)
# Try to find cpp_redis library
# cpp_redis main header is cpp_redis/cpp_redis (no extension)
find_path(CPP_REDIS_INCLUDE_DIR cpp_redis/cpp_redis
    PATHS
    /usr/include
    /usr/local/include
    /opt/cpp_redis/include
    ${CPP_REDIS_ROOT}/include
    ${REDIS_ROOT}/include
    ${CMAKE_SOURCE_DIR}/third_party/cpp_redis/include
    ${CMAKE_SOURCE_DIR}/../cpp_redis/include
)

# Try to find the library (cpp_redis may be header-only, so library might not exist)
find_library(CPP_REDIS_LIBRARY 
    NAMES cpp_redis cpp_redis_static cpp_redis_shared
    PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /opt/cpp_redis/lib
    ${CPP_REDIS_ROOT}/lib
    ${REDIS_ROOT}/lib
    ${CMAKE_SOURCE_DIR}/third_party/cpp_redis/lib
    ${CMAKE_SOURCE_DIR}/../cpp_redis/lib
)

# Find tacopie library (required dependency of cpp_redis)
find_library(TACOPIE_LIBRARY 
    NAMES tacopie
    PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /opt/cpp_redis/lib
    ${CPP_REDIS_ROOT}/lib
    ${REDIS_ROOT}/lib
    ${CMAKE_SOURCE_DIR}/third_party/cpp_redis/lib
    ${CMAKE_SOURCE_DIR}/../cpp_redis/lib
)

if(CPP_REDIS_INCLUDE_DIR)
    message(STATUS "cpp_redis found")
    message(STATUS "  CPP_REDIS_INCLUDE_DIR: ${CPP_REDIS_INCLUDE_DIR}")
    if(CPP_REDIS_LIBRARY)
        message(STATUS "  CPP_REDIS_LIBRARY: ${CPP_REDIS_LIBRARY}")
    else()
        message(STATUS "  cpp_redis appears to be header-only (no library found)")
    endif()
    if(TACOPIE_LIBRARY)
        message(STATUS "  TACOPIE_LIBRARY: ${TACOPIE_LIBRARY}")
    else()
        message(WARNING "tacopie library not found - cpp_redis requires it")
    endif()
else()
    message(WARNING "cpp_redis library not found.")
    message(STATUS "")
    message(STATUS "To install cpp_redis:")
    message(STATUS "  1. Clone the repository:")
    message(STATUS "     git clone https://github.com/cpp-redis/cpp_redis.git")
    message(STATUS "  2. Build and install:")
    message(STATUS "     cd cpp_redis")
    message(STATUS "     mkdir build && cd build")
    message(STATUS "     cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local")
    message(STATUS "     make && sudo make install")
    message(STATUS "")
    message(STATUS "Or set CPP_REDIS_ROOT to the installation directory:")
    message(STATUS "  cmake -DCPP_REDIS_ROOT=/path/to/cpp_redis ..")
    message(STATUS "")
    message(FATAL_ERROR "Please install cpp_redis or set CPP_REDIS_ROOT")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIR}
    ${TensorRT_INCLUDE_DIR}
    ${yaml-cpp_INCLUDE_DIRS}
    ${CPP_REDIS_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/main.cpp
    src/video_reader.cu
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
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${TensorRT_LIBRARY}
    yaml-cpp
    nvinfer
    nvinfer_plugin
    cudart
    pthread
    ${OpenMP_CXX_LIBRARIES}
)

# Add cpp_redis and tacopie libraries if found
if(CPP_REDIS_LIBRARY)
    list(APPEND ALL_LIBS ${CPP_REDIS_LIBRARY})
endif()
if(TACOPIE_LIBRARY)
    list(APPEND ALL_LIBS ${TACOPIE_LIBRARY})
endif()

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${ALL_LIBS})
