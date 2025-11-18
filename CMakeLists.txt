cmake_minimum_required(VERSION 3.15)
project(AICameraSolution VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
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

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    yaml-cpp
    pthread
)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -O3
)

