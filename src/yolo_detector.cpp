#include "yolo_detector.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <unordered_map>
#include <mutex>

// Simple TensorRT Logger implementation
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress INFO and VERBOSE messages, only show WARNING and ERROR
        if (severity == Severity::kWARNING || severity == Severity::kERROR) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TensorRTLogger gLogger;

// Static mutex map for thread-safe file writing (one mutex per file path)
static std::unordered_map<std::string, std::unique_ptr<std::mutex>> file_mutexes;
static std::mutex file_mutexes_map_mutex;  // Protects the map itself

YOLODetector::YOLODetector(const std::string& engine_path, ModelType model_type, int batch_size,
                           int input_width, int input_height, float conf_threshold, 
                           float nms_threshold, int gpu_id)
    : engine_path_(engine_path), model_type_(model_type), batch_size_(batch_size),
      input_width_(input_width), input_height_(input_height), gpu_id_(gpu_id),
      runtime_(nullptr), engine_(nullptr), context_(nullptr), input_buffer_(nullptr), 
      output_buffer_(nullptr), input_size_(0), output_size_(0), output_height_(0), 
      output_width_(0), num_anchors_(0), num_classes_(0), output_channels_(0),
      conf_threshold_(conf_threshold), nms_threshold_(nms_threshold), max_detections_(300) {
    // Set CUDA device before creating stream
    cudaSetDevice(gpu_id_);
    cudaStreamCreate(&stream_);
    preprocessor_ = std::make_unique<Preprocessor>(input_width_, input_height_);
}

YOLODetector::~YOLODetector() {
    freeBuffers();
    if (context_) {
        context_->destroy();
    }
    if (engine_) {
        engine_->destroy();
    }
    if (runtime_) {
        runtime_->destroy();
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool YOLODetector::loadEngine() {
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_path_ << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size, nullptr);
    if (!engine_) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }
    
    return true;
}

void YOLODetector::allocateBuffers() {
    if (!engine_) return;
    
    // Get input and output buffer sizes
    int num_bindings = engine_->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            input_size_ = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                input_size_ *= dims.d[j];
            }
            input_size_ *= sizeof(float);
            cudaMalloc(&input_buffer_, input_size_);
        } else {
            // Output format: [batch, num_anchors, num_channels]
            // For YOLO without NMS: [1, 8400, num_classes+5] for detection or [1, 8400, 56] for pose
            auto dims = engine_->getBindingDimensions(i);
            output_size_ = 1;
            
            if (dims.nbDims == 3) {
                // Format: [batch, num_anchors, channels]
                output_height_ = 1;  // batch
                output_width_ = dims.d[1];  // num_anchors (e.g., 8400)
                output_channels_ = dims.d[2];  // channels per anchor
                num_anchors_ = output_width_;
            } else if (dims.nbDims == 2) {
                // Format: [num_anchors, channels] (no batch dimension)
                output_height_ = 1;
                output_width_ = dims.d[0];
                output_channels_ = dims.d[1];
                num_anchors_ = output_width_;
            } else {
                // Fallback: treat as flattened
                output_height_ = 1;
                output_width_ = 1;
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (j == dims.nbDims - 1) {
                        output_channels_ = dims.d[j];
                    } else {
                        output_width_ *= dims.d[j];
                    }
                }
                num_anchors_ = output_width_;
            }
            
            // Determine number of classes based on model type
            if (model_type_ == ModelType::POSE) {
                // Pose: 56 channels = 4(bbox) + 1(objectness) + 1(class) + 17*3(keypoints)
                num_classes_ = 1;  // Usually single class for pose
            } else {
                // Detection: channels = 4(bbox) + 1(objectness) + num_classes
                num_classes_ = output_channels_ - 5;
            }
            
            for (int j = 0; j < dims.nbDims; ++j) {
                output_size_ *= dims.d[j];
            }
            output_size_ *= sizeof(float);
            
            cudaMalloc(&output_buffer_, output_size_);
        }
    }
}

void YOLODetector::freeBuffers() {
    if (input_buffer_) {
        cudaFree(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (output_buffer_) {
        cudaFree(output_buffer_);
        output_buffer_ = nullptr;
    }
}

bool YOLODetector::initialize() {
    // Set CUDA device before initialization
    cudaError_t cuda_status = cudaSetDevice(gpu_id_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error: Failed to set CUDA device " << gpu_id_ 
                  << ": " << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }
    
    if (!loadEngine()) {
        return false;
    }
    
    // Verify batch size matches engine batch size
    if (engine_) {
        auto input_dims = engine_->getBindingDimensions(0);
        int engine_batch_size = input_dims.d[0];
        if (engine_batch_size != batch_size_) {
            std::cerr << "Warning: Engine batch size (" << engine_batch_size 
                      << ") does not match config batch size (" << batch_size_ 
                      << "). Using engine batch size." << std::endl;
            batch_size_ = engine_batch_size;
        }
    }
    
    allocateBuffers();
    return true;
}

bool YOLODetector::detect(const cv::Mat& frame, const std::string& output_path, int frame_number) {
    if (!context_ || !input_buffer_ || !output_buffer_) {
        std::cerr << "Error: Detector not initialized" << std::endl;
        return false;
    }
    
    // Set CUDA device for this detector (important for multi-GPU setups)
    cudaSetDevice(gpu_id_);
    
    // Preprocess frame for this engine's input size
    std::vector<float> input_data = preprocessor_->preprocessToFloat(frame);
    
    // Copy input data to GPU
    cudaMemcpyAsync(input_buffer_, input_data.data(), input_size_, cudaMemcpyHostToDevice, stream_);
    
    // Prepare bindings array (input at index 0, output at index 1)
    void* bindings[] = {input_buffer_, output_buffer_};
    
    // Run inference
    bool success = context_->enqueueV2(bindings, stream_, nullptr);
    if (!success) {
        std::cerr << "Error: Inference execution failed" << std::endl;
        return false;
    }
    
    // Copy output from GPU to CPU
    std::vector<float> output_data(output_size_ / sizeof(float));
    cudaMemcpyAsync(output_data.data(), output_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    // Parse raw YOLO output (without NMS) based on model type
    std::vector<Detection> detections;
    if (model_type_ == ModelType::POSE) {
        detections = parseRawPoseOutput(output_data);
    } else {
        detections = parseRawDetectionOutput(output_data);
    }
    
    // Apply NMS to filter overlapping detections
    detections = applyNMS(detections);
    
    // Limit to max_detections
    if (static_cast<int>(detections.size()) > max_detections_) {
        detections.resize(max_detections_);
    }
    
    // Write parsed detections to file
    // Frame number is not available here, will be passed separately
    // For now, we'll need to modify the detect signature or pass it differently
    // Let's check how it's called
    return writeDetectionsToFile(detections, output_path, frame_number);
}

std::vector<Detection> YOLODetector::parseRawDetectionOutput(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // Raw YOLO detection format: [batch, num_anchors, num_classes + 5]
    // Each anchor: [x_center, y_center, width, height, objectness, class_scores...]
    // Coordinates are typically normalized (0-1) or in grid coordinates
    
    for (int i = 0; i < num_anchors_; ++i) {
        int offset = i * output_channels_;
        if (offset + output_channels_ - 1 >= static_cast<int>(output_data.size())) break;
        
        // Extract bbox and objectness
        float x_center = output_data[offset + 0];
        float y_center = output_data[offset + 1];
        float width = output_data[offset + 2];
        float height = output_data[offset + 3];
        float objectness = output_data[offset + 4];
        
        // Find class with highest score
        float max_class_score = 0.0f;
        int best_class = -1;
        for (int c = 0; c < num_classes_; ++c) {
            float class_score = output_data[offset + 5 + c];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                best_class = c;
            }
        }
        
        // Calculate final confidence: objectness * class_score
        float confidence = objectness * max_class_score;
        
        // Apply confidence threshold
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center (normalized or absolute)
        det.bbox[1] = y_center;   // y_center
        det.bbox[2] = width;      // width
        det.bbox[3] = height;     // height
        det.confidence = confidence;
        det.class_id = best_class;
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::parseRawPoseOutput(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // Raw YOLO pose format: [batch, num_anchors, 56]
    // Each anchor: [x_center, y_center, width, height, objectness, class_score,
    //                kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ..., kpt16_x, kpt16_y, kpt16_conf]
    // Total: 4(bbox) + 1(objectness) + 1(class) + 17*3(keypoints) = 56
    
    for (int i = 0; i < num_anchors_; ++i) {
        int offset = i * output_channels_;
        if (offset + output_channels_ - 1 >= static_cast<int>(output_data.size())) break;
        
        // Extract bbox, objectness, and class
        float x_center = output_data[offset + 0];
        float y_center = output_data[offset + 1];
        float width = output_data[offset + 2];
        float height = output_data[offset + 3];
        float objectness = output_data[offset + 4];
        float class_score = output_data[offset + 5];
        
        // Calculate final confidence: objectness * class_score
        float confidence = objectness * class_score;
        
        // Apply confidence threshold
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center
        det.bbox[1] = y_center;   // y_center
        det.bbox[2] = width;      // width
        det.bbox[3] = height;     // height
        det.confidence = confidence;
        det.class_id = 0;  // Usually single class for pose models
        
        // Parse 17 keypoints (COCO format)
        det.keypoints.resize(17);
        for (int k = 0; k < 17; ++k) {
            int kpt_offset = offset + 6 + k * 3;
            det.keypoints[k].x = output_data[kpt_offset + 0];
            det.keypoints[k].y = output_data[kpt_offset + 1];
            det.keypoints[k].confidence = output_data[kpt_offset + 2];
        }
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::applyNMS(const std::vector<Detection>& detections) {
    if (detections.empty()) {
        return detections;
    }
    
    // Create indices and sort by confidence (descending)
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) {
            continue;
        }
        
        result.push_back(detections[indices[i]]);
        
        // Suppress overlapping detections
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) {
                continue;
            }
            
            // Check if same class
            if (detections[indices[i]].class_id != detections[indices[j]].class_id) {
                continue;
            }
            
            // Calculate IoU
            float iou = calculateIoU(detections[indices[i]].bbox, detections[indices[j]].bbox);
            
            if (iou > nms_threshold_) {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    return result;
}

float YOLODetector::calculateIoU(const float* box1, const float* box2) {
    // Convert from center format [x_center, y_center, width, height] to corner format
    float x1_min = box1[0] - box1[2] / 2.0f;
    float y1_min = box1[1] - box1[3] / 2.0f;
    float x1_max = box1[0] + box1[2] / 2.0f;
    float y1_max = box1[1] + box1[3] / 2.0f;
    
    float x2_min = box2[0] - box2[2] / 2.0f;
    float y2_min = box2[1] - box2[3] / 2.0f;
    float x2_max = box2[0] + box2[2] / 2.0f;
    float y2_max = box2[1] + box2[3] / 2.0f;
    
    // Calculate intersection
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    float inter_width = std::max(0.0f, inter_x_max - inter_x_min);
    float inter_height = std::max(0.0f, inter_y_max - inter_y_min);
    float inter_area = inter_width * inter_height;
    
    // Calculate union
    float box1_area = box1[2] * box1[3];
    float box2_area = box2[2] * box2[3];
    float union_area = box1_area + box2_area - inter_area;
    
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    
    return inter_area / union_area;
}

bool YOLODetector::writeDetectionsToFile(const std::vector<Detection>& detections, const std::string& output_path, int frame_number) {
    // Get or create mutex for this file path (thread-safe)
    std::mutex* file_mutex = nullptr;
    {
        std::lock_guard<std::mutex> map_lock(file_mutexes_map_mutex);
        if (file_mutexes.find(output_path) == file_mutexes.end()) {
            file_mutexes[output_path] = std::make_unique<std::mutex>();
        }
        file_mutex = file_mutexes[output_path].get();
    }
    
    // Lock this specific file for writing
    std::lock_guard<std::mutex> file_lock(*file_mutex);
    
    // Open file in append mode (ios::app) for binary writing
    // Use ios::ate to seek to end, then ios::in|ios::out|ios::binary for read/write
    std::fstream out_file(output_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
    
    // If file doesn't exist, create it and write header
    bool file_exists = out_file.is_open();
    if (!file_exists) {
        out_file.open(output_path, std::ios::out | std::ios::binary);
        if (!out_file.is_open()) {
            std::cerr << "Error: Cannot create output file: " << output_path << std::endl;
            return false;
        }
        // Write header: model type (written once at file creation)
        int model_type_int = static_cast<int>(model_type_);
        out_file.write(reinterpret_cast<const char*>(&model_type_int), sizeof(int));
        out_file.close();
        // Reopen in append mode
        out_file.open(output_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
    }
    
    if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
        return false;
    }
    
    // Seek to end for appending
    out_file.seekp(0, std::ios::end);
    
    // Write frame number
    out_file.write(reinterpret_cast<const char*>(&frame_number), sizeof(int));
    
    // Write number of detections for this frame
    int num_dets = static_cast<int>(detections.size());
    out_file.write(reinterpret_cast<const char*>(&num_dets), sizeof(int));
    
    // Write each detection
    for (const auto& det : detections) {
        // Write bbox (4 floats)
        out_file.write(reinterpret_cast<const char*>(det.bbox), 4 * sizeof(float));
        
        // Write confidence and class_id
        out_file.write(reinterpret_cast<const char*>(&det.confidence), sizeof(float));
        out_file.write(reinterpret_cast<const char*>(&det.class_id), sizeof(int));
        
        // Write keypoints if pose model
        if (model_type_ == ModelType::POSE) {
            int num_keypoints = static_cast<int>(det.keypoints.size());
            out_file.write(reinterpret_cast<const char*>(&num_keypoints), sizeof(int));
            for (const auto& kpt : det.keypoints) {
                out_file.write(reinterpret_cast<const char*>(&kpt.x), sizeof(float));
                out_file.write(reinterpret_cast<const char*>(&kpt.y), sizeof(float));
                out_file.write(reinterpret_cast<const char*>(&kpt.confidence), sizeof(float));
            }
        } else {
            // For detection models, write 0 keypoints
            int num_keypoints = 0;
            out_file.write(reinterpret_cast<const char*>(&num_keypoints), sizeof(int));
        }
    }
    
    out_file.close();
    return true;
}

