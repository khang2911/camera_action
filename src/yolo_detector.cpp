#include "yolo_detector.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
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
      output_buffer_(nullptr), input_size_(0), output_size_(0), input_elements_(0),
      output_height_(0), 
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
            if (batch_size_ > 0) {
                input_elements_ = (input_size_ / sizeof(float)) / batch_size_;
            } else {
                input_elements_ = input_size_ / sizeof(float);
            }
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
            
            // YOLOv11 format determination - ALL models are single-class:
            if (model_type_ == ModelType::POSE) {
                // YOLOv11 Pose: 56 channels = 4(bbox) + 1(confidence) + 1(class) + 17*3(keypoints)
                // Format: [x_center, y_center, width, height, confidence, class_score, 17*3 keypoints]
                num_classes_ = 1;  // All models are single-class (class_id=0)
            } else {
                // YOLOv11 Detection: 5 channels = 4(bbox) + 1(confidence)
                // Format: [x_center, y_center, width, height, confidence]
                // All models are single-class (class_id=0)
                num_classes_ = 1;
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
    if (batch_size_ != 1) {
        std::cerr << "Error: detect() called with batch_size=" << batch_size_ 
                  << ". Use detectBatch() instead." << std::endl;
        return false;
    }
    
    auto buffer = std::make_shared<std::vector<float>>(input_elements_);
    preprocessor_->preprocessToFloat(frame, *buffer);
    return runWithPreprocessedData(buffer, output_path, frame_number, frame.cols, frame.rows);
}

bool YOLODetector::detectBatch(const std::vector<cv::Mat>& frames, 
                                const std::vector<std::string>& output_paths, 
                                const std::vector<int>& frame_numbers) {
    if (frames.size() != static_cast<size_t>(batch_size_)) {
        std::cerr << "Error: Expected " << batch_size_ << " frames, got " << frames.size() << std::endl;
        return false;
    }
    
    if (output_paths.size() != frames.size() || frame_numbers.size() != frames.size()) {
        std::cerr << "Error: Mismatch between frames, output_paths, and frame_numbers sizes" << std::endl;
        return false;
    }
    
    std::vector<std::shared_ptr<std::vector<float>>> tensors(batch_size_);
    std::vector<int> original_widths(batch_size_);
    std::vector<int> original_heights(batch_size_);
    for (int b = 0; b < batch_size_; ++b) {
        auto buffer = std::make_shared<std::vector<float>>(input_elements_);
        preprocessor_->preprocessToFloat(frames[b], *buffer);
        tensors[b] = buffer;
        original_widths[b] = frames[b].cols;
        original_heights[b] = frames[b].rows;
    }
    
    return runWithPreprocessedBatch(tensors, output_paths, frame_numbers, original_widths, original_heights);
}

bool YOLODetector::runWithPreprocessedData(const std::shared_ptr<std::vector<float>>& input_data,
                                           const std::string& output_path,
                                           int frame_number,
                                           int original_width,
                                           int original_height) {
    if (batch_size_ != 1) {
        std::cerr << "Error: runWithPreprocessedData requires batch_size=1 (current batch_size="
                  << batch_size_ << ")" << std::endl;
        return false;
    }
    
    return runWithPreprocessedBatch({input_data}, {output_path}, {frame_number},
                                    {original_width}, {original_height});
}

bool YOLODetector::runWithPreprocessedBatch(
    const std::vector<std::shared_ptr<std::vector<float>>>& inputs,
    const std::vector<std::string>& output_paths,
    const std::vector<int>& frame_numbers,
    const std::vector<int>& original_widths,
    const std::vector<int>& original_heights) {
    
    if (!context_ || !input_buffer_ || !output_buffer_) {
        std::cerr << "Error: Detector not initialized" << std::endl;
        return false;
    }
    
    if (static_cast<int>(inputs.size()) != batch_size_ ||
        static_cast<int>(output_paths.size()) != batch_size_ ||
        static_cast<int>(frame_numbers.size()) != batch_size_) {
        std::cerr << "Error: Preprocessed input batch size mismatch (expected "
                  << batch_size_ << ")" << std::endl;
        return false;
    }
    
    // Ensure correct device is active before copying data
    cudaSetDevice(gpu_id_);
    
    // Copy each preprocessed tensor into the GPU input buffer
    size_t single_input_bytes = input_size_ / batch_size_;
    for (int b = 0; b < batch_size_; ++b) {
        if (!inputs[b] || inputs[b]->size() < input_elements_) {
            std::cerr << "Error: Preprocessed input " << b << " has incorrect size" << std::endl;
            return false;
        }
        
        cudaMemcpyAsync(static_cast<char*>(input_buffer_) + b * single_input_bytes,
                        inputs[b]->data(),
                        single_input_bytes,
                        cudaMemcpyHostToDevice,
                        stream_);
    }
    
    return runInference(output_paths, frame_numbers, original_widths, original_heights);
}

std::vector<Detection> YOLODetector::parseRawDetectionOutput(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // YOLOv11 Detection Format (matching Python postprocess):
    // Format: [x_center, y_center, width, height, confidence]
    // - 5 channels total for single-class detection
    // - All values are already normalized [0,1], no sigmoid needed
    // - Python: predictions = output[0].T, boxes = predictions[:, :4], confidences = predictions[:, 4]
    // - ALL models are single-class (class_id=0)
    
    for (int i = 0; i < num_anchors_; ++i) {
        int offset = i * output_channels_;
        if (offset + 4 >= static_cast<int>(output_data.size())) break;
        
        // Extract bbox coordinates (YOLOv11: already normalized 0-1, no sigmoid)
        // Matching Python: boxes = predictions[:, :4]
        float x_center = output_data[offset + 0];
        float y_center = output_data[offset + 1];
        float width = output_data[offset + 2];
        float height = output_data[offset + 3];
        
        // Debug: Log first few bbox values to verify range
        static int debug_bbox_count = 0;
        if (debug_bbox_count < 5) {
            std::cout << "[DEBUG] Detection bbox: x_center=" << x_center 
                      << ", y_center=" << y_center 
                      << ", width=" << width 
                      << ", height=" << height << std::endl;
            debug_bbox_count++;
        }
        
        // Extract confidence (YOLOv11: index 4 is confidence, already normalized)
        // Matching Python: confidences = predictions[:, 4]
        float confidence = output_data[offset + 4];
        
        // Debug: Log first few confidence values to verify range
        static int debug_count = 0;
        if (debug_count < 5) {
            std::cout << "[DEBUG] Detection confidence value: " << confidence 
                      << " (threshold: " << conf_threshold_ << ")" << std::endl;
            debug_count++;
        }
        
        // Apply confidence threshold (matching Python: valid_indices = confidences > self.conf_threshold)
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center (normalized 0-1)
        det.bbox[1] = y_center;   // y_center (normalized 0-1)
        det.bbox[2] = width;      // width (normalized 0-1)
        det.bbox[3] = height;      // height (normalized 0-1)
        det.confidence = confidence;
        det.class_id = 0;  // ALL models are single-class (matching Python: class_id=0)
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::parseRawDetectionOutputTransposed(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // YOLOv11 Detection Format (matching Python postprocess after transpose):
    // After transpose: [channels, num_anchors] format
    // - Channel 0: x_center for all anchors
    // - Channel 1: y_center for all anchors
    // - Channel 2: width for all anchors
    // - Channel 3: height for all anchors
    // - Channel 4: confidence for all anchors
    // Python: predictions = output[0].T, boxes = predictions[:, :4], confidences = predictions[:, 4]
    
    for (int i = 0; i < num_anchors_; ++i) {
        // After transpose, data is [channels, num_anchors]
        // For anchor i: x_center is at [0*num_anchors + i], y_center at [1*num_anchors + i], etc.
        float x_center = output_data[0 * num_anchors_ + i];
        float y_center = output_data[1 * num_anchors_ + i];
        float width = output_data[2 * num_anchors_ + i];
        float height = output_data[3 * num_anchors_ + i];
        float confidence = output_data[4 * num_anchors_ + i];
        
        // Debug: Log first few values
        static int debug_count = 0;
        if (debug_count < 5) {
            std::cout << "[DEBUG Transposed] Anchor " << i << ": x_center=" << x_center 
                      << ", y_center=" << y_center << ", width=" << width 
                      << ", height=" << height << ", confidence=" << confidence << std::endl;
            debug_count++;
        }
        
        // Apply confidence threshold
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center (normalized 0-1)
        det.bbox[1] = y_center;   // y_center (normalized 0-1)
        det.bbox[2] = width;      // width (normalized 0-1)
        det.bbox[3] = height;      // height (normalized 0-1)
        det.confidence = confidence;
        det.class_id = 0;  // ALL models are single-class
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::parseRawPoseOutput(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // YOLOv11 Pose Format (matching Python postprocess_pose):
    // Format: [x_center, y_center, width, height, confidence, class_score, 
    //          kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ..., kpt16_x, kpt16_y, kpt16_conf]
    // - 56 channels total: 4(bbox) + 1(confidence) + 1(class) + 17*3(keypoints) = 56
    // - All values are already normalized [0,1], no sigmoid needed
    // - Python: boxes = output[:, :4], scores = output[:, 4], keypoints = output[:, 5:]
    // - Keypoints reshaped to (num_detections, 17, 3) - 17 keypoints, each with [x, y, conf]
    
    for (int i = 0; i < num_anchors_; ++i) {
        int offset = i * output_channels_;
        if (offset + 55 >= static_cast<int>(output_data.size())) break;  // Need at least 56 values
        
        // Extract bbox coordinates (YOLOv11: already normalized 0-1, no sigmoid)
        // Matching Python: boxes = output[:, :4]
        float x_center = output_data[offset + 0];
        float y_center = output_data[offset + 1];
        float width = output_data[offset + 2];
        float height = output_data[offset + 3];
        
        // Extract confidence (YOLOv11: index 4 is confidence, already normalized)
        // Matching Python: scores = output[:, 4]
        float confidence = output_data[offset + 4];
        
        // Apply confidence threshold (matching Python: mask = scores > self.conf_threshold)
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center (normalized 0-1)
        det.bbox[1] = y_center;   // y_center (normalized 0-1)
        det.bbox[2] = width;       // width (normalized 0-1)
        det.bbox[3] = height;      // height (normalized 0-1)
        det.confidence = confidence;
        det.class_id = 0;  // ALL models are single-class (matching Python: class_id=0)
        
        // Parse 17 keypoints (COCO format) - YOLOv11: already normalized, no sigmoid
        // Matching Python: keypoints = output[:, 5:], reshaped to (17, 3)
        det.keypoints.resize(17);
        for (int k = 0; k < 17; ++k) {
            int kpt_offset = offset + 6 + k * 3;  // Start from index 6 (after bbox + confidence + class)
            det.keypoints[k].x = output_data[kpt_offset + 0];
            det.keypoints[k].y = output_data[kpt_offset + 1];
            det.keypoints[k].confidence = output_data[kpt_offset + 2];
        }
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::parseRawPoseOutputTransposed(const std::vector<float>& output_data) {
    std::vector<Detection> detections;
    
    // YOLOv11 Pose Format (matching Python postprocess_pose after transpose):
    // After transpose: [channels, num_anchors] format
    // - Channel 0-3: bbox (x_center, y_center, width, height)
    // - Channel 4: confidence
    // - Channel 5: class_score
    // - Channel 6-56: keypoints (17 keypoints × 3 values each)
    // Python: output = output[0].T, boxes = output[:, :4], scores = output[:, 4]
    
    for (int i = 0; i < num_anchors_; ++i) {
        // After transpose, data is [channels, num_anchors]
        float x_center = output_data[0 * num_anchors_ + i];
        float y_center = output_data[1 * num_anchors_ + i];
        float width = output_data[2 * num_anchors_ + i];
        float height = output_data[3 * num_anchors_ + i];
        float confidence = output_data[4 * num_anchors_ + i];
        
        // Apply confidence threshold
        if (confidence < conf_threshold_) {
            continue;
        }
        
        Detection det;
        det.bbox[0] = x_center;   // x_center (normalized 0-1)
        det.bbox[1] = y_center;   // y_center (normalized 0-1)
        det.bbox[2] = width;       // width (normalized 0-1)
        det.bbox[3] = height;      // height (normalized 0-1)
        det.confidence = confidence;
        det.class_id = 0;  // ALL models are single-class
        
        // Parse 17 keypoints - after transpose, keypoints are in channels 6-56
        det.keypoints.resize(17);
        for (int k = 0; k < 17; ++k) {
            int kpt_channel_base = 6 + k * 3;  // Start from channel 6
            det.keypoints[k].x = output_data[kpt_channel_base * num_anchors_ + i];
            det.keypoints[k].y = output_data[(kpt_channel_base + 1) * num_anchors_ + i];
            det.keypoints[k].confidence = output_data[(kpt_channel_base + 2) * num_anchors_ + i];
        }
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> YOLODetector::applyNMS(const std::vector<Detection>& detections) {
    if (detections.empty()) {
        return detections;
    }
    
    // Convert detections to xyxy format and calculate areas (matching Python implementation)
    struct BoxXYXY {
        float x1, y1, x2, y2;
        float area;
        int original_index;
    };
    
    std::vector<BoxXYXY> boxes_xyxy(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        // Convert from center format [x_center, y_center, width, height] to xyxy
        float x_center = detections[i].bbox[0];
        float y_center = detections[i].bbox[1];
        float width = detections[i].bbox[2];
        float height = detections[i].bbox[3];
        
        boxes_xyxy[i].x1 = x_center - width / 2.0f;
        boxes_xyxy[i].y1 = y_center - height / 2.0f;
        boxes_xyxy[i].x2 = x_center + width / 2.0f;
        boxes_xyxy[i].y2 = y_center + height / 2.0f;
        boxes_xyxy[i].area = width * height;
        boxes_xyxy[i].original_index = static_cast<int>(i);
    }
    
    // Sort by confidence (descending) - matching Python: order = scores.argsort()[::-1]
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    // NMS algorithm matching Python implementation
    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        
        if (order.size() == 1) {
            break;
        }
        
        // Calculate IoU with remaining boxes
        std::vector<float> ious;
        std::vector<int> valid_inds;
        
        for (size_t j = 1; j < order.size(); ++j) {
            int idx = order[j];
            
            // Calculate intersection
            float xx1 = std::max(boxes_xyxy[i].x1, boxes_xyxy[idx].x1);
            float yy1 = std::max(boxes_xyxy[i].y1, boxes_xyxy[idx].y1);
            float xx2 = std::min(boxes_xyxy[i].x2, boxes_xyxy[idx].x2);
            float yy2 = std::min(boxes_xyxy[i].y2, boxes_xyxy[idx].y2);
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;
            
            // IoU = intersection / (area[i] + area[others] - intersection)
            float iou = intersection / (boxes_xyxy[i].area + boxes_xyxy[idx].area - intersection);
            ious.push_back(iou);
            valid_inds.push_back(j);
        }
        
        // Keep boxes with IoU <= threshold (matching Python: inds = np.where(iou <= iou_threshold)[0])
        std::vector<int> new_order;
        for (size_t k = 0; k < ious.size(); ++k) {
            if (ious[k] <= nms_threshold_) {
                new_order.push_back(order[valid_inds[k]]);
            }
        }
        order = new_order;
    }
    
    // Build result from kept indices
    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        result.push_back(detections[idx]);
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

bool YOLODetector::runInference(const std::vector<std::string>& output_paths,
                                const std::vector<int>& frame_numbers,
                                const std::vector<int>& original_widths,
                                const std::vector<int>& original_heights) {
    if (static_cast<int>(output_paths.size()) != batch_size_ ||
        static_cast<int>(frame_numbers.size()) != batch_size_) {
        std::cerr << "Error: Output metadata batch size mismatch (expected "
                  << batch_size_ << ")" << std::endl;
        return false;
    }
    
    // Set CUDA device for this detector (important for multi-GPU setups)
    cudaSetDevice(gpu_id_);
    
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
    
    size_t output_per_frame = static_cast<size_t>(num_anchors_) * output_channels_;
    size_t expected_total_output = static_cast<size_t>(batch_size_) * output_per_frame;
    if (output_data.size() != expected_total_output) {
        std::cerr << "Error: Output size mismatch. Expected " << expected_total_output
                  << " elements, got " << output_data.size() << std::endl;
        std::cerr << "  batch_size=" << batch_size_ << ", num_anchors_=" << num_anchors_
                  << ", output_channels_=" << output_channels_ << std::endl;
        return false;
    }
    
    for (int b = 0; b < batch_size_; ++b) {
        std::vector<float> frame_output(output_per_frame);
        size_t batch_offset = static_cast<size_t>(b) * output_per_frame;
        std::copy(output_data.begin() + batch_offset,
                  output_data.begin() + batch_offset + output_per_frame,
                  frame_output.begin());
        
        std::vector<Detection> detections;
        if (model_type_ == ModelType::POSE) {
            detections = parseRawPoseOutput(frame_output);
        } else {
            detections = parseRawDetectionOutput(frame_output);
        }
        
        detections = applyNMS(detections);
        if (static_cast<int>(detections.size()) > max_detections_) {
            detections.resize(max_detections_);
        }
        
        // Scale detections to original frame coordinates if dimensions provided
        int orig_w = (b < static_cast<int>(original_widths.size()) && original_widths[b] > 0) 
                     ? original_widths[b] : 0;
        int orig_h = (b < static_cast<int>(original_heights.size()) && original_heights[b] > 0) 
                     ? original_heights[b] : 0;
        
        if (orig_w > 0 && orig_h > 0) {
            // Debug: Log scaling info for first detection (once per batch)
            static bool logged_scaling_info = false;
            if (!logged_scaling_info && detections.size() > 0) {
                std::cout << "[YOLODetector] Scaling detections: original=" << orig_w << "x" << orig_h 
                          << ", preprocessed=" << input_width_ << "x" << input_height_ << std::endl;
                logged_scaling_info = true;
            }
            
            for (auto& det : detections) {
                scaleDetectionToOriginal(det, orig_w, orig_h);
            }
        } else {
            // Warning: No original dimensions provided, bbox coordinates are in normalized [0,1] format
            // relative to preprocessed image (input_width_ x input_height_)
            // This means they need to be scaled manually or the scaling step was skipped
            if (detections.size() > 0) {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "Warning: No original frame dimensions provided for scaling. "
                              << "Bbox coordinates are normalized [0,1] relative to preprocessed image ("
                              << input_width_ << "x" << input_height_ << "). "
                              << "Detections may appear incorrectly scaled." << std::endl;
                    std::cerr << "  Frame " << frame_numbers[b] << " has " << detections.size() 
                              << " detections but orig_w=" << orig_w << ", orig_h=" << orig_h << std::endl;
                    warned = true;
                }
            }
        }
        
        if (!writeDetectionsToFile(detections, output_paths[b], frame_numbers[b])) {
            std::cerr << "Error: Failed to write detections for frame " << frame_numbers[b] << std::endl;
            return false;
        }
    }
    
    return true;
}

bool YOLODetector::runInferenceWithDetections(const std::vector<std::string>& output_paths,
                                               const std::vector<int>& frame_numbers,
                                               const std::vector<int>& original_widths,
                                               const std::vector<int>& original_heights,
                                               std::vector<std::vector<Detection>>& all_detections) {
    if (static_cast<int>(output_paths.size()) != batch_size_ ||
        static_cast<int>(frame_numbers.size()) != batch_size_) {
        std::cerr << "Error: Output metadata batch size mismatch (expected "
                  << batch_size_ << ")" << std::endl;
        return false;
    }
    
    // Set CUDA device for this detector (important for multi-GPU setups)
    cudaSetDevice(gpu_id_);
    
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
    
    size_t output_per_frame = static_cast<size_t>(num_anchors_) * output_channels_;
    size_t expected_total_output = static_cast<size_t>(batch_size_) * output_per_frame;
    if (output_data.size() != expected_total_output) {
        std::cerr << "Error: Output size mismatch. Expected " << expected_total_output
                  << " elements, got " << output_data.size() << std::endl;
        return false;
    }
    
    all_detections.clear();
    all_detections.resize(batch_size_);
    
    for (int b = 0; b < batch_size_; ++b) {
        // Extract output for this batch item
        // TensorRT output format: [batch, num_anchors, channels] flattened as [batch*num_anchors*channels]
        // Python does: output[i:i+1] which gets [1, num_anchors, channels], then .T transposes to [channels, num_anchors]
        // But we read it directly as [num_anchors, channels] which should be correct
        std::vector<float> frame_output(output_per_frame);
        size_t batch_offset = static_cast<size_t>(b) * output_per_frame;
        
        if (batch_offset + output_per_frame > output_data.size()) {
            std::cerr << "Error: Batch " << b << " offset out of bounds. "
                      << "batch_offset=" << batch_offset << ", output_per_frame=" << output_per_frame
                      << ", output_data.size()=" << output_data.size() << std::endl;
            return false;
        }
        
        std::copy(output_data.begin() + batch_offset,
                  output_data.begin() + batch_offset + output_per_frame,
                  frame_output.begin());
        
        // Transpose from [num_anchors, channels] to [channels, num_anchors] to match Python
        // Python: predictions = output[0].T
        // After transpose: row i is channel i, column j is anchor j
        std::vector<float> transposed_output(output_per_frame);
        for (int anchor = 0; anchor < num_anchors_; ++anchor) {
            for (int channel = 0; channel < output_channels_; ++channel) {
                // Original: [anchor, channel] -> Transposed: [channel, anchor]
                size_t orig_idx = anchor * output_channels_ + channel;
                size_t transposed_idx = channel * num_anchors_ + anchor;
                transposed_output[transposed_idx] = frame_output[orig_idx];
            }
        }
        
        // Debug: Log first batch's first few values to verify format
        static bool logged_batch_format = false;
        if (!logged_batch_format && b == 0 && transposed_output.size() >= 4 * num_anchors_ + num_anchors_) {
            // After transpose, for anchor 0:
            // x_center = transposed_output[0] (channel 0, anchor 0)
            // y_center = transposed_output[num_anchors_] (channel 1, anchor 0)
            // width = transposed_output[2*num_anchors_] (channel 2, anchor 0)
            // height = transposed_output[3*num_anchors_] (channel 3, anchor 0)
            // confidence = transposed_output[4*num_anchors_] (channel 4, anchor 0)
            std::cout << "[DEBUG Batch] After transpose - Anchor 0: " 
                      << "x_center=" << transposed_output[0] 
                      << ", y_center=" << transposed_output[num_anchors_] 
                      << ", width=" << transposed_output[2 * num_anchors_] 
                      << ", height=" << transposed_output[3 * num_anchors_] 
                      << ", confidence=" << transposed_output[4 * num_anchors_] << std::endl;
            logged_batch_format = true;
        }
        
        std::vector<Detection> detections;
        if (model_type_ == ModelType::POSE) {
            detections = parseRawPoseOutputTransposed(transposed_output);
        } else {
            detections = parseRawDetectionOutputTransposed(transposed_output);
        }
        
        detections = applyNMS(detections);
        if (static_cast<int>(detections.size()) > max_detections_) {
            detections.resize(max_detections_);
        }
        
        // Store detections in preprocessed coordinate space (normalized [0,1] relative to input_width_ x input_height_)
        // Don't scale back to original - keep them in preprocessed space for debug images
        all_detections[b] = detections;
        
        // Scale detections to original frame coordinates for writing to file
        int orig_w = (b < static_cast<int>(original_widths.size()) && original_widths[b] > 0) 
                     ? original_widths[b] : 0;
        int orig_h = (b < static_cast<int>(original_heights.size()) && original_heights[b] > 0) 
                     ? original_heights[b] : 0;
        
        if (orig_w > 0 && orig_h > 0) {
            // Create a copy for file writing (scaled to original)
            std::vector<Detection> detections_for_file = detections;
            for (auto& det : detections_for_file) {
                scaleDetectionToOriginal(det, orig_w, orig_h);
            }
            if (!writeDetectionsToFile(detections_for_file, output_paths[b], frame_numbers[b])) {
                std::cerr << "Error: Failed to write detections for frame " << frame_numbers[b] << std::endl;
                return false;
            }
        } else {
            if (!writeDetectionsToFile(detections, output_paths[b], frame_numbers[b])) {
                std::cerr << "Error: Failed to write detections for frame " << frame_numbers[b] << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

void YOLODetector::scaleDetectionToOriginal(Detection& det, int original_width, int original_height) {
    if (original_width <= 0 || original_height <= 0) {
        // Cannot scale without valid dimensions - coordinates remain normalized [0,1]
        return;
    }
    
    // YOLO outputs are normalized (0-1) relative to preprocessed image (input_width_ x input_height_)
    // Preprocessor adds padding to maintain aspect ratio, then resizes to target size
    
    // Calculate scale factor used during preprocessing (same as in Preprocessor::addPadding)
    // This is the factor by which the original image was scaled down to fit in the target size
    float scale = std::min(static_cast<float>(input_width_) / original_width,
                          static_cast<float>(input_height_) / original_height);
    
    // Calculate dimensions after scaling (before padding)
    float scaled_w = original_width * scale;
    float scaled_h = original_height * scale;
    
    // Calculate padding offsets (centered padding)
    float pad_w = (input_width_ - scaled_w) / 2.0f;
    float pad_h = (input_height_ - scaled_h) / 2.0f;
    
    // Convert normalized coordinates (0-1) to pixel coordinates in preprocessed image
    // det.bbox values are in [0,1] range relative to input_width_ x input_height_
    float x_center_prep = det.bbox[0] * input_width_;
    float y_center_prep = det.bbox[1] * input_height_;
    float width_prep = det.bbox[2] * input_width_;
    float height_prep = det.bbox[3] * input_height_;
    
    // Remove padding offset to get coordinates in the scaled (unpadded) region
    float x_center_scaled = x_center_prep - pad_w;
    float y_center_scaled = y_center_prep - pad_h;
    
    // Scale back to original image dimensions
    float x_center_orig = x_center_scaled / scale;
    float y_center_orig = y_center_scaled / scale;
    float width_orig = width_prep / scale;
    float height_orig = height_prep / scale;
    
    // Clamp to original image bounds
    x_center_orig = std::max(0.0f, std::min(static_cast<float>(original_width), x_center_orig));
    y_center_orig = std::max(0.0f, std::min(static_cast<float>(original_height), y_center_orig));
    width_orig = std::max(0.0f, std::min(static_cast<float>(original_width), width_orig));
    height_orig = std::max(0.0f, std::min(static_cast<float>(original_height), height_orig));
    
    // Update bbox (still in center-width-height format, now in original image pixel coordinates)
    det.bbox[0] = x_center_orig;
    det.bbox[1] = y_center_orig;
    det.bbox[2] = width_orig;
    det.bbox[3] = height_orig;
    
    // Scale keypoints if present (for pose models)
    for (auto& kpt : det.keypoints) {
        // Keypoints are also normalized (0-1) relative to preprocessed image
        float kpt_x_prep = kpt.x * input_width_;
        float kpt_y_prep = kpt.y * input_height_;
        
        // Remove padding and scale back
        float kpt_x_scaled = kpt_x_prep - pad_w;
        float kpt_y_scaled = kpt_y_prep - pad_h;
        
        float kpt_x_orig = kpt_x_scaled / scale;
        float kpt_y_orig = kpt_y_scaled / scale;
        
        // Clamp to original image bounds
        kpt.x = std::max(0.0f, std::min(static_cast<float>(original_width), kpt_x_orig));
        kpt.y = std::max(0.0f, std::min(static_cast<float>(original_height), kpt_y_orig));
    }
}

void YOLODetector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    // Color palette for different classes (BGR format for OpenCV)
    std::vector<cv::Scalar> class_colors = {
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(255, 0, 0),      // Blue
        cv::Scalar(0, 0, 255),      // Red
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 128),    // Purple
        cv::Scalar(255, 165, 0),    // Orange
    };
    
    // Keypoint connections for skeleton (COCO format - 17 keypoints)
    std::vector<std::pair<int, int>> keypoint_connections = {
        {0, 1}, {0, 2},   // nose to eyes
        {1, 3}, {2, 4},   // eyes to ears
        {5, 6},           // shoulders
        {5, 7}, {7, 9},   // left arm
        {6, 8}, {8, 10},  // right arm
        {5, 11}, {6, 12}, // shoulders to hips
        {11, 12},         // hips
        {11, 13}, {13, 15}, // left leg
        {12, 14}, {14, 16}, // right leg
    };
    
    // Detections are in normalized [0,1] coordinates relative to preprocessed image (input_width_ x input_height_)
    // Convert to pixel coordinates in the preprocessed frame
    for (const auto& det : detections) {
        // Get color for this class
        cv::Scalar color = class_colors[det.class_id % class_colors.size()];
        
        // Get bbox coordinates - check if they're already in pixel space or normalized
        float x_center_raw = det.bbox[0];
        float y_center_raw = det.bbox[1];
        float width_raw = det.bbox[2];
        float height_raw = det.bbox[3];
        
        // Determine if coordinates are normalized [0,1] or in pixel space
        // If values are > 1.0, they're likely in pixel space; otherwise normalized
        bool is_normalized = (x_center_raw <= 1.0f && y_center_raw <= 1.0f && 
                             width_raw <= 1.0f && height_raw <= 1.0f);
        
        float x_center, y_center, width, height;
        if (is_normalized) {
            // Normalized [0,1] - convert to pixel coordinates
            x_center = x_center_raw * input_width_;
            y_center = y_center_raw * input_height_;
            width = width_raw * input_width_;
            height = height_raw * input_height_;
        } else {
            // Already in pixel space - use directly (but clamp to reasonable bounds)
            x_center = std::max(0.0f, std::min(static_cast<float>(input_width_), x_center_raw));
            y_center = std::max(0.0f, std::min(static_cast<float>(input_height_), y_center_raw));
            width = std::max(0.0f, std::min(static_cast<float>(input_width_), width_raw));
            height = std::max(0.0f, std::min(static_cast<float>(input_height_), height_raw));
        }
        
        // Convert from center format to corner format
        int x1 = static_cast<int>(x_center - width / 2.0f);
        int y1 = static_cast<int>(y_center - height / 2.0f);
        int x2 = static_cast<int>(x_center + width / 2.0f);
        int y2 = static_cast<int>(y_center + height / 2.0f);
        
        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));
        
        // Draw bounding box
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        
        // Draw label - handle confidence display
        // YOLOv11 should output confidence in [0,1] range, but check if it's already a percentage
        float display_confidence;
        if (det.confidence > 1.0f) {
            // Confidence is already a percentage (0-100), use directly
            display_confidence = det.confidence;
        } else {
            // Confidence is in [0,1] range, convert to percentage
            display_confidence = det.confidence * 100.0f;
        }
        // Clamp to reasonable range [0, 100]
        display_confidence = std::max(0.0f, std::min(100.0f, display_confidence));
        std::string label = "Class " + std::to_string(det.class_id) + 
                           " (" + std::to_string(static_cast<int>(display_confidence)) + "%)";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, 
                     cv::Point(x1, y1 - text_size.height - baseline - 5),
                     cv::Point(x1 + text_size.width, y1),
                     color, -1);
        cv::putText(frame, label, cv::Point(x1, y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Draw keypoints and skeleton for pose models
        if (model_type_ == ModelType::POSE && !det.keypoints.empty()) {
            // Draw keypoints - check if coordinates are normalized or in pixel space
            for (size_t i = 0; i < det.keypoints.size() && i < 17; ++i) {
                const auto& kpt = det.keypoints[i];
                if (kpt.confidence > 0.1f) {  // Only draw visible keypoints
                    // Check if keypoint coordinates are normalized [0,1] or in pixel space
                    int kpt_x, kpt_y;
                    if (kpt.x <= 1.0f && kpt.y <= 1.0f) {
                        // Normalized - convert to pixel coordinates
                        kpt_x = static_cast<int>(kpt.x * input_width_);
                        kpt_y = static_cast<int>(kpt.y * input_height_);
                    } else {
                        // Already in pixel space - use directly
                        kpt_x = static_cast<int>(kpt.x);
                        kpt_y = static_cast<int>(kpt.y);
                    }
                    kpt_x = std::max(0, std::min(kpt_x, frame.cols - 1));
                    kpt_y = std::max(0, std::min(kpt_y, frame.rows - 1));
                    cv::circle(frame, cv::Point(kpt_x, kpt_y), 3, color, -1);
                }
            }
            
            // Draw skeleton connections
            for (const auto& conn : keypoint_connections) {
                if (conn.first < static_cast<int>(det.keypoints.size()) &&
                    conn.second < static_cast<int>(det.keypoints.size())) {
                    const auto& kpt1 = det.keypoints[conn.first];
                    const auto& kpt2 = det.keypoints[conn.second];
                    if (kpt1.confidence > 0.1f && kpt2.confidence > 0.1f) {
                        // Check if keypoint coordinates are normalized [0,1] or in pixel space
                        int x1_kpt, y1_kpt, x2_kpt, y2_kpt;
                        if (kpt1.x <= 1.0f && kpt1.y <= 1.0f && kpt2.x <= 1.0f && kpt2.y <= 1.0f) {
                            // Normalized - convert to pixel coordinates
                            x1_kpt = static_cast<int>(kpt1.x * input_width_);
                            y1_kpt = static_cast<int>(kpt1.y * input_height_);
                            x2_kpt = static_cast<int>(kpt2.x * input_width_);
                            y2_kpt = static_cast<int>(kpt2.y * input_height_);
                        } else {
                            // Already in pixel space - use directly
                            x1_kpt = static_cast<int>(kpt1.x);
                            y1_kpt = static_cast<int>(kpt1.y);
                            x2_kpt = static_cast<int>(kpt2.x);
                            y2_kpt = static_cast<int>(kpt2.y);
                        }
                        x1_kpt = std::max(0, std::min(x1_kpt, frame.cols - 1));
                        y1_kpt = std::max(0, std::min(y1_kpt, frame.rows - 1));
                        x2_kpt = std::max(0, std::min(x2_kpt, frame.cols - 1));
                        y2_kpt = std::max(0, std::min(y2_kpt, frame.rows - 1));
                        cv::line(frame, cv::Point(x1_kpt, y1_kpt), cv::Point(x2_kpt, y2_kpt), color, 2);
                    }
                }
            }
        }
    }
}

