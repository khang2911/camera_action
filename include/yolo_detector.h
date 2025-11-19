#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "config_parser.h"
#include "preprocessor.h"

// Pose keypoint structure
struct Keypoint {
    float x;
    float y;
    float confidence;
    
    Keypoint() : x(0.0f), y(0.0f), confidence(0.0f) {}
    Keypoint(float x_val, float y_val, float conf) : x(x_val), y(y_val), confidence(conf) {}
};

// Detection result structure
struct Detection {
    float bbox[4];      // [x_center, y_center, width, height] or [x1, y1, x2, y2]
    float confidence;
    int class_id;
    std::vector<Keypoint> keypoints;  // Empty for detection models, 17 keypoints for pose models
    
    Detection() : confidence(0.0f), class_id(-1) {
        bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0.0f;
    }
};

class YOLODetector {
public:
    YOLODetector(const std::string& engine_path, ModelType model_type = ModelType::DETECTION, 
                 int batch_size = 1, int input_width = 640, int input_height = 640,
                 float conf_threshold = 0.25f, float nms_threshold = 0.45f, int gpu_id = 0);
    ~YOLODetector();
    
    bool initialize();
    // Detect single frame (for batch_size=1) or batch of frames (for batch_size>1)
    bool detect(const cv::Mat& frame, const std::string& output_path, int frame_number);
    bool detectBatch(const std::vector<cv::Mat>& frames, const std::vector<std::string>& output_paths, const std::vector<int>& frame_numbers);
    bool runWithPreprocessedData(const std::shared_ptr<std::vector<float>>& input_data,
                                 const std::string& output_path, int frame_number,
                                 int original_width = 0, int original_height = 0);
    bool runWithPreprocessedBatch(const std::vector<std::shared_ptr<std::vector<float>>>& inputs,
                                  const std::vector<std::string>& output_paths,
                                  const std::vector<int>& frame_numbers,
                                  const std::vector<int>& original_widths = {},
                                  const std::vector<int>& original_heights = {});
    
    ModelType getModelType() const { return model_type_; }
    int getBatchSize() const { return batch_size_; }
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getGpuId() const { return gpu_id_; }
    
private:
    std::string engine_path_;
    ModelType model_type_;
    int batch_size_;
    int input_width_;
    int input_height_;
    int gpu_id_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    
    void* input_buffer_;
    void* output_buffer_;
    size_t input_size_;
    size_t output_size_;
    size_t input_elements_;
    
    // Output dimensions
    int output_height_;      // Grid height (e.g., 80 for 640x640 input)
    int output_width_;       // Grid width
    int num_anchors_;        // Total number of anchors (height * width)
    int num_classes_;        // Number of classes
    int output_channels_;    // Number of output channels per anchor
    
    // NMS parameters
    float conf_threshold_;
    float nms_threshold_;
    int max_detections_;
    
    // Preprocessor for this detector
    std::unique_ptr<Preprocessor> preprocessor_;
    
    bool loadEngine();
    void allocateBuffers();
    void freeBuffers();
    
    // Parse raw YOLO output (without NMS)
    std::vector<Detection> parseRawDetectionOutput(const std::vector<float>& output_data);
    std::vector<Detection> parseRawPoseOutput(const std::vector<float>& output_data);
    
    // Apply NMS to detections
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections);
    
    // Calculate IoU between two boxes
    float calculateIoU(const float* box1, const float* box2);
    
    // Write results to binary file
    bool writeDetectionsToFile(const std::vector<Detection>& detections, const std::string& output_path, int frame_number);
    
    bool runInference(const std::vector<std::string>& output_paths,
                      const std::vector<int>& frame_numbers,
                      const std::vector<int>& original_widths = {},
                      const std::vector<int>& original_heights = {});
    
    // Run inference and return detections (for debug mode)
    bool runInferenceWithDetections(const std::vector<std::string>& output_paths,
                                     const std::vector<int>& frame_numbers,
                                     const std::vector<int>& original_widths,
                                     const std::vector<int>& original_heights,
                                     std::vector<std::vector<Detection>>& all_detections);
    
    // Scale detection coordinates from preprocessed image to original frame
    void scaleDetectionToOriginal(Detection& det, int original_width, int original_height);
    
    // Draw detections on frame (for debug visualization)
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
};

#endif // YOLO_DETECTOR_H

