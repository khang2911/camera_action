#include "preprocessor.h"
#include <algorithm>

Preprocessor::Preprocessor(int target_width, int target_height)
    : target_width_(target_width), target_height_(target_height) {}

cv::Mat Preprocessor::addPadding(const cv::Mat& frame) {
    int h = frame.rows;
    int w = frame.cols;
    
    // Calculate aspect ratio
    float scale = std::min(static_cast<float>(target_width_) / w,
                          static_cast<float>(target_height_) / h);
    
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    
    // Resize maintaining aspect ratio
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // Calculate padding
    int pad_w = (target_width_ - new_w) / 2;
    int pad_h = (target_height_ - new_h) / 2;
    
    // Add padding (black padding)
    cv::Mat padded;
    // Python preprocessing fills padding with value 114 (gray) before normalization.
    // Match that behavior instead of default black padding.
    cv::copyMakeBorder(resized, padded, pad_h, target_height_ - new_h - pad_h,
                      pad_w, target_width_ - new_w - pad_w,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return padded;
}

cv::Mat Preprocessor::resizeWithPadding(const cv::Mat& frame) {
    return addPadding(frame);
}

cv::Mat Preprocessor::preprocess(const cv::Mat& frame) {
    // Step 1: Add padding and resize to 640x640
    cv::Mat resized = resizeWithPadding(frame);
    
    // Step 2: Normalize by dividing all pixels by 255
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    return normalized;
}

void Preprocessor::preprocessToFloat(const cv::Mat& frame, std::vector<float>& output) {
    cv::Mat processed = preprocess(frame);
    
    // Convert BGR to RGB (OpenCV uses BGR by default, but model expects RGB)
    cv::Mat rgb;
    cv::cvtColor(processed, rgb, cv::COLOR_BGR2RGB);
    
    // Flatten to CHW format for TensorRT: [C, H, W]
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    
    // TensorRT expects CHW format: [C, H, W]
    const size_t total_size = static_cast<size_t>(target_width_) * target_height_ * 3;
    output.resize(total_size);
    
    size_t offset = 0;
    for (int c = 0; c < 3; ++c) {
        const float* data = channels[c].ptr<float>();
        for (int i = 0; i < target_height_ * target_width_; ++i) {
            output[offset++] = data[i];
        }
    }
}

std::vector<float> Preprocessor::preprocessToFloat(const cv::Mat& frame) {
    std::vector<float> output;
    preprocessToFloat(frame, output);
    return output;
}

