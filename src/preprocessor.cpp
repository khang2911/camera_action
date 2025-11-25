#include "preprocessor.h"
#include <algorithm>
#include <cstring>  // For memcpy

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
    
    // Add padding using zeros (black) - matching working C++ script
    // Create zero-padded image and copy resized image into it
    cv::Mat padded = cv::Mat::zeros(target_height_, target_width_, resized.type());
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    return padded;
}

cv::Mat Preprocessor::resizeWithPadding(const cv::Mat& frame) {
    return addPadding(frame);
}

cv::Mat Preprocessor::preprocess(const cv::Mat& frame) {
    // Step 1: Convert BGR to RGB FIRST (matching working C++ script order)
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    
    // Step 2: Resize maintaining aspect ratio
    int h = rgb.rows;
    int w = rgb.cols;
    float scale = std::min(static_cast<float>(target_width_) / w,
                          static_cast<float>(target_height_) / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // Step 3: Add padding using zeros (black) - matching working C++ script
    int pad_w = (target_width_ - new_w) / 2;
    int pad_h = (target_height_ - new_h) / 2;
    cv::Mat padded = cv::Mat::zeros(target_height_, target_width_, resized.type());
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    // Step 4: Normalize by dividing all pixels by 255 (AFTER padding)
    cv::Mat normalized;
    padded.convertTo(normalized, CV_32F, 1.0f / 255.0f);
    
    return normalized;
}

void Preprocessor::preprocessToFloat(const cv::Mat& frame, std::vector<float>& output) {
    // Optimized preprocessing: merge BGR->RGB, resize, pad, normalize, and CHW conversion
    int h = frame.rows;
    int w = frame.cols;
    
    // Calculate scale and new dimensions
    float scale = std::min(static_cast<float>(target_width_) / w,
                          static_cast<float>(target_height_) / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    int pad_w = (target_width_ - new_w) / 2;
    int pad_h = (target_height_ - new_h) / 2;
    
    // Resize with BGR->RGB conversion in one step (using cv::resize with color conversion)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // Allocate output vector (CHW format: [C, H, W])
    const size_t total_size = static_cast<size_t>(target_width_) * target_height_ * 3;
    output.resize(total_size);
    std::fill(output.begin(), output.end(), 0.0f);  // Initialize with zeros (padding)
    
    // Direct copy with normalization and CHW conversion in one pass
    // This avoids intermediate Mat allocations and channel splitting
    size_t csize = static_cast<size_t>(target_height_) * target_width_;
    const float norm_factor = 1.0f / 255.0f;
    
    // Copy resized image to padded position with normalization and CHW conversion
    // Parallelize row processing using OpenMP (if available) or std::thread
    // This is beneficial for larger images (640x640 = 640 rows)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 0; y < new_h; ++y) {
        const cv::Vec3b* src_row = resized.ptr<cv::Vec3b>(y);
        int dst_y = pad_h + y;
        
        for (int x = 0; x < new_w; ++x) {
            int dst_x = pad_w + x;
            size_t base_idx = dst_y * target_width_ + dst_x;
            
            // BGR->RGB conversion, normalization, and CHW layout in one step
            // OpenCV uses BGR, TensorRT expects RGB
            output[0 * csize + base_idx] = static_cast<float>(src_row[x][2]) * norm_factor;  // R
            output[1 * csize + base_idx] = static_cast<float>(src_row[x][1]) * norm_factor;  // G
            output[2 * csize + base_idx] = static_cast<float>(src_row[x][0]) * norm_factor;  // B
        }
    }
}

std::vector<float> Preprocessor::preprocessToFloat(const cv::Mat& frame) {
    std::vector<float> output;
    preprocessToFloat(frame, output);
    return output;
}

