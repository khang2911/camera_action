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

namespace {
struct ScratchBuffers {
    cv::Mat rgb;
    cv::Mat resized;
    cv::Mat padded;
    cv::Mat normalized;
    std::vector<cv::Mat> channels;
};
}  // namespace

void Preprocessor::preprocessToFloat(const cv::Mat& frame, std::vector<float>& output) {
    static thread_local ScratchBuffers scratch;
    
    // Step 1: Convert BGR to RGB
    cv::cvtColor(frame, scratch.rgb, cv::COLOR_BGR2RGB);
    
    int h = scratch.rgb.rows;
    int w = scratch.rgb.cols;
    float scale = std::min(static_cast<float>(target_width_) / w,
                          static_cast<float>(target_height_) / h);
    int new_w = std::max(1, static_cast<int>(std::round(w * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(h * scale)));
    int pad_w = (target_width_ - new_w) / 2;
    int pad_h = (target_height_ - new_h) / 2;
    
    // Step 2: Resize
    cv::resize(scratch.rgb, scratch.resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // Step 3: Add padding (reuse buffer)
    scratch.padded.create(target_height_, target_width_, CV_8UC3);
    scratch.padded.setTo(cv::Scalar(0, 0, 0));
    scratch.resized.copyTo(scratch.padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    // Step 4: Normalize to float
    scratch.normalized.create(target_height_, target_width_, CV_32FC3);
    scratch.padded.convertTo(scratch.normalized, CV_32F, 1.0f / 255.0f);
    
    // Step 5: Convert to CHW
    scratch.channels.resize(3);
    cv::split(scratch.normalized, scratch.channels);
    
    const size_t csize = static_cast<size_t>(target_width_) * target_height_;
    output.resize(csize * 3);
    for (int c = 0; c < 3; ++c) {
        const float* data = scratch.channels[c].ptr<float>();
        std::memcpy(output.data() + static_cast<size_t>(c) * csize, data, csize * sizeof(float));
    }
}

std::vector<float> Preprocessor::preprocessToFloat(const cv::Mat& frame) {
    std::vector<float> output;
    preprocessToFloat(frame, output);
    return output;
}

