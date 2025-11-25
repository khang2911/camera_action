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
#if defined(__AVX2__)
#include <immintrin.h>
#endif

struct ScratchBuffers {
    cv::Mat rgb;
    cv::Mat resized;
    cv::Mat padded;
    cv::Mat normalized;
    std::vector<cv::Mat> channels;
};

inline bool hasAVX2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

void convertToCHWScalar(const cv::Mat& normalized, std::vector<float>& output) {
    const size_t total = static_cast<size_t>(normalized.rows) * normalized.cols;
    output.resize(total * 3);
    float* dst_r = output.data();
    float* dst_g = dst_r + total;
    float* dst_b = dst_g + total;
    
    for (size_t i = 0; i < total; ++i) {
        const float* src = normalized.ptr<float>() + i * 3;
        dst_b[i] = src[0];
        dst_g[i] = src[1];
        dst_r[i] = src[2];
    }
}

#if defined(__AVX2__)
void convertToCHWAVX(const cv::Mat& normalized, std::vector<float>& output) {
    const size_t total = static_cast<size_t>(normalized.rows) * normalized.cols;
    output.resize(total * 3);
    float* dst_r = output.data();
    float* dst_g = dst_r + total;
    float* dst_b = dst_g + total;
    
    const float* src = normalized.ptr<float>();
    const __m256i offsets = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i stride = _mm256_set1_epi32(24);
    __m256i base = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i two = _mm256_set1_epi32(2);
    
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256i idx_b = _mm256_add_epi32(base, offsets);
        __m256i idx_g = _mm256_add_epi32(idx_b, one);
        __m256i idx_r = _mm256_add_epi32(idx_b, two);
        
        __m256 b = _mm256_i32gather_ps(src, idx_b, 4);
        __m256 g = _mm256_i32gather_ps(src, idx_g, 4);
        __m256 r = _mm256_i32gather_ps(src, idx_r, 4);
        
        _mm256_storeu_ps(dst_b + i, b);
        _mm256_storeu_ps(dst_g + i, g);
        _mm256_storeu_ps(dst_r + i, r);
        
        base = _mm256_add_epi32(base, stride);
    }
    
    for (; i < total; ++i) {
        const float* src_ptr = src + i * 3;
        dst_b[i] = src_ptr[0];
        dst_g[i] = src_ptr[1];
        dst_r[i] = src_ptr[2];
    }
}
#endif

void convertToCHW(const cv::Mat& normalized, std::vector<float>& output) {
#if defined(__AVX2__)
    if (hasAVX2()) {
        convertToCHWAVX(normalized, output);
        return;
    }
#endif
    convertToCHWScalar(normalized, output);
}
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
    
    // Step 5: Convert to CHW using SIMD if available
    convertToCHW(scratch.normalized, output);
}

std::vector<float> Preprocessor::preprocessToFloat(const cv::Mat& frame) {
    std::vector<float> output;
    preprocessToFloat(frame, output);
    return output;
}

