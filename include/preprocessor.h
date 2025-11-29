#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Preprocessor {
public:
    Preprocessor(int target_width = 640, int target_height = 640);
    
    // Preprocess frame: padding, resize to 640x640, normalize by 255
    cv::Mat preprocess(const cv::Mat& frame);
    
    // Get preprocessed data as float array (for TensorRT input)
    std::vector<float> preprocessToFloat(const cv::Mat& frame);
    void preprocessToFloat(const cv::Mat& frame, std::vector<float>& output);
    
    // Get resized frame with padding (for debug visualization)
    cv::Mat addPadding(const cv::Mat& frame);
    
private:
    int target_width_;
    int target_height_;
    
    cv::Mat resizeWithPadding(const cv::Mat& frame);
};

#endif // PREPROCESSOR_H

