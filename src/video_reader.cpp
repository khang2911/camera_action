#include "video_reader.h"

VideoReader::VideoReader(const std::string& video_path, int video_id)
    : video_path_(video_path), video_id_(video_id), frame_number_(0) {
    cap_.open(video_path);
}

VideoReader::~VideoReader() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoReader::isOpened() const {
    return cap_.isOpened();
}

bool VideoReader::readFrame(cv::Mat& frame) {
    if (!cap_.isOpened()) {
        return false;
    }
    
    bool success = cap_.read(frame);
    if (success) {
        frame_number_++;
    }
    return success;
}

