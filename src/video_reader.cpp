#include "video_reader.h"

#include <algorithm>
#include <cmath>
#include <limits>

VideoReader::VideoReader(const std::string& video_path, int video_id)
    : video_path_(video_path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      fps_(0.0),
      has_clip_metadata_(false),
      original_width_(0),
      original_height_(0) {
    cap_.open(video_path_);
    initializeMetadata();
}

VideoReader::VideoReader(const VideoClip& clip, int video_id)
    : video_path_(clip.path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      fps_(0.0),
      has_clip_metadata_(true),
      clip_(clip),
      original_width_(0),
      original_height_(0) {
    cap_.open(video_path_);
    initializeMetadata();
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

    while (true) {
        if (!cap_.read(frame)) {
            return false;
        }

        ++total_frames_read_;

        if (has_clip_metadata_ && clip_.has_time_window) {
            const double effective_fps = (fps_ > 0.0) ? fps_ : 30.0;
            const double current_ts =
                clip_.moment_time + static_cast<double>(total_frames_read_) / effective_fps;
            if (current_ts < clip_.start_timestamp) {
                continue;
            }
            if (current_ts > clip_.end_timestamp) {
                return false;
            }
        }

        // Store original dimensions (for ROI offset calculation)
        original_width_ = frame.cols;
        original_height_ = frame.rows;

        // Calculate ROI offset if ROI is defined (for scaling detections back to original frame)
        // Note: ROI cropping is now applied per-engine in preprocessor, not here
        // We keep the original frame intact and let each engine decide if it needs ROI cropping
        if (has_clip_metadata_ && clip_.has_roi && !frame.empty()) {
            int x1 = static_cast<int>(clip_.roi_x1 * original_width_);
            int y1 = static_cast<int>(clip_.roi_y1 * original_height_);
            clip_.roi_offset_x = std::max(0, std::min(x1, original_width_ - 1));
            clip_.roi_offset_y = std::max(0, std::min(y1, original_height_ - 1));
        } else {
            if (has_clip_metadata_) {
                clip_.roi_offset_x = 0;
                clip_.roi_offset_y = 0;
            }
        }

        ++frame_number_;
        return true;
    }
}

void VideoReader::initializeMetadata() {
    if (!cap_.isOpened()) {
        return;
    }

    const double duration_hint = has_clip_metadata_ ? clip_.duration_seconds : 0.0;
    fps_ = computeFps(duration_hint);
    if (fps_ <= 0.0) {
        fps_ = 30.0;
    }

    if (has_clip_metadata_ && clip_.has_time_window && std::isfinite(clip_.start_timestamp) &&
        std::isfinite(clip_.moment_time) && fps_ > 0.0) {
        const double offset_seconds = clip_.start_timestamp - clip_.moment_time;
        if (offset_seconds > 0.0) {
            const double start_frame = std::max(0.0, std::floor(offset_seconds * fps_));
            cap_.set(cv::CAP_PROP_POS_FRAMES, start_frame);
            total_frames_read_ = static_cast<long long>(start_frame);
        }
    }
}

double VideoReader::computeFps(double reported_duration) const {
    double fps_value = cap_.get(cv::CAP_PROP_FPS);
    if (std::isfinite(fps_value) && fps_value > 0.1) {
        return fps_value;
    }

    double total_frames = cap_.get(cv::CAP_PROP_FRAME_COUNT);
    if (reported_duration > 0.0 && total_frames > 0.0) {
        double derived = total_frames / reported_duration;
        if (std::isfinite(derived) && derived > 0.1) {
            return derived;
        }
    }

    return (fps_value > 0.0) ? fps_value : 30.0;
}

