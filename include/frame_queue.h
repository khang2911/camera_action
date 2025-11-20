#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct FrameData {
    cv::Mat frame;  // Raw frame (shared among readers/preprocessors)
    std::shared_ptr<std::vector<float>> preprocessed_data;  // Reusable normalized tensor
    int video_id;
    int frame_number;
    std::string video_path;
    int original_width;   // Original frame width before preprocessing (after ROI crop if applied)
    int original_height;  // Original frame height before preprocessing (after ROI crop if applied)
    int roi_offset_x = 0;  // ROI offset X (for scaling detections back to true original frame)
    int roi_offset_y = 0;  // ROI offset Y (for scaling detections back to true original frame)
    
    FrameData() : video_id(-1), frame_number(-1), original_width(0), original_height(0), roi_offset_x(0), roi_offset_y(0) {}
    FrameData(const cv::Mat& f, int vid_id, int fnum, const std::string& vpath)
        : frame(f), video_id(vid_id), frame_number(fnum), video_path(vpath),
          original_width(f.cols), original_height(f.rows), roi_offset_x(0), roi_offset_y(0) {}
    FrameData(const std::shared_ptr<std::vector<float>>& tensor,
              int vid_id, int fnum, const std::string& vpath, int orig_w = 0, int orig_h = 0, int roi_x = 0, int roi_y = 0)
        : preprocessed_data(tensor), video_id(vid_id), frame_number(fnum), video_path(vpath),
          original_width(orig_w), original_height(orig_h), roi_offset_x(roi_x), roi_offset_y(roi_y) {}
};

class FrameQueue {
public:
    FrameQueue(size_t max_size = 100);
    
    void push(const FrameData& frame);
    bool pop(FrameData& frame, int timeout_ms = -1);
    bool empty() const;
    size_t size() const;
    void clear();
    
private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<FrameData> queue_;
    size_t max_size_;
};

#endif // FRAME_QUEUE_H

