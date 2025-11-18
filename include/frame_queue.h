#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

struct FrameData {
    cv::Mat frame;  // Raw frame (shared among readers/preprocessors)
    std::shared_ptr<std::vector<float>> preprocessed_data;  // Reusable normalized tensor
    int video_id;
    int frame_number;
    std::string video_path;
    
    FrameData() : video_id(-1), frame_number(-1) {}
    FrameData(const cv::Mat& f, int vid_id, int fnum, const std::string& vpath)
        : frame(f), video_id(vid_id), frame_number(fnum), video_path(vpath) {}
    FrameData(const std::shared_ptr<std::vector<float>>& tensor,
              int vid_id, int fnum, const std::string& vpath)
        : preprocessed_data(tensor), video_id(vid_id), frame_number(fnum), video_path(vpath) {}
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

