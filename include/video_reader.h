#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

#include "video_clip.h"

class VideoReader {
public:
    VideoReader(const std::string& video_path, int video_id);
    VideoReader(const VideoClip& clip, int video_id);
    ~VideoReader();
    
    bool isOpened() const;
    bool readFrame(cv::Mat& frame);
    int getFrameNumber() const { return frame_number_; }
    int getVideoId() const { return video_id_; }
    std::string getVideoPath() const { return video_path_; }
    
private:
    void initializeMetadata();
    double computeFps(double reported_duration) const;
    
    cv::VideoCapture cap_;
    std::string video_path_;
    int video_id_;
    int frame_number_;
    long long total_frames_read_;
    double fps_;
    bool has_clip_metadata_;
    VideoClip clip_;
};

#endif // VIDEO_READER_H

