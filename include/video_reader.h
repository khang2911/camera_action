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
    int getActualFramePosition() const;  // Get actual frame position in video file
    int getVideoId() const { return video_id_; }
    std::string getVideoPath() const { return video_path_; }
    int getOriginalWidth() const { return original_width_; }
    int getOriginalHeight() const { return original_height_; }
    
private:
    void initializeMetadata();
    double computeFps(double reported_duration) const;
    
    cv::VideoCapture cap_;
    std::string video_path_;
    int video_id_;
    int frame_number_;
    long long total_frames_read_;
    int actual_frame_position_;  // Track the actual frame position that was processed (for bin file)
    double fps_;
    bool has_clip_metadata_;
    VideoClip clip_;
    int original_width_;   // Original frame width before ROI cropping
    int original_height_;  // Original frame height before ROI cropping
};

#endif // VIDEO_READER_H

