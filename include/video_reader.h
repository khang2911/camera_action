#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

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
    int getOriginalWidth() const { return original_width_; }
    int getOriginalHeight() const { return original_height_; }
    
private:
    void initializeMetadata();
    void cleanup();
    bool seekToTimestamp(double timestamp);
    bool decodeFrame(cv::Mat& frame);
    double computeFps(double reported_duration) const;
    int64_t timestampToPts(double timestamp) const;
    double ptsToTimestamp(int64_t pts) const;
    
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    const AVCodec* codec_;
    AVFrame* frame_;
    AVFrame* frame_rgb_;
    uint8_t* rgb_buffer_;
    SwsContext* sws_ctx_;
    AVPacket* packet_;
    int video_stream_index_;
    
    std::string video_path_;
    int video_id_;
    int frame_number_;
    long long total_frames_read_;
    double fps_;
    double time_base_;
    int64_t start_pts_;
    int64_t end_pts_;
    bool has_clip_metadata_;
    VideoClip clip_;
    int original_width_;
    int original_height_;
    bool initialized_;
};

#endif // VIDEO_READER_H
