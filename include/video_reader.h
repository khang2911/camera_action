#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
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
    int getActualFramePosition() const;  // Get actual frame position in video file
    int getVideoId() const { return video_id_; }
    std::string getVideoPath() const { return video_path_; }
    int getOriginalWidth() const { return original_width_; }
    int getOriginalHeight() const { return original_height_; }
    
private:
    void initializeMetadata();
    double computeFps(double reported_duration) const;
    bool initialize(const VideoClip* clip);
    bool openInput();
    bool setupDecoder();
    bool initHardwareDecoder(const AVCodec* decoder);
    static AVPixelFormat getHWFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);
    bool sendNextPacket();
    bool receiveFrame(cv::Mat& frame);
    bool convertFrameToMat(AVFrame* frame, cv::Mat& out);
    void cleanup();
    static void ensureFFmpegInitialized();
    
    AVFormatContext* fmt_ctx_;
    AVCodecContext* codec_ctx_;
    AVStream* video_stream_;
    AVPacket* packet_;
    AVFrame* frame_;
    AVFrame* sw_frame_;
    SwsContext* sws_ctx_;
    AVBufferRef* hw_device_ctx_;
    AVPixelFormat hw_pix_fmt_;
    bool use_hw_decode_;
    bool end_of_stream_;

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

