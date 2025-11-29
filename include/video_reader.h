#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <cuda_runtime_api.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include "video_clip.h"
#include "reader_options.h"

class VideoReader {
public:
    VideoReader(const std::string& video_path, int video_id, const ReaderOptions& options = ReaderOptions());
    VideoReader(const VideoClip& clip, int video_id, const ReaderOptions& options = ReaderOptions());
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
    bool convertFrameToMatGPU(AVFrame* frame, cv::Mat& out);
    bool convertFrameToMatCPU(AVFrame* frame, cv::Mat& out);
    bool ensureCudaBuffer(int width, int height);
    void cleanup();
    static void ensureFFmpegInitialized();
    void startPrefetchThread();
    void stopPrefetchThread();
    void prefetchLoop();
    
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
    uint8_t* gpu_bgr_buffer_ = nullptr;
    size_t gpu_bgr_pitch_ = 0;
    int gpu_buffer_width_ = 0;
    int gpu_buffer_height_ = 0;
    cudaStream_t cuda_stream_ = nullptr;
    bool cuda_stream_created_ = false;

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
    
    ReaderOptions options_;
    
    // Packet prefetching
    struct PrefetchQueue;
    std::unique_ptr<PrefetchQueue> packet_queue_;
    std::thread prefetch_thread_;
    std::atomic<bool> prefetch_stop_{false};
    std::atomic<bool> prefetch_eof_{false};
    std::atomic<bool> prefetch_error_{false};
    std::atomic<bool> prefetch_started_{false};
    bool prefetch_enabled_ = false;
};

#endif // VIDEO_READER_H

