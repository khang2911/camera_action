#include "video_reader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

VideoReader::VideoReader(const std::string& video_path, int video_id)
    : format_ctx_(nullptr),
      codec_ctx_(nullptr),
      codec_(nullptr),
      frame_(nullptr),
      frame_rgb_(nullptr),
      rgb_buffer_(nullptr),
      sws_ctx_(nullptr),
      packet_(nullptr),
      video_stream_index_(-1),
      video_path_(video_path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      fps_(0.0),
      time_base_(0.0),
      start_pts_(AV_NOPTS_VALUE),
      end_pts_(AV_NOPTS_VALUE),
      has_clip_metadata_(false),
      original_width_(0),
      original_height_(0),
      initialized_(false) {
    initializeMetadata();
}

VideoReader::VideoReader(const VideoClip& clip, int video_id)
    : format_ctx_(nullptr),
      codec_ctx_(nullptr),
      codec_(nullptr),
      frame_(nullptr),
      frame_rgb_(nullptr),
      rgb_buffer_(nullptr),
      sws_ctx_(nullptr),
      packet_(nullptr),
      video_stream_index_(-1),
      video_path_(clip.path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      fps_(0.0),
      time_base_(0.0),
      start_pts_(AV_NOPTS_VALUE),
      end_pts_(AV_NOPTS_VALUE),
      has_clip_metadata_(true),
      clip_(clip),
      original_width_(0),
      original_height_(0),
      initialized_(false) {
    initializeMetadata();
}

VideoReader::~VideoReader() {
    cleanup();
}

void VideoReader::cleanup() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (rgb_buffer_) {
        av_free(rgb_buffer_);
        rgb_buffer_ = nullptr;
    }
    if (frame_rgb_) {
        av_frame_free(&frame_rgb_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (packet_) {
        av_packet_free(&packet_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
    }
}

bool VideoReader::isOpened() const {
    return initialized_ && format_ctx_ != nullptr && codec_ctx_ != nullptr;
}

void VideoReader::initializeMetadata() {
    // Allocate format context
    format_ctx_ = avformat_alloc_context();
    if (!format_ctx_) {
        std::cerr << "Error: Could not allocate format context" << std::endl;
        return;
    }

    // Open input file
    if (avformat_open_input(&format_ctx_, video_path_.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error: Could not open video file: " << video_path_ << std::endl;
        cleanup();
        return;
    }

    // Find stream info
    if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
        std::cerr << "Error: Could not find stream info" << std::endl;
        cleanup();
        return;
    }

    // Find video stream
    video_stream_index_ = -1;
    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index_ = i;
            break;
        }
    }

    if (video_stream_index_ == -1) {
        std::cerr << "Error: Could not find video stream" << std::endl;
        cleanup();
        return;
    }

    // Get codec parameters
    AVCodecParameters* codecpar = format_ctx_->streams[video_stream_index_]->codecpar;
    
    // Find decoder
    codec_ = avcodec_find_decoder(codecpar->codec_id);
    if (!codec_) {
        std::cerr << "Error: Could not find decoder" << std::endl;
        cleanup();
        return;
    }

    // Allocate codec context
    codec_ctx_ = avcodec_alloc_context3(codec_);
    if (!codec_ctx_) {
        std::cerr << "Error: Could not allocate codec context" << std::endl;
        cleanup();
        return;
    }

    // Copy codec parameters to context
    if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
        std::cerr << "Error: Could not copy codec parameters" << std::endl;
        cleanup();
        return;
    }

    // Enable multi-threaded decoding for better performance
    // Set thread_count to 0 to auto-detect optimal thread count
    codec_ctx_->thread_count = 0;
    // Use frame-level threading if available (faster for most codecs)
    codec_ctx_->thread_type = FF_THREAD_FRAME;
    
    // Open codec
    if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
        std::cerr << "Error: Could not open codec" << std::endl;
        cleanup();
        return;
    }

    // Get time base
    AVRational time_base = format_ctx_->streams[video_stream_index_]->time_base;
    time_base_ = av_q2d(time_base);

    // Compute FPS
    AVRational fps_rational = format_ctx_->streams[video_stream_index_]->avg_frame_rate;
    if (fps_rational.num > 0 && fps_rational.den > 0) {
        fps_ = av_q2d(fps_rational);
    } else {
        fps_ = computeFps(has_clip_metadata_ ? clip_.duration_seconds : 0.0);
    }

    if (fps_ <= 0.0) {
        fps_ = 30.0;
    }

    // Store original dimensions
    original_width_ = codec_ctx_->width;
    original_height_ = codec_ctx_->height;

    // Allocate frames
    frame_ = av_frame_alloc();
    frame_rgb_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    if (!frame_ || !frame_rgb_ || !packet_) {
        std::cerr << "Error: Could not allocate frames or packet" << std::endl;
        cleanup();
        return;
    }

    // Allocate buffer for RGB frame (kept for compatibility, but we'll use frame_mat_ directly)
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, original_width_, original_height_, 1);
    rgb_buffer_ = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
    if (!rgb_buffer_) {
        std::cerr << "Error: Could not allocate RGB buffer" << std::endl;
        cleanup();
        return;
    }
    av_image_fill_arrays(frame_rgb_->data, frame_rgb_->linesize, rgb_buffer_, AV_PIX_FMT_BGR24,
                         original_width_, original_height_, 1);

    // Pre-allocate OpenCV Mat to avoid reallocation and reduce cloning overhead
    frame_mat_ = cv::Mat(original_height_, original_width_, CV_8UC3);

    // Initialize SWS context for color conversion
    // Use SWS_FAST_BILINEAR for better performance (faster than SWS_BILINEAR)
    sws_ctx_ = sws_getContext(
        codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
        original_width_, original_height_, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    if (!sws_ctx_) {
        std::cerr << "Error: Could not initialize SWS context" << std::endl;
        cleanup();
        return;
    }

    // Calculate ROI offset if ROI is defined
    if (has_clip_metadata_ && clip_.has_roi) {
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

    // Seek to start timestamp if time window is defined
    if (has_clip_metadata_ && clip_.has_time_window && 
        std::isfinite(clip_.start_timestamp) && std::isfinite(clip_.moment_time)) {
        
        // Calculate start and end PTS
        double offset_seconds = clip_.start_timestamp - clip_.moment_time;
        if (offset_seconds > 0.0) {
            start_pts_ = timestampToPts(clip_.start_timestamp);
            if (std::isfinite(clip_.end_timestamp)) {
                end_pts_ = timestampToPts(clip_.end_timestamp);
            }
            
            // Seek to start timestamp
            if (!seekToTimestamp(clip_.start_timestamp)) {
                std::cerr << "Warning: Could not seek to start timestamp, starting from beginning" << std::endl;
            }
        } else {
            // Start from beginning
            start_pts_ = 0;
            if (std::isfinite(clip_.end_timestamp)) {
                end_pts_ = timestampToPts(clip_.end_timestamp);
            }
        }
    } else {
        start_pts_ = 0;
        end_pts_ = AV_NOPTS_VALUE;
    }

    initialized_ = true;
}

int64_t VideoReader::timestampToPts(double timestamp) const {
    if (time_base_ <= 0.0 || !std::isfinite(timestamp)) {
        return AV_NOPTS_VALUE;
    }
    
    AVStream* stream = format_ctx_->streams[video_stream_index_];
    
    // If we have clip metadata with moment_time, we need to calculate the offset
    if (has_clip_metadata_ && std::isfinite(clip_.moment_time)) {
        // Calculate relative seconds from moment_time
        double relative_seconds = timestamp - clip_.moment_time;
        
        // Convert to PTS using the stream's time base
        int64_t pts_offset = static_cast<int64_t>(relative_seconds / time_base_);
        
        // Get the base PTS (first frame PTS) of the stream
        // If start_time is not set, we assume the first frame is at PTS 0
        int64_t base_pts = 0;
        if (stream->start_time != AV_NOPTS_VALUE) {
            base_pts = stream->start_time;
        }
        
        return base_pts + pts_offset;
    }
    
    // Without clip metadata, try to use timestamp directly
    // This is less accurate but might work for some cases
    return static_cast<int64_t>(timestamp / time_base_);
}

double VideoReader::ptsToTimestamp(int64_t pts) const {
    if (time_base_ <= 0.0 || pts == AV_NOPTS_VALUE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    AVStream* stream = format_ctx_->streams[video_stream_index_];
    
    // Get base PTS (first frame PTS)
    int64_t base_pts = 0;
    if (stream->start_time != AV_NOPTS_VALUE) {
        base_pts = stream->start_time;
    }
    
    // Calculate relative seconds from base
    double relative_seconds = (pts - base_pts) * time_base_;
    
    // Convert to absolute timestamp if we have moment_time
    if (has_clip_metadata_ && std::isfinite(clip_.moment_time)) {
        return clip_.moment_time + relative_seconds;
    }
    
    return relative_seconds;
}

bool VideoReader::seekToTimestamp(double timestamp) {
    if (!isOpened()) {
        return false;
    }

    int64_t target_pts = timestampToPts(timestamp);
    if (target_pts == AV_NOPTS_VALUE) {
        return false;
    }

    // Seek to the target PTS
    // AVSEEK_FLAG_BACKWARD ensures we get a keyframe before the target
    int ret = av_seek_frame(format_ctx_, video_stream_index_, target_pts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codec_ctx_);

    // Reset frame counters
    total_frames_read_ = 0;
    frame_number_ = 0;

    return true;
}

bool VideoReader::decodeFrame(cv::Mat& frame) {
    if (!isOpened()) {
        return false;
    }

    // Main decode loop - try to get buffered frames first, then read packets
    while (true) {
        // Try to receive a frame from decoder (might have buffered frames from previous packets)
        int ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret == 0) {
            // We have a frame, check if it's within time window
            int64_t frame_pts = (frame_->pts != AV_NOPTS_VALUE) ? frame_->pts : AV_NOPTS_VALUE;
            
            if (frame_pts != AV_NOPTS_VALUE && has_clip_metadata_ && clip_.has_time_window) {
                double current_timestamp = ptsToTimestamp(frame_pts);
                if (std::isfinite(current_timestamp)) {
                    if (current_timestamp < clip_.start_timestamp) {
                        // Frame is before start, continue to next frame
                        continue;
                    }
                    if (std::isfinite(clip_.end_timestamp) && current_timestamp > clip_.end_timestamp) {
                        return false;
                    }
                }
            }

            // Convert frame to BGR24 (OpenCV format) directly into pre-allocated Mat
            sws_scale(sws_ctx_,
                     frame_->data, frame_->linesize, 0, codec_ctx_->height,
                     frame_mat_.data, frame_mat_.step);

            // Copy to output frame (need copy since frame_mat_ is reused)
            frame = frame_mat_.clone();
            return true;
        } else if (ret == AVERROR(EAGAIN)) {
            // Need more input, read a packet
            if (av_read_frame(format_ctx_, packet_) < 0) {
                return false;  // End of file or error
            }
            
            if (packet_->stream_index == video_stream_index_) {
                // Send packet to decoder
                ret = avcodec_send_packet(codec_ctx_, packet_);
                if (ret < 0 && ret != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    continue;  // Skip this packet and try next
                }
                // Continue loop to receive frame
            }
            av_packet_unref(packet_);
        } else {
            // Error or EOF
            return false;
        }
    }
}

bool VideoReader::readFrame(cv::Mat& frame) {
    if (!isOpened()) {
        return false;
    }

    if (decodeFrame(frame)) {
        ++total_frames_read_;
        ++frame_number_;
        return true;
    }

    return false;
}

double VideoReader::computeFps(double reported_duration) const {
    if (!isOpened()) {
        return 30.0;
    }

    // Try to get FPS from stream
    AVRational fps_rational = format_ctx_->streams[video_stream_index_]->avg_frame_rate;
    if (fps_rational.num > 0 && fps_rational.den > 0) {
        double fps_value = av_q2d(fps_rational);
        if (std::isfinite(fps_value) && fps_value > 0.1) {
            return fps_value;
        }
    }

    // Try to compute from duration and frame count
    if (reported_duration > 0.0) {
        int64_t duration = format_ctx_->duration;
        if (duration != AV_NOPTS_VALUE && duration > 0) {
            double duration_seconds = duration / (double)AV_TIME_BASE;
            if (duration_seconds > 0.0) {
                // Get frame count from stream
                int64_t frame_count = format_ctx_->streams[video_stream_index_]->nb_frames;
                if (frame_count > 0) {
                    double derived = frame_count / duration_seconds;
                    if (std::isfinite(derived) && derived > 0.1) {
                        return derived;
                    }
                }
            }
        }
    }

    return 30.0;
}
