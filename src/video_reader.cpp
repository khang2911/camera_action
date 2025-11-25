#include "video_reader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

#include "logger.h"

namespace {
void log_ffmpeg_error(const std::string& prefix, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    LOG_ERROR("VideoReader", prefix + ": " + std::string(buf));
}
}  // namespace

VideoReader::VideoReader(const std::string& video_path, int video_id)
    : fmt_ctx_(nullptr),
      codec_ctx_(nullptr),
      video_stream_(nullptr),
      packet_(nullptr),
      frame_(nullptr),
      sw_frame_(nullptr),
      sws_ctx_(nullptr),
      hw_device_ctx_(nullptr),
      hw_pix_fmt_(AV_PIX_FMT_NONE),
      use_hw_decode_(false),
      end_of_stream_(false),
      video_path_(video_path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      actual_frame_position_(0),
      fps_(0.0),
      has_clip_metadata_(false),
      original_width_(0),
      original_height_(0) {
    initialize(nullptr);
}

VideoReader::VideoReader(const VideoClip& clip, int video_id)
    : fmt_ctx_(nullptr),
      codec_ctx_(nullptr),
      video_stream_(nullptr),
      packet_(nullptr),
      frame_(nullptr),
      sw_frame_(nullptr),
      sws_ctx_(nullptr),
      hw_device_ctx_(nullptr),
      hw_pix_fmt_(AV_PIX_FMT_NONE),
      use_hw_decode_(false),
      end_of_stream_(false),
      video_path_(clip.path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      actual_frame_position_(0),
      fps_(0.0),
      has_clip_metadata_(true),
      clip_(clip),
      original_width_(0),
      original_height_(0) {
    initialize(&clip_);
}

VideoReader::~VideoReader() {
    cleanup();
}

void VideoReader::ensureFFmpegInitialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        av_log_set_level(AV_LOG_ERROR);
        avformat_network_init();
    });
}

bool VideoReader::initialize(const VideoClip* clip) {
    ensureFFmpegInitialized();
    if (!openInput()) {
        return false;
    }
    if (!setupDecoder()) {
        return false;
    }
    initializeMetadata();
    return true;
}

bool VideoReader::openInput() {
    int ret = avformat_open_input(&fmt_ctx_, video_path_.c_str(), nullptr, nullptr);
    if (ret < 0) {
        log_ffmpeg_error("avformat_open_input", ret);
        return false;
    }
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        log_ffmpeg_error("avformat_find_stream_info", ret);
        return false;
    }
    return true;
}

bool VideoReader::setupDecoder() {
    int ret = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0) {
        log_ffmpeg_error("av_find_best_stream", ret);
        return false;
    }
    video_stream_ = fmt_ctx_->streams[ret];
    
    const AVCodec* decoder = avcodec_find_decoder(video_stream_->codecpar->codec_id);
    if (!decoder) {
        LOG_ERROR("VideoReader", "Failed to find decoder");
        return false;
    }
    
    codec_ctx_ = avcodec_alloc_context3(decoder);
    if (!codec_ctx_) {
        LOG_ERROR("VideoReader", "Failed to allocate codec context");
        return false;
    }
    ret = avcodec_parameters_to_context(codec_ctx_, video_stream_->codecpar);
    if (ret < 0) {
        log_ffmpeg_error("avcodec_parameters_to_context", ret);
        return false;
    }
    
    codec_ctx_->pkt_timebase = video_stream_->time_base;
    codec_ctx_->thread_count = 1;
    
    if (initHardwareDecoder(decoder)) {
        codec_ctx_->opaque = this;
        codec_ctx_->get_format = &VideoReader::getHWFormat;
    }
    
    ret = avcodec_open2(codec_ctx_, decoder, nullptr);
    if (ret < 0) {
        log_ffmpeg_error("avcodec_open2", ret);
        return false;
    }
    
    packet_ = av_packet_alloc();
    frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    return packet_ && frame_ && sw_frame_;
}

bool VideoReader::initHardwareDecoder(const AVCodec* decoder) {
    AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        return false;
    }
    
    for (int i = 0;; ++i) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            return false;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == type) {
            hw_pix_fmt_ = config->pix_fmt;
            int err = av_hwdevice_ctx_create(&hw_device_ctx_, type, nullptr, nullptr, 0);
            if (err < 0) {
                log_ffmpeg_error("av_hwdevice_ctx_create", err);
                return false;
            }
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            use_hw_decode_ = true;
            return true;
        }
    }
    return false;
}

AVPixelFormat VideoReader::getHWFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    auto* self = reinterpret_cast<VideoReader*>(ctx->opaque);
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == self->hw_pix_fmt_) {
            return *p;
        }
    }
    return pix_fmts[0];
}

bool VideoReader::isOpened() const {
    return codec_ctx_ != nullptr;
}

bool VideoReader::readFrame(cv::Mat& frame) {
    if (!codec_ctx_) {
        return false;
    }
    
    while (true) {
        if (!receiveFrame(frame)) {
            if (!sendNextPacket()) {
                return false;
            }
            continue;
        }
        
        ++frame_number_;
        ++total_frames_read_;
        actual_frame_position_ = static_cast<int>(total_frames_read_ - 1);
        
        original_width_ = frame.cols;
        original_height_ = frame.rows;
        
        if (has_clip_metadata_ && clip_.has_roi && !frame.empty()) {
            int x1 = static_cast<int>(clip_.roi_x1 * original_width_);
            int y1 = static_cast<int>(clip_.roi_y1 * original_height_);
            clip_.roi_offset_x = std::max(0, std::min(x1, original_width_ - 1));
            clip_.roi_offset_y = std::max(0, std::min(y1, original_height_ - 1));
        } else if (has_clip_metadata_) {
            clip_.roi_offset_x = 0;
            clip_.roi_offset_y = 0;
        }
        
        if (has_clip_metadata_ && clip_.has_time_window) {
            double effective_fps = (fps_ > 0.0) ? fps_ : 30.0;
            double current_ts = clip_.moment_time + static_cast<double>(total_frames_read_) / effective_fps;
            if (current_ts < clip_.start_timestamp) {
                continue;
            }
            if (current_ts > clip_.end_timestamp) {
                return false;
            }
        }
        
        return true;
    }
}

bool VideoReader::sendNextPacket() {
    if (end_of_stream_) {
        return false;
    }
    
    while (true) {
        int ret = av_read_frame(fmt_ctx_, packet_);
        if (ret == AVERROR_EOF) {
            end_of_stream_ = true;
            avcodec_send_packet(codec_ctx_, nullptr);
            return true;
        }
        if (ret < 0) {
            log_ffmpeg_error("av_read_frame", ret);
            return false;
        }
        if (packet_->stream_index == static_cast<int>(video_stream_->index)) {
            ret = avcodec_send_packet(codec_ctx_, packet_);
            av_packet_unref(packet_);
            if (ret < 0) {
                log_ffmpeg_error("avcodec_send_packet", ret);
                return false;
            }
            return true;
        }
        av_packet_unref(packet_);
    }
}

bool VideoReader::receiveFrame(cv::Mat& out) {
    while (true) {
        int ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret == AVERROR(EAGAIN)) {
            return false;
        }
        if (ret == AVERROR_EOF) {
            return false;
        }
        if (ret < 0) {
            log_ffmpeg_error("avcodec_receive_frame", ret);
            return false;
        }
        
        AVFrame* src = frame_;
        if (use_hw_decode_ && frame_->format == hw_pix_fmt_) {
            if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                LOG_ERROR("VideoReader", "av_hwframe_transfer_data failed");
                continue;
            }
            src = sw_frame_;
        }
        
        if (convertFrameToMat(src, out)) {
            return true;
        }
    }
}

bool VideoReader::convertFrameToMat(AVFrame* src, cv::Mat& out) {
    AVPixelFormat src_fmt = static_cast<AVPixelFormat>(src->format);
    sws_ctx_ = sws_getCachedContext(
        sws_ctx_,
        src->width,
        src->height,
        src_fmt,
        src->width,
        src->height,
        AV_PIX_FMT_BGR24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr);
    if (!sws_ctx_) {
        LOG_ERROR("VideoReader", "sws_getCachedContext failed");
        return false;
    }
    
    out.create(src->height, src->width, CV_8UC3);
    uint8_t* dst_data[4] = { out.data, nullptr, nullptr, nullptr };
    int dst_linesize[4] = { static_cast<int>(out.step), 0, 0, 0 };
    
    sws_scale(sws_ctx_, src->data, src->linesize, 0, src->height, dst_data, dst_linesize);
    return true;
}

int VideoReader::getActualFramePosition() const {
    return actual_frame_position_;
}

void VideoReader::initializeMetadata() {
    double duration_hint = has_clip_metadata_ ? clip_.duration_seconds : 0.0;
    fps_ = computeFps(duration_hint);
    if (fps_ <= 0.0) {
        fps_ = 30.0;
    }
    
    if (has_clip_metadata_ && clip_.has_time_window && std::isfinite(clip_.start_timestamp) &&
        std::isfinite(clip_.moment_time) && fps_ > 0.0) {
        double offset_seconds = clip_.start_timestamp - clip_.moment_time;
        if (offset_seconds > 0.0) {
            int64_t target = static_cast<int64_t>(offset_seconds / av_q2d(video_stream_->time_base));
            av_seek_frame(fmt_ctx_, video_stream_->index, target, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(codec_ctx_);
            total_frames_read_ = static_cast<long long>(std::max(0.0, std::floor(offset_seconds * fps_)));
            actual_frame_position_ = static_cast<int>(total_frames_read_);
        }
    }
}

double VideoReader::computeFps(double reported_duration) const {
    if (!video_stream_) {
        return 30.0;
    }
    AVRational fr = video_stream_->avg_frame_rate;
    if (fr.num > 0 && fr.den > 0) {
        return av_q2d(fr);
    }
    if (reported_duration > 0.0 && video_stream_->nb_frames > 0) {
        double derived = static_cast<double>(video_stream_->nb_frames) / reported_duration;
        if (std::isfinite(derived) && derived > 0.1) {
            return derived;
        }
    }
    return 30.0;
}

void VideoReader::cleanup() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
    }
    if (packet_) {
        av_packet_free(&packet_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
}

