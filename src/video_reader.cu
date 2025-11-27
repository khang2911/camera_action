#include "video_reader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <condition_variable>

#include "logger.h"

#include <cuda_runtime.h>

namespace {
void log_ffmpeg_error(const std::string& prefix, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    LOG_ERROR("VideoReader", prefix + ": " + std::string(buf));
}

__device__ inline uint8_t clampToUInt8(int v) {
    return static_cast<uint8_t>(min(max(v, 0), 255));
}

__global__ void nv12ToBgrKernel(const uint8_t* y_plane,
                                const uint8_t* uv_plane,
                                uint8_t* bgr,
                                int width,
                                int height,
                                int y_pitch,
                                int uv_pitch,
                                size_t bgr_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    
    int Y = static_cast<int>(y_plane[y * y_pitch + x]) - 16;
    int uv_row = y / 2;
    int uv_col = (x / 2) * 2;
    int U = static_cast<int>(uv_plane[uv_row * uv_pitch + uv_col]) - 128;
    int V = static_cast<int>(uv_plane[uv_row * uv_pitch + uv_col + 1]) - 128;
    
    int C = (Y < 0 ? 0 : Y);
    int R = (298 * C + 409 * V + 128) >> 8;
    int G = (298 * C - 100 * U - 208 * V + 128) >> 8;
    int B = (298 * C + 516 * U + 128) >> 8;
    
    uint8_t* dst_row = bgr + y * bgr_pitch;
    dst_row[x * 3 + 2] = clampToUInt8(R);
    dst_row[x * 3 + 1] = clampToUInt8(G);
    dst_row[x * 3 + 0] = clampToUInt8(B);
}
}  // namespace

struct VideoReader::PrefetchQueue {
    enum class PopResult { kPacket, kTimeout, kAborted };
    
    explicit PrefetchQueue(size_t capacity)
        : capacity_(capacity) {}
    
    bool push(AVPacket* pkt) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return aborted_ || queue_.size() < capacity_; });
        if (aborted_) {
            return false;
        }
        queue_.push_back(pkt);
        lock.unlock();
        cv_.notify_all();
        return true;
    }
    
    PopResult pop(AVPacket*& pkt, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto predicate = [&] { return aborted_ || !queue_.empty() || eof_; };
        if (timeout_ms < 0) {
            cv_.wait(lock, predicate);
        } else {
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), predicate);
        }
        
        if (aborted_) {
            return PopResult::kAborted;
        }
        if (queue_.empty()) {
            return PopResult::kTimeout;
        }
        
        pkt = queue_.front();
        queue_.pop_front();
        lock.unlock();
        cv_.notify_all();
        return PopResult::kPacket;
    }
    
    void signalEof() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            eof_ = true;
        }
        cv_.notify_all();
    }
    
    void abort() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            aborted_ = true;
        }
        cv_.notify_all();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            AVPacket* pkt = queue_.front();
            queue_.pop_front();
            av_packet_free(&pkt);
        }
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
private:
    size_t capacity_;
    std::deque<AVPacket*> queue_;
    bool eof_ = false;
    bool aborted_ = false;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

VideoReader::VideoReader(const std::string& video_path, int video_id, const ReaderOptions& options)
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
      gpu_bgr_buffer_(nullptr),
      gpu_bgr_pitch_(0),
      gpu_buffer_width_(0),
      gpu_buffer_height_(0),
      cuda_stream_(nullptr),
      cuda_stream_created_(false),
      video_path_(video_path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      actual_frame_position_(0),
      fps_(0.0),
      has_clip_metadata_(false),
      original_width_(0),
      original_height_(0),
      options_(options) {
    initialize(nullptr);
}

VideoReader::VideoReader(const VideoClip& clip, int video_id, const ReaderOptions& options)
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
      gpu_bgr_buffer_(nullptr),
      gpu_bgr_pitch_(0),
      gpu_buffer_width_(0),
      gpu_buffer_height_(0),
      cuda_stream_(nullptr),
      cuda_stream_created_(false),
      video_path_(clip.path),
      video_id_(video_id),
      frame_number_(0),
      total_frames_read_(0),
      actual_frame_position_(0),
      fps_(0.0),
      has_clip_metadata_(true),
      clip_(clip),
      original_width_(0),
      original_height_(0),
      options_(options) {
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
    LOG_INFO("VideoReader", std::string("Initialized reader for ") + video_path_ +
                 (use_hw_decode_ ? " [NVDEC]" : " [CPU]"));
    if (options_.enable_prefetch) {
        startPrefetchThread();
    }
    return true;
}

bool VideoReader::openInput() {
    AVDictionary* format_opts = nullptr;
    if (options_.ffmpeg_buffer_size > 0) {
        av_dict_set_int(&format_opts, "buffer_size", options_.ffmpeg_buffer_size, 0);
    }
    if (options_.ffmpeg_probe_size > 0) {
        av_dict_set_int(&format_opts, "probesize", options_.ffmpeg_probe_size, 0);
    }
    if (options_.ffmpeg_analyze_duration > 0) {
        av_dict_set_int(&format_opts, "analyzeduration", options_.ffmpeg_analyze_duration, 0);
    }
    if (options_.ffmpeg_read_timeout_ms > 0) {
        av_dict_set(&format_opts, "rw_timeout",
                    std::to_string(static_cast<long long>(options_.ffmpeg_read_timeout_ms) * 1000).c_str(), 0);
    }
    if (options_.ffmpeg_fast_seek) {
        av_dict_set(&format_opts, "fflags", "fastseek", 0);
    }
    if (options_.ffmpeg_fast_io) {
        av_dict_set(&format_opts, "avioflags", "direct", 0);
    }
    
    int ret = avformat_open_input(&fmt_ctx_, video_path_.c_str(), nullptr, &format_opts);
    av_dict_free(&format_opts);
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
    
    // Try to find hardware-accelerated decoder first
    const AVCodec* decoder = nullptr;
    AVCodecID codec_id = video_stream_->codecpar->codec_id;
    
    // Try CUVID decoder first for supported codecs
    if (codec_id == AV_CODEC_ID_H264) {
        decoder = avcodec_find_decoder_by_name("h264_cuvid");
        if (decoder) {
            LOG_INFO("VideoReader", "Found h264_cuvid hardware decoder");
        }
    } else if (codec_id == AV_CODEC_ID_HEVC) {
        decoder = avcodec_find_decoder_by_name("hevc_cuvid");
        if (decoder) {
            LOG_INFO("VideoReader", "Found hevc_cuvid hardware decoder");
        }
    } else if (codec_id == AV_CODEC_ID_VP8) {
        decoder = avcodec_find_decoder_by_name("vp8_cuvid");
        if (decoder) {
            LOG_INFO("VideoReader", "Found vp8_cuvid hardware decoder");
        }
    } else if (codec_id == AV_CODEC_ID_VP9) {
        decoder = avcodec_find_decoder_by_name("vp9_cuvid");
        if (decoder) {
            LOG_INFO("VideoReader", "Found vp9_cuvid hardware decoder");
        }
    }
    
    // Fall back to generic decoder if CUVID not found
    if (!decoder) {
        decoder = avcodec_find_decoder(codec_id);
        if (!decoder) {
            LOG_ERROR("VideoReader", "Failed to find decoder");
            return false;
        }
        LOG_INFO("VideoReader", std::string("Using generic decoder: ") + decoder->name);
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
    codec_ctx_->thread_count = 1;  // Hardware decoding doesn't use CPU threads
    
    // Check if we're using CUVID decoder (which is hardware-accelerated by default)
    std::string decoder_name = decoder->name;
    bool is_cuvid_decoder = (decoder_name.find("cuvid") != std::string::npos);
    
    if (is_cuvid_decoder) {
        // CUVID decoders decode directly to GPU memory (CUDA frames)
        // They use NVDEC directly and should output frames in AV_PIX_FMT_CUDA format
        LOG_INFO("VideoReader", std::string("Using CUVID decoder: ") + decoder_name + 
                 " (NVDEC hardware-accelerated)");
        
        // Set up get_format callback to request CUDA format
        codec_ctx_->opaque = this;
        codec_ctx_->get_format = &VideoReader::getHWFormat;
        
        use_hw_decode_ = true;
        hw_pix_fmt_ = AV_PIX_FMT_CUDA;  // CUVID decoders should output CUDA frames
    } else {
        // For generic decoders, try to set up hardware acceleration
        codec_ctx_->extra_hw_frames = 8;  // Allow more buffering for async decode
        if (initHardwareDecoder(decoder)) {
            codec_ctx_->opaque = this;
            codec_ctx_->get_format = &VideoReader::getHWFormat;
            LOG_INFO("VideoReader", std::string("Hardware decoder initialized successfully, hw_pix_fmt: ") + 
                     std::to_string(hw_pix_fmt_) + ", device_ctx: " + (hw_device_ctx_ ? "OK" : "NULL"));
        } else {
            LOG_INFO("VideoReader", "Hardware decoder not available, using CPU decoding");
        }
    }
    
    // Set decoder options for CUVID decoders to ensure optimal performance
    AVDictionary* opts = nullptr;
    if (is_cuvid_decoder) {
        // CUVID decoder options - these help ensure NVDEC is used efficiently
        av_dict_set(&opts, "surfaces", "32", 0);  // Increase surfaces for better async decode and higher utilization
        av_dict_set(&opts, "deint", "0", 0);     // No deinterlacing needed
        av_dict_set(&opts, "output", "cuda", 0); // Output frames in CUDA format (GPU memory)
        LOG_INFO("VideoReader", "Setting CUVID decoder with 32 surfaces for async decode");
    }
    
    ret = avcodec_open2(codec_ctx_, decoder, &opts);
    if (ret < 0) {
        log_ffmpeg_error("avcodec_open2", ret);
        av_dict_free(&opts);
        return false;
    }
    
    // Log any unused options (for debugging)
    if (opts) {
        AVDictionaryEntry* entry = nullptr;
        while ((entry = av_dict_get(opts, "", entry, AV_DICT_IGNORE_SUFFIX))) {
            LOG_WARNING("VideoReader", std::string("Unused decoder option: ") + entry->key + "=" + entry->value);
        }
    }
    av_dict_free(&opts);
    
    if (use_hw_decode_) {
        LOG_INFO("VideoReader", std::string("Codec opened with hardware decoder, decoder: ") + 
                 std::string(decoder->name) + ", hw_pix_fmt: " + std::to_string(hw_pix_fmt_) +
                 ", codec_id: " + std::to_string(codec_ctx_->codec_id));
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
    static thread_local int call_count = 0;
    if (call_count++ < 5) {
        std::string formats;
        for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
            formats += std::to_string(*p) + " ";
        }
        LOG_INFO("VideoReader", std::string("getHWFormat called, available formats: ") + formats + 
                 ", looking for: " + std::to_string(self->hw_pix_fmt_));
    }
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == self->hw_pix_fmt_) {
            if (call_count <= 5) {
                LOG_INFO("VideoReader", std::string("getHWFormat returning hardware format: ") + std::to_string(*p));
            }
            return *p;
        }
    }
    if (call_count <= 5) {
        LOG_WARNING("VideoReader", std::string("getHWFormat: hardware format not found, returning: ") + std::to_string(pix_fmts[0]));
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
    
    // For hardware decoders, keep the decoder busy by maintaining a queue of packets.
    // The decoder can accept multiple packets before needing to output frames.
    while (true) {
        // Try to receive a frame first
        if (receiveFrame(frame)) {
            // Successfully received a frame - process it
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
                    continue;  // Skip this frame, try next one
                }
                if (current_ts > clip_.end_timestamp) {
                    return false;  // Past end time, done with this clip
                }
            }
            
            return true;  // Frame is ready
        }
        
        // Decoder needs more input. Try to send packets.
        // For hardware decoders, send more packets to keep decoder busy and hide I/O latency
        // This is especially important for network storage where I/O can be slow
        int max_packets = use_hw_decode_
                              ? std::max(1, options_.max_packets_per_loop)
                              : 1;
        bool sent_any = false;
        
        for (int i = 0; i < max_packets && !end_of_stream_; ++i) {
            if (sendNextPacket()) {
                sent_any = true;
            } else {
                // EAGAIN means decoder buffer is full - need to receive frames first
                // Break and try receiving again
                break;
            }
        }
        
        // If we couldn't send any packets and we're at end of stream, we're done
        if (!sent_any && end_of_stream_) {
            return false;
        }
        
        // If we sent packets or got EAGAIN, loop back to try receiving frames
        continue;
    }
}

bool VideoReader::sendNextPacket() {
    if (end_of_stream_) {
        return false;
    }
    
    if (prefetch_enabled_ && packet_queue_) {
        while (true) {
            if (prefetch_stop_) {
                return false;
            }
            
            AVPacket* pkt = nullptr;
            auto result = packet_queue_->pop(pkt, 50);
            if (result == PrefetchQueue::PopResult::kPacket && pkt) {
                int ret = avcodec_send_packet(codec_ctx_, pkt);
                av_packet_free(&pkt);
                if (ret == AVERROR(EAGAIN)) {
                    return false;
                }
                if (ret < 0) {
                    log_ffmpeg_error("avcodec_send_packet", ret);
                    return false;
                }
                return true;
            }
            
            if (result == PrefetchQueue::PopResult::kAborted) {
                return false;
            }
            
            if (prefetch_error_) {
                LOG_ERROR("VideoReader", "Prefetch thread encountered an error, stopping reader");
                return false;
            }
            
            if (prefetch_eof_ && packet_queue_->empty()) {
                end_of_stream_ = true;
                avcodec_send_packet(codec_ctx_, nullptr);
                return true;
            }
            
            // Timeout - queue currently empty but not at EOF, continue waiting
        }
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
            if (ret == AVERROR(EAGAIN)) {
                // Decoder input buffer is full, need to receive frames first
                // Don't log as error, this is expected when keeping decoder busy
                return false;  // Signal caller to receive frames first
            }
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
        
        // Frame format check - verify CUVID decoder is working
        static thread_local int frame_count = 0;
        if (frame_count++ < 3) {
            std::string decoder_info = "format: " + std::to_string(frame_->format);
            if (use_hw_decode_) {
                decoder_info += ", expected: " + std::to_string(hw_pix_fmt_) + 
                               ", match: " + (frame_->format == hw_pix_fmt_ ? "YES" : "NO");
            }
            if (frame_->hw_frames_ctx) {
                decoder_info += ", hw_frames_ctx: YES";
            }
            LOG_INFO("VideoReader", std::string("Frame ") + std::to_string(frame_count) + ": " + decoder_info);
        }
        
        if (convertFrameToMat(frame_, out)) {
            return true;
        }
    }
}

bool VideoReader::convertFrameToMat(AVFrame* src, cv::Mat& out) {
    static thread_local int gpu_count = 0;
    static thread_local int cpu_count = 0;
    static thread_local int last_log_frame = 0;
    
    if (use_hw_decode_ && src->format == hw_pix_fmt_) {
        if (convertFrameToMatGPU(src, out)) {
            gpu_count++;
            // Only log every 1000 frames to reduce overhead
            if (gpu_count % 1000 == 0 && gpu_count > last_log_frame) {
                LOG_INFO("VideoReader", std::string("GPU conversion stats - GPU: ") + std::to_string(gpu_count) + 
                         ", CPU: " + std::to_string(cpu_count) + ", GPU%: " + 
                         std::to_string(100.0 * gpu_count / (gpu_count + cpu_count)));
                last_log_frame = gpu_count;
            }
            return true;
        }
        LOG_ERROR("VideoReader", "GPU conversion failed, falling back to CPU path");
    } else {
        static thread_local int format_mismatch_count = 0;
        if (use_hw_decode_ && format_mismatch_count++ < 5) {
            LOG_WARNING("VideoReader", std::string("Frame NOT in hardware format! hw_decode: ") + 
                     std::to_string(use_hw_decode_) + ", format: " + std::to_string(src->format) + 
                     ", expected: " + std::to_string(hw_pix_fmt_));
        }
    }
    
    cpu_count++;
    if (cpu_count % 1000 == 0 && cpu_count > last_log_frame) {
        LOG_INFO("VideoReader", std::string("CPU conversion stats - GPU: ") + std::to_string(gpu_count) + 
                 ", CPU: " + std::to_string(cpu_count) + ", GPU%: " + 
                 std::to_string(100.0 * gpu_count / (gpu_count + cpu_count)));
        last_log_frame = cpu_count;
    }
    
    AVFrame* cpu_frame = src;
    if (use_hw_decode_ && src->format == hw_pix_fmt_) {
        if (av_hwframe_transfer_data(sw_frame_, src, 0) < 0) {
            LOG_ERROR("VideoReader", "av_hwframe_transfer_data failed");
            return false;
        }
        cpu_frame = sw_frame_;
    }
    return convertFrameToMatCPU(cpu_frame, out);
}

bool VideoReader::ensureCudaBuffer(int width, int height) {
    if (gpu_bgr_buffer_ && width <= gpu_buffer_width_ && height <= gpu_buffer_height_) {
        return true;
    }
    if (gpu_bgr_buffer_) {
        cudaFree(gpu_bgr_buffer_);
        gpu_bgr_buffer_ = nullptr;
        gpu_bgr_pitch_ = 0;
        gpu_buffer_width_ = 0;
        gpu_buffer_height_ = 0;
    }
    if (!cuda_stream_created_) {
        if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
            LOG_ERROR("VideoReader", "Failed to create CUDA stream");
            return false;
        }
        cuda_stream_created_ = true;
    }
    size_t pitch = 0;
    cudaError_t err = cudaMallocPitch(reinterpret_cast<void**>(&gpu_bgr_buffer_),
                                      &pitch,
                                      static_cast<size_t>(width) * 3,
                                      height);
    if (err != cudaSuccess) {
        LOG_ERROR("VideoReader", std::string("cudaMallocPitch failed: ") + cudaGetErrorString(err));
        gpu_bgr_buffer_ = nullptr;
        return false;
    }
    gpu_bgr_pitch_ = pitch;
    gpu_buffer_width_ = width;
    gpu_buffer_height_ = height;
    return true;
}

bool VideoReader::convertFrameToMatGPU(AVFrame* src, cv::Mat& out) {
    if (!ensureCudaBuffer(src->width, src->height)) {
        return false;
    }

    const uint8_t* y_plane = src->data[0];
    const uint8_t* uv_plane = src->data[1];
    int y_pitch = src->linesize[0];
    int uv_pitch = src->linesize[1];

    dim3 block(32, 8);
    dim3 grid((src->width + block.x - 1) / block.x,
              (src->height + block.y - 1) / block.y);

    nv12ToBgrKernel<<<grid, block, 0, cuda_stream_>>>(
        y_plane,
        uv_plane,
        gpu_bgr_buffer_,
        src->width,
        src->height,
        y_pitch,
        uv_pitch,
        gpu_bgr_pitch_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("VideoReader", std::string("nv12ToBgrKernel launch failed: ") + cudaGetErrorString(err));
        return false;
    }

    out.create(src->height, src->width, CV_8UC3);
    err = cudaMemcpy2DAsync(out.data,
                            out.step,
                            gpu_bgr_buffer_,
                            gpu_bgr_pitch_,
                            static_cast<size_t>(src->width) * 3,
                            src->height,
                            cudaMemcpyDeviceToHost,
                            cuda_stream_);
    if (err != cudaSuccess) {
        LOG_ERROR("VideoReader", std::string("cudaMemcpy2DAsync failed: ") + cudaGetErrorString(err));
        return false;
    }
    // Use cudaStreamQuery to check completion without blocking, then sync only if needed
    cudaError_t query_err = cudaStreamQuery(cuda_stream_);
    if (query_err == cudaErrorNotReady) {
        // Stream is still running, must sync
        err = cudaStreamSynchronize(cuda_stream_);
        if (err != cudaSuccess) {
            LOG_ERROR("VideoReader", std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
            return false;
        }
    } else if (query_err != cudaSuccess) {
        LOG_ERROR("VideoReader", std::string("cudaStreamQuery failed: ") + cudaGetErrorString(err));
        return false;
    }

    return true;
}

bool VideoReader::convertFrameToMatCPU(AVFrame* src, cv::Mat& out) {
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
    stopPrefetchThread();
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
    if (gpu_bgr_buffer_) {
        cudaFree(gpu_bgr_buffer_);
        gpu_bgr_buffer_ = nullptr;
        gpu_bgr_pitch_ = 0;
        gpu_buffer_width_ = 0;
        gpu_buffer_height_ = 0;
    }
    if (cuda_stream_created_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
        cuda_stream_created_ = false;
    }
}

void VideoReader::startPrefetchThread() {
    if (!options_.enable_prefetch || !fmt_ctx_) {
        prefetch_enabled_ = false;
        return;
    }
    const size_t depth = static_cast<size_t>(std::max(4, options_.prefetch_queue_depth));
    packet_queue_ = std::make_unique<PrefetchQueue>(depth);
    prefetch_stop_ = false;
    prefetch_eof_ = false;
    prefetch_error_ = false;
    prefetch_enabled_ = true;
    prefetch_thread_ = std::thread(&VideoReader::prefetchLoop, this);
    prefetch_started_ = true;
}

void VideoReader::stopPrefetchThread() {
    if (!prefetch_started_) {
        return;
    }
    prefetch_stop_ = true;
    if (packet_queue_) {
        packet_queue_->abort();
        packet_queue_->clear();
    }
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    packet_queue_.reset();
    prefetch_started_ = false;
}

void VideoReader::prefetchLoop() {
    while (!prefetch_stop_) {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) {
            prefetch_error_ = true;
            break;
        }
        int ret = av_read_frame(fmt_ctx_, pkt);
        if (ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            prefetch_eof_ = true;
            if (packet_queue_) {
                packet_queue_->signalEof();
            }
            break;
        }
        if (ret < 0) {
            log_ffmpeg_error("av_read_frame", ret);
            av_packet_free(&pkt);
            prefetch_error_ = true;
            if (packet_queue_) {
                packet_queue_->abort();
            }
            break;
        }
        
        if (!packet_queue_ || !packet_queue_->push(pkt)) {
            av_packet_free(&pkt);
            break;
        }
    }
    
    if (packet_queue_) {
        packet_queue_->signalEof();
    }
}

