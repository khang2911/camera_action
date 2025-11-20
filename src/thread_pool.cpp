#include "thread_pool.h"
#include "video_reader.h"
#include "logger.h"
#include "yolo_detector.h"
#include "redis_queue.h"
#include "config_parser.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <fstream>

EngineGroup::~EngineGroup() {
    std::lock_guard<std::mutex> lock(buffer_pool_mutex);
    for (auto* buffer : buffer_pool) {
        delete buffer;
    }
    buffer_pool.clear();
}

std::shared_ptr<std::vector<float>> EngineGroup::acquireBuffer() {
    std::vector<float>* buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        if (!buffer_pool.empty()) {
            buffer = buffer_pool.back();
            buffer_pool.pop_back();
        }
    }
    
    if (!buffer) {
        buffer = new std::vector<float>(tensor_elements);
    } else {
        buffer->resize(tensor_elements);
    }
    
    auto deleter = [this](std::vector<float>* ptr) {
        this->releaseBuffer(ptr);
    };
    
    return std::shared_ptr<std::vector<float>>(buffer, deleter);
}

void EngineGroup::releaseBuffer(std::vector<float>* buffer) {
    std::lock_guard<std::mutex> lock(buffer_pool_mutex);
    buffer_pool.push_back(buffer);
}

ThreadPool::ThreadPool(int num_readers,
                       int num_preprocessors,
                       const std::vector<VideoClip>& video_clips,
                       const std::vector<EngineConfig>& engine_configs,
                       const std::string& output_dir,
                       bool debug_mode,
                       int max_frames_per_video)
    : num_readers_(num_readers),
      num_preprocessors_(num_preprocessors > 0 ? num_preprocessors : num_readers),
      video_clips_(video_clips),
      output_dir_(output_dir),
      debug_mode_(debug_mode),
      max_frames_per_video_(max_frames_per_video),
      use_redis_queue_(false),
      stop_flag_(false) {
    
    
    // Log debug mode status
    if (debug_mode_) {
        LOG_INFO("ThreadPool", "Debug mode enabled: max_frames_per_video=" + 
                 std::to_string(max_frames_per_video_));
    }
    
    // Initialize statistics
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        stats_.frames_detected.resize(engine_configs.size(), 0);
        stats_.frames_failed.resize(engine_configs.size(), 0);
        stats_.engine_total_time_ms.resize(engine_configs.size(), 0);
        stats_.engine_frame_count.resize(engine_configs.size(), 0);
    }
    stats_.start_time = std::chrono::steady_clock::now();
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    raw_frame_queue_ = std::make_unique<FrameQueue>(500);
    
    // Initialize video processed flags
    {
        std::lock_guard<std::mutex> lock(video_mutex_);
        video_processed_.assign(video_clips_.size(), false);
    }
    
    // Initialize Redis queue mode flag
    use_redis_queue_ = false;
    
    // Initialize engine groups (one per engine)
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        const auto& config = engine_configs[i];
        auto engine_group = std::make_unique<EngineGroup>(
            static_cast<int>(i), config.path, config.name, config.num_detectors,
            config.input_width, config.input_height, config.roi_cropping
        );
        
        LOG_INFO("ThreadPool", "Initializing engine " + config.name + " with " + 
                 std::to_string(config.num_detectors) + " detector threads");
        
        // Initialize detectors for this engine
        for (int j = 0; j < config.num_detectors; ++j) {
            auto detector = std::make_unique<YOLODetector>(
                config.path, config.type, config.batch_size,
                config.input_width, config.input_height,
                config.conf_threshold, config.nms_threshold, config.gpu_id
            );
            if (!detector->initialize()) {
                LOG_ERROR("ThreadPool", "Failed to initialize detector " + std::to_string(j) + 
                         " for engine " + config.name);
            } else {
                std::string type_str = (config.type == ModelType::POSE) ? "pose" : "detection";
                LOG_DEBUG("ThreadPool", "Detector " + std::to_string(j) + " (" + type_str + 
                         ", batch_size=" + std::to_string(detector->getBatchSize()) +
                         ", input_size=" + std::to_string(detector->getInputWidth()) + "x" + 
                         std::to_string(detector->getInputHeight()) +
                         ", conf_threshold=" + std::to_string(config.conf_threshold) +
                         ", nms_threshold=" + std::to_string(config.nms_threshold) +
                         ", gpu_id=" + std::to_string(detector->getGpuId()) +
                         ") initialized for engine " + config.name);
            }
            engine_group->detectors.push_back(std::move(detector));
        }
        
        engine_groups_.push_back(std::move(engine_group));
    }
    
    LOG_INFO("ThreadPool", "ThreadPool initialized with " + std::to_string(num_readers) + 
             " reader threads, " + std::to_string(num_preprocessors_) + " preprocessor threads, and " +
             std::to_string(engine_configs.size()) + " engines");
}

// Redis queue mode constructor
ThreadPool::ThreadPool(int num_readers,
                       int num_preprocessors,
                       const std::vector<EngineConfig>& engine_configs,
                       const std::string& output_dir,
                       std::shared_ptr<RedisQueue> input_queue,
                       std::shared_ptr<RedisQueue> output_queue,
                       const std::string& input_queue_name,
                       const std::string& output_queue_name,
                       bool debug_mode,
                       int max_frames_per_video)
    : num_readers_(num_readers),
      num_preprocessors_(num_preprocessors > 0 ? num_preprocessors : num_readers),
      output_dir_(output_dir),
      debug_mode_(debug_mode),
      max_frames_per_video_(max_frames_per_video),
      use_redis_queue_(true),
      input_queue_(input_queue),
      output_queue_(output_queue),
      input_queue_name_(input_queue_name),
      output_queue_name_(output_queue_name),
      stop_flag_(false) {
    
    // Log debug mode status
    if (debug_mode_) {
        LOG_INFO("ThreadPool", "Debug mode enabled: max_frames_per_video=" + 
                 std::to_string(max_frames_per_video_));
    }
    
    // Initialize statistics
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        stats_.frames_detected.resize(engine_configs.size(), 0);
        stats_.frames_failed.resize(engine_configs.size(), 0);
        stats_.engine_total_time_ms.resize(engine_configs.size(), 0);
        stats_.engine_frame_count.resize(engine_configs.size(), 0);
    }
    stats_.start_time = std::chrono::steady_clock::now();
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    raw_frame_queue_ = std::make_unique<FrameQueue>(500);
    
    // Initialize engine groups (one per engine)
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        const auto& config = engine_configs[i];
        auto engine_group = std::make_unique<EngineGroup>(
            static_cast<int>(i), config.path, config.name, config.num_detectors,
            config.input_width, config.input_height, config.roi_cropping
        );
        
        LOG_INFO("ThreadPool", "Initializing engine " + config.name + " with " + 
                 std::to_string(config.num_detectors) + " detector threads");
        
        // Initialize detectors for this engine
        for (int j = 0; j < config.num_detectors; ++j) {
            auto detector = std::make_unique<YOLODetector>(
                config.path, config.type, config.batch_size,
                config.input_width, config.input_height,
                config.conf_threshold, config.nms_threshold, config.gpu_id
            );
            if (!detector->initialize()) {
                LOG_ERROR("ThreadPool", "Failed to initialize detector " + std::to_string(j) + 
                         " for engine " + config.name);
            } else {
                std::string type_str = (config.type == ModelType::POSE) ? "pose" : "detection";
                LOG_DEBUG("ThreadPool", "Detector " + std::to_string(j) + " (" + type_str + 
                         ", batch_size=" + std::to_string(detector->getBatchSize()) +
                         ", input_size=" + std::to_string(detector->getInputWidth()) + "x" + 
                         std::to_string(detector->getInputHeight()) +
                         ", conf_threshold=" + std::to_string(config.conf_threshold) +
                         ", nms_threshold=" + std::to_string(config.nms_threshold) +
                         ", gpu_id=" + std::to_string(detector->getGpuId()) +
                         ") initialized for engine " + config.name);
            }
            engine_group->detectors.push_back(std::move(detector));
        }
        
        engine_groups_.push_back(std::move(engine_group));
    }
    
    LOG_INFO("ThreadPool", "ThreadPool initialized (Redis mode) with " + std::to_string(num_readers) + 
             " reader threads, " + std::to_string(num_preprocessors_) + " preprocessor threads, and " +
             std::to_string(engine_configs.size()) + " engines");
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::start() {
    stop_flag_ = false;
    stats_.start_time = std::chrono::steady_clock::now();
    stats_.frames_read = 0;
    stats_.frames_preprocessed = 0;
    stats_.reader_total_time_ms = 0;
    stats_.preprocessor_total_time_ms = 0;
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        for (size_t i = 0; i < stats_.frames_detected.size(); ++i) {
            stats_.frames_detected[i] = 0;
            stats_.frames_failed[i] = 0;
            stats_.engine_total_time_ms[i] = 0;
            stats_.engine_frame_count[i] = 0;
        }
    }
    if (raw_frame_queue_) {
        raw_frame_queue_->clear();
    }
    
    // Reset video processed flags
    {
        std::lock_guard<std::mutex> lock(video_mutex_);
        for (size_t i = 0; i < video_processed_.size(); ++i) {
            video_processed_[i] = false;
        }
    }
    
    LOG_INFO("ThreadPool", "Starting processing...");
    
    // Start reader threads (dynamic number, can process any video)
    for (int i = 0; i < num_readers_; ++i) {
        reader_threads_.emplace_back(&ThreadPool::readerWorker, this, i);
        LOG_DEBUG("ThreadPool", "Started reader thread " + std::to_string(i));
    }
    
    // Start preprocessor threads
    for (int i = 0; i < num_preprocessors_; ++i) {
        preprocessor_threads_.emplace_back(&ThreadPool::preprocessorWorker, this, i);
        LOG_DEBUG("ThreadPool", "Started preprocessor thread " + std::to_string(i));
    }
    
    // Start detector threads for each engine
    for (size_t engine_id = 0; engine_id < engine_groups_.size(); ++engine_id) {
        auto& engine_group = engine_groups_[engine_id];
        for (int detector_id = 0; detector_id < engine_group->num_detectors; ++detector_id) {
            engine_group->detector_threads.emplace_back(
                &ThreadPool::detectorWorker, this, static_cast<int>(engine_id), detector_id
            );
            LOG_DEBUG("ThreadPool", "Started detector thread " + std::to_string(detector_id) + 
                     " for engine " + engine_group->engine_name);
        }
    }
    
    // Start monitoring thread
    monitor_thread_ = std::thread(&ThreadPool::monitorWorker, this);
    LOG_INFO("ThreadPool", "Monitoring thread started");
}

void ThreadPool::stop() {
    stop_flag_ = true;
    LOG_INFO("ThreadPool", "Stopping all threads...");
    
    // Wait for monitoring thread
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    // Wait for all reader threads to finish
    for (auto& thread : reader_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    reader_threads_.clear();
    
    // Wait for all preprocessor threads to finish
    for (auto& thread : preprocessor_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    preprocessor_threads_.clear();
    
    // Wait for all detector threads in all engine groups
    for (auto& engine_group : engine_groups_) {
        for (auto& thread : engine_group->detector_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        engine_group->detector_threads.clear();
    }
    
    LOG_INFO("ThreadPool", "All threads stopped");
}

void ThreadPool::waitForCompletion() {
    if (use_redis_queue_) {
        // Redis queue mode: keep running indefinitely, waiting for new messages
        LOG_INFO("ThreadPool", "Redis queue mode: Process will run continuously, waiting for messages...");
        LOG_INFO("ThreadPool", "Press Ctrl+C to stop the process");
        
        // Wait indefinitely until stop_flag_ is set (by signal handler or external stop)
        while (!stop_flag_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        LOG_INFO("ThreadPool", "Stop signal received. Stopping all threads...");
        stop();
    } else {
        // File-based mode: wait for all videos to be processed
        LOG_INFO("ThreadPool", "Waiting for all processing to complete...");
        
        // Wait for all videos to be processed and all queues to be empty
        bool all_done = false;
        int consecutive_empty_checks = 0;
        const int REQUIRED_EMPTY_CHECKS = 20;  // Wait 2 seconds (20 * 100ms) with empty queues
        
        while (!all_done && !stop_flag_) {
            // Check if all videos have been processed
            bool all_videos_processed = true;
            {
                std::lock_guard<std::mutex> lock(video_mutex_);
                for (size_t i = 0; i < video_processed_.size(); ++i) {
                    if (!video_processed_[i]) {
                        all_videos_processed = false;
                        break;
                    }
                }
            }
            
            // Check if all queues are empty
            bool all_queues_empty = true;
            if (raw_frame_queue_ && !raw_frame_queue_->empty()) {
                all_queues_empty = false;
            }
            for (auto& engine_group : engine_groups_) {
                if (!engine_group->frame_queue->empty()) {
                    all_queues_empty = false;
                    break;
                }
            }
            
            if (all_videos_processed && all_queues_empty) {
                // Both conditions met, but wait a bit more to ensure detectors finish processing
                consecutive_empty_checks++;
                if (consecutive_empty_checks >= REQUIRED_EMPTY_CHECKS) {
                    // Queues have been empty for required time, processing should be done
                    all_done = true;
                    LOG_INFO("ThreadPool", "All videos processed and queues empty for " + 
                             std::to_string(REQUIRED_EMPTY_CHECKS * 100) + "ms. Processing complete.");
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            } else {
                // Reset counter if conditions not met
                consecutive_empty_checks = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                // Log status periodically
                if (consecutive_empty_checks == 0) {
                    int videos_remaining = 0;
                    {
                        std::lock_guard<std::mutex> lock(video_mutex_);
                        for (size_t i = 0; i < video_processed_.size(); ++i) {
                            if (!video_processed_[i]) videos_remaining++;
                        }
                    }
                    size_t total_queue_size = 0;
                    for (auto& engine_group : engine_groups_) {
                        total_queue_size += engine_group->frame_queue->size();
                    }
                    LOG_DEBUG("ThreadPool", "Waiting... Videos remaining: " + std::to_string(videos_remaining) + 
                             ", Total queue size: " + std::to_string(total_queue_size));
                }
            }
        }
        
        LOG_INFO("ThreadPool", "All processing complete. Stopping all threads...");
        
        stop_flag_ = true;
        stop();
    }
}

int ThreadPool::getNextVideo() {
    std::lock_guard<std::mutex> lock(video_mutex_);
    for (size_t i = 0; i < video_processed_.size(); ++i) {
        if (!video_processed_[i]) {
            video_processed_[i] = true;
            return static_cast<int>(i);
        }
    }
    return -1;  // No more videos to process
}

void ThreadPool::readerWorker(int reader_id) {
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " started");
    
    if (use_redis_queue_) {
        // Redis queue mode: read messages from Redis
        readerWorkerRedis(reader_id);
        return;
    }
    
    // File-based mode: process videos from video_clips_
    // Each reader thread processes videos until all are done
    while (!stop_flag_) {
        int video_id = getNextVideo();
        
        if (video_id < 0) {
            // No more videos to process
            LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " finished (no more videos)");
            break;
        }
        
        const auto& clip = video_clips_[video_id];
        processVideo(reader_id, clip, video_id);
    }
}

void ThreadPool::readerWorkerRedis(int reader_id) {
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " started (Redis mode)");
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " waiting for messages from queue: " + input_queue_name_);
    
    int video_id_counter = 0;
    int consecutive_empty_polls = 0;
    
    while (!stop_flag_) {
        std::string message;
        if (!input_queue_->popMessage(message, 1, input_queue_name_)) {  // 1 second timeout
            // No message available, continue waiting
            consecutive_empty_polls++;
            
            // Log waiting status every 30 seconds to show process is alive
            if (consecutive_empty_polls % 30 == 0) {
                LOG_INFO("Reader", "[WAITING] Reader " + std::to_string(reader_id) + 
                         " waiting for messages from queue: " + input_queue_name_ + 
                         " (alive and ready)");
            }
            continue;
        }
        
        // Reset counter when we get a message
        consecutive_empty_polls = 0;
        
        try {
            // Parse JSON message (one line from video_list.jsonl format)
            // Each message is a single JSON object, same format as video_list.jsonl
            VideoClip clip = parseJsonToVideoClip(message);
            
            if (clip.path.empty()) {
                LOG_WARNING("Reader", "Reader " + std::to_string(reader_id) + 
                         " received message with empty video path, skipping");
                continue;
            }
            
            LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
                     " processing video from Redis: " + clip.path);
            
            // Process the video
            processVideo(reader_id, clip, video_id_counter, message);
            
            video_id_counter++;
        } catch (const std::exception& e) {
            LOG_ERROR("Reader", "Failed to parse Redis message: " + std::string(e.what()));
            if (!message.empty()) {
                size_t preview_len = std::min(200UL, message.length());
                LOG_ERROR("Reader", "Message content (first " + std::to_string(preview_len) + " chars): " + 
                         message.substr(0, preview_len));
            }
            continue;
        }
    }
    
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " finished (Redis mode)");
}

VideoClip ThreadPool::parseJsonToVideoClip(const std::string& json_str) {
    VideoClip clip;
    
    // Simple JSON parsing using string operations
    // This is a basic parser - for production, consider using a proper JSON library
    
    // Parse playback_location
    size_t playback_pos = json_str.find("\"playback_location\"");
    if (playback_pos != std::string::npos) {
        size_t colon_pos = json_str.find(":", playback_pos);
        if (colon_pos != std::string::npos) {
            size_t start = json_str.find("\"", colon_pos) + 1;
            if (start != std::string::npos && start > colon_pos) {
                size_t end = json_str.find("\"", start);
                if (end != std::string::npos) {
                    clip.path = json_str.substr(start, end - start);
                }
            }
        }
    }
    
    // Parse raw_alarm.video_start_time and video_end_time
    size_t start_time_pos = json_str.find("\"video_start_time\"");
    if (start_time_pos != std::string::npos) {
        size_t colon_pos = json_str.find(":", start_time_pos);
        if (colon_pos != std::string::npos) {
            size_t start = json_str.find("\"", colon_pos) + 1;
            if (start != std::string::npos && start > colon_pos) {
                size_t end = json_str.find("\"", start);
                if (end != std::string::npos) {
                    std::string start_time_str = json_str.substr(start, end - start);
                    clip.start_timestamp = ConfigParser::parseTimestamp(start_time_str);
                }
            }
        }
    }
    
    size_t end_time_pos = json_str.find("\"video_end_time\"");
    if (end_time_pos != std::string::npos) {
        size_t colon_pos = json_str.find(":", end_time_pos);
        if (colon_pos != std::string::npos) {
            size_t start = json_str.find("\"", colon_pos) + 1;
            if (start != std::string::npos && start > colon_pos) {
                size_t end = json_str.find("\"", start);
                if (end != std::string::npos) {
                    std::string end_time_str = json_str.substr(start, end - start);
                    clip.end_timestamp = ConfigParser::parseTimestamp(end_time_str);
                }
            }
        }
    }
    
    // Parse record_list[0].moment_time and duration
    size_t moment_time_pos = json_str.find("\"moment_time\"");
    if (moment_time_pos != std::string::npos) {
        size_t colon_pos = json_str.find(":", moment_time_pos);
        if (colon_pos != std::string::npos) {
            size_t start = json_str.find("\"", colon_pos) + 1;
            if (start != std::string::npos && start > colon_pos) {
                size_t end = json_str.find("\"", start);
                if (end != std::string::npos) {
                    std::string moment_time_str = json_str.substr(start, end - start);
                    clip.moment_time = ConfigParser::parseTimestamp(moment_time_str);
                }
            }
        }
    }
    
    size_t duration_pos = json_str.find("\"duration\"");
    if (duration_pos != std::string::npos) {
        size_t colon_pos = json_str.find(":", duration_pos);
        if (colon_pos != std::string::npos) {
            std::string duration_str = json_str.substr(colon_pos + 1);
            // Remove whitespace and find number
            size_t num_start = duration_str.find_first_of("0123456789.-");
            if (num_start != std::string::npos) {
                size_t num_end = duration_str.find_first_not_of("0123456789.-", num_start);
                if (num_end == std::string::npos) num_end = duration_str.length();
                try {
                    clip.duration_seconds = std::stod(duration_str.substr(num_start, num_end - num_start));
                } catch (...) {
                    clip.duration_seconds = 0.0;
                }
            }
        }
    }
    
    if (std::isfinite(clip.start_timestamp) && std::isfinite(clip.end_timestamp) && 
        std::isfinite(clip.moment_time)) {
        clip.has_time_window = true;
    }
    
    // Parse config.box (simplified - assumes format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    size_t box_pos = json_str.find("\"box\"");
    if (box_pos != std::string::npos) {
        size_t bracket_start = json_str.find("[[", box_pos);
        if (bracket_start != std::string::npos) {
            clip.has_roi = true;
            // Simple parsing - extract first and third coordinates
            // This is a simplified parser - for production use proper JSON library
            std::istringstream iss(json_str.substr(bracket_start));
            char c;
            float x1, y1, x2, y2;
            if (iss >> c && c == '[' && 
                iss >> c && c == '[' &&
                iss >> x1 >> c && c == ',' &&
                iss >> y1 >> c && c == ']') {
                clip.roi_x1 = x1;
                clip.roi_y1 = y1;
                // Find third coordinate [x2, y2]
                size_t third_bracket = json_str.find("[[", bracket_start + 1);
                if (third_bracket != std::string::npos) {
                    std::istringstream iss2(json_str.substr(third_bracket));
                    if (iss2 >> c && c == '[' &&
                        iss2 >> x2 >> c && c == ',' &&
                        iss2 >> y2 >> c && c == ']') {
                        clip.roi_x2 = x2;
                        clip.roi_y2 = y2;
                    }
                }
            }
        }
    }
    
    return clip;
}

void ThreadPool::processVideo(int reader_id, const VideoClip& clip, int video_id, const std::string& redis_message) {
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
             " processing video " + std::to_string(video_id) + ": " + clip.path);
    
    VideoReader reader(clip, video_id);
    
    if (!reader.isOpened()) {
        LOG_ERROR("Reader", "Cannot open video " + std::to_string(video_id) + ": " + clip.path);
        // Push message to output queue even if failed
        if (use_redis_queue_ && output_queue_ && !redis_message.empty()) {
            output_queue_->pushMessage(output_queue_name_, redis_message);
        }
        return;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    while (!stop_flag_ && reader.readFrame(frame)) {
            // Check debug mode limit first (takes priority over global limit)
            // frame_count is the number of frames already processed, so check BEFORE processing this one
            if (debug_mode_ && max_frames_per_video_ > 0 && frame_count >= max_frames_per_video_) {
                LOG_INFO("Reader", "Reader " + std::to_string(reader_id) + 
                         " stopping after processing " + std::to_string(frame_count) + " frames" +
                         " (max_frames_per_video=" + std::to_string(max_frames_per_video_) + ")");
                break;
            }
            
            
            auto frame_start = std::chrono::steady_clock::now();
            
            stats_.frames_read++;
            
            // Create frame data with original frame (shared to preprocessor queue)
            // Note: Frame is NOT cropped here - ROI cropping is applied per-engine in preprocessor
            FrameData frame_data(frame.clone(), video_id, reader.getFrameNumber(), clip.path);
            // Store original frame dimensions (before any ROI cropping)
            frame_data.original_width = frame.cols;
            frame_data.original_height = frame.rows;
            frame_data.true_original_width = frame.cols;  // True original dimensions (same as original when no cropping)
            frame_data.true_original_height = frame.rows;
            // Store ROI offset for scaling detections back to true original frame
            if (clip.has_roi) {
                frame_data.roi_offset_x = clip.roi_offset_x;
                frame_data.roi_offset_y = clip.roi_offset_y;
            } else {
                frame_data.roi_offset_x = 0;
                frame_data.roi_offset_y = 0;
            }
            if (raw_frame_queue_) {
                raw_frame_queue_->push(frame_data);
            }
            
            auto frame_end = std::chrono::steady_clock::now();
            auto frame_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
            stats_.reader_total_time_ms += frame_time_ms;
            
            frame_count++;
            
            // Print progress every 20 frames to show continuous operation
            if (frame_count % 20 == 0) {
                LOG_INFO("Reader", "[RUNNING] Reader " + std::to_string(reader_id) + 
                         " processing video " + std::to_string(video_id) + 
                         " - Frame " + std::to_string(frame_count) + " read and queued");
            }
            
            // Also log every 100 frames with more detail
            if (frame_count % 100 == 0) {
                LOG_DEBUG("Reader", "Reader " + std::to_string(reader_id) + 
                         " processed " + std::to_string(frame_count) + " frames from video " + 
                         std::to_string(video_id));
            }
        }
        
        LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
                 " finished video " + std::to_string(video_id) + " (" + 
                 std::to_string(frame_count) + " frames)");
    
    // Push message to output queue after processing (Redis mode)
    if (use_redis_queue_ && output_queue_ && !redis_message.empty()) {
        if (output_queue_->pushMessage(output_queue_name_, redis_message)) {
            LOG_INFO("Reader", "Pushed processed message to output queue: " + output_queue_name_);
        } else {
            LOG_ERROR("Reader", "Failed to push message to output queue: " + output_queue_name_);
        }
    }
}

void ThreadPool::preprocessorWorker(int worker_id) {
    LOG_INFO("Preprocessor", "Preprocessor thread " + std::to_string(worker_id) + " started");
    
    while (!stop_flag_) {
        FrameData raw_frame;
        if (!raw_frame_queue_ || !raw_frame_queue_->pop(raw_frame, 100)) {
            if (stop_flag_) break;
            continue;
        }
        
        auto preprocess_start = std::chrono::steady_clock::now();
        
        for (auto& engine_group : engine_groups_) {
            cv::Mat frame_to_process = raw_frame.frame;
            // Store TRUE original frame dimensions (before any cropping)
            int true_original_width = raw_frame.true_original_width > 0 ? raw_frame.true_original_width : raw_frame.original_width;
            int true_original_height = raw_frame.true_original_height > 0 ? raw_frame.true_original_height : raw_frame.original_height;
            int cropped_width = raw_frame.original_width;
            int cropped_height = raw_frame.original_height;
            int roi_offset_x = raw_frame.roi_offset_x;
            int roi_offset_y = raw_frame.roi_offset_y;
            
            // Apply ROI cropping if enabled for this engine and ROI is available
            if (engine_group->roi_cropping) {
                // Find the video clip to get ROI coordinates
                // We need to find which video this frame belongs to
                int video_id = raw_frame.video_id;
                if (video_id >= 0 && video_id < static_cast<int>(video_clips_.size())) {
                    const VideoClip& clip = video_clips_[video_id];
                    if (clip.has_roi && !frame_to_process.empty()) {
                        // Use true original dimensions for ROI coordinate calculation
                        int orig_w = true_original_width;
                        int orig_h = true_original_height;
                        
                        // Convert normalized ROI coordinates to pixel coordinates in the true original frame
                        int x1 = static_cast<int>(clip.roi_x1 * orig_w);
                        int y1 = static_cast<int>(clip.roi_y1 * orig_h);
                        int x2 = static_cast<int>(clip.roi_x2 * orig_w);
                        int y2 = static_cast<int>(clip.roi_y2 * orig_h);
                        
                        // Clamp to frame bounds
                        x1 = std::max(0, std::min(x1, orig_w - 1));
                        y1 = std::max(0, std::min(y1, orig_h - 1));
                        x2 = std::max(x1 + 1, std::min(x2, orig_w));
                        y2 = std::max(y1 + 1, std::min(y2, orig_h));
                        
                        // Crop the frame for this engine
                        cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                        frame_to_process = frame_to_process(roi_rect).clone();
                        cropped_width = frame_to_process.cols;
                        cropped_height = frame_to_process.rows;
                        // Update ROI offset to the calculated crop coordinates (x1, y1) in the original frame
                        roi_offset_x = x1;
                        roi_offset_y = y1;
                        
                        // Debug: Log ROI cropping (first frame only)
                        static bool logged_roi_crop = false;
                        if (!logged_roi_crop && worker_id == 0) {
                            LOG_INFO("Preprocessor", "ROI cropping applied for engine " + engine_group->engine_name +
                                     ": original=" + std::to_string(orig_w) + "x" + std::to_string(orig_h) +
                                     ", cropped=" + std::to_string(cropped_width) + "x" + std::to_string(cropped_height) +
                                     ", ROI=[" + std::to_string(x1) + "," + std::to_string(y1) + "," + 
                                     std::to_string(x2) + "," + std::to_string(y2) + "]");
                            logged_roi_crop = true;
                        }
                    } else {
                        // Debug: Log when ROI cropping is enabled but ROI not available
                        static bool logged_roi_missing = false;
                        if (!logged_roi_missing && worker_id == 0) {
                            if (!clip.has_roi) {
                                LOG_WARNING("Preprocessor", "ROI cropping enabled for engine " + engine_group->engine_name +
                                         " but video " + std::to_string(video_id) + " has no ROI defined");
                            }
                            logged_roi_missing = true;
                        }
                    }
                }
            }
            
            auto buffer = engine_group->acquireBuffer();
            engine_group->preprocessor->preprocessToFloat(frame_to_process, *buffer);
            
            FrameData processed;
            processed.preprocessed_data = buffer;
            processed.video_id = raw_frame.video_id;
            processed.frame_number = raw_frame.frame_number;
            processed.video_path = raw_frame.video_path;
            processed.frame = frame_to_process;  // Keep reference for potential debugging/logging
            processed.original_width = cropped_width;  // Cropped frame width (for scale calculation)
            processed.original_height = cropped_height;  // Cropped frame height (for scale calculation)
            processed.true_original_width = true_original_width;  // True original frame width (for clamping after ROI offset)
            processed.true_original_height = true_original_height;  // True original frame height (for clamping after ROI offset)
            processed.roi_offset_x = roi_offset_x;  // ROI offset X in true original frame
            processed.roi_offset_y = roi_offset_y;  // ROI offset Y in true original frame
            
            engine_group->frame_queue->push(processed);
        }
        
        auto preprocess_end = std::chrono::steady_clock::now();
        auto preprocess_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            preprocess_end - preprocess_start).count();
        
        stats_.frames_preprocessed++;
        stats_.preprocessor_total_time_ms += preprocess_time_ms;
    }
    
    LOG_INFO("Preprocessor", "Preprocessor thread " + std::to_string(worker_id) + " finished");
}

void ThreadPool::detectorWorker(int engine_id, int detector_id) {
    if (engine_id >= static_cast<int>(engine_groups_.size())) {
        return;
    }
    
    auto& engine_group = engine_groups_[engine_id];
    if (detector_id >= static_cast<int>(engine_group->detectors.size()) || 
        !engine_group->detectors[detector_id]) {
        return;
    }
    
    LOG_INFO("Detector", "Detector thread " + std::to_string(detector_id) + 
             " for engine " + engine_group->engine_name + " started");
    
    FrameData frame_data;
    int processed_count = 0;
    auto last_wait_log = std::chrono::steady_clock::now();
    
    // Get batch size for this detector
    int batch_size = engine_group->detectors[detector_id]->getBatchSize();
    
    while (!stop_flag_) {
            // For batch_size > 1, accumulate frames into a batch
        if (batch_size > 1) {
            std::vector<std::shared_ptr<std::vector<float>>> batch_tensors;
            std::vector<std::string> batch_output_paths;
            std::vector<int> batch_frame_numbers;
            std::vector<int> batch_original_widths;
            std::vector<int> batch_original_heights;
            std::vector<int> batch_true_original_widths;  // True original frame widths (for clamping after ROI offset)
            std::vector<int> batch_true_original_heights;  // True original frame heights (for clamping after ROI offset)
            std::vector<int> batch_roi_offset_x;  // ROI offset X for each frame
            std::vector<int> batch_roi_offset_y;  // ROI offset Y for each frame
            std::vector<cv::Mat> batch_frames;  // Store frames for debug mode
            std::vector<int> batch_video_ids;   // Store video IDs for debug mode
            
            // Collect batch_size frames
            while (static_cast<int>(batch_tensors.size()) < batch_size && !stop_flag_) {
                if (!engine_group->frame_queue->pop(frame_data, 100)) {
                    if (stop_flag_) break;
                    
                    // Log waiting status periodically (every 5 seconds)
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_wait_log).count();
                    if (elapsed >= 5) {
                        LOG_INFO("Detector", "[WAITING] Detector " + std::to_string(detector_id) + 
                                 " (" + engine_group->engine_name + ") waiting for frames... " +
                                 "(Queue size: " + std::to_string(engine_group->frame_queue->size()) + ")");
                        last_wait_log = now;
                    }
                    continue;
                }
                
                // Generate output path with engine name
                std::string output_path = generateOutputPath(
                    frame_data.video_id, 
                    frame_data.frame_number, 
                    engine_id,
                    detector_id,
                    engine_group->engine_name
                );
                
                std::shared_ptr<std::vector<float>> tensor = frame_data.preprocessed_data;
                if (!tensor) {
                    tensor = engine_group->acquireBuffer();
                    engine_group->preprocessor->preprocessToFloat(frame_data.frame, *tensor);
                }
                
                // CRITICAL FIX: Make a deep copy of the preprocessed data to prevent buffer reuse corruption
                // The shared_ptr might point to a buffer that gets reused by the preprocessor worker
                // before the batch is processed. By making a copy, we ensure the data is stable.
                auto tensor_copy = std::make_shared<std::vector<float>>(*tensor);
                batch_tensors.push_back(tensor_copy);
                batch_output_paths.push_back(output_path);
                batch_frame_numbers.push_back(frame_data.frame_number);
                batch_original_widths.push_back(frame_data.original_width);
                batch_original_heights.push_back(frame_data.original_height);
                batch_true_original_widths.push_back(frame_data.true_original_width);
                batch_true_original_heights.push_back(frame_data.true_original_height);
                batch_roi_offset_x.push_back(frame_data.roi_offset_x);
                batch_roi_offset_y.push_back(frame_data.roi_offset_y);
                
                // Store frame and video_id for debug mode
                if (debug_mode_) {
                    batch_frames.push_back(frame_data.frame.clone());
                    batch_video_ids.push_back(frame_data.video_id);
                }
            }
            
            // Process batch if we have enough frames
            if (static_cast<int>(batch_tensors.size()) == batch_size) {
                auto batch_start = std::chrono::steady_clock::now();
                bool success = false;
                std::vector<std::vector<Detection>> batch_detections;
                
                if (debug_mode_) {
                    // In debug mode, get detections to draw on images
                    success = engine_group->detectors[detector_id]->runInferenceWithDetections(
                        batch_tensors,
                        batch_output_paths, batch_frame_numbers,
                        batch_original_widths, batch_original_heights,
                        batch_detections,
                        batch_roi_offset_x, batch_roi_offset_y,
                        batch_true_original_widths, batch_true_original_heights
                    );
                } else {
                    success = engine_group->detectors[detector_id]->runWithPreprocessedBatch(
                        batch_tensors, batch_output_paths, batch_frame_numbers,
                        batch_original_widths, batch_original_heights,
                        batch_roi_offset_x, batch_roi_offset_y,
                        batch_true_original_widths, batch_true_original_heights
                    );
                }
                
                auto batch_end = std::chrono::steady_clock::now();
                auto batch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
                
                // Save debug images for batch
                if (debug_mode_ && success && batch_detections.size() == static_cast<size_t>(batch_size) &&
                    batch_frames.size() == static_cast<size_t>(batch_size)) {
                    for (int b = 0; b < batch_size; ++b) {
                        // Create preprocessed frame (resized with padding) for debug visualization
                        cv::Mat debug_frame = engine_group->preprocessor->addPadding(batch_frames[b]);
                        engine_group->detectors[detector_id]->drawDetections(debug_frame, batch_detections[b]);
                        
                        // Create debug output directory: output_dir/debug_images/video_id/engine_name/
                        std::filesystem::path debug_dir = std::filesystem::path(output_dir_) / "debug_images" / 
                                                           ("video_" + std::to_string(batch_video_ids[b])) /
                                                           engine_group->engine_name;
                        std::filesystem::create_directories(debug_dir);
                        
                        // Save image: frame_XXXXX_engine.jpg
                        std::string image_filename = "frame_" + 
                            std::to_string(batch_frame_numbers[b]) + "_" + 
                            engine_group->engine_name + ".jpg";
                        std::filesystem::path image_path = debug_dir / image_filename;
                        
                        if (cv::imwrite(image_path.string(), debug_frame)) {
                            LOG_DEBUG("Detector", "Saved debug image: " + image_path.string());
                        } else {
                            LOG_ERROR("Detector", "Failed to save debug image: " + image_path.string());
                        }
                    }
                }
                
                if (success) {
                    {
                        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                        stats_.frames_detected[engine_id] += batch_size;
                        stats_.engine_total_time_ms[engine_id] += batch_time_ms;
                        stats_.engine_frame_count[engine_id] += batch_size;
                    }
                    processed_count += batch_size;
                    
                    // Print progress every 10 batches to show continuous operation
                    if (processed_count % (10 * batch_size) == 0) {
                        LOG_INFO("Detector", "[RUNNING] Detector " + std::to_string(detector_id) + 
                                 " (" + engine_group->engine_name + ") processed " + 
                                 std::to_string(processed_count) + " frames (batch_size=" + 
                                 std::to_string(batch_size) + ")");
                    }
                } else {
                    {
                        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                        stats_.frames_failed[engine_id] += batch_size;
                    }
                    LOG_ERROR("Detector", "Batch detection failed for " + std::to_string(batch_size) + " frames");
                }
            }
        } else {
            // batch_size = 1: process frames one at a time (original behavior)
            if (!engine_group->frame_queue->pop(frame_data, 100)) {
                if (stop_flag_) break;
                
                // Log waiting status periodically (every 5 seconds)
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_wait_log).count();
                if (elapsed >= 5) {
                    LOG_INFO("Detector", "[WAITING] Detector " + std::to_string(detector_id) + 
                             " (" + engine_group->engine_name + ") waiting for frames... " +
                             "(Queue size: " + std::to_string(engine_group->frame_queue->size()) + ")");
                    last_wait_log = now;
                }
                continue;
            }
            
            // Generate output path with engine name
            std::string output_path = generateOutputPath(
                frame_data.video_id, 
                frame_data.frame_number, 
                engine_id,
                detector_id,
                engine_group->engine_name
            );
            
            // Run YOLO detection on GPU (TensorRT) - frame will be preprocessed by detector
            // Note: Multiple detector threads may write to the same file (same video+engine)
            // File writes are protected by per-file mutex inside writeDetectionsToFile
            auto detect_start = std::chrono::steady_clock::now();
            std::shared_ptr<std::vector<float>> tensor = frame_data.preprocessed_data;
            if (!tensor) {
                tensor = engine_group->acquireBuffer();
                engine_group->preprocessor->preprocessToFloat(frame_data.frame, *tensor);
            }
            
            bool success = false;
            std::vector<Detection> detections;
            
            if (debug_mode_) {
                // In debug mode, get detections to draw on image
                std::vector<std::vector<Detection>> all_detections;
                std::vector<std::shared_ptr<std::vector<float>>> single_input = {tensor};
                success = engine_group->detectors[detector_id]->runInferenceWithDetections(
                    single_input,
                    {output_path}, {frame_data.frame_number},
                    {frame_data.original_width}, {frame_data.original_height},
                    all_detections,
                    {frame_data.roi_offset_x}, {frame_data.roi_offset_y},
                    {frame_data.true_original_width}, {frame_data.true_original_height}
                );
                if (success && !all_detections.empty()) {
                    detections = all_detections[0];
                }
            } else {
                success = engine_group->detectors[detector_id]->runWithPreprocessedData(
                    tensor, output_path, frame_data.frame_number,
                    frame_data.original_width, frame_data.original_height,
                    frame_data.roi_offset_x, frame_data.roi_offset_y,
                    frame_data.true_original_width, frame_data.true_original_height
                );
            }
            
            auto detect_end = std::chrono::steady_clock::now();
            auto detect_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();
            
            // Save debug image if enabled
            if (debug_mode_ && success && !frame_data.frame.empty()) {
                // Create preprocessed frame (resized with padding) for debug visualization
                cv::Mat debug_frame = engine_group->preprocessor->addPadding(frame_data.frame);
                engine_group->detectors[detector_id]->drawDetections(debug_frame, detections);
                
                // Create debug output directory: output_dir/debug_images/video_id/engine_name/
                std::filesystem::path debug_dir = std::filesystem::path(output_dir_) / "debug_images" / 
                                                   ("video_" + std::to_string(frame_data.video_id)) /
                                                   engine_group->engine_name;
                std::filesystem::create_directories(debug_dir);
                
                // Save image: frame_XXXXX_engine.png
                std::string image_filename = "frame_" + 
                    std::to_string(frame_data.frame_number) + "_" + 
                    engine_group->engine_name + ".jpg";
                std::filesystem::path image_path = debug_dir / image_filename;
                
                if (cv::imwrite(image_path.string(), debug_frame)) {
                    LOG_DEBUG("Detector", "Saved debug image: " + image_path.string());
                } else {
                    LOG_ERROR("Detector", "Failed to save debug image: " + image_path.string());
                }
            }
            
            if (success) {
                {
                    std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                    stats_.frames_detected[engine_id]++;
                    stats_.engine_total_time_ms[engine_id] += detect_time_ms;
                    stats_.engine_frame_count[engine_id]++;
                }
                processed_count++;
                
                // Print progress every 10 frames to show continuous operation
                if (processed_count % 10 == 0) {
                    LOG_INFO("Detector", "[RUNNING] Detector " + std::to_string(detector_id) + 
                             " (" + engine_group->engine_name + ") processed " + 
                             std::to_string(processed_count) + " frames - " +
                             "Video: " + std::to_string(frame_data.video_id) + 
                             ", Frame: " + std::to_string(frame_data.frame_number));
                }
                
                LOG_DEBUG("Detector", "Detection completed: " + output_path + 
                         " (Video: " + std::to_string(frame_data.video_id) + 
                         ", Frame: " + std::to_string(frame_data.frame_number) + 
                         ", Engine: " + engine_group->engine_name +
                         ", Detector: " + std::to_string(detector_id) + ")");
            } else {
                {
                    std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                    stats_.frames_failed[engine_id]++;
                }
                LOG_ERROR("Detector", "Detection failed: " + output_path);
            }
            
            // Also log every 50 frames with summary
            if (processed_count % 50 == 0) {
                LOG_DEBUG("Detector", "Detector " + std::to_string(detector_id) + 
                         " (" + engine_group->engine_name + ") processed " + 
                         std::to_string(processed_count) + " frames");
            }
        }
    }
    
    LOG_INFO("Detector", "Detector thread " + std::to_string(detector_id) + 
             " for engine " + engine_group->engine_name + " finished (" + 
             std::to_string(processed_count) + " frames processed)");
}

void ThreadPool::monitorWorker() {
    LOG_INFO("Monitor", "Monitoring thread started");
    
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));  // Print stats every 5 seconds
        
        if (stop_flag_) break;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - stats_.start_time).count();
        
        if (elapsed == 0) continue;
        
        long long total_read = stats_.frames_read.load();
        long long total_preprocessed = stats_.frames_preprocessed.load();
        
        std::ostringstream stats_oss;
        stats_oss << "=== Statistics (Runtime: " << elapsed << "s) ===" << std::endl;
        stats_oss << "Frames Read: " << total_read 
                  << " | Preprocessed: " << total_preprocessed;
        if (elapsed > 0) {
            stats_oss << " | Read FPS: " << (total_read / elapsed)
                      << " | Preprocess FPS: " << (total_preprocessed / elapsed);
        }
        stats_oss << std::endl;
        
        // Queue sizes
        stats_oss << "Queue Sizes: ";
        for (size_t i = 0; i < engine_groups_.size(); ++i) {
            stats_oss << engine_groups_[i]->engine_name << "=" 
                      << engine_groups_[i]->frame_queue->size();
            if (i < engine_groups_.size() - 1) stats_oss << ", ";
        }
        stats_oss << std::endl;
        
        // Per-engine statistics
        {
            std::lock_guard<std::mutex> lock(stats_.stats_mutex);
            for (size_t i = 0; i < engine_groups_.size(); ++i) {
                long long detected = stats_.frames_detected[i];
                long long failed = stats_.frames_failed[i];
                long long total_time_ms = stats_.engine_total_time_ms[i];
                long long frame_count = stats_.engine_frame_count[i];
                
                stats_oss << "Engine " << engine_groups_[i]->engine_name 
                          << ": Detected=" << detected 
                          << " | Failed=" << failed;
                if (elapsed > 0) {
                    stats_oss << " | FPS=" << (detected / elapsed);
                }
                if (frame_count > 0) {
                    double avg_time_ms = static_cast<double>(total_time_ms) / frame_count;
                    stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_time_ms << "ms/frame";
                    stats_oss << " | TotalTime=" << (total_time_ms / 1000.0) << "s";
                }
                stats_oss << std::endl;
            }
        }
        
        // Reader statistics
        long long reader_time_ms = stats_.reader_total_time_ms.load();
        long long reader_frames = stats_.frames_read.load();
        stats_oss << "Reader: Frames=" << reader_frames;
        if (elapsed > 0) {
            stats_oss << " | FPS=" << (reader_frames / elapsed);
        }
        if (reader_frames > 0) {
            double avg_reader_time_ms = static_cast<double>(reader_time_ms) / reader_frames;
            stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_reader_time_ms << "ms/frame";
            stats_oss << " | TotalTime=" << (reader_time_ms / 1000.0) << "s";
        }
        stats_oss << std::endl;
        
        long long preproc_time_ms = stats_.preprocessor_total_time_ms.load();
        long long preproc_frames = stats_.frames_preprocessed.load();
        stats_oss << "Preprocessor: Frames=" << preproc_frames;
        if (elapsed > 0) {
            stats_oss << " | FPS=" << (preproc_frames / elapsed);
        }
        if (preproc_frames > 0) {
            double avg_preproc_time_ms = static_cast<double>(preproc_time_ms) / preproc_frames;
            stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_preproc_time_ms << "ms/frame";
            stats_oss << " | TotalTime=" << (preproc_time_ms / 1000.0) << "s";
        }
        stats_oss << std::endl;
        
        LOG_STATS("Monitor", stats_oss.str());
    }
    
    LOG_INFO("Monitor", "Monitoring thread finished");
}

void ThreadPool::getStatisticsSnapshot(long long& frames_read, long long& frames_preprocessed,
                                      std::vector<long long>& frames_detected,
                                      std::vector<long long>& frames_failed,
                                      long long& reader_total_time_ms,
                                      long long& preprocessor_total_time_ms,
                                      std::vector<long long>& engine_total_time_ms,
                                      std::vector<long long>& engine_frame_count,
                                      std::chrono::steady_clock::time_point& start_time) const {
    frames_read = stats_.frames_read.load();
    frames_preprocessed = stats_.frames_preprocessed.load();
    reader_total_time_ms = stats_.reader_total_time_ms.load();
    preprocessor_total_time_ms = stats_.preprocessor_total_time_ms.load();
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        frames_detected.resize(stats_.frames_detected.size());
        frames_failed.resize(stats_.frames_failed.size());
        engine_total_time_ms.resize(stats_.engine_total_time_ms.size());
        engine_frame_count.resize(stats_.engine_frame_count.size());
        for (size_t i = 0; i < stats_.frames_detected.size(); ++i) {
            frames_detected[i] = stats_.frames_detected[i];
            frames_failed[i] = stats_.frames_failed[i];
            engine_total_time_ms[i] = stats_.engine_total_time_ms[i];
            engine_frame_count[i] = stats_.engine_frame_count[i];
        }
    }
    start_time = stats_.start_time;
}

std::string ThreadPool::generateOutputPath(int video_id, int frame_number, 
                                            int engine_id, int detector_id, 
                                            const std::string& engine_name) {
    // Generate path for one file per video per engine (frame_number and detector_id not used in path)
    std::ostringstream oss;
    oss << output_dir_ << "/video_" << std::setfill('0') << std::setw(4) << video_id
        << "_" << engine_name << ".bin";
    return oss.str();
}

