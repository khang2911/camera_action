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
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <future>
#include <fstream>
#include <yaml-cpp/yaml.h>

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
    postprocess_queue_ = std::make_unique<PostProcessQueue>();
    
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
    postprocess_queue_ = std::make_unique<PostProcessQueue>();
    
    max_active_redis_readers_ = num_readers_;
    
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
    
    // Start post-processing threads (2 threads per engine for parallel post-processing)
    int num_postprocess_threads = std::max(2, static_cast<int>(engine_groups_.size()) * 2);
    for (int i = 0; i < num_postprocess_threads; ++i) {
        postprocess_threads_.emplace_back(&ThreadPool::postprocessWorker, this, i);
        LOG_DEBUG("ThreadPool", "Started post-processing thread " + std::to_string(i));
    }
    
    // Start async Redis output thread (if using Redis)
    if (use_redis_queue_) {
        redis_output_thread_ = std::thread(&ThreadPool::redisOutputWorker, this);
        LOG_DEBUG("ThreadPool", "Started async Redis output thread");
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
    
    // Stop post-processing queue
    if (postprocess_queue_) {
        postprocess_queue_->stop();
    }
    
    // Wait for all post-processing threads
    for (auto& thread : postprocess_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    postprocess_threads_.clear();
    
    // Wait for async Redis output thread
    if (redis_output_thread_.joinable()) {
        redis_output_thread_.join();
    }
    
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
        (void)processVideo(reader_id, clip, video_id);
    }
}

void ThreadPool::readerWorkerRedis(int reader_id) {
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " started (Redis mode)");
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " waiting for messages from queue: " + input_queue_name_);
    
    int video_id_counter = 0;
    int consecutive_empty_polls = 0;
    
    while (!stop_flag_) {
        if (!acquireReaderSlot()) {
            break;
        }
        std::string message;
        // Use 1 second timeout for BLPOP - returns false on timeout (no message)
        if (!input_queue_->popMessage(message, 1, input_queue_name_)) {
            // No message available (timeout), continue waiting
            releaseReaderSlot();
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
        
        // Get queue length to show how many messages are remaining
        int queue_length = input_queue_->getQueueLength(input_queue_name_);
        std::string queue_status = (queue_length >= 0) ? 
            " (messages remaining in queue: " + std::to_string(queue_length) + ")" : "";
        
        try {
            auto clips = parseJsonToVideoClips(message);
            std::vector<VideoClip> valid_clips;
            for (auto& clip : clips) {
                if (!clip.path.empty()) {
                    valid_clips.push_back(std::move(clip));
                }
            }
            
            if (valid_clips.empty()) {
                LOG_WARNING("Reader", "Reader " + std::to_string(reader_id) + 
                         " parsed Redis message but found no playable videos" + queue_status);
                releaseReaderSlot();
                continue;
            }
            
            int frame_offset = 0;
            if (use_redis_queue_ && !valid_clips.empty()) {
                const std::string& message_key = valid_clips[0].message_key;
                registerVideoMessage(message_key, message);
            }
            
            for (size_t idx = 0; idx < valid_clips.size(); ++idx) {
                auto& clip = valid_clips[idx];
                LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
                         " processing video from Redis: " + clip.path + queue_status);
                
                bool register_message = false;  // already registered before loop
                bool finalize_message = (idx == valid_clips.size() - 1);
                int frames_processed = processVideo(reader_id, clip, video_id_counter, message,
                                                    register_message, finalize_message, frame_offset);
                frame_offset += frames_processed;
                video_id_counter++;
            }
            
            releaseReaderSlot();
        } catch (const std::exception& e) {
            LOG_ERROR("Reader", "Failed to parse Redis message: " + std::string(e.what()));
            if (!message.empty()) {
                size_t preview_len = std::min(200UL, message.length());
                LOG_ERROR("Reader", "Message content (first " + std::to_string(preview_len) + " chars): " + 
                         message.substr(0, preview_len));
            }
            releaseReaderSlot();
            continue;
        }
    }
    
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " finished (Redis mode)");
}

std::vector<VideoClip> ThreadPool::parseJsonToVideoClips(const std::string& json_str) {
    std::vector<VideoClip> clips;
    try {
        YAML::Node root = YAML::Load(json_str);
        YAML::Node alarm = root["alarm"];
        YAML::Node raw_alarm = alarm ? alarm["raw_alarm"] : YAML::Node();
        
        auto getString = [](const YAML::Node& node, const std::string& key) -> std::string {
            if (!node || !node[key]) return "";
            try {
                return node[key].as<std::string>();
            } catch (...) {
                return "";
            }
        };
        
        std::string serial = getString(raw_alarm, "serial");
        if (serial.empty()) serial = getString(alarm, "serial");
        std::string record_id = getString(alarm, "record_id");
        if (record_id.empty()) record_id = getString(raw_alarm, "record_id");
        std::string tracking_key = getString(root, "tracking_key");
        
        std::string record_date;
        std::string send_at = getString(raw_alarm, "send_at");
        if (send_at.length() >= 8) {
            record_date = send_at.substr(0, 4) + "-" + send_at.substr(4, 2) + "-" + send_at.substr(6, 2);
        } else {
            std::string record_start_time = getString(alarm, "record_start_time");
            if (record_start_time.length() >= 10) {
                record_date = record_start_time.substr(0, 10);
            }
        }
        
        double start_ts = ConfigParser::parseTimestamp(getString(raw_alarm, "video_start_time"));
        double end_ts = ConfigParser::parseTimestamp(getString(raw_alarm, "video_end_time"));
        
        std::string message_key = !tracking_key.empty() ? tracking_key : buildMessageKey(serial, record_id);

        YAML::Node playback = root["playback_location"];
        YAML::Node record_list = raw_alarm ? raw_alarm["record_list"] : YAML::Node();
        size_t clip_count = 0;
        if (playback && playback.IsSequence()) {
            clip_count = playback.size();
        }
        if (record_list && record_list.IsSequence()) {
            clip_count = std::max(clip_count, record_list.size());
        }
        if (clip_count == 0 && playback && playback.IsSequence()) {
            clip_count = playback.size();
        }
        
        YAML::Node config_node = root["config"];
        YAML::Node box_node = config_node ? config_node["box"] : YAML::Node();
        bool has_roi = false;
        float roi_x1 = 0.0f, roi_y1 = 0.0f, roi_x2 = 1.0f, roi_y2 = 1.0f;
        if (box_node && box_node.IsSequence() && box_node.size() >= 3) {
            try {
                auto first = box_node[0];
                auto third = box_node[2];
                if (first.IsSequence() && first.size() >= 2 &&
                    third.IsSequence() && third.size() >= 2) {
                    roi_x1 = first[0].as<float>();
                    roi_y1 = first[1].as<float>();
                    roi_x2 = third[0].as<float>();
                    roi_y2 = third[1].as<float>();
                    has_roi = true;
                }
            } catch (...) {
                has_roi = false;
            }
        }
        
        YAML::Node download_info = root["download_info"];
        
        for (size_t i = 0; i < clip_count; ++i) {
            VideoClip clip;
            clip.serial = serial;
            clip.record_id = record_id;
            clip.record_date = record_date;
            clip.message_key = message_key;
            clip.video_index = static_cast<int>(i);
            clip.start_timestamp = start_ts;
            clip.end_timestamp = end_ts;
            if (std::isfinite(start_ts) && std::isfinite(end_ts)) {
                clip.has_time_window = true;
            }
            clip.has_roi = has_roi;
            clip.roi_x1 = roi_x1;
            clip.roi_y1 = roi_y1;
            clip.roi_x2 = roi_x2;
            clip.roi_y2 = roi_y2;
            
            if (playback && playback.IsSequence() && i < playback.size()) {
                try {
                    clip.path = playback[i].as<std::string>("");
                } catch (...) {
                    clip.path.clear();
                }
            }
            if (clip.path.empty() && download_info && download_info.IsSequence() && i < download_info.size()) {
                clip.path = getString(download_info[i], "local_filepath");
            }
            
            if (record_list && record_list.IsSequence() && i < record_list.size()) {
                const auto& entry = record_list[i];
                if (entry["moment_time"]) {
                    clip.moment_time = entry["moment_time"].as<double>(0.0);
                } else if (entry["recordtimestamp"]) {
                    clip.moment_time = ConfigParser::parseTimestamp(entry["recordtimestamp"].as<std::string>());
                }
                if (entry["duration"]) {
                    try {
                        clip.duration_seconds = entry["duration"].as<double>(0.0);
                    } catch (...) {
                        clip.duration_seconds = 0.0;
                    }
                }
            }
            
            clips.push_back(clip);
        }
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Reader", "Failed to parse Redis message JSON: " + std::string(e.what()));
    }
    
    return clips;
}

int ThreadPool::processVideo(int reader_id, const VideoClip& clip, int video_id,
                             const std::string& redis_message,
                             bool register_message,
                             bool finalize_message,
                             int frame_start_offset) {
    LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
             " processing video " + std::to_string(video_id) + ": " + clip.path);
    
    const std::string message_key = !clip.message_key.empty()
        ? clip.message_key
        : buildMessageKey(clip.serial, clip.record_id);
    const std::string video_key = buildVideoKey(message_key, clip.video_index);
    
    if (register_message && use_redis_queue_ && output_queue_ && !redis_message.empty()) {
        registerVideoMessage(message_key, redis_message);
    }
    
    VideoReader reader(clip, video_id);
    
    if (!reader.isOpened()) {
        LOG_ERROR("Reader", "Cannot open video " + std::to_string(video_id) + ": " + clip.path);
        if (finalize_message) {
            markVideoReadingComplete(message_key);
        }
        return 0;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    int global_frame_number = frame_start_offset;  // Keep for internal tracking if needed
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
            
            // Get the actual frame position in the video file (accounts for time-based filtering)
            int actual_frame_position = reader.getActualFramePosition();
            // Use actual_frame_position (actual frame number in video) for bin file output
            // This ensures frame numbers match the video file, even when we skip frames at the beginning
            FrameData frame_data(frame.clone(), video_id, actual_frame_position, clip.path, 
                                clip.record_id, clip.record_date, clip.serial,
                                message_key, video_key, clip.video_index,
                                clip.has_roi, clip.roi_x1, clip.roi_y1, clip.roi_x2, clip.roi_y2);
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
            global_frame_number++;
            
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
    
    if (finalize_message) {
        markVideoReadingComplete(message_key);
    }
    
    return frame_count;
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
        
        // Process all engines sequentially (std::async overhead was too high)
        // The multiple preprocessor threads already provide parallelism
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
                if (raw_frame.has_roi && !frame_to_process.empty() &&
                    true_original_width > 0 && true_original_height > 0) {
                    float norm_x1 = std::clamp(raw_frame.roi_norm_x1, 0.0f, 1.0f);
                    float norm_y1 = std::clamp(raw_frame.roi_norm_y1, 0.0f, 1.0f);
                    float norm_x2 = std::clamp(raw_frame.roi_norm_x2, 0.0f, 1.0f);
                    float norm_y2 = std::clamp(raw_frame.roi_norm_y2, 0.0f, 1.0f);
                    if (norm_x2 <= norm_x1) norm_x2 = std::min(1.0f, norm_x1 + 0.001f);
                    if (norm_y2 <= norm_y1) norm_y2 = std::min(1.0f, norm_y1 + 0.001f);
                    
                    int x1 = static_cast<int>(norm_x1 * true_original_width);
                    int y1 = static_cast<int>(norm_y1 * true_original_height);
                    int x2 = static_cast<int>(norm_x2 * true_original_width);
                    int y2 = static_cast<int>(norm_y2 * true_original_height);
                    
                    // Clamp to frame bounds
                    x1 = std::max(0, std::min(x1, true_original_width - 1));
                    y1 = std::max(0, std::min(y1, true_original_height - 1));
                    x2 = std::max(x1 + 1, std::min(x2, true_original_width));
                    y2 = std::max(y1 + 1, std::min(y2, true_original_height));
                    
                    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                    frame_to_process = frame_to_process(roi_rect).clone();
                    cropped_width = frame_to_process.cols;
                    cropped_height = frame_to_process.rows;
                    roi_offset_x = x1;
                    roi_offset_y = y1;
                    
                    static bool logged_roi_crop = false;
                    if (!logged_roi_crop && worker_id == 0) {
                        LOG_INFO("Preprocessor", "ROI cropping applied for engine " + engine_group->engine_name +
                                 ": original=" + std::to_string(true_original_width) + "x" + std::to_string(true_original_height) +
                                 ", cropped=" + std::to_string(cropped_width) + "x" + std::to_string(cropped_height) +
                                 ", ROI=[" + std::to_string(x1) + "," + std::to_string(y1) + "," + 
                                 std::to_string(x2) + "," + std::to_string(y2) + "]");
                        logged_roi_crop = true;
                    }
                } else if (worker_id == 0) {
                    static bool logged_roi_missing = false;
                    if (!logged_roi_missing) {
                        LOG_WARNING("Preprocessor", "ROI cropping enabled for engine " + engine_group->engine_name +
                                 " but no ROI metadata available for video " + std::to_string(raw_frame.video_id));
                        logged_roi_missing = true;
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
            processed.record_id = raw_frame.record_id;  // Copy record_id for output path
            processed.record_date = raw_frame.record_date;  // Copy record_date for output path
            processed.serial = raw_frame.serial;
            processed.message_key = raw_frame.message_key;
            processed.video_key = raw_frame.video_key;
            processed.video_index = raw_frame.video_index;
            processed.has_roi = raw_frame.has_roi;
            processed.roi_norm_x1 = raw_frame.roi_norm_x1;
            processed.roi_norm_y1 = raw_frame.roi_norm_y1;
            processed.roi_norm_x2 = raw_frame.roi_norm_x2;
            processed.roi_norm_y2 = raw_frame.roi_norm_y2;
            processed.frame = frame_to_process;  // Keep reference for potential debugging/logging
            processed.original_width = cropped_width;  // Cropped frame width (for scale calculation)
            processed.original_height = cropped_height;  // Cropped frame height (for scale calculation)
            processed.true_original_width = true_original_width;  // True original frame width (for clamping after ROI offset)
            processed.true_original_height = true_original_height;  // True original frame height (for clamping after ROI offset)
            processed.roi_offset_x = roi_offset_x;  // ROI offset X in true original frame
            processed.roi_offset_y = roi_offset_y;  // ROI offset Y in true original frame
            
            registerPendingFrame(processed.message_key, engine_group->engine_name);
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

    auto formatDate = [](const std::string& record_date) {
        if (record_date.length() >= 10 && record_date.find('-') != std::string::npos) {
            std::string year = record_date.substr(0, 4);
            std::string month = record_date.substr(5, 2);
            std::string day = record_date.substr(8, 2);
            return day + "-" + month + "-" + year.substr(2, 2);
        }
        return std::string("unknown-date");
    };

    auto serialPart = [](const std::string& serial, const std::string& fallback_key, int video_id) {
        if (!serial.empty()) return serial;
        if (!fallback_key.empty()) return fallback_key;
        return std::string("video_") + std::to_string(video_id);
    };

    auto recordPart = [](const std::string& record_id, const std::string& fallback_key, int video_id) {
        if (!record_id.empty()) return record_id;
        if (!fallback_key.empty()) return fallback_key;
        return std::string("video_") + std::to_string(video_id);
    };
    
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
            std::vector<std::string> batch_message_keys;
            std::vector<int> batch_video_indices;
            std::vector<std::string> batch_serials;
            std::vector<std::string> batch_record_ids;
            std::vector<std::string> batch_record_dates;
            
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
                std::string serial_value = serialPart(frame_data.serial, frame_data.message_key, frame_data.video_id);
                std::string record_value = recordPart(frame_data.record_id, frame_data.message_key, frame_data.video_id);
                std::string output_path = generateOutputPath(
                    serial_value, record_value, frame_data.record_date, engine_group->engine_name,
                    frame_data.video_index);
                
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
                batch_message_keys.push_back(frame_data.message_key);
                batch_video_indices.push_back(frame_data.video_index);
                batch_serials.push_back(frame_data.serial);
                batch_record_ids.push_back(frame_data.record_id);
                batch_record_dates.push_back(frame_data.record_date);
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
                    // Get raw inference output and push to post-processing queue
                    std::vector<std::vector<float>> raw_outputs;
                    success = engine_group->detectors[detector_id]->getRawInferenceOutput(
                        batch_tensors, raw_outputs
                    );
                    
                    if (success && !raw_outputs.empty()) {
                        // Create post-processing task
                        PostProcessTask task;
                        task.detector = engine_group->detectors[detector_id].get();
                        task.raw_outputs = std::move(raw_outputs);
                        task.output_paths = batch_output_paths;
                        task.frame_numbers = batch_frame_numbers;
                        task.original_widths = batch_original_widths;
                        task.original_heights = batch_original_heights;
                        task.roi_offset_x = batch_roi_offset_x;
                        task.roi_offset_y = batch_roi_offset_y;
                        task.true_original_widths = batch_true_original_widths;
                        task.true_original_heights = batch_true_original_heights;
                        task.message_keys = batch_message_keys;
                        task.video_indices = batch_video_indices;
                        task.engine_name = engine_group->engine_name;
                        task.engine_id = engine_id;
                        task.batch_size = static_cast<int>(batch_tensors.size());
                        
                        // Register pending frames BEFORE pushing to post-processing queue
                        for (const auto& msg_key : batch_message_keys) {
                            if (!msg_key.empty()) {
                                registerPendingFrame(msg_key, engine_group->engine_name);
                            }
                        }
                        
                        // Push to post-processing queue (non-blocking)
                        if (!postprocess_queue_ || !postprocess_queue_->push(task)) {
                            LOG_ERROR("Detector", "Failed to push batch to post-processing queue");
                            success = false;
                        }
                    }
                }
                
                auto batch_end = std::chrono::steady_clock::now();
                auto batch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
                
                // Save debug images for batch
                if (debug_mode_ && success && batch_detections.size() == static_cast<size_t>(batch_size) &&
                    batch_frames.size() == static_cast<size_t>(batch_size)) {
                    for (int b = 0; b < batch_size; ++b) {
                        cv::Mat debug_frame = engine_group->preprocessor->addPadding(batch_frames[b]);
                        engine_group->detectors[detector_id]->drawDetections(debug_frame, batch_detections[b]);
                        
                        std::string serial_part = serialPart(batch_serials[b], batch_message_keys[b], batch_video_ids[b]);
                        std::string record_part = recordPart(batch_record_ids[b], batch_message_keys[b], batch_video_ids[b]);
                        std::string date_part = formatDate(batch_record_dates[b]);
                        
                        std::filesystem::path debug_dir = std::filesystem::path(output_dir_) / "debug_images" /
                                                           engine_group->engine_name / date_part /
                                                           (serial_part + "_" + record_part + "_v" + std::to_string(batch_video_indices[b]));
                        std::filesystem::create_directories(debug_dir);
                        
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

                // Note: markFrameProcessed is now called in postprocessWorker after file writing
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
            std::string serial_value = serialPart(frame_data.serial, frame_data.message_key, frame_data.video_id);
            std::string record_value = recordPart(frame_data.record_id, frame_data.message_key, frame_data.video_id);
            std::string output_path = generateOutputPath(
                serial_value,
                record_value,
                frame_data.record_date,
                engine_group->engine_name,
                frame_data.video_index
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
                // Get raw inference output and push to post-processing queue
                std::vector<std::shared_ptr<std::vector<float>>> single_input = {tensor};
                std::vector<std::vector<float>> raw_outputs;
                success = engine_group->detectors[detector_id]->getRawInferenceOutput(
                    single_input, raw_outputs
                );
                
                if (success && !raw_outputs.empty()) {
                    // Create post-processing task
                    PostProcessTask task;
                    task.detector = engine_group->detectors[detector_id].get();
                    task.raw_outputs = std::move(raw_outputs);
                    task.output_paths = {output_path};
                    task.frame_numbers = {frame_data.frame_number};
                    task.original_widths = {frame_data.original_width};
                    task.original_heights = {frame_data.original_height};
                    task.roi_offset_x = {frame_data.roi_offset_x};
                    task.roi_offset_y = {frame_data.roi_offset_y};
                    task.true_original_widths = {frame_data.true_original_width};
                    task.true_original_heights = {frame_data.true_original_height};
                    task.message_keys = {frame_data.message_key};
                    task.video_indices = {frame_data.video_index};
                    task.engine_name = engine_group->engine_name;
                    task.engine_id = engine_id;
                    task.batch_size = 1;
                    
                    // Register pending frame
                    if (!frame_data.message_key.empty()) {
                        registerPendingFrame(frame_data.message_key, engine_group->engine_name);
                    }
                    
                    // Push to post-processing queue (non-blocking)
                    if (!postprocess_queue_ || !postprocess_queue_->push(task)) {
                        LOG_ERROR("Detector", "Failed to push frame to post-processing queue");
                        success = false;
                    }
                }
            }
            
            auto detect_end = std::chrono::steady_clock::now();
            auto detect_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();
            
            // Save debug image if enabled
            if (debug_mode_ && success && !frame_data.frame.empty()) {
                cv::Mat debug_frame = engine_group->preprocessor->addPadding(frame_data.frame);
                engine_group->detectors[detector_id]->drawDetections(debug_frame, detections);
                
                std::string serial_part = serialPart(frame_data.serial, frame_data.message_key, frame_data.video_id);
                std::string record_part = recordPart(frame_data.record_id, frame_data.message_key, frame_data.video_id);
                std::string date_part = formatDate(frame_data.record_date);
                
                std::filesystem::path debug_dir = std::filesystem::path(output_dir_) / "debug_images" /
                                                   engine_group->engine_name / date_part /
                                                   (serial_part + "_" + record_part + "_v" + std::to_string(frame_data.video_index));
                std::filesystem::create_directories(debug_dir);
                
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

            // Note: markFrameProcessed is now called in postprocessWorker after file writing
            
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

void ThreadPool::postprocessWorker(int worker_id) {
    LOG_INFO("PostProcess", "Post-processing thread " + std::to_string(worker_id) + " started");
    
    PostProcessTask task;
    int processed_count = 0;
    
    while (!stop_flag_) {
        if (!postprocess_queue_ || !postprocess_queue_->pop(task, 100)) {
            if (stop_flag_) break;
            continue;
        }
        
        if (!task.detector || task.raw_outputs.empty() || task.output_paths.empty()) {
            LOG_ERROR("PostProcess", "Invalid post-processing task");
            continue;
        }
        
        auto postprocess_start = std::chrono::steady_clock::now();
        
        // Process each frame in the batch
        int num_anchors = task.detector->getNumAnchors();
        int output_channels = task.detector->getOutputChannels();
        size_t output_per_frame = static_cast<size_t>(num_anchors) * output_channels;
        
        bool all_success = true;
        
        for (size_t b = 0; b < task.raw_outputs.size(); ++b) {
            if (b >= task.output_paths.size() || b >= task.frame_numbers.size()) {
                LOG_ERROR("PostProcess", "Batch index out of bounds: " + std::to_string(b));
                all_success = false;
                continue;
            }
            
            // Validate output size
            if (task.raw_outputs[b].size() != output_per_frame) {
                LOG_ERROR("PostProcess", "Output size mismatch for frame " + std::to_string(b) + 
                         ": expected " + std::to_string(output_per_frame) + 
                         ", got " + std::to_string(task.raw_outputs[b].size()));
                all_success = false;
                continue;
            }
            
            // Transpose from [channels, num_anchors] to [num_anchors, channels]
            std::vector<float> frame_output(output_per_frame);
            for (int anchor = 0; anchor < num_anchors; ++anchor) {
                for (int channel = 0; channel < output_channels; ++channel) {
                    int src_idx = channel * num_anchors + anchor;  // [channels, num_anchors]
                    int dst_idx = anchor * output_channels + channel;  // [num_anchors, channels]
                    if (src_idx < static_cast<int>(task.raw_outputs[b].size()) && 
                        dst_idx < static_cast<int>(frame_output.size())) {
                        frame_output[dst_idx] = task.raw_outputs[b][src_idx];
                    }
                }
            }
            
            // Parse raw output
            std::vector<Detection> detections;
            if (task.detector->getModelType() == ModelType::POSE) {
                detections = task.detector->parseRawPoseOutput(frame_output);
            } else {
                detections = task.detector->parseRawDetectionOutput(frame_output);
            }
            
            // Apply NMS
            detections = task.detector->applyNMS(detections);
            
            // Limit detections (max_detections is per-frame, not per-batch)
            // We'll use a reasonable limit of 1000 detections per frame
            if (static_cast<int>(detections.size()) > 1000) {
                detections.resize(1000);
            }
            
            // Scale to original frame coordinates
            if (b < task.original_widths.size() && b < task.original_heights.size()) {
                int orig_w = task.original_widths[b];
                int orig_h = task.original_heights[b];
                int roi_x = (b < task.roi_offset_x.size()) ? task.roi_offset_x[b] : 0;
                int roi_y = (b < task.roi_offset_y.size()) ? task.roi_offset_y[b] : 0;
                int true_orig_w = (b < task.true_original_widths.size()) ? task.true_original_widths[b] : 0;
                int true_orig_h = (b < task.true_original_heights.size()) ? task.true_original_heights[b] : 0;
                
                if (orig_w > 0 && orig_h > 0) {
                    for (auto& det : detections) {
                        task.detector->scaleDetectionToOriginal(det, orig_w, orig_h, roi_x, roi_y, true_orig_w, true_orig_h);
                    }
                }
            }
            
            // Write to file
            if (!task.detector->writeDetectionsToFile(detections, task.output_paths[b], task.frame_numbers[b])) {
                LOG_ERROR("PostProcess", "Failed to write detections for frame " + std::to_string(task.frame_numbers[b]));
                all_success = false;
            } else {
                // Mark frame as processed (for Redis message tracking)
                if (use_redis_queue_ && b < task.message_keys.size() && b < task.video_indices.size()) {
                    markFrameProcessed(task.message_keys[b], task.engine_name, task.output_paths[b], task.video_indices[b]);
                }
            }
        }
        
        auto postprocess_end = std::chrono::steady_clock::now();
        auto postprocess_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            postprocess_end - postprocess_start).count();
        
        if (all_success) {
            {
                std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                if (task.engine_id >= 0 && task.engine_id < static_cast<int>(stats_.frames_detected.size())) {
                    stats_.frames_detected[task.engine_id] += static_cast<int>(task.raw_outputs.size());
                    stats_.engine_total_time_ms[task.engine_id] += postprocess_time_ms;
                    stats_.engine_frame_count[task.engine_id] += static_cast<int>(task.raw_outputs.size());
                }
            }
            processed_count += static_cast<int>(task.raw_outputs.size());
        } else {
            {
                std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                if (task.engine_id >= 0 && task.engine_id < static_cast<int>(stats_.frames_failed.size())) {
                    stats_.frames_failed[task.engine_id] += static_cast<int>(task.raw_outputs.size());
                }
            }
        }
    }
    
    LOG_INFO("PostProcess", "Post-processing thread " + std::to_string(worker_id) + 
             " finished (" + std::to_string(processed_count) + " frames processed)");
}

void ThreadPool::redisOutputWorker() {
    LOG_INFO("RedisOutput", "Async Redis output worker started");
    
    // This worker will handle async Redis message sending
    // For now, markFrameProcessed handles Redis directly, but we can add a queue here if needed
    // The current implementation already calls markFrameProcessed from postprocessWorker
    // which is non-blocking for the detector threads
    
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    LOG_INFO("RedisOutput", "Async Redis output worker finished");
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

        if (use_redis_queue_ && input_queue_) {
            int input_len = input_queue_->getQueueLength(input_queue_name_);
            stats_oss << "Redis Input Queue (" << input_queue_name_ << "): " << input_len;
            if (output_queue_) {
                int output_len = output_queue_->getQueueLength(output_queue_name_);
                stats_oss << " | Redis Output Queue (" << output_queue_name_ << "): " << output_len;
            }
            stats_oss << std::endl;
        }
        
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

std::string ThreadPool::generateOutputPath(const std::string& serial,
                                           const std::string& record_id,
                                           const std::string& record_date,
                                           const std::string& engine_name,
                                           int video_index) {
    // Generate path in format: /detector_name/dd-mm-yy/serial_recordid_videoindex.bin
    
    auto sanitize_component = [](const std::string& value, const std::string& fallback) {
        return value.empty() ? fallback : value;
    };
    
    std::string serial_part = sanitize_component(serial, "unknown_serial");
    std::string record_part = sanitize_component(record_id, "video");
    
    // Convert date from YYYY-MM-DD to dd-mm-yy
    std::string date_formatted = record_date;
    if (record_date.length() >= 10 && record_date.find("-") != std::string::npos) {
        std::string year = record_date.substr(0, 4);
        std::string month = record_date.substr(5, 2);
        std::string day = record_date.substr(8, 2);
        date_formatted = day + "-" + month + "-" + year.substr(2, 2);
    } else {
        date_formatted = "unknown-date";
    }
    
    std::ostringstream oss;
    oss << output_dir_ << "/" << engine_name << "/" << date_formatted << "/"
        << serial_part << "_" << record_part;
    if (video_index > 0) {
        oss << "_v" << video_index;
    }
    oss << ".bin";
    
    return oss.str();
}

std::string ThreadPool::buildMessageKey(const std::string& serial, const std::string& record_id) const {
    if (!serial.empty() && !record_id.empty()) {
        return serial + "_" + record_id;
    }
    if (!record_id.empty()) {
        return record_id;
    }
    if (!serial.empty()) {
        return serial;
    }
    return "message";
}

std::string ThreadPool::buildVideoKey(const std::string& message_key, int video_index) const {
    std::ostringstream oss;
    if (!message_key.empty()) {
        oss << message_key;
    } else {
        oss << "video";
    }
    oss << "_v" << video_index;
    return oss.str();
}

void ThreadPool::registerVideoMessage(const std::string& message_key, const std::string& message) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty() || message.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(video_output_mutex_);
    VideoOutputStatus& status = video_output_status_[message_key];
    status.original_message = message;
    status.reading_completed = false;
    status.message_pushed = false;
}

void ThreadPool::registerPendingFrame(const std::string& message_key, const std::string& engine_name) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty() || engine_name.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(video_output_mutex_);
    VideoOutputStatus& status = video_output_status_[message_key];
    status.pending_counts[engine_name]++;
}

void ThreadPool::markFrameProcessed(const std::string& message_key, const std::string& engine_name,
                                    const std::string& output_path, int video_index) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty() || engine_name.empty()) {
        return;
    }
    
    std::string message_to_push;
    {
        std::lock_guard<std::mutex> lock(video_output_mutex_);
        auto& status = video_output_status_[message_key];
        if (!output_path.empty()) {
            status.detector_outputs[engine_name][video_index] = output_path;
        }
        int& pending = status.pending_counts[engine_name];
        if (pending > 0) {
            pending--;
        }
        message_to_push = tryPushOutputLocked(message_key, status);
    }
    
    if (!message_to_push.empty()) {
        if (output_queue_->pushMessage(output_queue_name_, message_to_push)) {
            LOG_INFO("RedisOutput", "Pushed processed message to output queue: " + output_queue_name_);
        } else {
            LOG_ERROR("RedisOutput", "Failed to push message to output queue: " + output_queue_name_);
        }
    }
}

void ThreadPool::markVideoReadingComplete(const std::string& message_key) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty()) {
        return;
    }
    
    std::string message_to_push;
    {
        std::lock_guard<std::mutex> lock(video_output_mutex_);
        auto it = video_output_status_.find(message_key);
        if (it == video_output_status_.end()) {
            return;
        }
        it->second.reading_completed = true;
        message_to_push = tryPushOutputLocked(message_key, it->second);
    }
    
    if (!message_to_push.empty()) {
        if (output_queue_->pushMessage(output_queue_name_, message_to_push)) {
            LOG_INFO("RedisOutput", "Pushed processed message to output queue: " + output_queue_name_);
        } else {
            LOG_ERROR("RedisOutput", "Failed to push message to output queue: " + output_queue_name_);
        }
    }
}

bool ThreadPool::canPushOutputLocked(const VideoOutputStatus& status) const {
    if (!status.reading_completed || status.message_pushed) {
        return false;
    }
    for (const auto& kv : status.pending_counts) {
        if (kv.second > 0) {
            return false;
        }
    }
    return true;
}

std::string ThreadPool::augmentMessageWithDetectors(const std::string& message,
                                                    const std::unordered_map<std::string, std::map<int, std::string>>& outputs) const {
    if (message.empty()) {
        return message;
    }
    bool has_data = false;
    for (const auto& kv : outputs) {
        if (!kv.second.empty()) {
            has_data = true;
            break;
        }
    }
    if (!has_data) {
        return message;
    }
    
    std::string augmented = message;
    size_t insert_pos = augmented.find_last_of('}');
    if (insert_pos == std::string::npos) {
        return message;
    }
    
    std::ostringstream extra;
    extra << ", ";
    bool first_engine = true;
    for (const auto& kv : outputs) {
        if (kv.second.empty()) {
            continue;
        }
        if (!first_engine) {
            extra << ", ";
        }
        first_engine = false;
        extra << "\"" << kv.first << "\": [";
        bool first_path = true;
        for (const auto& idx_path : kv.second) {
            if (idx_path.second.empty()) {
                continue;
            }
            if (!first_path) {
                extra << ", ";
            }
            first_path = false;
            extra << "\"" << idx_path.second << "\"";
        }
        extra << "]";
    }

    if (first_engine) {
        return message;
    }

    augmented.insert(insert_pos, extra.str());
    return augmented;
}

std::string ThreadPool::tryPushOutputLocked(const std::string& message_key, VideoOutputStatus& status) {
    if (!canPushOutputLocked(status)) {
        return "";
    }
    if (status.original_message.empty()) {
        return "";
    }
    bool has_outputs = false;
    for (const auto& kv : status.detector_outputs) {
        if (!kv.second.empty()) {
            has_outputs = true;
            break;
        }
    }
    if (!has_outputs) {
        LOG_WARNING("RedisOutput", "tryPushOutputLocked: message '" + message_key +
                                   "' has no detector outputs yet, delaying push");
        return "";
    }
    
    status.message_pushed = true;
    std::string final_message = augmentMessageWithDetectors(status.original_message, status.detector_outputs);
    
    // Persist the final message to JSON file for auditing/debugging
    try {
        std::filesystem::path output_dir = std::filesystem::path(output_dir_) / "redis_messages";
        std::filesystem::create_directories(output_dir);
        std::string safe_key = message_key.empty() ? "unknown" : message_key;
        std::filesystem::path file_path = output_dir / (safe_key + ".json");
        std::ofstream outfile(file_path);
        if (outfile.is_open()) {
            outfile << final_message;
            outfile.close();
            LOG_INFO("RedisOutput", "Saved output message to " + file_path.string());
        } else {
            LOG_ERROR("RedisOutput", "Failed to open file for saving message: " + file_path.string());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("RedisOutput", "Failed to save Redis output message: " + std::string(e.what()));
    }
    
    LOG_INFO("RedisOutput", "Ready to push message for key '" + message_key + "': " + final_message);
    video_output_status_.erase(message_key);
    return final_message;
}

bool ThreadPool::acquireReaderSlot() {
    if (!use_redis_queue_ || max_active_redis_readers_ <= 0) {
        return true;
    }
    
    std::unique_lock<std::mutex> lock(reader_slot_mutex_);
    reader_slot_cv_.wait(lock, [this]() {
        return stop_flag_ || active_redis_readers_.load() < max_active_redis_readers_;
    });
    
    if (stop_flag_) {
        return false;
    }
    
    active_redis_readers_++;
    return true;
}

void ThreadPool::releaseReaderSlot() {
    if (!use_redis_queue_ || max_active_redis_readers_ <= 0) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(reader_slot_mutex_);
        if (active_redis_readers_ > 0) {
            active_redis_readers_--;
        }
    }
    reader_slot_cv_.notify_one();
}

