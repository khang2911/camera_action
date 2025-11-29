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
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <future>
#include <fstream>
#include <atomic>
#include <deque>
#include <optional>
#include <set>
#include <map>
#include <yaml-cpp/yaml.h>
#include <cuda_runtime_api.h>
#include "nlohmann/json.hpp"

// Helper functions for path generation
static std::string formatDate(const std::string& record_date) {
    if (record_date.length() >= 10 && record_date.find('-') != std::string::npos) {
        std::string year = record_date.substr(0, 4);
        std::string month = record_date.substr(5, 2);
        std::string day = record_date.substr(8, 2);
        return day + "-" + month + "-" + year.substr(2, 2);
    }
    return std::string("unknown-date");
}

static std::string serialPart(const std::string& serial, const std::string& fallback_key, int video_id) {
    if (!serial.empty()) return serial;
    if (!fallback_key.empty()) return fallback_key;
    return std::string("video_") + std::to_string(video_id);
}

static std::string recordPart(const std::string& record_id, const std::string& fallback_key, int video_id) {
    if (!record_id.empty()) return record_id;
    if (!fallback_key.empty()) return fallback_key;
    return std::string("video_") + std::to_string(video_id);
}

SharedPreprocessGroup::SharedPreprocessGroup(int id, int in_w, int in_h, bool roi, size_t queue_cap)
    : group_id(id),
      input_width(in_w),
      input_height(in_h),
      roi_cropping(roi),
      tensor_elements(static_cast<size_t>(in_w) * in_h * 3),
      queue_capacity(queue_cap),
      queue(std::make_unique<FrameQueue>(queue_cap)),
      preprocessor(std::make_unique<Preprocessor>(in_w, in_h)) {}

SharedPreprocessGroup::~SharedPreprocessGroup() {
    std::lock_guard<std::mutex> lock(buffer_pool_mutex);
    for (auto* buffer : buffer_pool) {
        if (buffer && !buffer->empty()) {
            cudaHostUnregister(buffer->data());
        }
        delete buffer;
    }
    buffer_pool.clear();
}

std::shared_ptr<std::vector<float>> SharedPreprocessGroup::acquireBuffer() {
    std::vector<float>* buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        if (!buffer_pool.empty()) {
            buffer = buffer_pool.back();
            buffer_pool.pop_back();
        }
    }
    
    if (!buffer) {
        buffer = new std::vector<float>();
        buffer->reserve(tensor_elements);
        buffer->resize(tensor_elements);
        if (!buffer->empty()) {
            cudaHostRegister(buffer->data(), tensor_elements * sizeof(float), cudaHostRegisterPortable);
        }
    } else if (buffer->size() != tensor_elements) {
        buffer->assign(tensor_elements, 0.0f);
    }
    
    auto deleter = [this](std::vector<float>* ptr) {
        this->releaseBuffer(ptr);
    };
    return std::shared_ptr<std::vector<float>>(buffer, deleter);
}

void SharedPreprocessGroup::releaseBuffer(std::vector<float>* buffer) {
    if (!buffer) {
        return;
    }
    std::lock_guard<std::mutex> lock(buffer_pool_mutex);
    buffer_pool.push_back(buffer);
}

ThreadPool::ThreadPool(int num_readers,
                       int num_preprocessors,
                       const std::vector<VideoClip>& video_clips,
                       const std::vector<EngineConfig>& engine_configs,
                       const std::string& output_dir,
                       bool debug_mode,
                       int max_frames_per_video,
                       const ReaderOptions& reader_options)
    : num_readers_(num_readers),
      num_preprocessors_(num_preprocessors > 0 ? num_preprocessors : num_readers),
      video_clips_(video_clips),
      output_dir_(output_dir),
      debug_mode_(debug_mode),
      max_frames_per_video_(max_frames_per_video),
      reader_options_(reader_options),
      use_redis_queue_(false),
      redis_message_timeout_seconds_(300),  // Default timeout for file-based mode (not used)
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
    
    // Initialize previous stats for windowed FPS calculation
    {
        std::lock_guard<std::mutex> lock(prev_stats_mutex_);
        prev_monitor_time_ = stats_.start_time;
        prev_frames_read_ = 0;
        prev_frames_preprocessed_ = 0;
        prev_frames_postprocessed_ = 0;
        prev_frames_detected_.resize(engine_configs.size(), 0);
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    // Increased queue size to reduce blocking - readers can push more frames before waiting
    // With multiple readers, we need larger buffer to prevent blocking when preprocessors are slower
    raw_frame_queue_ = std::make_unique<FrameQueue>(2000);
    postprocess_queue_ = std::make_unique<PostProcessQueue>();
    
    // Initialize video processed flags
    {
        std::lock_guard<std::mutex> lock(video_mutex_);
        video_processed_.assign(video_clips_.size(), false);
    }
    
    // Initialize Redis queue mode flag
    use_redis_queue_ = false;
    
    // Initialize engine groups (one per engine)
    std::unordered_map<std::string, SharedPreprocessGroup*> preprocess_group_map;
    auto build_preprocess_key = [](const EngineConfig& cfg) {
        std::ostringstream oss;
        oss << cfg.input_width << "x" << cfg.input_height << (cfg.roi_cropping ? "_roi" : "_full");
        return oss.str();
    };
    auto get_shared_group = [&](const EngineConfig& cfg, size_t queue_size) {
        const std::string key = build_preprocess_key(cfg);
        auto it = preprocess_group_map.find(key);
        if (it != preprocess_group_map.end()) {
            return it->second;
        }
        auto group = std::make_unique<SharedPreprocessGroup>(
            static_cast<int>(preprocess_groups_.size()),
            cfg.input_width,
            cfg.input_height,
            cfg.roi_cropping,
            queue_size);
        auto* ptr = group.get();
        preprocess_groups_.push_back(std::move(group));
        preprocess_group_map.emplace(key, ptr);
        return ptr;
    };
    
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        const auto& config = engine_configs[i];
        // CRITICAL: Calculate optimal queue size based on batch_size and num_detectors
        // Need enough buffer for: batch_size * num_detectors * 2 (for pipelining) + safety margin
        // With batch_size=16 and num_detectors=2, we need at least 64 frames, use 500 for safety
        size_t optimal_queue_size = std::max(500UL, static_cast<size_t>(config.batch_size * config.num_detectors * 4));
        auto engine_group = std::make_unique<EngineGroup>(
            static_cast<int>(i), config.path, config.name, config.num_detectors,
            config.input_width, config.input_height, config.roi_cropping, optimal_queue_size
        );
        
        LOG_INFO("ThreadPool", "Initializing engine " + config.name + " with " + 
                 std::to_string(config.num_detectors) + " detector threads (queue_size=" + 
                 std::to_string(optimal_queue_size) + ")");
        
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
        auto* shared_group = get_shared_group(config, optimal_queue_size);
        shared_group->engines.push_back(engine_group.get());
        engine_group->shared_preprocess = shared_group;
        
        engine_groups_.push_back(std::move(engine_group));
    }
    
    // Configure dispatcher and shared preprocess worker counts
    preprocess_dispatcher_count_ = std::max(1, std::min(num_preprocessors_, 4));
    // Configure dispatcher and shared preprocess worker counts
    preprocess_dispatcher_count_ = std::max(1, std::min(num_preprocessors_, 4));
    
    LOG_INFO("ThreadPool", "ThreadPool initialized with " + std::to_string(num_readers) + 
             " reader threads, " + std::to_string(preprocess_dispatcher_count_) + " dispatcher threads, " +
             std::to_string(num_preprocessors_) + " shared preprocess workers, and " +
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
                       int max_frames_per_video,
                       const ReaderOptions& reader_options,
                       int redis_message_timeout_seconds)
    : num_readers_(num_readers),
      num_preprocessors_(num_preprocessors > 0 ? num_preprocessors : num_readers),
      output_dir_(output_dir),
      debug_mode_(debug_mode),
      max_frames_per_video_(max_frames_per_video),
      reader_options_(reader_options),
      use_redis_queue_(true),
      input_queue_(input_queue),
      output_queue_(output_queue),
      input_queue_name_(input_queue_name),
      output_queue_name_(output_queue_name),
      redis_message_timeout_seconds_(redis_message_timeout_seconds),
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
    
    // Initialize previous stats for windowed FPS calculation
    {
        std::lock_guard<std::mutex> lock(prev_stats_mutex_);
        prev_monitor_time_ = stats_.start_time;
        prev_frames_read_ = 0;
        prev_frames_preprocessed_ = 0;
        prev_frames_postprocessed_ = 0;
        prev_frames_detected_.resize(engine_configs.size(), 0);
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    // Increased queue size to reduce blocking - readers can push more frames before waiting
    raw_frame_queue_ = std::make_unique<FrameQueue>(2000);
    postprocess_queue_ = std::make_unique<PostProcessQueue>();
    
    max_active_redis_readers_ = num_readers_;
    
    // Initialize engine groups (one per engine)
    std::unordered_map<std::string, SharedPreprocessGroup*> preprocess_group_map;
    auto build_preprocess_key = [](const EngineConfig& cfg) {
        std::ostringstream oss;
        oss << cfg.input_width << "x" << cfg.input_height << (cfg.roi_cropping ? "_roi" : "_full");
        return oss.str();
    };
    auto get_shared_group = [&](const EngineConfig& cfg, size_t queue_size) {
        const std::string key = build_preprocess_key(cfg);
        auto it = preprocess_group_map.find(key);
        if (it != preprocess_group_map.end()) {
            return it->second;
        }
        auto group = std::make_unique<SharedPreprocessGroup>(
            static_cast<int>(preprocess_groups_.size()),
            cfg.input_width,
            cfg.input_height,
            cfg.roi_cropping,
            queue_size);
        auto* ptr = group.get();
        preprocess_groups_.push_back(std::move(group));
        preprocess_group_map.emplace(key, ptr);
        return ptr;
    };
    
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        const auto& config = engine_configs[i];
        // CRITICAL: Calculate optimal queue size based on batch_size and num_detectors
        // Need enough buffer for: batch_size * num_detectors * 2 (for pipelining) + safety margin
        // With batch_size=16 and num_detectors=2, we need at least 64 frames, use 500 for safety
        size_t optimal_queue_size = std::max(500UL, static_cast<size_t>(config.batch_size * config.num_detectors * 4));
        auto engine_group = std::make_unique<EngineGroup>(
            static_cast<int>(i), config.path, config.name, config.num_detectors,
            config.input_width, config.input_height, config.roi_cropping, optimal_queue_size
        );
        
        LOG_INFO("ThreadPool", "Initializing engine " + config.name + " with " + 
                 std::to_string(config.num_detectors) + " detector threads (queue_size=" + 
                 std::to_string(optimal_queue_size) + ")");
        
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
        auto* shared_group = get_shared_group(config, optimal_queue_size);
        shared_group->engines.push_back(engine_group.get());
        engine_group->shared_preprocess = shared_group;
        
        engine_groups_.push_back(std::move(engine_group));
    }
    
    preprocess_dispatcher_count_ = std::max(1, std::min(num_preprocessors_, 4));
    
    LOG_INFO("ThreadPool", "ThreadPool initialized (Redis mode) with " + std::to_string(num_readers) + 
             " reader threads, " + std::to_string(preprocess_dispatcher_count_) + " dispatcher threads, " +
             std::to_string(num_preprocessors_) + " shared preprocess workers, and " +
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
    // Initialize previous stats for windowed FPS calculation
    {
        std::lock_guard<std::mutex> lock(prev_stats_mutex_);
        prev_monitor_time_ = stats_.start_time;
        prev_frames_read_ = 0;
        prev_frames_preprocessed_ = 0;
        prev_frames_postprocessed_ = 0;
        prev_frames_detected_.resize(stats_.frames_detected.size(), 0);
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
    
    // Start dispatcher threads (feed engine preprocess queues)
    for (int i = 0; i < preprocess_dispatcher_count_; ++i) {
        preprocessor_threads_.emplace_back(&ThreadPool::preprocessorWorker, this, i);
        LOG_DEBUG("ThreadPool", "Started preprocess dispatcher thread " + std::to_string(i));
    }
    
    for (auto& group : preprocess_groups_) {
        if (group && group->queue) {
            group->queue->clear();
        }
    }
    int shared_workers = std::max(1, num_preprocessors_);
    
    // Warn if too many preprocess workers for the number of groups (causes contention)
    if (shared_workers > static_cast<int>(preprocess_groups_.size()) * 10) {
        LOG_WARNING("ThreadPool", "Too many preprocess workers (" + std::to_string(shared_workers) + 
                   ") for number of preprocess groups (" + std::to_string(preprocess_groups_.size()) + 
                   "). This may cause queue contention. Consider reducing preprocess workers to ~" + 
                   std::to_string(preprocess_groups_.size() * 5) + " or fewer.");
    }
    
    for (int worker = 0; worker < shared_workers; ++worker) {
        shared_preprocess_threads_.emplace_back(&ThreadPool::enginePreprocessWorker, this, worker);
        LOG_DEBUG("ThreadPool", "Started shared preprocess thread " + std::to_string(worker));
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
    
    // Start post-processing threads (increase to handle debug image saving and file I/O)
    // With debug mode, postprocessing is slower due to image saving, so we need more threads
    // Use 4 threads per engine to handle the load better
    int num_postprocess_threads = std::max(4, static_cast<int>(engine_groups_.size()) * 4);
    for (int i = 0; i < num_postprocess_threads; ++i) {
        postprocess_threads_.emplace_back(&ThreadPool::postprocessWorker, this, i);
        LOG_DEBUG("ThreadPool", "Started post-processing thread " + std::to_string(i));
    }
    LOG_INFO("ThreadPool", "Started " + std::to_string(num_postprocess_threads) + " post-processing threads");
    
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
    
    // Wait for dispatcher threads to finish
    for (auto& thread : preprocessor_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    preprocessor_threads_.clear();
    shared_preprocess_threads_.clear();
    
    // Wait for shared preprocess workers
    for (auto& thread : shared_preprocess_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    shared_preprocess_threads_.clear();
    for (auto& group : preprocess_groups_) {
        if (group && group->queue) {
            group->queue->clear();
        }
    }
    
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
            // Also check postprocessor queue (critical - this is often the bottleneck)
            if (postprocess_queue_ && postprocess_queue_->size() > 0) {
                all_queues_empty = false;
                // Log periodically if postprocessor queue is backing up (debug only)
                static int postproc_warning_count = 0;
                if (postproc_warning_count++ % 50 == 0) {
                    LOG_DEBUG("ThreadPool", "Waiting for postprocessor queue to drain: " + 
                             std::to_string(postprocess_queue_->size()) + " tasks pending");
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
        
        LOG_INFO("ThreadPool", "=== All Processing Complete ===");
        LOG_INFO("ThreadPool", "All videos processed, all queues empty, and all threads stopped.");
        LOG_INFO("ThreadPool", "Stopping all threads...");
        
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
        
        // Removed getQueueLength call - it's a Redis operation that blocks and slows down reader
        // Queue status can be monitored by the monitor thread instead
        
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
                         " parsed Redis message but found no playable videos");
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
                // Removed video_key generation and logging - too expensive for hot path
                
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
        
        std::string video_start_time_str = getString(raw_alarm, "video_start_time");
        std::string video_end_time_str = getString(raw_alarm, "video_end_time");
        double start_ts = ConfigParser::parseTimestamp(video_start_time_str)-30;
        double end_ts = ConfigParser::parseTimestamp(video_end_time_str)+5;
        
        // Log parsed time window for debugging
        LOG_DEBUG("Reader", "Parsed time window: video_start_time='" + video_start_time_str + 
                 "' -> " + std::to_string(start_ts) +
                 ", video_end_time='" + video_end_time_str + 
                 "' -> " + std::to_string(end_ts) +
                 ", duration=" + std::to_string(end_ts - start_ts) + "s");
        
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
                
                // Log clip time info for debugging
                LOG_DEBUG("Reader", "Clip " + std::to_string(i) + ": moment_time=" + std::to_string(clip.moment_time) +
                         ", duration=" + std::to_string(clip.duration_seconds) +
                         ", start_ts=" + std::to_string(clip.start_timestamp) +
                         ", end_ts=" + std::to_string(clip.end_timestamp) +
                         ", has_time_window=" + (clip.has_time_window ? "true" : "false"));
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
    const std::string message_key = !clip.message_key.empty()
        ? clip.message_key
        : buildMessageKey(clip.serial, clip.record_id);
    const std::string video_key = buildVideoKey(message_key, clip.video_index);
    
    // Reduced logging - only log occasionally
    static std::atomic<int> process_log_counter{0};
    bool should_log = (++process_log_counter % 20 == 0);
    if (should_log) {
        LOG_DEBUG("Reader", "Reader thread " + std::to_string(reader_id) + 
                 " processing video: " + clip.path + 
                 " (video_key='" + video_key + "')");
    }
    
    if (register_message && use_redis_queue_ && output_queue_ && !redis_message.empty()) {
        registerVideoMessage(message_key, redis_message);
    }
    
    VideoReader reader(clip, video_id, reader_options_);
    
    if (!reader.isOpened()) {
        LOG_ERROR("Reader", "Cannot open video (video_key='" + video_key + "'): " + clip.path);
        if (finalize_message) {
            markVideoReadingComplete(message_key);
        }
        return 0;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    int global_frame_number = frame_start_offset;  // Keep for internal tracking if needed
    // Store previous frame for validation (only in debug mode to avoid overhead)
    cv::Mat prev_frame_for_validation;
    int prev_frame_number = -1;
    
    while (!stop_flag_ && reader.readFrame(frame)) {
            // Check debug mode limit first (takes priority over global limit)
            // frame_count is the number of frames already processed, so check BEFORE processing this one
            if (debug_mode_ && max_frames_per_video_ > 0 && frame_count >= max_frames_per_video_) {
                LOG_INFO("Reader", "Reader " + std::to_string(reader_id) + 
                         " stopping after processing " + std::to_string(frame_count) + " frames" +
                         " (max_frames_per_video=" + std::to_string(max_frames_per_video_) + ")");
                break;
            }
            
            
            stats_.frames_read++;
            
            // Get the actual frame position in the video file (accounts for time-based filtering)
            // Note: getActualFramePosition() is fast (just returns cached value from readFrame)
            int actual_frame_position = reader.getActualFramePosition();
            
            // CRITICAL: Validate that consecutive frames are actually different
            // This helps catch bugs in the video reader or frame tracking
            if (debug_mode_ && !prev_frame_for_validation.empty() && !frame.empty() &&
                prev_frame_number >= 0 && actual_frame_position == prev_frame_number + 1) {
                // Check if this frame is identical to the previous frame
                if (prev_frame_for_validation.size() == frame.size() &&
                    prev_frame_for_validation.type() == frame.type()) {
                    cv::Mat diff;
                    cv::absdiff(prev_frame_for_validation, frame, diff);
                    cv::Mat diff_gray;
                    if (diff.channels() > 1) {
                        cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
                    } else {
                        diff_gray = diff;
                    }
                    int non_zero = cv::countNonZero(diff_gray);
                    if (non_zero == 0) {
                        // Duplicate frames are common in some video encodings (e.g., frame rate conversion, B-frames)
                        // This is expected behavior and not an error - the video file itself contains duplicate frames
                        static thread_local int duplicate_frame_count = 0;
                        duplicate_frame_count++;
                        if (duplicate_frame_count <= 10) {
                            std::string warning_msg = std::string("Video contains duplicate frames (expected in some encodings): ") +
                                                      "frame_number=" + std::to_string(actual_frame_position) +
                                                      ", previous_frame_number=" + std::to_string(prev_frame_number) +
                                                      " (video=" + video_key + ")";
                            LOG_WARNING("Reader", warning_msg);
                        } else if (duplicate_frame_count == 11) {
                            LOG_WARNING("Reader", "Suppressing further duplicate frame warnings for this video (this is normal for some encodings)");
                        }
                        // Still process the frame - duplicate frames are valid in video encoding
                    }
                }
            }
            
            // Use actual_frame_position (actual frame number in video) for bin file output
            // CRITICAL: Clone the frame to ensure each FrameData has an independent copy
            // The 'frame' variable is reused in the loop, so we must clone it to prevent
            // all FrameData objects from sharing the same frame data
            // Pre-compute dimensions to avoid repeated frame.cols/rows access
            int frame_width = frame.cols;
            int frame_height = frame.rows;
            cv::Mat frame_clone;
            if (!frame.empty()) {
                frame.copyTo(frame_clone);  // Use copyTo() for explicit deep copy
            }
            
            // Store current frame for next iteration validation
            if (debug_mode_) {
                if (!frame_clone.empty()) {
                    frame_clone.copyTo(prev_frame_for_validation);
                }
                prev_frame_number = actual_frame_position;
            }
            FrameData frame_data(frame_clone, video_id, actual_frame_position, clip.path, 
                                clip.record_id, clip.record_date, clip.serial,
                                message_key, video_key, clip.video_index,
                                clip.has_roi, clip.roi_x1, clip.roi_y1, clip.roi_x2, clip.roi_y2);
            // Store original frame dimensions (before any ROI cropping) - use pre-computed values
            frame_data.original_width = frame_width;
            frame_data.original_height = frame_height;
            frame_data.true_original_width = frame_width;  // True original dimensions
            frame_data.true_original_height = frame_height;
            // Store ROI offset for scaling detections back to true original frame
            frame_data.roi_offset_x = clip.has_roi ? clip.roi_offset_x : 0;
            frame_data.roi_offset_y = clip.has_roi ? clip.roi_offset_y : 0;
            // OPTIMIZATION: Use progressive timeouts to reduce blocking
            // Try non-blocking first (10ms), then medium (50ms), then longer (200ms)
            // This prevents readers from blocking too quickly while still ensuring frames are processed
            if (raw_frame_queue_) {
                // CRITICAL: Use minimal timeouts to reduce blocking (target: 5ms/frame)
                int queue_size = raw_frame_queue_->size();
                int push_timeout1 = (queue_size < raw_frame_queue_->max_size() * 0.9) ? 1 : 5;
                if (!raw_frame_queue_->push(frame_data, push_timeout1)) {
                    if (!raw_frame_queue_->push(frame_data, 5)) {
                        if (!raw_frame_queue_->push(frame_data, 10)) {
                            // Queue still full after timeouts - this indicates a serious bottleneck
                    // Log warning but still push (blocking) to ensure frame is not lost
                    static thread_local int slow_push_count = 0;
                    if (++slow_push_count % 100 == 0) {
                        LOG_WARNING("Reader", "Reader " + std::to_string(reader_id) + 
                                   " queue push slow (" + std::to_string(slow_push_count) + 
                                   " times) - preprocessors may be bottleneck");
                    }
                    raw_frame_queue_->push(frame_data);  // Blocking push as last resort
                        }
                    }
                }
            }
            
            // Removed per-frame timing - too expensive, use batch timing instead
            // Timing is now tracked at batch level in statistics
            
            frame_count++;
            global_frame_number++;
            
            // Removed frequent logging - too expensive in hot path
        }
        
        // Removed logging - too expensive
    
    // Log frame count for this video
    LOG_INFO("Reader", "Reader " + std::to_string(reader_id) + " finished video " + video_key +
             ": frames_read=" + std::to_string(frame_count) +
             ", path=" + clip.path);
    
    // Update total frames read for this message (for validation)
    if (use_redis_queue_ && output_queue_ && !message_key.empty()) {
        std::lock_guard<std::mutex> lock(video_output_mutex_);
        auto it = video_output_status_.find(message_key);
        if (it != video_output_status_.end()) {
            it->second.total_frames_read += frame_count;
        }
    }
    
    // Always mark reading complete when video processing finishes (even if not finalize_message)
    // This ensures reading_completed is set as soon as possible, not waiting for all videos
    // The finalize_message flag is only used to determine if this is the last video in a multi-video message
    if (use_redis_queue_ && output_queue_ && !message_key.empty()) {
        // For multi-video messages, we should only mark complete when ALL videos are done
        // But if this is the last video (finalize_message=true), mark it complete
        // For single-video messages, finalize_message will be true
        if (finalize_message) {
            markVideoReadingComplete(message_key);
        }
    }
    
    return frame_count;
}

void ThreadPool::preprocessorWorker(int worker_id) {
    LOG_INFO("Preprocessor", "Preprocess dispatcher thread " + std::to_string(worker_id) + " started");
    
    // REMOVED pending buffer to ensure strict FIFO ordering
    // Frames must be pushed in order to maintain frame-detection matching
    // If a queue is full, we block (with timeout) rather than buffering out of order
    
    while (!stop_flag_) {
        // Pop a new frame from raw queue
        // CRITICAL: Use 1ms timeout when queue has frames to minimize latency
        // Target is 5ms/frame, so we can't afford 50ms waits
        FrameData raw_frame;
        int raw_pop_timeout = (raw_frame_queue_ && raw_frame_queue_->size() > 0) ? 1 : 5;
        if (!raw_frame_queue_ || !raw_frame_queue_->pop(raw_frame, raw_pop_timeout)) {
            if (stop_flag_) break;
            continue;
        }
        
        // OPTIMIZATION: Use cv::Mat reference counting - frames are only read, never modified
        // Preprocessors create new tensors, they don't modify the original frame
        // Only clone if debug mode (where frames need to be preserved for visualization)
        // This eliminates 3x frame cloning per frame (one per preprocess group)!
        FrameData frame_copy = raw_frame;
        if (debug_mode_ && !raw_frame.frame.empty()) {
            // In debug mode, we need independent copies since frames are kept longer
            cv::Mat frame_clone;
            raw_frame.frame.copyTo(frame_clone);
            frame_copy.frame = frame_clone;
        }
        // Otherwise, cv::Mat reference counting handles sharing safely
        
        // Push to each group's queue in order - use non-blocking push first to avoid delays
        // If one queue is full, we wait (with shorter timeout) rather than blocking too long
        for (size_t g = 0; g < preprocess_groups_.size(); ++g) {
            auto& shared_group_ptr = preprocess_groups_[g];
            if (!shared_group_ptr || !shared_group_ptr->queue) {
                continue;
            }
            auto* shared_group = shared_group_ptr.get();
            
            // CRITICAL: Use minimal timeouts to reduce blocking (target: 5ms/frame)
            // Try 1ms first (queue likely has space), then 5ms, then 10ms
            int queue_size = shared_group->queue ? shared_group->queue->size() : 0;
            int push_timeout1 = (queue_size < shared_group->queue->max_size() * 0.9) ? 1 : 5;
            if (!shared_group->queue->push(frame_copy, push_timeout1)) {
                if (!shared_group->queue->push(frame_copy, 5)) {
                    if (!shared_group->queue->push(frame_copy, 10)) {
                        // Queue still full after timeouts - log warning but continue
                // This indicates a serious bottleneck, but we don't want to lose frames
                static thread_local int slow_push_count = 0;
                if (++slow_push_count % 100 == 0) {
                    LOG_WARNING("Preprocessor", "Dispatcher " + std::to_string(worker_id) + 
                               " queue push slow for group " + std::to_string(g) + 
                               " (" + std::to_string(slow_push_count) + " times)");
                }
                // Final blocking push to ensure frame is not lost
                shared_group->queue->push(frame_copy);
                    }
                }
            }
        }
    }
    
    LOG_INFO("Preprocessor", "Preprocess dispatcher thread " + std::to_string(worker_id) + " finished");
}

void ThreadPool::enginePreprocessWorker(int worker_id) {
    LOG_INFO("Preprocessor", "Shared preprocess thread " + std::to_string(worker_id) + " started");
    
    while (!stop_flag_) {
        if (preprocess_groups_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        
        FrameData frame_data;
        SharedPreprocessGroup* target_group = nullptr;
        const size_t group_count = preprocess_groups_.size();
        size_t start_index = static_cast<size_t>(worker_id) % group_count;
        
        for (size_t attempt = 0; attempt < group_count; ++attempt) {
            size_t idx = (start_index + attempt) % group_count;
            auto* candidate = preprocess_groups_[idx].get();
            if (!candidate || !candidate->queue) {
                continue;
            }
            // CRITICAL: Use 1ms timeout when queue has frames to minimize latency
            // Target is 5ms/frame, so we can't afford 100ms waits
            int queue_size = candidate->queue ? candidate->queue->size() : 0;
            int timeout = (queue_size > 0) ? 1 : ((attempt == 0) ? 5 : 1);
            if (candidate->queue->pop(frame_data, timeout)) {
                target_group = candidate;
                start_index = (idx + 1) % group_count;
                break;
            }
        }
        
        if (!target_group) {
            if (stop_flag_) break;
            continue;
        }
        
        // OPTIMIZATION: Start timing only for actual preprocessing work, not blocking operations
        auto preprocess_start = std::chrono::steady_clock::now();
        
        int true_original_width = frame_data.true_original_width > 0 ? frame_data.true_original_width : frame_data.original_width;
        int true_original_height = frame_data.true_original_height > 0 ? frame_data.true_original_height : frame_data.original_height;
        
        // OPTIMIZATION: Only copy/clone frame if needed (ROI cropping or debug mode)
        // For normal operation, we only need the preprocessed tensor, not the frame
        cv::Mat frame_to_process;
        bool need_frame_copy = target_group->roi_cropping || debug_mode_;
        if (need_frame_copy && !frame_data.frame.empty()) {
            frame_data.frame.copyTo(frame_to_process);
        } else if (!frame_data.frame.empty()) {
            // Use reference for preprocessing (faster, no copy needed)
            frame_to_process = frame_data.frame;
        }
        int cropped_width = frame_data.original_width;
        int cropped_height = frame_data.original_height;
        int roi_offset_x = frame_data.roi_offset_x;
        int roi_offset_y = frame_data.roi_offset_y;
        
        if (target_group->roi_cropping) {
            if (frame_data.has_roi && !frame_to_process.empty() &&
                true_original_width > 0 && true_original_height > 0) {
                float norm_x1 = std::clamp(frame_data.roi_norm_x1, 0.0f, 1.0f);
                float norm_y1 = std::clamp(frame_data.roi_norm_y1, 0.0f, 1.0f);
                float norm_x2 = std::clamp(frame_data.roi_norm_x2, 0.0f, 1.0f);
                float norm_y2 = std::clamp(frame_data.roi_norm_y2, 0.0f, 1.0f);
                if (norm_x2 <= norm_x1) norm_x2 = std::min(1.0f, norm_x1 + 0.001f);
                if (norm_y2 <= norm_y1) norm_y2 = std::min(1.0f, norm_y1 + 0.001f);
                
                int x1 = static_cast<int>(norm_x1 * true_original_width);
                int y1 = static_cast<int>(norm_y1 * true_original_height);
                int x2 = static_cast<int>(norm_x2 * true_original_width);
                int y2 = static_cast<int>(norm_y2 * true_original_height);
                
                x1 = std::max(0, std::min(x1, true_original_width - 1));
                y1 = std::max(0, std::min(y1, true_original_height - 1));
                x2 = std::max(x1 + 1, std::min(x2, true_original_width));
                y2 = std::max(y1 + 1, std::min(y2, true_original_height));
                
                cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                frame_to_process = frame_to_process(roi_rect).clone();
                cropped_width = x2 - x1;
                cropped_height = y2 - y1;
                roi_offset_x = x1;
                roi_offset_y = y1;
            }
        }
        
        auto tensor = target_group->acquireBuffer();
        target_group->preprocessor->preprocessToFloat(frame_to_process, *tensor);
        
        // OPTIMIZATION: Stop timing before blocking operations (queue pushes, mutex locks)
        // This gives accurate measurement of actual preprocessing work, not blocking time
        auto preprocess_end = std::chrono::steady_clock::now();
        auto preprocess_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
        
        // OPTIMIZATION: Batch registerPendingFrame calls to reduce mutex contention
        std::vector<std::pair<std::string, std::string>> pending_frame_registrations;
        pending_frame_registrations.reserve(target_group->engines.size());
        
        for (auto* engine_group : target_group->engines) {
            FrameData processed = frame_data;
            processed.preprocessed_data = tensor;
            // OPTIMIZATION: Only clone frame if debug mode is enabled (frame only used for debug images)
            // In normal operation, engines only need the preprocessed tensor
            if (debug_mode_ && !frame_to_process.empty()) {
            processed.frame = frame_to_process.clone();
            } else {
                // Clear frame to save memory - engines don't need it
                processed.frame = cv::Mat();
            }
            processed.original_width = cropped_width;
            processed.original_height = cropped_height;
            processed.true_original_width = true_original_width;
            processed.true_original_height = true_original_height;
            processed.roi_offset_x = roi_offset_x;
            processed.roi_offset_y = roi_offset_y;
            
            // CRITICAL: Use minimal timeouts to reduce blocking (target: 5ms/frame)
            // Note: Queue push blocking time is NOT included in timing measurement
            int queue_size = engine_group->frame_queue->size();
            int push_timeout1 = (queue_size < engine_group->frame_queue->max_size() * 0.9) ? 1 : 5;
            if (!engine_group->frame_queue->push(processed, push_timeout1)) {
                // Queue might be full, try with longer timeout
                if (!engine_group->frame_queue->push(processed, 5)) {
                    // Last resort: blocking push (should be rare)
                    engine_group->frame_queue->push(processed);
                }
            }
            
            // Collect pending frame registrations to batch them
            if (!processed.message_key.empty()) {
                pending_frame_registrations.emplace_back(processed.message_key, engine_group->engine_name);
            }
        }
        
        // OPTIMIZATION: Batch registerPendingFrame calls in single mutex lock
        // Note: Mutex lock time is NOT included in timing measurement
        if (!pending_frame_registrations.empty() && use_redis_queue_ && output_queue_) {
            std::lock_guard<std::mutex> lock(video_output_mutex_);
            for (const auto& reg : pending_frame_registrations) {
                VideoOutputStatus& status = video_output_status_[reg.first];
                status.pending_counts[reg.second]++;
                status.registered_counts[reg.second]++;
            }
        }
        
        // Update statistics with actual preprocessing time (excluding blocking operations)
        stats_.preprocessor_total_time_ms += preprocess_time_ms;
        stats_.frames_preprocessed += static_cast<long long>(target_group->engines.size());
    }
    
    LOG_INFO("Preprocessor", "Shared preprocess thread " + std::to_string(worker_id) + " finished");
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
    
    // Get batch size for this detector
    int batch_size = engine_group->detectors[detector_id]->getBatchSize();
    
    // CRITICAL: Pending frame to handle frames from different videos
    // When we get a frame from a different video, we save it here and process current batch first
    std::optional<FrameData> pending_frame;
    
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
            
            // CRITICAL FIX: Allow frames from different videos in same batch to avoid deadlock
            // The original logic required same video, but this causes deadlock when videos switch
            // Prod branch doesn't enforce same video - it just collects batch_size frames
            // We'll track video_key for logging, but allow mixed videos
            std::string batch_video_key;  // Track for logging, but allow mixed videos
            
            // CRITICAL: Match prod branch - simple approach that works!
            // Prod branch: Simple loop with 100ms timeout, NO partial batch processing
            // Only processes when batch_tensors.size() == batch_size (exactly full batch)
            // This works because with TensorRT fixed batch_size, partial batches waste GPU time
            auto last_wait_log = std::chrono::steady_clock::now();
            
            // CRITICAL: Match prod branch - no need to track video finishing
            // We only process full batches, so we just skip different video frames and continue
            
            // Collect batch_size frames (match prod branch exactly)
            while (static_cast<int>(batch_tensors.size()) < batch_size && !stop_flag_) {
                // First, check if we have a pending frame from previous iteration (video switching)
                if (pending_frame.has_value()) {
                    frame_data = pending_frame.value();
                    pending_frame.reset();
                    
                    const std::string& pending_video_key = frame_data.video_key;
                    
                    // CRITICAL FIX: Allow collecting batches from different videos to avoid getting stuck
                    // Don't skip frames from different videos - allow them in the batch
                    // This ensures we can always collect full batches even when videos switch
                    if (batch_video_key.empty()) {
                        batch_video_key = pending_video_key;
                    }
                    // If different video, we still use the frame (don't skip it)
                } else {
                    // CRITICAL: Use 1ms timeout when queue has frames to minimize latency
                    // Target is 5ms/frame, so we can't afford 100ms waits
                    // With queues 422-500/500 full, we should collect frames immediately
                    int queue_size = engine_group->frame_queue->size();
                    int pop_timeout = (queue_size > 0) ? 1 : 5;
                    if (!engine_group->frame_queue->pop(frame_data, pop_timeout)) {
                        if (stop_flag_) break;
                        
                        // Log waiting status periodically (every 5 seconds) - match prod branch
                            auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_wait_log).count();
                        if (elapsed >= 5) {
                            LOG_INFO("Detector", "[WAITING] Detector " + std::to_string(detector_id) + 
                                     " (" + engine_group->engine_name + ") waiting for frames... " +
                                     "(Queue size: " + std::to_string(engine_group->frame_queue->size()) + ")");
                            last_wait_log = now;
                        }
                        continue;  // Just continue waiting - prod branch doesn't process partial batches!
                    }
                }
                
                // OPTIMIZATION: Use pre-computed video_key from frame_data instead of recomputing
                // video_key is already computed in reader and stored in FrameData
                const std::string& current_video_key = frame_data.video_key;
                
                // CRITICAL FIX: Allow collecting batches from different videos to avoid getting stuck
                // The original logic required all frames from same video, but this causes deadlock when videos switch
                // Prod branch doesn't enforce same video - it just collects batch_size frames
                // We'll allow mixed videos in a batch, but prefer same video when possible
                if (batch_video_key.empty()) {
                    batch_video_key = current_video_key;
                } else if (current_video_key != batch_video_key) {
                    // Different video - but don't skip it! Allow it in the batch to avoid deadlock
                    // This ensures we can always collect full batches even when videos switch
                    // The postprocessor can handle frames from different videos
                }
                
                // OPTIMIZATION: Cache output paths to avoid expensive string operations
                // Use (video_key, engine_name) as cache key - video_key already computed and stored in frame_data
                std::string cache_key = frame_data.video_key + "_" + engine_group->engine_name;
                std::string output_path;
                {
                    std::lock_guard<std::mutex> lock(path_cache_mutex_);
                    auto it = output_path_cache_.find(cache_key);
                    if (it != output_path_cache_.end()) {
                        output_path = it->second;
                    } else {
                        // Cache miss - generate path and cache it
                std::string serial_value = serialPart(frame_data.serial, frame_data.message_key, frame_data.video_id);
                std::string record_value = recordPart(frame_data.record_id, frame_data.message_key, frame_data.video_id);
                        output_path = generateOutputPath(
                    serial_value, record_value, frame_data.record_date, engine_group->engine_name,
                    frame_data.video_index);
                        output_path_cache_[cache_key] = output_path;
                    }
                }
                
                // CRITICAL: Get preprocessed tensor BEFORE clearing frame
                // If preprocessed_data doesn't exist, we need frame_data.frame for preprocessing
                std::shared_ptr<std::vector<float>> tensor = frame_data.preprocessed_data;
                if (!tensor) {
                    auto shared_group = engine_group->shared_preprocess;
                    tensor = shared_group ? shared_group->acquireBuffer()
                                          : std::make_shared<std::vector<float>>(engine_group->tensor_elements);
                    auto* preproc = shared_group ? shared_group->preprocessor.get()
                                                 : engine_group->preprocessor.get();
                    preproc->preprocessToFloat(frame_data.frame, *tensor);
                }
                
                // OPTIMIZATION: Clear frame AFTER we've ensured we have the preprocessed tensor
                // This saves memory while ensuring we don't lose the frame before preprocessing
                if (!debug_mode_) {
                    frame_data.frame = cv::Mat();
                }
                
                // OPTIMIZATION: Use shared_ptr directly - the reference counting ensures the buffer
                // stays alive until the batch is processed. The preprocessor releases the buffer
                // back to the pool, but the shared_ptr keeps it alive. No deep copy needed.
                // Deep copy was causing massive overhead (4.8MB per frame * 16 = 76.8MB per batch).
                batch_tensors.push_back(tensor);
                batch_output_paths.push_back(output_path);
                batch_frame_numbers.push_back(frame_data.frame_number);
                batch_original_widths.push_back(frame_data.original_width);
                batch_original_heights.push_back(frame_data.original_height);
                batch_true_original_widths.push_back(frame_data.true_original_width);
                batch_true_original_heights.push_back(frame_data.true_original_height);
                batch_roi_offset_x.push_back(frame_data.roi_offset_x);
                batch_roi_offset_y.push_back(frame_data.roi_offset_y);
                
                // Store frame and video_id for debug mode
                // CRITICAL: Clone the frame to ensure it matches the tensor used for inference
                // frame_data.frame is the preprocessed frame (after ROI cropping if applicable)
                // This should match the tensor which was created from the same preprocessed frame
                // IMPORTANT: Ensure we create a deep copy that's independent of the source
                if (debug_mode_) {
                    // CRITICAL: Create a truly independent deep copy of the frame
                    // Use copyTo() to ensure we get a completely independent Mat with its own data buffer
                    cv::Mat frame_copy;
                    if (!frame_data.frame.empty()) {
                        frame_data.frame.copyTo(frame_copy);
                    } else {
                        frame_copy = cv::Mat();
                        LOG_ERROR("Detector", "CRITICAL: Empty frame at frame_number=" + 
                                 std::to_string(frame_data.frame_number) + 
                                 " (engine=" + engine_group->engine_name + 
                                 ", detector=" + std::to_string(detector_id) + ")");
                    }
                    
                    // OPTIMIZATION: Removed expensive duplicate frame validation during batch collection
                    // This was doing absdiff + cvtColor + countNonZero for every frame, which is very slow
                    // Duplicate frame detection can be done later if needed, or removed entirely
                    // The frame_number check is sufficient for most cases
                    
                    batch_frames.push_back(frame_copy);
                    batch_video_ids.push_back(frame_data.video_id);
                    
                    // Note: Frames may arrive out of order due to multiple detectors pulling from the same queue.
                    // This is expected and will be fixed by sorting the batch before processing.
                    // Only log at debug level since sorting will correct the order.
                    if (batch_frame_numbers.size() > 1 && 
                        frame_data.frame_number < batch_frame_numbers[batch_frame_numbers.size() - 2]) {
                        std::string debug_msg = std::string("Frame number out of order when collecting batch (will be sorted): ") +
                                                "previous=" + std::to_string(batch_frame_numbers[batch_frame_numbers.size() - 2]) +
                                                ", current=" + std::to_string(frame_data.frame_number) +
                                                " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + 
                                                ", video_key=" + current_video_key + ")";
                        LOG_DEBUG("Detector", debug_msg);
                    }
                }
                batch_message_keys.push_back(frame_data.message_key);
                batch_video_indices.push_back(frame_data.video_index);
                batch_serials.push_back(frame_data.serial);
                batch_record_ids.push_back(frame_data.record_id);
                batch_record_dates.push_back(frame_data.record_date);
            }
            
            // CRITICAL: Match prod branch EXACTLY - ONLY process full batches!
            // Prod branch strategy: NEVER process partial batches, even when videos finish
            // This is the key to achieving 200 FPS - partial batches waste GPU time
            // If a video finishes with a partial batch, we drop those frames and continue
            // The alternative (processing partial batches) causes 3-5 FPS instead of 200 FPS
            bool should_process = false;
            if (static_cast<int>(batch_tensors.size()) == batch_size) {
                // Full batch - always process (matches prod branch)
                should_process = true;
            } else if (!batch_tensors.empty() && static_cast<int>(batch_tensors.size()) < batch_size) {
                // Partial batch - DROP IT and restart batch collection
                // CRITICAL: We dropped a partial batch, so we need to restart batch collection
                // Clear everything and go back to the batch collection loop
                LOG_DEBUG("Detector", "Dropping partial batch of " + std::to_string(batch_tensors.size()) + 
                         " frames (only processing full batches of " + std::to_string(batch_size) + 
                         " frames) (engine=" + engine_group->engine_name + 
                         ", detector=" + std::to_string(detector_id) + ")");
                // Clear the partial batch
                batch_tensors.clear();
                batch_output_paths.clear();
                batch_frame_numbers.clear();
                batch_original_widths.clear();
                batch_original_heights.clear();
                batch_true_original_widths.clear();
                batch_true_original_heights.clear();
                batch_roi_offset_x.clear();
                batch_roi_offset_y.clear();
                batch_message_keys.clear();
                batch_video_indices.clear();
                batch_serials.clear();
                batch_record_ids.clear();
                batch_record_dates.clear();
                if (debug_mode_) {
                    batch_frames.clear();
                    batch_video_ids.clear();
                }
                batch_video_key.clear();
                // CRITICAL: Don't continue here - restart batch collection by going back to the while loop
                // The continue statement will go back to the outer while loop, which will restart batch collection
            }
            
            // If we get here with should_process=false, it means we had a partial batch that was dropped
            // We need to restart batch collection, so continue to the outer loop
            if (!should_process) {
                continue;
            }
            
            if (should_process && !batch_tensors.empty()) {
                int actual_batch_count = static_cast<int>(batch_tensors.size());
                
                // CRITICAL: Sort batch by frame number to ensure frames are processed in order
                // This fixes the issue where multiple detectors pull frames from the same queue
                // and frames arrive out of order due to parallel processing
                // OPTIMIZATION: Quick check first - if frames are already in order, skip expensive sorting
                bool needs_sorting = false;
                if (actual_batch_count > 1) {
                    // Quick check: are frames already in order?
                    for (int i = 1; i < actual_batch_count; ++i) {
                        if (batch_frame_numbers[i] < batch_frame_numbers[i-1]) {
                            needs_sorting = true;
                            break;
                        }
                    }
                }
                
                if (needs_sorting && actual_batch_count > 1) {
                    // Create index vector and sort by frame number
                    std::vector<size_t> indices(actual_batch_count);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::sort(indices.begin(), indices.end(), 
                        [&batch_frame_numbers](size_t a, size_t b) {
                            return batch_frame_numbers[a] < batch_frame_numbers[b];
                        });
                    
                    // Reorder all batch vectors according to sorted indices
                    if (needs_sorting) {
                        // Create temporary vectors
                        std::vector<std::shared_ptr<std::vector<float>>> sorted_tensors(actual_batch_count);
                        std::vector<std::string> sorted_output_paths(actual_batch_count);
                        std::vector<int> sorted_frame_numbers(actual_batch_count);
                        std::vector<int> sorted_original_widths(actual_batch_count);
                        std::vector<int> sorted_original_heights(actual_batch_count);
                        std::vector<int> sorted_true_original_widths(actual_batch_count);
                        std::vector<int> sorted_true_original_heights(actual_batch_count);
                        std::vector<int> sorted_roi_offset_x(actual_batch_count);
                        std::vector<int> sorted_roi_offset_y(actual_batch_count);
                        std::vector<std::string> sorted_message_keys(actual_batch_count);
                        std::vector<int> sorted_video_indices(actual_batch_count);
                        std::vector<std::string> sorted_serials(actual_batch_count);
                        std::vector<std::string> sorted_record_ids(actual_batch_count);
                        std::vector<std::string> sorted_record_dates(actual_batch_count);
                        std::vector<cv::Mat> sorted_frames;
                        std::vector<int> sorted_video_ids;
                        
                        for (int i = 0; i < actual_batch_count; ++i) {
                            size_t src_idx = indices[i];
                            sorted_tensors[i] = batch_tensors[src_idx];
                            sorted_output_paths[i] = batch_output_paths[src_idx];
                            sorted_frame_numbers[i] = batch_frame_numbers[src_idx];
                            sorted_original_widths[i] = batch_original_widths[src_idx];
                            sorted_original_heights[i] = batch_original_heights[src_idx];
                            sorted_true_original_widths[i] = batch_true_original_widths[src_idx];
                            sorted_true_original_heights[i] = batch_true_original_heights[src_idx];
                            sorted_roi_offset_x[i] = batch_roi_offset_x[src_idx];
                            sorted_roi_offset_y[i] = batch_roi_offset_y[src_idx];
                            sorted_message_keys[i] = batch_message_keys[src_idx];
                            sorted_video_indices[i] = batch_video_indices[src_idx];
                            sorted_serials[i] = batch_serials[src_idx];
                            sorted_record_ids[i] = batch_record_ids[src_idx];
                            sorted_record_dates[i] = batch_record_dates[src_idx];
                            
                            // CRITICAL: Ensure batch_frames and batch_video_ids are sorted in the same order
                            // They should have the same size as other batch vectors if in debug mode
                            // OPTIMIZATION: Frames are already cloned when added to batch_frames (line 1500),
                            // so we can just move them during sorting - no need to clone again
                            if (debug_mode_) {
                                if (src_idx < batch_frames.size() && src_idx < batch_video_ids.size()) {
                                    // Move frame (already cloned, so safe to move)
                                    sorted_frames.push_back(std::move(batch_frames[src_idx]));
                                    sorted_video_ids.push_back(batch_video_ids[src_idx]);
                                } else {
                                    // Size mismatch - this shouldn't happen, but handle gracefully
                                    std::string error_msg = std::string("CRITICAL: batch_frames size mismatch during sorting! ") +
                                                           "src_idx=" + std::to_string(src_idx) +
                                                           ", batch_frames.size()=" + std::to_string(batch_frames.size()) +
                                                           ", batch_video_ids.size()=" + std::to_string(batch_video_ids.size()) +
                                                           ", actual_batch_count=" + std::to_string(actual_batch_count) +
                                                           " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")";
                                    LOG_ERROR("Detector", error_msg);
                                    // Push empty frame to maintain alignment (will cause issues but prevent crash)
                                    sorted_frames.push_back(cv::Mat());
                                    sorted_video_ids.push_back(-1);
                                }
                            }
                        }
                        
                        // CRITICAL: Validate that sorted vectors have correct sizes BEFORE moving
                        // If validation fails, skip sorting and use original batch
                        bool sorting_valid = true;
                        if (sorted_tensors.size() != static_cast<size_t>(actual_batch_count)) {
                            std::string error_msg = std::string("CRITICAL: sorted_tensors size mismatch! ") +
                                                   "size=" + std::to_string(sorted_tensors.size()) +
                                                   ", expected=" + std::to_string(actual_batch_count) +
                                                   " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")";
                            LOG_ERROR("Detector", error_msg);
                            sorting_valid = false;
                        }
                        
                        size_t expected_frames = debug_mode_ ? static_cast<size_t>(actual_batch_count) : 0;
                        if (sorted_frames.size() != expected_frames) {
                            std::string error_msg = std::string("CRITICAL: sorted_frames size mismatch! ") +
                                                   "size=" + std::to_string(sorted_frames.size()) +
                                                   ", expected=" + std::to_string(expected_frames) +
                                                   ", debug_mode=" + std::to_string(debug_mode_) +
                                                   " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")";
                            LOG_ERROR("Detector", error_msg);
                            sorting_valid = false;
                        }
                        
                        // Only replace original vectors with sorted ones if validation passed
                        if (!sorting_valid) {
                            LOG_WARNING("Detector", "Skipping sorted batch due to validation failure, using original batch order");
                            // Don't move sorted vectors, keep original batch
                        } else {
                            // Replace original vectors with sorted ones
                            batch_tensors = std::move(sorted_tensors);
                            batch_output_paths = std::move(sorted_output_paths);
                            batch_frame_numbers = std::move(sorted_frame_numbers);
                            batch_original_widths = std::move(sorted_original_widths);
                            batch_original_heights = std::move(sorted_original_heights);
                            batch_true_original_widths = std::move(sorted_true_original_widths);
                            batch_true_original_heights = std::move(sorted_true_original_heights);
                            batch_roi_offset_x = std::move(sorted_roi_offset_x);
                            batch_roi_offset_y = std::move(sorted_roi_offset_y);
                            batch_message_keys = std::move(sorted_message_keys);
                            batch_video_indices = std::move(sorted_video_indices);
                            batch_serials = std::move(sorted_serials);
                            batch_record_ids = std::move(sorted_record_ids);
                            batch_record_dates = std::move(sorted_record_dates);
                            if (debug_mode_) {
                                batch_frames = std::move(sorted_frames);
                                batch_video_ids = std::move(sorted_video_ids);
                            }
                            
                            std::string sort_msg = std::string("Sorted batch by frame number (engine=") +
                                                   engine_group->engine_name + ", detector=" + std::to_string(detector_id) + 
                                                   ", frames=[" + std::to_string(batch_frame_numbers[0]);
                            if (actual_batch_count > 1) {
                                sort_msg += ".." + std::to_string(batch_frame_numbers[actual_batch_count-1]) + "])";
                            } else {
                                sort_msg += "])";
                            }
                            LOG_DEBUG("Detector", sort_msg);
                        }
                    }
                }
                
                // Validate frame numbers are in order (should always be true after sorting)
                for (size_t i = 1; i < batch_frame_numbers.size(); ++i) {
                    if (batch_frame_numbers[i] < batch_frame_numbers[i-1]) {
                        std::string error_msg = "Frame order violation in batch after sorting: frame[" + 
                                                std::to_string(i-1) + "]=" + std::to_string(batch_frame_numbers[i-1]) +
                                                ", frame[" + std::to_string(i) + "]=" + std::to_string(batch_frame_numbers[i]) +
                                                " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")";
                        LOG_ERROR("Detector", error_msg);
                        break;
                    }
                }
                
                // Batch processing is normal operation - no need to log unless in debug mode
                
                auto batch_start = std::chrono::steady_clock::now();
                bool success = false;
                
                // CRITICAL: Validate all batch vectors have the same size after sorting
                bool batch_valid = true;
                if (actual_batch_count > 0) {
                    size_t expected_size = static_cast<size_t>(actual_batch_count);
                    if (batch_tensors.size() != expected_size ||
                        batch_output_paths.size() != expected_size ||
                        batch_frame_numbers.size() != expected_size ||
                        batch_original_widths.size() != expected_size ||
                        batch_original_heights.size() != expected_size ||
                        batch_true_original_widths.size() != expected_size ||
                        batch_true_original_heights.size() != expected_size ||
                        batch_roi_offset_x.size() != expected_size ||
                        batch_roi_offset_y.size() != expected_size ||
                        batch_message_keys.size() != expected_size ||
                        batch_video_indices.size() != expected_size ||
                        batch_serials.size() != expected_size ||
                        batch_record_ids.size() != expected_size ||
                        batch_record_dates.size() != expected_size) {
                        std::string error_msg = "CRITICAL: Batch vector size mismatch after sorting! expected=" + 
                                                std::to_string(expected_size) +
                                                ", tensors=" + std::to_string(batch_tensors.size()) +
                                                ", paths=" + std::to_string(batch_output_paths.size()) +
                                                ", frames=" + std::to_string(batch_frame_numbers.size()) +
                                                " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")";
                        LOG_ERROR("Detector", error_msg);
                        batch_valid = false;
                    }
                }
                
                if (!batch_valid) {
                    // Skip processing this invalid batch
                    {
                        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                        stats_.frames_failed[engine_id] += actual_batch_count;
                    }
                    LOG_ERROR("Detector", "Skipping invalid batch (size mismatch) for " + std::to_string(actual_batch_count) + " frames");
                    // Clear batch vectors and continue to next iteration
                    batch_tensors.clear();
                    batch_output_paths.clear();
                    batch_frame_numbers.clear();
                    batch_original_widths.clear();
                    batch_original_heights.clear();
                    batch_true_original_widths.clear();
                    batch_true_original_heights.clear();
                    batch_roi_offset_x.clear();
                    batch_roi_offset_y.clear();
                    batch_message_keys.clear();
                    batch_video_indices.clear();
                    batch_serials.clear();
                    batch_record_ids.clear();
                    batch_record_dates.clear();
                    if (debug_mode_) {
                        batch_frames.clear();
                        batch_video_ids.clear();
                    }
                    continue;  // Continue to next batch collection
                }
                
                // Use the same processing path for both debug and normal mode
                // Get raw inference output and push to post-processing queue
                std::vector<std::vector<float>> raw_outputs;
                success = engine_group->detectors[detector_id]->getRawInferenceOutput(
                    batch_tensors, raw_outputs
                );
                
                if (!success) {
                    LOG_ERROR("Detector", "getRawInferenceOutput failed for batch_size=" + 
                             std::to_string(actual_batch_count) + 
                             " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")");
                } else if (raw_outputs.empty()) {
                    LOG_ERROR("Detector", "getRawInferenceOutput returned empty raw_outputs for batch_size=" + 
                             std::to_string(actual_batch_count) + 
                             " (engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")");
                    success = false;
                }
                
                // OPTIMIZATION: Stop timing before blocking operations (queue push)
                // This gives accurate measurement of actual inference work, not blocking time
                auto batch_end = std::chrono::steady_clock::now();
                auto batch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
                
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
                    
                    // Add debug mode fields if in debug mode
                    if (debug_mode_) {
                        task.frames = batch_frames;  // Copy frames for debug image saving
                        task.video_ids = batch_video_ids;
                        task.serials = batch_serials;
                        task.record_ids = batch_record_ids;
                        task.record_dates = batch_record_dates;
                    }
                    
                    // Note: registerPendingFrame is already called in preprocessorWorker (line 907)
                    // No need to call it again here to avoid double-counting
                    
                    // Push to post-processing queue (non-blocking)
                    // Note: Queue push blocking time is NOT included in timing measurement
                    if (!postprocess_queue_ || !postprocess_queue_->push(task)) {
                        LOG_ERROR("Detector", "Failed to push batch to post-processing queue");
                        success = false;
                    }
                }
                
                if (success) {
                    int actual_batch_size = static_cast<int>(batch_tensors.size());
                    // CRITICAL: Only track inference time here, NOT frame count
                    // Frame count is tracked in postprocessor after successful postprocessing
                    // This prevents double-counting and ensures accurate statistics
                    {
                        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                        // Only update timing statistics here, not frame counts
                        // Frame counts are updated in postprocessor after successful processing
                        stats_.engine_total_time_ms[engine_id] += batch_time_ms;
                        stats_.engine_frame_count[engine_id] += actual_batch_size;  // For timing average only
                    }
                    processed_count += actual_batch_size;
                    
                    // Removed frequent logging - too expensive in hot path
                } else {
                    {
                        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                        stats_.frames_failed[engine_id] += actual_batch_count;
                    }
                    LOG_ERROR("Detector", "Batch detection failed for " + std::to_string(actual_batch_count) + " frames " +
                             "(engine=" + engine_group->engine_name + ", detector=" + std::to_string(detector_id) + ")");
                    // Mark failed frames as processed to avoid blocking message push
                    if (use_redis_queue_) {
                        for (size_t b = 0; b < batch_message_keys.size() && b < batch_video_indices.size(); ++b) {
                            markFrameProcessed(batch_message_keys[b], engine_group->engine_name, "", batch_video_indices[b]);
                        }
                    }
                }
            }
        } else {
            // batch_size = 1: process frames one at a time (original behavior)
            if (!engine_group->frame_queue->pop(frame_data, 100)) {
                if (stop_flag_) break;
                
                // Removed waiting log - too expensive, queue size check involves mutex
                continue;
            }
            
            // OPTIMIZATION: Cache output paths to avoid expensive string operations
            std::string cache_key = frame_data.video_key + "_" + engine_group->engine_name;
            std::string output_path;
            {
                std::lock_guard<std::mutex> lock(path_cache_mutex_);
                auto it = output_path_cache_.find(cache_key);
                if (it != output_path_cache_.end()) {
                    output_path = it->second;
                } else {
                    // Cache miss - generate path and cache it
            std::string serial_value = serialPart(frame_data.serial, frame_data.message_key, frame_data.video_id);
            std::string record_value = recordPart(frame_data.record_id, frame_data.message_key, frame_data.video_id);
                    output_path = generateOutputPath(
                serial_value,
                record_value,
                frame_data.record_date,
                engine_group->engine_name,
                        frame_data.video_index);
                    output_path_cache_[cache_key] = output_path;
                }
            }
            
            // Run YOLO detection on GPU (TensorRT) - frame will be preprocessed by detector
            // Note: Multiple detector threads may write to the same file (same video+engine)
            // File writes are protected by per-file mutex inside writeDetectionsToFile
            auto detect_start = std::chrono::steady_clock::now();
            std::shared_ptr<std::vector<float>> tensor = frame_data.preprocessed_data;
            if (!tensor) {
                auto shared_group = engine_group->shared_preprocess;
                tensor = shared_group ? shared_group->acquireBuffer()
                                      : std::make_shared<std::vector<float>>(engine_group->tensor_elements);
                auto* preproc = shared_group ? shared_group->preprocessor.get()
                                             : engine_group->preprocessor.get();
                preproc->preprocessToFloat(frame_data.frame, *tensor);
            }
            
            bool success = false;
            
            // Use the same processing path for both debug and normal mode
            // Get raw inference output and push to post-processing queue
            std::vector<std::shared_ptr<std::vector<float>>> single_input = {tensor};
            std::vector<std::vector<float>> raw_outputs;
            success = engine_group->detectors[detector_id]->getRawInferenceOutput(
                single_input, raw_outputs
            );
            
            // OPTIMIZATION: Stop timing before blocking operations (queue push)
            // This gives accurate measurement of actual inference work, not blocking time
            auto detect_end = std::chrono::steady_clock::now();
            auto detect_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();
            
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
                
                // Add debug mode fields if in debug mode
                if (debug_mode_) {
                    task.frames = {frame_data.frame.clone()};  // Copy frame for debug image saving
                    task.video_ids = {frame_data.video_id};
                    task.serials = {frame_data.serial};
                    task.record_ids = {frame_data.record_id};
                    task.record_dates = {frame_data.record_date};
                }
                
                // Note: registerPendingFrame is already called in preprocessorWorker (line 907)
                // No need to call it again here to avoid double-counting
                
                // Push to post-processing queue (non-blocking)
                // Note: Queue push blocking time is NOT included in timing measurement
                if (!postprocess_queue_ || !postprocess_queue_->push(task)) {
                    LOG_ERROR("Detector", "Failed to push frame to post-processing queue");
                    success = false;
                }
            }
            
            if (success) {
                // CRITICAL: Only track inference time here, NOT frame count
                // Frame count is tracked in postprocessor after successful postprocessing
                // This prevents double-counting and ensures accurate statistics
                {
                    std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                    // Only update timing statistics here, not frame counts
                    // Frame counts are updated in postprocessor after successful processing
                    stats_.engine_total_time_ms[engine_id] += detect_time_ms;
                    stats_.engine_frame_count[engine_id]++;  // For timing average only
                }
                processed_count++;
                
                // Removed expensive logging - string operations accumulate overhead over time
            } else {
                // Mark failed frame as processed to avoid blocking message push
                if (use_redis_queue_ && !frame_data.message_key.empty()) {
                    markFrameProcessed(frame_data.message_key, engine_group->engine_name, "", frame_data.video_index);
                }
                {
                    std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                    stats_.frames_failed[engine_id]++;
                }
                LOG_ERROR("Detector", "Detection failed: " + output_path);
            }

            // Note: markFrameProcessed is now called in postprocessWorker after file writing
            
            // Also log every 50 frames with summary
            if (processed_count % 50 == 0) {
                // Removed expensive logging - string operations accumulate overhead
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
    int skipped_count = 0;
    
    while (!stop_flag_) {
        if (!postprocess_queue_ || !postprocess_queue_->pop(task, 100)) {
            if (stop_flag_) break;
            continue;
        }
        
        if (!task.detector || task.raw_outputs.empty() || task.output_paths.empty()) {
            LOG_ERROR("PostProcess", "Invalid post-processing task: detector=" + 
                     std::to_string(task.detector != nullptr) + 
                     ", outputs=" + std::to_string(task.raw_outputs.size()) + 
                     ", paths=" + std::to_string(task.output_paths.size()));
            skipped_count++;
            continue;
        }
        
        // Validate task has all required fields
        if (task.raw_outputs.size() != task.output_paths.size() || 
            task.raw_outputs.size() != task.frame_numbers.size()) {
            LOG_ERROR("PostProcess", "Task size mismatch: outputs=" + std::to_string(task.raw_outputs.size()) + 
                     ", paths=" + std::to_string(task.output_paths.size()) + 
                     ", frames=" + std::to_string(task.frame_numbers.size()));
            skipped_count++;
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
            
            // Save debug images if in debug mode (before scaling to original coordinates)
            // Debug images should be drawn on the preprocessed frame (resized scale)
            // OPTIMIZATION: Use faster resize and avoid unnecessary clones where possible
            if (debug_mode_ && b < task.frames.size() && !task.frames[b].empty()) {
                // Get the preprocessor for this engine (needed for addPadding)
                // We need to find the engine group to get the preprocessor
                // For now, we'll use a simpler approach: just draw on the frame directly
                // The detections are in normalized [0,1] coordinates relative to the preprocessed frame
                const cv::Mat& debug_frame = task.frames[b];  // Use reference to avoid clone if possible
                
                // Add padding to match the model input size
                // We need to get the input size from the detector
                int input_w = task.detector->getInputWidth();
                int input_h = task.detector->getInputHeight();
                
                // Calculate padding (same logic as Preprocessor::addPadding - centered)
                int orig_w = debug_frame.cols;
                int orig_h = debug_frame.rows;
                float scale = std::min(static_cast<float>(input_w) / orig_w, static_cast<float>(input_h) / orig_h);
                int new_w = static_cast<int>(orig_w * scale);
                int new_h = static_cast<int>(orig_h * scale);
                
                // Calculate padding to center the image (same as Preprocessor::addPadding)
                int pad_w = (input_w - new_w) / 2;
                int pad_h = (input_h - new_h) / 2;
                
                // OPTIMIZATION: Create padded image directly and resize into it (avoids intermediate Mat)
                cv::Mat padded = cv::Mat::zeros(input_h, input_w, debug_frame.type());
                cv::Mat roi = padded(cv::Rect(pad_w, pad_h, new_w, new_h));
                // Use INTER_AREA for downscaling (faster) or INTER_LINEAR for upscaling
                cv::InterpolationFlags interp = (new_w < orig_w || new_h < orig_h) ? cv::INTER_AREA : cv::INTER_LINEAR;
                cv::resize(debug_frame, roi, cv::Size(new_w, new_h), 0, 0, interp);
                
                // Draw detections (they're in normalized [0,1] coordinates relative to padded frame)
                task.detector->drawDetections(padded, detections);
                
                // Generate debug image path
                if (b < task.serials.size() && b < task.record_ids.size() && b < task.record_dates.size() && 
                    b < task.message_keys.size() && b < task.video_ids.size() && b < task.video_indices.size()) {
                    std::string serial_part = serialPart(task.serials[b], task.message_keys[b], task.video_ids[b]);
                    std::string record_part = recordPart(task.record_ids[b], task.message_keys[b], task.video_ids[b]);
                    std::string date_part = formatDate(task.record_dates[b]);
                    
                    std::filesystem::path debug_dir = std::filesystem::path(output_dir_) / "debug_images" /
                                                       task.engine_name / date_part /
                                                       (serial_part + "_" + record_part + "_v" + std::to_string(task.video_indices[b]));
                    // OPTIMIZATION: create_directories is idempotent and fast if directory exists
                    std::filesystem::create_directories(debug_dir);
                    
                    std::string image_filename = "frame_" + 
                        std::to_string(task.frame_numbers[b]) + "_" + 
                        task.engine_name + ".jpg";
                    std::filesystem::path image_path = debug_dir / image_filename;
                    
                    // OPTIMIZATION: Use JPEG quality 95 (default) but compress more for faster writes
                    // Use imencode + file write for better control, but for now keep imwrite for simplicity
                    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};  // Slightly lower quality for faster writes
                    if (cv::imwrite(image_path.string(), padded, compression_params)) {
                        LOG_DEBUG("PostProcess", "Saved debug image: frame=" + 
                                 std::to_string(task.frame_numbers[b]) + 
                                 ", detections=" + std::to_string(detections.size()) +
                                 ", path=" + image_path.string());
                    } else {
                        LOG_ERROR("PostProcess", "Failed to save debug image: " + image_path.string());
                    }
                }
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
            bool write_success = task.detector->writeDetectionsToFile(detections, task.output_paths[b], task.frame_numbers[b]);
            if (!write_success) {
                LOG_ERROR("PostProcess", "Failed to write detections for frame " + std::to_string(task.frame_numbers[b]));
                all_success = false;
            }
            
            // Mark frame as processed (for Redis message tracking) - always call this to decrement pending count
            // Even if write fails, we need to decrement pending count to avoid blocking message push
            if (use_redis_queue_ && b < task.message_keys.size() && b < task.video_indices.size()) {
                std::string output_path = write_success ? task.output_paths[b] : "";
                markFrameProcessed(task.message_keys[b], task.engine_name, output_path, task.video_indices[b]);
            }
        }
        
        auto postprocess_end = std::chrono::steady_clock::now();
        auto postprocess_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            postprocess_end - postprocess_start).count();
        
        // Update postprocessing statistics
        int frames_in_task = static_cast<int>(task.raw_outputs.size());
        stats_.frames_postprocessed += frames_in_task;
        stats_.postprocessor_total_time_ms += postprocess_time_ms;
        
        // CRITICAL: Only count frames as "detected" after successful postprocessing
        // This is the single source of truth for frame counts
        // Note: engine_total_time_ms and engine_frame_count are already updated in detector
        // for inference timing purposes - we don't update them here to avoid double-counting
        if (all_success) {
            {
                std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                if (task.engine_id >= 0 && task.engine_id < static_cast<int>(stats_.frames_detected.size())) {
                    // This is where frames are actually counted as successfully processed
                    stats_.frames_detected[task.engine_id] += frames_in_task;
                }
            }
            processed_count += frames_in_task;
        } else {
            {
                std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                if (task.engine_id >= 0 && task.engine_id < static_cast<int>(stats_.frames_failed.size())) {
                    stats_.frames_failed[task.engine_id] += frames_in_task;
                }
            }
        }
    }
    
    LOG_INFO("PostProcess", "Post-processing thread " + std::to_string(worker_id) + 
             " finished (" + std::to_string(processed_count) + " frames processed" +
             (skipped_count > 0 ? ", " + std::to_string(skipped_count) + " tasks skipped" : "") + ")");
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
        
        // Calculate windowed FPS (FPS over the last monitoring period)
        long long window_read_fps = 0;
        long long window_preprocess_fps = 0;
        double window_seconds = 5.0;  // Default to 5 seconds
        
        {
            std::lock_guard<std::mutex> lock(prev_stats_mutex_);
            auto window_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - prev_monitor_time_).count();
            if (window_duration > 0) {
                window_seconds = window_duration / 1000.0;
                long long read_diff = total_read - prev_frames_read_;
                long long preprocess_diff = total_preprocessed - prev_frames_preprocessed_;
                window_read_fps = static_cast<long long>(read_diff / window_seconds);
                window_preprocess_fps = static_cast<long long>(preprocess_diff / window_seconds);
            }
        }
        
        std::ostringstream stats_oss;
        stats_oss << "=== Statistics (Runtime: " << elapsed << "s) ===" << std::endl;
        stats_oss << "Frames Read: " << total_read 
                  << " | Preprocessed: " << total_preprocessed;
        stats_oss << " | Read FPS: " << window_read_fps
                  << " | Preprocess FPS: " << window_preprocess_fps;
        stats_oss << std::endl;
        
        if (raw_frame_queue_) {
            stats_oss << "RawQueue: " << raw_frame_queue_->size() << "/2000" << std::endl;
        }
        for (const auto& group : preprocess_groups_) {
            if (!group) continue;
            size_t queue_size = group->queue ? group->queue->size() : 0;
            stats_oss << "PreQ " << group->input_width << "x" << group->input_height
                      << (group->roi_cropping ? " [ROI]" : " [FULL]")
                      << " engines=" << group->engines.size()
                      << " queue=" << queue_size << "/" << group->queue_capacity << std::endl;
        }
        
        // Show postprocessor queue size
        if (postprocess_queue_) {
            size_t postproc_queue_size = postprocess_queue_->size();
            stats_oss << "PostQ: " << postproc_queue_size << " tasks pending" << std::endl;
        }
        
        // OPTIMIZATION: Removed queue size checks - they involve mutex locks which can block
        // Queue sizes are not critical for monitoring and cause performance degradation

        // OPTIMIZATION: Removed Redis queue length checks - they involve network operations
        // that can block and degrade performance over time. These are not critical for monitoring.
        
        // Per-engine statistics
        std::vector<long long> window_engine_fps(engine_groups_.size(), 0);
        std::vector<long long> engine_detected;
        std::vector<long long> engine_failed;
        std::vector<long long> engine_total_time_ms;
        std::vector<long long> engine_frame_count;
        {
            std::lock_guard<std::mutex> lock(stats_.stats_mutex);
            engine_detected.resize(engine_groups_.size());
            engine_failed.resize(engine_groups_.size());
            engine_total_time_ms.resize(engine_groups_.size());
            engine_frame_count.resize(engine_groups_.size());
            for (size_t i = 0; i < engine_groups_.size(); ++i) {
                engine_detected[i] = stats_.frames_detected[i];
                engine_failed[i] = stats_.frames_failed[i];
                engine_total_time_ms[i] = stats_.engine_total_time_ms[i];
                engine_frame_count[i] = stats_.engine_frame_count[i];
            }
        }
        // Calculate windowed FPS
        {
            std::lock_guard<std::mutex> prev_lock(prev_stats_mutex_);
            for (size_t i = 0; i < engine_groups_.size(); ++i) {
                if (i < prev_frames_detected_.size() && window_seconds > 0) {
                    long long detected_diff = engine_detected[i] - prev_frames_detected_[i];
                    window_engine_fps[i] = static_cast<long long>(detected_diff / window_seconds);
                }
            }
        }
        // Output engine statistics
        for (size_t i = 0; i < engine_groups_.size(); ++i) {
            stats_oss << "Engine " << engine_groups_[i]->engine_name 
                      << ": Detected=" << engine_detected[i] 
                      << " | Failed=" << engine_failed[i];
            stats_oss << " | FPS=" << window_engine_fps[i];
            if (engine_frame_count[i] > 0) {
                double avg_time_ms = static_cast<double>(engine_total_time_ms[i]) / engine_frame_count[i];
                    stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_time_ms << "ms/frame";
                stats_oss << " | TotalTime=" << (engine_total_time_ms[i] / 1000.0) << "s";
                }
                stats_oss << std::endl;
        }
        
        // Reader statistics
        long long reader_time_ms = stats_.reader_total_time_ms.load();
        long long reader_frames = stats_.frames_read.load();
        stats_oss << "Reader: Frames=" << reader_frames;
        stats_oss << " | FPS=" << window_read_fps;
        if (reader_frames > 0) {
            double avg_reader_time_ms = static_cast<double>(reader_time_ms) / reader_frames;
            stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_reader_time_ms << "ms/frame";
            stats_oss << " | TotalTime=" << (reader_time_ms / 1000.0) << "s";
        }
        stats_oss << std::endl;
        
        long long preproc_time_ms = stats_.preprocessor_total_time_ms.load();
        long long preproc_frames = stats_.frames_preprocessed.load();
        stats_oss << "Preprocessor: Frames=" << preproc_frames;
        stats_oss << " | FPS=" << window_preprocess_fps;
        if (preproc_frames > 0) {
            double avg_preproc_time_ms = static_cast<double>(preproc_time_ms) / preproc_frames;
            stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_preproc_time_ms << "ms/frame";
            stats_oss << " | TotalTime=" << (preproc_time_ms / 1000.0) << "s";
        }
        stats_oss << std::endl;
        
        long long postproc_time_ms = stats_.postprocessor_total_time_ms.load();
        long long postproc_frames = stats_.frames_postprocessed.load();
        long long window_postproc_fps = 0;
        {
            std::lock_guard<std::mutex> lock(prev_stats_mutex_);
            if (window_seconds > 0) {
                long long postproc_diff = postproc_frames - prev_frames_postprocessed_;
                window_postproc_fps = static_cast<long long>(postproc_diff / window_seconds);
            }
        }
        stats_oss << "Postprocessor: Frames=" << postproc_frames;
        stats_oss << " | FPS=" << window_postproc_fps;
        if (postproc_frames > 0) {
            double avg_postproc_time_ms = static_cast<double>(postproc_time_ms) / postproc_frames;
            stats_oss << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_postproc_time_ms << "ms/frame";
            stats_oss << " | TotalTime=" << (postproc_time_ms / 1000.0) << "s";
        }
        stats_oss << std::endl;
        
        LOG_STATS("Monitor", stats_oss.str());
        
        // Update previous stats for next window calculation
        // Read current detected frames first, then update previous stats
        std::vector<long long> current_detected;
        {
            std::lock_guard<std::mutex> stats_lock(stats_.stats_mutex);
            current_detected.resize(stats_.frames_detected.size());
            for (size_t i = 0; i < stats_.frames_detected.size(); ++i) {
                current_detected[i] = stats_.frames_detected[i];
            }
        }
        {
            std::lock_guard<std::mutex> lock(prev_stats_mutex_);
            prev_monitor_time_ = now;
            prev_frames_read_ = total_read;
            prev_frames_preprocessed_ = total_preprocessed;
            prev_frames_postprocessed_ = postproc_frames;
            prev_frames_detected_.resize(current_detected.size());
            for (size_t i = 0; i < current_detected.size(); ++i) {
                prev_frames_detected_[i] = current_detected[i];
            }
        }
        
        // Check for timed out messages (only in Redis mode)
        if (use_redis_queue_) {
            const int message_timeout_seconds = redis_message_timeout_seconds_;
            std::lock_guard<std::mutex> lock(video_output_mutex_);
            for (auto& kv : video_output_status_) {
                auto& status = kv.second;
                if (status.message_pushed || status.timed_out) {
                    continue;  // Already handled
                }
                
                auto age = std::chrono::duration_cast<std::chrono::seconds>(
                    now - status.created_at).count();
                
                if (age > message_timeout_seconds) {
                    // Message timed out - mark it and log warning
                    status.timed_out = true;
                    
                    int total_pending = 0;
                    int total_registered = 0;
                    int total_processed = 0;
                    for (const auto& pc : status.pending_counts) {
                        total_pending += pc.second;
                    }
                    for (const auto& rc : status.registered_counts) {
                        total_registered += rc.second;
                    }
                    for (const auto& pc : status.processed_counts) {
                        total_processed += pc.second;
                    }
                    
                    LOG_ERROR("RedisOutput", "Message TIMED OUT (not pushed): key='" + kv.first + 
                             "', age=" + std::to_string(age) + "s" +
                             ", reading_completed=" + (status.reading_completed ? "true" : "false") +
                             ", pending=" + std::to_string(total_pending) +
                             ", registered=" + std::to_string(total_registered) +
                             ", processed=" + std::to_string(total_processed));
                }
            }
        }
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
    status.timed_out = false;
    status.total_frames_read = 0;
    status.created_at = std::chrono::steady_clock::now();
}

void ThreadPool::registerPendingFrame(const std::string& message_key, const std::string& engine_name) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty() || engine_name.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(video_output_mutex_);
    VideoOutputStatus& status = video_output_status_[message_key];
    status.pending_counts[engine_name]++;
    status.registered_counts[engine_name]++;
}

void ThreadPool::markFrameProcessed(const std::string& message_key, const std::string& engine_name,
                                    const std::string& output_path, int video_index) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty() || engine_name.empty()) {
        static thread_local int skip_count = 0;
        if (skip_count++ < 10) {
            LOG_WARNING("RedisOutput", "markFrameProcessed skipped: use_redis_queue_=" + 
                       std::to_string(use_redis_queue_) + ", output_queue_=" + 
                       (output_queue_ ? "valid" : "null") + ", message_key='" + message_key + 
                       "', engine_name='" + engine_name + "'");
        }
        return;
    }
    
    std::string message_to_push;
    {
        std::lock_guard<std::mutex> lock(video_output_mutex_);
        auto it = video_output_status_.find(message_key);
        if (it == video_output_status_.end()) {
            // Entry might have been erased or never created - this is OK if message was already pushed
            // But log it as warning since this might indicate messages aren't being registered
            static thread_local std::set<std::string> logged_missing_keys;
            if (logged_missing_keys.find(message_key) == logged_missing_keys.end() && logged_missing_keys.size() < 10) {
                logged_missing_keys.insert(message_key);
                LOG_WARNING("RedisOutput", "markFrameProcessed: message_key '" + message_key + 
                           "' not found in video_output_status_ (message may not have been registered or already pushed)");
            }
            return;
        }
        
        auto& status = it->second;
        
        // If message was already pushed, don't process further updates
        if (status.message_pushed) {
            // Removed logging - too expensive in hot path
            return;
        }
        if (!output_path.empty()) {
            status.detector_outputs[engine_name][video_index] = output_path;
        }
        
        // Track processed count for validation
        status.processed_counts[engine_name]++;
        
        // Decrement pending count (should never go negative, but handle it gracefully)
        int& pending = status.pending_counts[engine_name];
        int old_pending = pending;
        if (pending > 0) {
            pending--;
        } else if (pending < 0) {
            // This shouldn't happen, but log it for debugging
            LOG_WARNING("RedisOutput", "markFrameProcessed: pending count for '" + message_key + 
                       "' engine '" + engine_name + "' is negative (" + std::to_string(pending) + 
                       "), resetting to 0");
            pending = 0;
        }
        
        // Log periodically to track progress
        static thread_local std::map<std::string, int> mark_log_counters;
        int& mark_counter = mark_log_counters[message_key + "_" + engine_name];
        if (++mark_counter % 100 == 0) {
            int registered = status.registered_counts[engine_name];
            int processed = status.processed_counts[engine_name];
            LOG_DEBUG("RedisOutput", "markFrameProcessed progress: message_key='" + message_key + 
                     "', engine='" + engine_name + "', pending=" + std::to_string(pending) + 
                     " (was " + std::to_string(old_pending) + "), registered=" + std::to_string(registered) + 
                     ", processed=" + std::to_string(processed));
        }
        
        message_to_push = tryPushOutputLocked(message_key, status);
    }
    
    if (!message_to_push.empty()) {
        auto push_start = std::chrono::steady_clock::now();
        bool push_success = output_queue_->pushMessage(output_queue_name_, message_to_push);
        auto push_end = std::chrono::steady_clock::now();
        auto push_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            push_end - push_start).count();
        
        if (push_success) {
            LOG_INFO("RedisOutput", "Pushed processed message to output queue: " + output_queue_name_ + 
                     " (push_time=" + std::to_string(push_time_ms) + "ms, message_key='" + message_key + "')");
        } else {
            LOG_ERROR("RedisOutput", "Failed to push message to output queue: " + output_queue_name_ + 
                     " (push_time=" + std::to_string(push_time_ms) + "ms, message_key='" + message_key + "')");
        }
    }
}

void ThreadPool::markVideoReadingComplete(const std::string& message_key) {
    if (!use_redis_queue_ || !output_queue_ || message_key.empty()) {
        LOG_WARNING("RedisOutput", "markVideoReadingComplete: called but Redis not enabled or invalid message_key");
        return;
    }
    
    std::string message_to_push;
    {
        std::lock_guard<std::mutex> lock(video_output_mutex_);
        auto it = video_output_status_.find(message_key);
        if (it == video_output_status_.end()) {
            LOG_WARNING("RedisOutput", "markVideoReadingComplete: message_key '" + message_key + 
                       "' not found in video_output_status_ (may have been pushed already or never registered)");
            return;
        }
        
        // Check if already marked complete (avoid duplicate calls)
        if (it->second.reading_completed) {
            return;  // Already marked, skip
        }
        
        // Removed expensive logging - string building operations accumulate overhead
        
        it->second.reading_completed = true;
        message_to_push = tryPushOutputLocked(message_key, it->second);
    }
    
    if (!message_to_push.empty()) {
        auto push_start = std::chrono::steady_clock::now();
        bool push_success = output_queue_->pushMessage(output_queue_name_, message_to_push);
        auto push_end = std::chrono::steady_clock::now();
        auto push_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            push_end - push_start).count();
        
        if (push_success) {
            LOG_INFO("RedisOutput", "Pushed processed message to output queue: " + output_queue_name_ + 
                     " (push_time=" + std::to_string(push_time_ms) + "ms, message_key='" + message_key + "')");
        } else {
            LOG_ERROR("RedisOutput", "Failed to push message to output queue: " + output_queue_name_ + 
                     " (push_time=" + std::to_string(push_time_ms) + "ms, message_key='" + message_key + "')");
        }
    }
}

bool ThreadPool::canPushOutputLocked(const VideoOutputStatus& status) const {
    // Don't push if timed out
    if (status.timed_out) {
        return false;
    }
    if (!status.reading_completed) {
        return false;
    }
    if (status.message_pushed) {
        return false;
    }
    
    // Check if all pending counts are zero
    for (const auto& kv : status.pending_counts) {
        if (kv.second > 0) {
            return false;
        }
    }
    
    // Validate that registered == processed for each engine (no frames lost)
    for (const auto& kv : status.registered_counts) {
        const std::string& engine_name = kv.first;
        int registered = kv.second;
        auto proc_it = status.processed_counts.find(engine_name);
        int processed = (proc_it != status.processed_counts.end()) ? proc_it->second : 0;
        
        if (registered != processed) {
            // Frames were lost - don't push incomplete results
            LOG_WARNING("RedisOutput", "Frame count mismatch for engine '" + engine_name + 
                       "': registered=" + std::to_string(registered) + 
                       ", processed=" + std::to_string(processed) + 
                       " - will not push incomplete results");
            return false;
        }
    }
    
    // Additional check: ensure we have outputs from at least one detector
    bool has_outputs = false;
    for (const auto& kv : status.detector_outputs) {
        if (!kv.second.empty()) {
            has_outputs = true;
            break;
        }
    }
    if (!has_outputs) {
        return false;
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
        // Log why we can't push (with throttling)
        static thread_local std::map<std::string, int> log_counters;
        int& counter = log_counters[message_key];
        bool should_log = (++counter % 100 == 0);  // Log every 100th call per message_key
        
        if (should_log) {
            std::ostringstream reason;
            if (status.timed_out) {
                reason << "timed_out=true";
            } else if (!status.reading_completed) {
                reason << "reading_completed=false";
            } else if (status.message_pushed) {
                reason << "message_pushed=true";
            } else {
                // Check pending counts
                int total_pending = 0;
                for (const auto& kv : status.pending_counts) {
                    total_pending += kv.second;
                    if (kv.second > 0) {
                        reason << "pending[" << kv.first << "]=" << kv.second << " ";
                    }
                }
                if (total_pending == 0) {
                    // Check registered vs processed
                    for (const auto& kv : status.registered_counts) {
                        const std::string& engine_name = kv.first;
                        int registered = kv.second;
                        auto proc_it = status.processed_counts.find(engine_name);
                        int processed = (proc_it != status.processed_counts.end()) ? proc_it->second : 0;
                        if (registered != processed) {
                            reason << "registered[" << engine_name << "]=" << registered << 
                                      " != processed[" << engine_name << "]=" << processed << " ";
                        }
                    }
                    // Check if has outputs
                    bool has_outputs = false;
                    for (const auto& kv : status.detector_outputs) {
                        if (!kv.second.empty()) {
                            has_outputs = true;
                            break;
                        }
                    }
                    if (!has_outputs) {
                        reason << "no_detector_outputs ";
                    }
                }
            }
            LOG_DEBUG("RedisOutput", "tryPushOutputLocked: message '" + message_key + 
                     "' cannot push: " + reason.str());
        }
        
        return "";
    }
    if (status.original_message.empty()) {
        LOG_WARNING("RedisOutput", "tryPushOutputLocked: message '" + message_key + 
                   "' has empty original_message");
        return "";
    }
    
    status.message_pushed = true;
    std::string final_message = augmentMessageWithDetectors(status.original_message, status.detector_outputs);
    
// Persist the final message to JSONL file for auditing/debugging
try {
    std::filesystem::path output_dir = 
        std::filesystem::path(output_dir_) / "redis_messages";
    std::filesystem::create_directories(output_dir);

    // Parse final_message to get send_at
    nlohmann::json j = nlohmann::json::parse(final_message);

    std::string send_at;
    try {
        send_at = j["alarm"]["raw_alarm"]["send_at"].get<std::string>();
    } catch (...) {
        send_at = "unknown";
    }

    // Convert send_at -> dd-mm-yy
    auto convertSendAt = [](const std::string& s) -> std::string {
        if (s.size() < 8) return "unknown";
        std::string dd = s.substr(6, 2);
        std::string mm = s.substr(4, 2);
        std::string yy = s.substr(2, 2);
        return dd + "-" + mm + "-" + yy;  // dd-mm-yy
    };

    std::string day_str = convertSendAt(send_at);

    // File theo ngày: dd-mm-yy.jsonl
    std::filesystem::path file_path = 
        output_dir / (day_str + ".jsonl");

    // Append mode
    std::ofstream outfile(file_path, std::ios::app);
    if (outfile.is_open()) {
        outfile << final_message << "\n";
        outfile.close();
        LOG_INFO("RedisOutput", 
                "Appended output message to " + file_path.string());
    } else {
        LOG_ERROR("RedisOutput", 
                "Failed to open file for saving message: " + file_path.string());
    }
}
catch (const std::exception& e) {
    LOG_ERROR("RedisOutput", 
            "Failed to save Redis output message: " + std::string(e.what()));
}
    
    LOG_INFO("RedisOutput", "Ready to push message for key '" + message_key + "': " + final_message);
    
    // DON'T erase the entry immediately - it might still be accessed by concurrent markFrameProcessed calls
    // The entry will be cleaned up later or can remain (it's marked as pushed, so won't be pushed again)
    // If we erase it now, subsequent markFrameProcessed calls will fail with "message_key not found"
    // video_output_status_.erase(message_key);  // REMOVED: causes race condition with concurrent markFrameProcessed calls
    
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

