#include "thread_pool.h"
#include "video_reader.h"
#include "logger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <thread>

ThreadPool::ThreadPool(int num_readers,
                       const std::vector<std::string>& video_paths,
                       const std::vector<EngineConfig>& engine_configs,
                       const std::string& output_dir)
    : num_readers_(num_readers), video_paths_(video_paths),
      output_dir_(output_dir), stop_flag_(false) {
    
    // Initialize statistics
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        stats_.frames_detected.resize(engine_configs.size(), 0);
        stats_.frames_failed.resize(engine_configs.size(), 0);
    }
    stats_.start_time = std::chrono::steady_clock::now();
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Initialize video processed flags
    {
        std::lock_guard<std::mutex> lock(video_mutex_);
        video_processed_.resize(video_paths.size(), false);
    }
    
    // Initialize engine groups (one per engine)
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        const auto& config = engine_configs[i];
        auto engine_group = std::make_unique<EngineGroup>(
            static_cast<int>(i), config.path, config.name, config.num_detectors
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
             " reader threads and " + std::to_string(engine_configs.size()) + " engines");
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::start() {
    stop_flag_ = false;
    stats_.start_time = std::chrono::steady_clock::now();
    stats_.frames_read = 0;
    stats_.frames_preprocessed = 0;
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        for (size_t i = 0; i < stats_.frames_detected.size(); ++i) {
            stats_.frames_detected[i] = 0;
            stats_.frames_failed[i] = 0;
        }
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
    // Wait for all frame queues to be empty and all readers done
    bool all_empty = false;
    while (!all_empty && !stop_flag_) {
        all_empty = true;
        for (auto& engine_group : engine_groups_) {
            if (!engine_group->frame_queue->empty()) {
                all_empty = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    stop_flag_ = true;
    stop();
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
    
    // Each reader thread processes videos until all are done
    while (!stop_flag_) {
        int video_id = getNextVideo();
        
        if (video_id < 0) {
            // No more videos to process
            LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + " finished (no more videos)");
            break;
        }
        
        LOG_INFO("Reader", "Reader thread " + std::to_string(reader_id) + 
                 " processing video " + std::to_string(video_id) + ": " + video_paths_[video_id]);
        
        VideoReader reader(video_paths_[video_id], video_id);
        
        if (!reader.isOpened()) {
            LOG_ERROR("Reader", "Cannot open video " + std::to_string(video_id) + ": " + video_paths_[video_id]);
            continue;
        }
        
        cv::Mat frame;
        int frame_count = 0;
        while (!stop_flag_ && reader.readFrame(frame)) {
            stats_.frames_read++;
            stats_.frames_preprocessed++;  // Count as preprocessed (will be done per-engine)
            
            // Create frame data with original frame (each engine will preprocess with its own size)
            FrameData frame_data(frame, video_id, reader.getFrameNumber(), video_paths_[video_id]);
            
            // Push original frame to ALL engine queues (each frame goes through all engines)
            for (auto& engine_group : engine_groups_) {
                engine_group->frame_queue->push(frame_data);
            }
            
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
    }
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
    
    while (!stop_flag_) {
        // Pop preprocessed frame data from this engine's queue
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
        bool success = engine_group->detectors[detector_id]->detect(
            frame_data.frame, output_path
        );
        
        if (success) {
            {
                std::lock_guard<std::mutex> lock(stats_.stats_mutex);
                stats_.frames_detected[engine_id]++;
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
                stats_oss << "Engine " << engine_groups_[i]->engine_name 
                          << ": Detected=" << detected 
                          << " | Failed=" << failed;
                if (elapsed > 0) {
                    stats_oss << " | FPS=" << (detected / elapsed);
                }
                stats_oss << std::endl;
            }
        }
        
        LOG_STATS("Monitor", stats_oss.str());
    }
    
    LOG_INFO("Monitor", "Monitoring thread finished");
}

void ThreadPool::getStatisticsSnapshot(long long& frames_read, long long& frames_preprocessed,
                                      std::vector<long long>& frames_detected,
                                      std::vector<long long>& frames_failed,
                                      std::chrono::steady_clock::time_point& start_time) const {
    frames_read = stats_.frames_read.load();
    frames_preprocessed = stats_.frames_preprocessed.load();
    {
        std::lock_guard<std::mutex> lock(stats_.stats_mutex);
        frames_detected.resize(stats_.frames_detected.size());
        frames_failed.resize(stats_.frames_failed.size());
        for (size_t i = 0; i < stats_.frames_detected.size(); ++i) {
            frames_detected[i] = stats_.frames_detected[i];
            frames_failed[i] = stats_.frames_failed[i];
        }
    }
    start_time = stats_.start_time;
}

std::string ThreadPool::generateOutputPath(int video_id, int frame_number, 
                                            int engine_id, int detector_id, 
                                            const std::string& engine_name) {
    std::ostringstream oss;
    oss << output_dir_ << "/video_" << std::setfill('0') << std::setw(4) << video_id
        << "_frame_" << std::setfill('0') << std::setw(6) << frame_number
        << "_" << engine_name
        << "_detector_" << detector_id << ".bin";
    return oss.str();
}

