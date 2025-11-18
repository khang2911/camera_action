#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <functional>
#include <memory>
#include "frame_queue.h"
#include "preprocessor.h"
#include "yolo_detector.h"
#include "config_parser.h"
#include "logger.h"

struct EngineGroup {
    int engine_id;
    std::string engine_path;
    std::string engine_name;
    int num_detectors;
    std::vector<std::unique_ptr<YOLODetector>> detectors;
    std::vector<std::thread> detector_threads;
    std::unique_ptr<FrameQueue> frame_queue;
    
    EngineGroup(int id, const std::string& path, const std::string& name, int num_det)
        : engine_id(id), engine_path(path), engine_name(name), num_detectors(num_det) {
        frame_queue = std::make_unique<FrameQueue>();
    }
};

class ThreadPool {
public:
    ThreadPool(int num_readers,
               const std::vector<std::string>& video_paths,
               const std::vector<EngineConfig>& engine_configs,
               const std::string& output_dir);
    
    ~ThreadPool();
    
    void start();
    void stop();
    void waitForCompletion();
    
    // Statistics getters
    struct Statistics {
        std::atomic<long long> frames_read{0};
        std::atomic<long long> frames_preprocessed{0};
        // Use regular long long with mutex for per-engine stats (atomic vectors are not resizable)
        mutable std::mutex stats_mutex;
        std::vector<long long> frames_detected;  // per engine
        std::vector<long long> frames_failed;    // per engine
        // Processing time tracking (in milliseconds)
        std::atomic<long long> reader_total_time_ms{0};  // Total time spent reading frames
        std::vector<long long> engine_total_time_ms;     // Total time per engine (detection)
        std::vector<long long> engine_frame_count;      // Frame count per engine (for average calculation)
        std::chrono::steady_clock::time_point start_time;
    };
    
    // Get statistics snapshot (non-atomic copy)
    void getStatisticsSnapshot(long long& frames_read, long long& frames_preprocessed,
                               std::vector<long long>& frames_detected,
                               std::vector<long long>& frames_failed,
                               long long& reader_total_time_ms,
                               std::vector<long long>& engine_total_time_ms,
                               std::vector<long long>& engine_frame_count,
                               std::chrono::steady_clock::time_point& start_time) const;
    
private:
    int num_readers_;
    std::vector<std::string> video_paths_;
    std::vector<std::unique_ptr<EngineGroup>> engine_groups_;
    std::string output_dir_;
    
    std::vector<std::thread> reader_threads_;
    std::thread monitor_thread_;
    
    std::atomic<bool> stop_flag_;
    std::vector<bool> video_processed_;  // Use regular bool with mutex (atomic vectors are not resizable)
    std::mutex video_mutex_;
    
    mutable Statistics stats_;
    
    void readerWorker(int reader_id);
    void detectorWorker(int engine_id, int detector_id);
    void monitorWorker();
    int getNextVideo();
    
    std::string generateOutputPath(int video_id, int frame_number, int engine_id, int detector_id, const std::string& engine_name);
};

#endif // THREAD_POOL_H

