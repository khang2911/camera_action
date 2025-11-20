#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
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
    int input_width;
    int input_height;
    bool roi_cropping;  // Enable ROI cropping for this engine
    size_t tensor_elements;
    std::vector<std::unique_ptr<YOLODetector>> detectors;
    std::vector<std::thread> detector_threads;
    std::unique_ptr<FrameQueue> frame_queue;
    std::unique_ptr<Preprocessor> preprocessor;
    
    std::vector<std::vector<float>*> buffer_pool;
    std::mutex buffer_pool_mutex;
    
    EngineGroup(int id, const std::string& path, const std::string& name, int num_det,
                int in_w, int in_h, bool roi = false)
        : engine_id(id), engine_path(path), engine_name(name), num_detectors(num_det),
          input_width(in_w), input_height(in_h), roi_cropping(roi) {
        tensor_elements = static_cast<size_t>(input_width) * input_height * 3;
        frame_queue = std::make_unique<FrameQueue>();
        preprocessor = std::make_unique<Preprocessor>(input_width, input_height);
    }
    
    ~EngineGroup();
    
    std::shared_ptr<std::vector<float>> acquireBuffer();
    void releaseBuffer(std::vector<float>* buffer);
};

class ThreadPool {
public:
    ThreadPool(int num_readers,
               int num_preprocessors,
               const std::vector<VideoClip>& video_clips,
               const std::vector<EngineConfig>& engine_configs,
               const std::string& output_dir,
               bool debug_mode = false,
               int max_frames_per_video = 0);
    
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
        std::atomic<long long> preprocessor_total_time_ms{0};  // Total time for preprocessors
        std::chrono::steady_clock::time_point start_time;
    };
    
    // Get statistics snapshot (non-atomic copy)
    void getStatisticsSnapshot(long long& frames_read, long long& frames_preprocessed,
                               std::vector<long long>& frames_detected,
                               std::vector<long long>& frames_failed,
                               long long& reader_total_time_ms,
                               long long& preprocessor_total_time_ms,
                               std::vector<long long>& engine_total_time_ms,
                               std::vector<long long>& engine_frame_count,
                               std::chrono::steady_clock::time_point& start_time) const;
    
private:
    int num_readers_;
    int num_preprocessors_;
    std::vector<VideoClip> video_clips_;
    std::vector<std::unique_ptr<EngineGroup>> engine_groups_;
    std::string output_dir_;
    bool debug_mode_;
    int max_frames_per_video_;
    
    std::vector<std::thread> reader_threads_;
    std::vector<std::thread> preprocessor_threads_;
    std::thread monitor_thread_;
    std::unique_ptr<FrameQueue> raw_frame_queue_;
    
    
    std::atomic<bool> stop_flag_;
    std::vector<bool> video_processed_;  // Use regular bool with mutex (atomic vectors are not resizable)
    std::mutex video_mutex_;
    
    mutable Statistics stats_;
    
    void readerWorker(int reader_id);
    void preprocessorWorker(int worker_id);
    void detectorWorker(int engine_id, int detector_id);
    void monitorWorker();
    int getNextVideo();
    
    std::string generateOutputPath(int video_id, int frame_number, int engine_id, int detector_id, const std::string& engine_name);
};

#endif // THREAD_POOL_H

