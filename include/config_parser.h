#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <vector>

#include "video_clip.h"
#include "reader_options.h"

enum class ModelType {
    DETECTION,
    POSE
};

struct EngineConfig {
    std::string path;
    int num_detectors;
    std::string name;  // Optional name identifier
    ModelType type;    // Model type: detection or pose
    int batch_size;    // Batch size for inference (must match engine batch size)
    int input_width;   // Input image width
    int input_height;  // Input image height
    float conf_threshold;  // Confidence threshold (0.0-1.0)
    float nms_threshold;  // NMS IoU threshold (0.0-1.0)
    int gpu_id;        // GPU device ID (0, 1, 2, etc.)
    int decode_gpu_id; // GPU device for NVDEC (optional)
    bool roi_cropping; // Enable ROI cropping for this engine (from config.box in video_list.jsonl)
    
    EngineConfig() : num_detectors(4), type(ModelType::DETECTION), batch_size(1),
                     input_width(640), input_height(640), conf_threshold(0.25f), 
                     nms_threshold(0.45f), gpu_id(0), decode_gpu_id(0), roi_cropping(false) {}
    EngineConfig(const std::string& p, int num_det, const std::string& n = "", 
                 ModelType t = ModelType::DETECTION, int batch = 1,
                 int in_w = 640, int in_h = 640, float conf = 0.25f, float nms = 0.45f, int gpu = 0, bool roi = false)
        : path(p), num_detectors(num_det), name(n), type(t), batch_size(batch),
          input_width(in_w), input_height(in_h), conf_threshold(conf), 
          nms_threshold(nms), gpu_id(gpu), decode_gpu_id(gpu), roi_cropping(roi) {}
};

class ConfigParser {
public:
    ConfigParser();
    ~ConfigParser();
    
    bool loadFromFile(const std::string& config_path);
    
    // Getters
    std::vector<EngineConfig> getEngineConfigs() const { return engine_configs_; }
    std::vector<std::string> getVideoPaths() const;
    const std::vector<VideoClip>& getVideoClips() const { return video_clips_; }
    int getNumReaders() const { return num_readers_; }
    int getNumPreprocessors() const { return num_preprocessors_; }
    std::string getOutputDir() const { return output_dir_; }
    bool isDebugMode() const { return debug_mode_; }
    int getMaxFramesPerVideo() const { return max_frames_per_video_; }
    bool isRoiCroppingEnabled() const { return roi_cropping_enabled_; }
    const ReaderOptions& getReaderOptions() const { return reader_options_; }
    
    // Backward compatibility getters
    std::string getModelPath() const { 
        return engine_configs_.empty() ? "" : engine_configs_[0].path; 
    }
    int getNumDetectors() const { 
        return engine_configs_.empty() ? 4 : engine_configs_[0].num_detectors; 
    }
    
    // Setters (for command line overrides)
    void setVideoPaths(const std::vector<std::string>& paths);
    void setNumReaders(int num) { 
        num_readers_ = num; 
        if (num_preprocessors_ <= 0) {
            num_preprocessors_ = num;
        }
    }
    void setNumPreprocessors(int num) { num_preprocessors_ = num; }
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }
    
    // Backward compatibility setters
    void setModelPath(const std::string& path) { 
        if (engine_configs_.empty()) {
            engine_configs_.push_back(EngineConfig(path, 4));
        } else {
            engine_configs_[0].path = path;
        }
    }
    void setNumDetectors(int num) { 
        if (engine_configs_.empty()) {
            engine_configs_.push_back(EngineConfig("", num));
        } else {
            engine_configs_[0].num_detectors = num;
        }
    }
    
    bool isValid() const;
    
    // Redis queue configuration
    bool useRedisQueue() const { return use_redis_queue_; }
    std::string getRedisHost() const { return redis_host_; }
    int getRedisPort() const { return redis_port_; }
    std::string getRedisPassword() const { return redis_password_; }
    int getRedisDb() const { return redis_db_; }
    std::string getInputQueueName() const { return input_queue_name_; }
    std::string getOutputQueueName() const { return output_queue_name_; }
    int getRedisMessageTimeout() const { return redis_message_timeout_seconds_; }
    
    // Utility helpers
    static double parseTimestamp(const std::string& iso);
    
private:
    std::vector<EngineConfig> engine_configs_;
    std::vector<VideoClip> video_clips_;
    int num_readers_;
    int num_preprocessors_;
    std::string output_dir_;
    double time_padding_seconds_ = 0.0;
    bool debug_mode_ = false;
    int max_frames_per_video_ = 0;  // 0 means no limit
    bool roi_cropping_enabled_ = false;  // Enable ROI cropping from config.box
    ReaderOptions reader_options_;
    
    // Redis queue settings
    bool use_redis_queue_ = false;
    std::string redis_host_ = "localhost";
    int redis_port_ = 6379;
    std::string redis_password_ = "";
    int redis_db_ = 0;
    std::string input_queue_name_ = "input_queue";
    std::string output_queue_name_ = "output_queue";
    int redis_message_timeout_seconds_ = 300;  // Default: 5 minutes
    
    void loadVideoListFromFile(const std::string& path);
    void loadPlainVideoList(std::istream& stream);
    void loadJsonVideoList(const std::string& path);
    void addPlainPath(const std::string& path);
    void addVideoClip(const VideoClip& clip);
    std::pair<double, double> computeTimeWindow(const std::string& start_iso,
                                                const std::string& end_iso) const;
};

#endif // CONFIG_PARSER_H

