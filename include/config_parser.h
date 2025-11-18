#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <vector>

#include "video_clip.h"

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
    
    EngineConfig() : num_detectors(4), type(ModelType::DETECTION), batch_size(1),
                     input_width(640), input_height(640), conf_threshold(0.25f), 
                     nms_threshold(0.45f), gpu_id(0) {}
    EngineConfig(const std::string& p, int num_det, const std::string& n = "", 
                 ModelType t = ModelType::DETECTION, int batch = 1,
                 int in_w = 640, int in_h = 640, float conf = 0.25f, float nms = 0.45f, int gpu = 0)
        : path(p), num_detectors(num_det), name(n), type(t), batch_size(batch),
          input_width(in_w), input_height(in_h), conf_threshold(conf), 
          nms_threshold(nms), gpu_id(gpu) {}
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
    
private:
    std::vector<EngineConfig> engine_configs_;
    std::vector<VideoClip> video_clips_;
    int num_readers_;
    int num_preprocessors_;
    std::string output_dir_;
    double time_padding_seconds_ = 0.0;
    
    void loadVideoListFromFile(const std::string& path);
    void loadPlainVideoList(std::istream& stream);
    void loadJsonVideoList(const std::string& path);
    void addPlainPath(const std::string& path);
    void addVideoClip(const VideoClip& clip);
    std::pair<double, double> computeTimeWindow(const std::string& start_iso,
                                                const std::string& end_iso) const;
    static double parseTimestamp(const std::string& iso);
};

#endif // CONFIG_PARSER_H

