#include "config_parser.h"
 
#include <yaml-cpp/yaml.h>
 
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
 
namespace {
 
std::string trim(const std::string& input) {
    const auto first = std::find_if_not(input.begin(), input.end(), [](unsigned char c) { return std::isspace(c); });
    if (first == input.end()) {
        return {};
    }
    const auto last = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return std::string(first, last);
}
 
bool isJsonExtension(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".json" || ext == ".jsonl" || ext == ".ndjson";
}
 
std::string nodeToString(const YAML::Node& node) {
    if (!node || !node.IsDefined()) {
        return {};
    }
    if (node.IsScalar()) {
        return node.as<std::string>();
    }
    std::stringstream ss;
    ss << node;
    return trim(ss.str());
}
 
std::string resolvePlaybackPath(const YAML::Node& node) {
    if (!node || !node.IsDefined()) {
        return {};
    }
    if (node.IsScalar()) {
        return node.as<std::string>();
    }
    if (node["file"]) {
        return node["file"].as<std::string>();
    }
    return nodeToString(node);
}
 
double nodeToDouble(const YAML::Node& node, double fallback = 0.0) {
    if (!node || !node.IsDefined()) {
        return fallback;
    }
    try {
        return node.as<double>();
    } catch (const YAML::Exception&) {
        try {
            return std::stod(node.as<std::string>());
        } catch (...) {
            return fallback;
        }
    }
}
 
}  // namespace
 
ConfigParser::ConfigParser()
    : num_readers_(10),
      num_preprocessors_(10),
      output_dir_("./output"),
      time_padding_seconds_(0.0),
      debug_mode_(false),
      max_frames_per_video_(0),
      roi_cropping_enabled_(false) {}
 
ConfigParser::~ConfigParser() = default;
 
bool ConfigParser::loadFromFile(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        engine_configs_.clear();
        video_clips_.clear();
        time_padding_seconds_ = 0.0;
 
        // Parse model definitions
        if (config["models"] && config["models"].IsSequence()) {
            for (const auto& node : config["models"]) {
                if (!node["path"]) {
                    std::cerr << "[Config] Skipping model entry without 'path'." << std::endl;
                    continue;
                }
 
                EngineConfig cfg;
                cfg.path = node["path"].as<std::string>();
                if (node["num_detectors"]) cfg.num_detectors = node["num_detectors"].as<int>();
                if (node["name"]) cfg.name = node["name"].as<std::string>();
                if (node["type"]) {
                    std::string type = node["type"].as<std::string>();
                    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
                    cfg.type = (type == "pose") ? ModelType::POSE : ModelType::DETECTION;
                }
                if (node["batch_size"]) cfg.batch_size = node["batch_size"].as<int>();
                if (node["input_width"]) cfg.input_width = node["input_width"].as<int>();
                if (node["input_height"]) cfg.input_height = node["input_height"].as<int>();
                if (node["conf_threshold"]) cfg.conf_threshold = node["conf_threshold"].as<float>();
                if (node["nms_threshold"]) cfg.nms_threshold = node["nms_threshold"].as<float>();
                if (node["gpu_id"]) cfg.gpu_id = node["gpu_id"].as<int>();
                if (node["roi_cropping"]) {
                    try {
                        cfg.roi_cropping = node["roi_cropping"].as<bool>();
                    } catch (const YAML::Exception&) {
                        std::string roi_str = node["roi_cropping"].as<std::string>();
                        std::transform(roi_str.begin(), roi_str.end(), roi_str.begin(), ::tolower);
                        cfg.roi_cropping = (roi_str == "true" || roi_str == "1" || roi_str == "yes");
                    }
                }

                engine_configs_.push_back(cfg);
            }
        } else if (config["model"]) {
            const auto& node = config["model"];
            if (!node["path"]) {
                std::cerr << "[Config] Missing model.path in legacy section." << std::endl;
            } else {
                EngineConfig cfg;
                cfg.path = node["path"].as<std::string>();
                if (node["num_detectors"]) cfg.num_detectors = node["num_detectors"].as<int>();
                if (node["type"]) {
                    std::string type = node["type"].as<std::string>();
                    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
                    cfg.type = (type == "pose") ? ModelType::POSE : ModelType::DETECTION;
                }
                engine_configs_.push_back(cfg);
            }
        }
 
        // Thread settings
        if (config["threads"]) {
            const auto& threads = config["threads"];
            if (threads["num_readers"]) {
                num_readers_ = threads["num_readers"].as<int>();
            }
            if (threads["num_preprocessors"]) {
                num_preprocessors_ = threads["num_preprocessors"].as<int>();
            } else if (num_preprocessors_ <= 0) {
                num_preprocessors_ = num_readers_;
            }
        }
 
        // Output directory
        if (config["output"] && config["output"]["dir"]) {
            output_dir_ = config["output"]["dir"].as<std::string>();
        }
        
        // Debug mode settings
        if (config["debug"]) {
            const auto& debug = config["debug"];
            if (debug["enabled"]) {
                try {
                    // Try as boolean first
                    debug_mode_ = debug["enabled"].as<bool>();
                } catch (const YAML::Exception&) {
                    // If that fails, try as string
                    std::string enabled_str = debug["enabled"].as<std::string>();
                    std::transform(enabled_str.begin(), enabled_str.end(), enabled_str.begin(), ::tolower);
                    debug_mode_ = (enabled_str == "true" || enabled_str == "1" || enabled_str == "yes");
                }
                std::cout << "[Config] Debug mode: " << (debug_mode_ ? "enabled" : "disabled") << std::endl;
            }
            if (debug["max_frames_per_video"]) {
                max_frames_per_video_ = debug["max_frames_per_video"].as<int>();
                if (max_frames_per_video_ < 0) {
                    max_frames_per_video_ = 0;  // 0 means no limit
                }
                std::cout << "[Config] Max frames per video: " << max_frames_per_video_ 
                          << (max_frames_per_video_ == 0 ? " (no limit)" : "") << std::endl;
            }
        }
 
        // ROI cropping settings
        if (config["roi_cropping"]) {
            const auto& roi = config["roi_cropping"];
            if (roi["enabled"]) {
                try {
                    roi_cropping_enabled_ = roi["enabled"].as<bool>();
                } catch (const YAML::Exception&) {
                    std::string enabled_str = roi["enabled"].as<std::string>();
                    std::transform(enabled_str.begin(), enabled_str.end(), enabled_str.begin(), ::tolower);
                    roi_cropping_enabled_ = (enabled_str == "true" || enabled_str == "1" || enabled_str == "yes");
                }
                std::cout << "[Config] ROI cropping: " << (roi_cropping_enabled_ ? "enabled" : "disabled") << std::endl;
            }
        }

        // Video sources
        if (config["videos"]) {
            const auto& videos = config["videos"];
            if (videos["time_padding_seconds"]) {
                time_padding_seconds_ = videos["time_padding_seconds"].as<double>();
                if (time_padding_seconds_ < 0.0) {
                    time_padding_seconds_ = 0.0;
                }
            }
            if (videos["list_file"]) {
                loadVideoListFromFile(videos["list_file"].as<std::string>());
            }
            if (videos["list_file_jsonl"]) {
                loadVideoListFromFile(videos["list_file_jsonl"].as<std::string>());
            }
            if (videos["paths"] && videos["paths"].IsSequence()) {
                for (const auto& p : videos["paths"]) {
                    addPlainPath(p.as<std::string>());
                }
            }
        }
        
        // Parse Redis queue configuration
        if (config["redis_queue"]) {
            const auto& redis = config["redis_queue"];
            if (redis["enabled"]) {
                try {
                    use_redis_queue_ = redis["enabled"].as<bool>();
                } catch (const YAML::Exception&) {
                    std::string enabled_str = redis["enabled"].as<std::string>();
                    std::transform(enabled_str.begin(), enabled_str.end(), enabled_str.begin(), ::tolower);
                    use_redis_queue_ = (enabled_str == "true" || enabled_str == "1" || enabled_str == "yes");
                }
            }
            
            if (use_redis_queue_) {
                if (redis["host"]) {
                    redis_host_ = redis["host"].as<std::string>();
                }
                if (redis["port"]) {
                    redis_port_ = redis["port"].as<int>();
                }
                if (redis["password"]) {
                    redis_password_ = redis["password"].as<std::string>();
                }
                if (redis["db"]) {
                    try {
                        redis_db_ = redis["db"].as<int>();
                    } catch (const YAML::Exception&) {
                        std::string db_str = redis["db"].as<std::string>();
                        redis_db_ = std::stoi(db_str);
                    }
                    if (redis_db_ < 0) {
                        redis_db_ = 0;
                    }
                }
                if (redis["input_queue"]) {
                    input_queue_name_ = redis["input_queue"].as<std::string>();
                }
                if (redis["output_queue"]) {
                    output_queue_name_ = redis["output_queue"].as<std::string>();
                }
            }
        }

        return true;
    } catch (const YAML::Exception& ex) {
        std::cerr << "[Config] Failed to load configuration '" << config_path << "': " << ex.what() << std::endl;
        return false;
    }
}
 
std::vector<std::string> ConfigParser::getVideoPaths() const {
    std::vector<std::string> paths;
    paths.reserve(video_clips_.size());
    for (const auto& clip : video_clips_) {
        paths.push_back(clip.path);
    }
    return paths;
}
 
void ConfigParser::setVideoPaths(const std::vector<std::string>& paths) {
    video_clips_.clear();
    for (const auto& path : paths) {
        addPlainPath(path);
    }
}
 
bool ConfigParser::isValid() const {
    if (engine_configs_.empty()) {
        return false;
    }
    if (video_clips_.empty()) {
        return false;
    }
    if (num_readers_ <= 0 || num_preprocessors_ <= 0) {
        return false;
    }
    return true;
}
 
void ConfigParser::loadVideoListFromFile(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        std::cerr << "[Config] Unable to open video list: " << path << std::endl;
        return;
    }
 
    if (isJsonExtension(path)) {
        input.close();
        loadJsonVideoList(path);
    } else {
        loadPlainVideoList(input);
    }
}
 
void ConfigParser::loadPlainVideoList(std::istream& stream) {
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        addPlainPath(line);
    }
}
 
void ConfigParser::loadJsonVideoList(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        std::cerr << "[Config] Unable to open JSON video list: " << path << std::endl;
        return;
    }
 
    std::string line;
    size_t line_index = 0;
    while (std::getline(input, line)) {
        ++line_index;
        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
 
        YAML::Node root;
        try {
            root = YAML::Load(trimmed);
        } catch (const YAML::Exception& ex) {
            std::cerr << "[Config] Failed to parse JSON line " << line_index << ": " << ex.what() << std::endl;
            continue;
        }
 
        YAML::Node alarm = root["alarm"];
        if (!alarm || !alarm["raw_alarm"]) {
            continue;
        }
        YAML::Node raw_alarm = alarm["raw_alarm"];
 
        YAML::Node record_list = raw_alarm["record_list"];
        YAML::Node playback = root["playback_location"];
        if (!record_list || !record_list.IsSequence() || !playback || !playback.IsSequence()) {
            continue;
        }
 
        const std::string start_value = nodeToString(raw_alarm["video_start_time"]);
        const std::string end_value = nodeToString(raw_alarm["video_end_time"]);
        const auto time_window = computeTimeWindow(start_value, end_value);
 
        // Parse ROI box from config (always parse, engines can decide whether to use it)
        YAML::Node config_node = root["config"];
        bool has_roi_box = false;
        float roi_x1 = 0.0f, roi_y1 = 0.0f, roi_x2 = 1.0f, roi_y2 = 1.0f;
        if (config_node && config_node["box"] && config_node["box"].IsSequence()) {
            const auto& box = config_node["box"];
            if (box.size() >= 4) {
                // Box format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                // Extract x1, y1, x2, y2 from the first and third points
                try {
                    if (box[0].IsSequence() && box[0].size() >= 2) {
                        roi_x1 = box[0][0].as<float>();
                        roi_y1 = box[0][1].as<float>();
                    }
                    if (box[2].IsSequence() && box[2].size() >= 2) {
                        roi_x2 = box[2][0].as<float>();
                        roi_y2 = box[2][1].as<float>();
                    }
                    // Validate and clamp to [0, 1]
                    if (roi_x1 >= 0.0f && roi_x1 <= 1.0f && roi_y1 >= 0.0f && roi_y1 <= 1.0f &&
                        roi_x2 >= 0.0f && roi_x2 <= 1.0f && roi_y2 >= 0.0f && roi_y2 <= 1.0f &&
                        roi_x1 < roi_x2 && roi_y1 < roi_y2) {
                        has_roi_box = true;
                    } else {
                        std::cerr << "[Config] Invalid ROI box coordinates, skipping ROI for this video" << std::endl;
                    }
                } catch (const YAML::Exception& ex) {
                    std::cerr << "[Config] Failed to parse ROI box: " << ex.what() << std::endl;
                }
            }
        }

        // Parse serial, record_id, and record_date from alarm/raw_alarm
        std::string serial = nodeToString(alarm["serial"]);
        std::string record_id = nodeToString(alarm["record_id"]);
        std::string record_start_time = nodeToString(alarm["record_start_time"]);
        std::string send_at = nodeToString(raw_alarm["send_at"]);
        std::string record_date;
        if (send_at.length() >= 8) {
            record_date = send_at.substr(0, 4) + "-" + send_at.substr(4, 2) + "-" + send_at.substr(6, 2);
        } else if (!record_start_time.empty()) {
            if (record_start_time.length() >= 10) {
                record_date = record_start_time.substr(0, 10);
            } else {
                record_date.clear();
            }
        }
        std::string raw_serial = nodeToString(raw_alarm["serial"]);

        const size_t clip_count = std::min(record_list.size(), playback.size());
        for (size_t i = 0; i < clip_count; ++i) {
            VideoClip clip;
            clip.path = resolvePlaybackPath(playback[i]);
            if (clip.path.empty()) {
                continue;
            }

            const auto& entry = record_list[i];
            clip.moment_time = nodeToDouble(entry["moment_time"], 0.0);
            clip.duration_seconds = nodeToDouble(entry["duration"], 0.0);
            clip.start_timestamp = time_window.first;
            clip.end_timestamp = time_window.second;
            clip.has_time_window = std::isfinite(clip.start_timestamp) || std::isfinite(clip.end_timestamp);
            
            // Set record_id and record_date for output file naming
            clip.record_id = record_id;
            clip.record_date = record_date;
            clip.serial = !raw_serial.empty() ? raw_serial : serial;
            
            // Set ROI if available
            if (has_roi_box) {
                clip.has_roi = true;
                clip.roi_x1 = roi_x1;
                clip.roi_y1 = roi_y1;
                clip.roi_x2 = roi_x2;
                clip.roi_y2 = roi_y2;
            }

            addVideoClip(clip);
        }
    }
}
 
void ConfigParser::addPlainPath(const std::string& path) {
    const auto trimmed = trim(path);
    if (trimmed.empty()) {
        return;
    }
    VideoClip clip(trimmed);
    clip.has_time_window = false;
    addVideoClip(clip);
}
 
void ConfigParser::addVideoClip(const VideoClip& clip) {
    if (clip.path.empty()) {
        return;
    }
    video_clips_.push_back(clip);
}
 
std::pair<double, double> ConfigParser::computeTimeWindow(const std::string& start_iso,
                                                          const std::string& end_iso) const {
    double start = parseTimestamp(start_iso);
    double end = parseTimestamp(end_iso);
 
    std::cout << "record timeeeeeeeeeeee" << end - start << "\n";
 
    if (!std::isfinite(start)) {
        start = -std::numeric_limits<double>::infinity();
    }
    if (!std::isfinite(end)) {
        end = std::numeric_limits<double>::infinity();
    }
 
    if (std::isfinite(start)) {
        start -= time_padding_seconds_;
    }
    if (std::isfinite(end)) {
        end += time_padding_seconds_;
    }
 
    if (end < start) {
        std::swap(start, end);
    }
 
    return {start, end};
}
 
 
double ConfigParser::parseTimestamp(const std::string& value) {
    if (value.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
 
    // Try parsing as numeric (Unix timestamp)
    try {
        size_t idx = 0;
        double numeric = std::stod(value, &idx);
        if (idx == value.size()) {
            return numeric;
        }
    } catch (...) {
        // fall through to try ISO parsing
    }
 
    // Try parsing as ISO 8601 timestamp string
    // Format: "YYYY-MM-DDTHH:MM:SS[.microseconds][+/-HH:MM]"
    try {
        std::string iso_str = value;
       
        // Find timezone position
        size_t tz_pos = iso_str.find_last_of("+-");
        if (tz_pos == std::string::npos || tz_pos == 0) {
            // Check for 'Z' (UTC)
            tz_pos = iso_str.find('Z');
            if (tz_pos == std::string::npos) {
                tz_pos = iso_str.length();
            }
        }
       
        // Extract datetime and microseconds part
        std::string datetime_part = iso_str.substr(0, tz_pos);
       
        // Extract microseconds if present
        double microseconds = 0.0;
        size_t dot_pos = datetime_part.find('.');
        if (dot_pos != std::string::npos) {
            std::string microsec_str = datetime_part.substr(dot_pos + 1);
            datetime_part = datetime_part.substr(0, dot_pos);
            try {
                microseconds = std::stod("0." + microsec_str);
            } catch (...) {
                microseconds = 0.0;
            }
        }
       
        // Replace 'T' with space
        size_t t_pos = datetime_part.find('T');
        if (t_pos != std::string::npos) {
            datetime_part[t_pos] = ' ';
        }
       
        // Parse datetime
        std::tm tm = {};
        std::istringstream ss(datetime_part);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
       
        if (ss.fail()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
       
        // Parse timezone offset
        int tz_offset_seconds = 0;
        if (tz_pos < iso_str.length()) {
            char tz_sign = iso_str[tz_pos];
            if (tz_sign == 'Z') {
                tz_offset_seconds = 0;
            } else if (tz_sign == '+' || tz_sign == '-') {
                std::string tz_str = iso_str.substr(tz_pos + 1);
                size_t colon_pos = tz_str.find(':');
                if (colon_pos != std::string::npos) {
                    int hours = std::stoi(tz_str.substr(0, colon_pos));
                    int minutes = std::stoi(tz_str.substr(colon_pos + 1));
                    tz_offset_seconds = (hours * 3600 + minutes * 60);
                    if (tz_sign == '-') {
                        tz_offset_seconds = -tz_offset_seconds;
                    }
                }
            }
        }
       
        // Convert to Unix timestamp
        // Use timegm or equivalent to avoid timezone issues
        #ifdef _WIN32
            // Windows: use _mkgmtime
            time_t unix_time = _mkgmtime(&tm);
        #else
            // Linux/Mac: use timegm
            time_t unix_time = timegm(&tm);
        #endif
       
        if (unix_time == -1) {
            return std::numeric_limits<double>::quiet_NaN();
        }
       
        // Subtract timezone offset to get UTC
        // If time is +07:00, it means it's 7 hours ahead of UTC
        // So UTC = local_time - offset
        double unix_timestamp = static_cast<double>(unix_time) - tz_offset_seconds;
       
        // Add microseconds
        unix_timestamp += microseconds;
        std::cout << "unix_timestamp " << std::to_string(unix_timestamp) << " ---->" << value << std::endl;  
        return unix_timestamp;
       
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}