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
      time_padding_seconds_(0.0) {}

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
        
        // Parse date/time part (before timezone)
        size_t tz_pos = iso_str.find_first_of("+-Z");
        if (tz_pos == std::string::npos) {
            tz_pos = iso_str.length();
        }
        
        std::string datetime_part = iso_str.substr(0, tz_pos);
        
        // Replace 'T' with space for std::get_time
        size_t t_pos = datetime_part.find('T');
        if (t_pos != std::string::npos) {
            datetime_part[t_pos] = ' ';
        }
        
        // Remove microseconds if present
        size_t dot_pos = datetime_part.find('.');
        if (dot_pos != std::string::npos) {
            datetime_part = datetime_part.substr(0, dot_pos);
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
        
        // Parse datetime using std::get_time
        std::tm tm = {};
        std::istringstream ss(datetime_part);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        
        if (ss.fail()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        // Convert to Unix timestamp manually to avoid timezone issues
        // Calculate days since epoch (1970-01-01)
        int year = tm.tm_year + 1900;
        int month = tm.tm_mon + 1;
        int day = tm.tm_mday;
        
        // Calculate day of year
        int days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        bool is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        if (is_leap) days_in_month[1] = 29;
        
        int day_of_year = day;
        for (int m = 0; m < month - 1; ++m) {
            day_of_year += days_in_month[m];
        }
        
        // Calculate days since 1970-01-01
        long long days_since_epoch = 0;
        for (int y = 1970; y < year; ++y) {
            bool y_is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
            days_since_epoch += y_is_leap ? 366 : 365;
        }
        days_since_epoch += day_of_year - 1;
        
        // Calculate total seconds
        long long total_seconds = days_since_epoch * 86400LL;
        total_seconds += tm.tm_hour * 3600LL;
        total_seconds += tm.tm_min * 60LL;
        total_seconds += tm.tm_sec;
        
        // Convert to UTC by subtracting timezone offset
        // (If timestamp is +07:00, it's 7 hours ahead of UTC, so subtract to get UTC)
        double unix_timestamp = static_cast<double>(total_seconds) - tz_offset_seconds;
        
        // Add microseconds if present (check original string)
        size_t orig_dot_pos = iso_str.find('.');
        if (orig_dot_pos != std::string::npos && orig_dot_pos < tz_pos) {
            std::string microsec_str = iso_str.substr(orig_dot_pos + 1, tz_pos - orig_dot_pos - 1);
            try {
                double microseconds = std::stod("0." + microsec_str);
                unix_timestamp += microseconds;
            } catch (...) {
                // ignore microsecond parsing errors
            }
        }
        
        return unix_timestamp;
    } catch (...) {
        // fall through to return NaN
    }

    return std::numeric_limits<double>::quiet_NaN();
}