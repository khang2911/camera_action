#include "logger.h"
#include <iostream>
#include <ctime>

Logger::Logger() : current_level_(LogLevel::INFO), console_output_(true), file_output_(false) {}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
        log_file_.close();
    }
    log_file_.open(filepath, std::ios::app);
    file_output_ = log_file_.is_open();
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& component, const std::string& message) {
    if (level < current_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::string timestamp = getTimestamp();
    std::string level_str = levelToString(level);
    std::string log_line = "[" + timestamp + "] [" + level_str + "] [" + component + "] " + message;
    
    if (console_output_) {
        std::cout << log_line << std::endl;
    }
    
    if (file_output_ && log_file_.is_open()) {
        log_file_ << log_line << std::endl;
        log_file_.flush();
    }
}

void Logger::log(LogLevel level, const std::string& component, const std::string& message,
                 int thread_id, int video_id, int frame_num) {
    if (level < current_level_) {
        return;
    }
    
    std::ostringstream oss;
    oss << "[T:" << thread_id;
    if (video_id >= 0) {
        oss << " V:" << video_id;
    }
    if (frame_num >= 0) {
        oss << " F:" << frame_num;
    }
    oss << "] " << message;
    
    log(level, component, oss.str());
}

void Logger::debug(const std::string& component, const std::string& message) {
    log(LogLevel::DEBUG, component, message);
}

void Logger::info(const std::string& component, const std::string& message) {
    log(LogLevel::INFO, component, message);
}

void Logger::warning(const std::string& component, const std::string& message) {
    log(LogLevel::WARNING, component, message);
}

void Logger::error(const std::string& component, const std::string& message) {
    log(LogLevel::ERROR, component, message);
}

void Logger::logStats(const std::string& component, const std::string& stats) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::string timestamp = getTimestamp();
    std::string log_line = "[" + timestamp + "] [STATS] [" + component + "] " + stats;
    
    if (console_output_) {
        std::cout << log_line << std::endl;
    }
    
    if (file_output_ && log_file_.is_open()) {
        log_file_ << log_line << std::endl;
        log_file_.flush();
    }
}

