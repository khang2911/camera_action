#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& filepath);
    void setLogLevel(LogLevel level);
    void enableConsoleOutput(bool enable) { console_output_ = enable; }
    
    void log(LogLevel level, const std::string& component, const std::string& message);
    void log(LogLevel level, const std::string& component, const std::string& message, 
             int thread_id, int video_id = -1, int frame_num = -1);
    
    // Convenience methods
    void debug(const std::string& component, const std::string& message);
    void info(const std::string& component, const std::string& message);
    void warning(const std::string& component, const std::string& message);
    void error(const std::string& component, const std::string& message);
    
    // Statistics logging
    void logStats(const std::string& component, const std::string& stats);
    
private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string getTimestamp();
    std::string levelToString(LogLevel level);
    
    std::ofstream log_file_;
    std::mutex log_mutex_;
    LogLevel current_level_;
    bool console_output_;
    bool file_output_;
};

// Macros for easier logging
#define LOG_DEBUG(component, msg) Logger::getInstance().debug(component, msg)
#define LOG_INFO(component, msg) Logger::getInstance().info(component, msg)
#define LOG_WARNING(component, msg) Logger::getInstance().warning(component, msg)
#define LOG_ERROR(component, msg) Logger::getInstance().error(component, msg)
#define LOG_STATS(component, stats) Logger::getInstance().logStats(component, stats)

#endif // LOGGER_H

