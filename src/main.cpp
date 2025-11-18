#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include "thread_pool.h"
#include "config_parser.h"
#include "logger.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --config PATH        Path to YAML config file (recommended)\n"
              << "  --num-readers N      Number of reader/preprocessor threads (overrides config)\n"
              << "  --num-detectors N    Number of YOLO detector threads (overrides config)\n"
              << "  --engine PATH        Path to TensorRT engine file (overrides config)\n"
              << "  --output-dir PATH    Output directory for bin files (overrides config)\n"
              << "  --videos PATH1 ...   List of video file paths (overrides config)\n"
              << "  --help               Show this help message\n"
              << "\n"
              << "Note: If --config is provided, other options override config values.\n"
              << "      If --config is not provided, --engine and --videos are required.\n";
}

int main(int argc, char* argv[]) {
    ConfigParser config;
    std::string config_file;
    bool config_loaded = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
            if (!config.loadFromFile(config_file)) {
                std::cerr << "Error: Failed to load config file: " << config_file << std::endl;
                return 1;
            }
            config_loaded = true;
        } else if (arg == "--num-readers" && i + 1 < argc) {
            config.setNumReaders(std::stoi(argv[++i]));
        } else if (arg == "--num-detectors" && i + 1 < argc) {
            config.setNumDetectors(std::stoi(argv[++i]));
        } else if (arg == "--engine" && i + 1 < argc) {
            config.setModelPath(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.setOutputDir(argv[++i]);
        } else if (arg == "--videos") {
            // Collect all remaining arguments as video paths
            std::vector<std::string> video_paths;
            while (i + 1 < argc) {
                video_paths.push_back(argv[++i]);
            }
            config.setVideoPaths(video_paths);
            break;
        }
    }
    
    // If no config file was loaded, try to load default config.yaml
    if (!config_loaded) {
        if (config.loadFromFile("config.yaml")) {
            std::cout << "Loaded default config.yaml" << std::endl;
            config_loaded = true;
        }
    }
    
    // Validate configuration
    if (!config.isValid()) {
        std::cerr << "Error: Invalid configuration. Missing required parameters:" << std::endl;
        auto engine_configs = config.getEngineConfigs();
        if (engine_configs.empty()) {
            std::cerr << "  - Model paths (use --engine or set in config file)" << std::endl;
        }
        if (config.getVideoClips().empty()) {
            std::cerr << "  - Video sources (use videos.list_file or set via --videos)" << std::endl;
        }
        if (config.getNumReaders() <= 0) {
            std::cerr << "  - Valid reader thread count (must be > 0)" << std::endl;
        }
        printUsage(argv[0]);
        return 1;
    }

    // Get configuration values
    int num_readers = config.getNumReaders();
    int num_preprocessors = config.getNumPreprocessors();
    std::string output_dir = config.getOutputDir();
    const auto& video_clips = config.getVideoClips();
    std::vector<EngineConfig> engine_configs = config.getEngineConfigs();
    
    // Initialize logger
    Logger& logger = Logger::getInstance();
    logger.setLogLevel(LogLevel::INFO);
    logger.enableConsoleOutput(true);
    logger.setLogFile(output_dir + "/processing.log");
    
    LOG_INFO("Main", "=== AI Camera Solution ===");
    if (config_loaded) {
        LOG_INFO("Main", "Config file: " + (config_file.empty() ? std::string("config.yaml") : config_file));
    }
    LOG_INFO("Main", "Number of reader threads: " + std::to_string(num_readers));
    LOG_INFO("Main", "Number of preprocessor threads: " + std::to_string(num_preprocessors));
    LOG_INFO("Main", "Number of engines: " + std::to_string(engine_configs.size()));
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        std::string type_str = (engine_configs[i].type == ModelType::POSE) ? "pose" : "detection";
        LOG_INFO("Main", "Engine " + std::to_string(i) + " (" + engine_configs[i].name + "): " + 
                 engine_configs[i].path + " [type=" + type_str + 
                 ", detectors=" + std::to_string(engine_configs[i].num_detectors) + 
                 ", batch_size=" + std::to_string(engine_configs[i].batch_size) +
                 ", input_size=" + std::to_string(engine_configs[i].input_width) + "x" + 
                 std::to_string(engine_configs[i].input_height) +
                 ", conf_threshold=" + std::to_string(engine_configs[i].conf_threshold) +
                 ", nms_threshold=" + std::to_string(engine_configs[i].nms_threshold) +
                 ", gpu_id=" + std::to_string(engine_configs[i].gpu_id) + "]");
    }
    LOG_INFO("Main", "Output directory: " + output_dir);
    LOG_INFO("Main", "Number of videos: " + std::to_string(video_clips.size()));
    
    // Create thread pool
    ThreadPool pool(num_readers, num_preprocessors, video_clips, engine_configs, output_dir);
    
    // Start processing
    LOG_INFO("Main", "Starting processing...");
    pool.start();
    
    // Wait for completion
    pool.waitForCompletion();
    
    // Print final statistics
    long long frames_read, frames_preprocessed, reader_total_time_ms, preprocessor_total_time_ms;
    std::vector<long long> frames_detected, frames_failed;
    std::vector<long long> engine_total_time_ms, engine_frame_count;
    std::chrono::steady_clock::time_point start_time;
    pool.getStatisticsSnapshot(frames_read, frames_preprocessed, frames_detected, frames_failed,
                               reader_total_time_ms, preprocessor_total_time_ms,
                               engine_total_time_ms, engine_frame_count, start_time);
    
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();
    
    LOG_INFO("Main", "=== Processing Completed ===");
    LOG_INFO("Main", "Total runtime: " + std::to_string(total_time) + " seconds");
    LOG_INFO("Main", "Total frames read: " + std::to_string(frames_read));
    LOG_INFO("Main", "Total frames preprocessed: " + std::to_string(frames_preprocessed));
    
    // Reader timing statistics
    if (frames_read > 0) {
        double avg_reader_time_ms = static_cast<double>(reader_total_time_ms) / frames_read;
        LOG_INFO("Main", "Reader: AvgTime=" + std::to_string(avg_reader_time_ms) + "ms/frame, " +
                 "TotalTime=" + std::to_string(reader_total_time_ms / 1000.0) + "s");
    }
    
    if (frames_preprocessed > 0) {
        double avg_preproc_time_ms = static_cast<double>(preprocessor_total_time_ms) / frames_preprocessed;
        LOG_INFO("Main", "Preprocessor: AvgTime=" + std::to_string(avg_preproc_time_ms) + "ms/frame, " +
                 "TotalTime=" + std::to_string(preprocessor_total_time_ms / 1000.0) + "s");
    }
    
    // Engine timing statistics
    for (size_t i = 0; i < engine_configs.size(); ++i) {
        std::ostringstream engine_stats;
        engine_stats << "Engine " << engine_configs[i].name 
                     << ": Detected=" << frames_detected[i]
                     << " | Failed=" << frames_failed[i];
        
        if (engine_frame_count[i] > 0) {
            double avg_time_ms = static_cast<double>(engine_total_time_ms[i]) / engine_frame_count[i];
            engine_stats << " | AvgTime=" << std::fixed << std::setprecision(2) << avg_time_ms << "ms/frame";
            engine_stats << " | TotalTime=" << (engine_total_time_ms[i] / 1000.0) << "s";
        }
        
        LOG_INFO("Main", engine_stats.str());
    }
    
    return 0;
}

