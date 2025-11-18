#include "config_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

ConfigParser::ConfigParser()
    : num_readers_(10), output_dir_("./output") {}

ConfigParser::~ConfigParser() {}

bool ConfigParser::loadFromFile(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        // Load model configurations (support multiple engines)
        engine_configs_.clear();
        
        if (config["models"]) {
            // New format: multiple engines
            if (config["models"].IsSequence()) {
                for (const auto& model_node : config["models"]) {
                    EngineConfig engine_config;
                    if (model_node["path"]) {
                        engine_config.path = model_node["path"].as<std::string>();
                    }
                    if (model_node["num_detectors"]) {
                        engine_config.num_detectors = model_node["num_detectors"].as<int>();
                    }
                    if (model_node["name"]) {
                        engine_config.name = model_node["name"].as<std::string>();
                    } else {
                        // Generate default name if not provided
                        engine_config.name = "engine" + std::to_string(engine_configs_.size());
                    }
                    // Parse model type
                    if (model_node["type"]) {
                        std::string type_str = model_node["type"].as<std::string>();
                        if (type_str == "pose") {
                            engine_config.type = ModelType::POSE;
                        } else {
                            engine_config.type = ModelType::DETECTION;
                        }
                    } else {
                        engine_config.type = ModelType::DETECTION;  // Default to detection
                    }
                    
                    // Parse batch size
                    if (model_node["batch_size"]) {
                        engine_config.batch_size = model_node["batch_size"].as<int>();
                    } else {
                        engine_config.batch_size = 1;  // Default to batch size 1
                    }
                    
                    engine_configs_.push_back(engine_config);
                }
            }
        } else if (config["model"]) {
            // Backward compatibility: single model
            EngineConfig engine_config;
            if (config["model"]["path"]) {
                engine_config.path = config["model"]["path"].as<std::string>();
            }
            if (config["model"]["num_detectors"]) {
                engine_config.num_detectors = config["model"]["num_detectors"].as<int>();
            } else if (config["threads"] && config["threads"]["num_detectors"]) {
                engine_config.num_detectors = config["threads"]["num_detectors"].as<int>();
            }
            if (config["model"]["type"]) {
                std::string type_str = config["model"]["type"].as<std::string>();
                engine_config.type = (type_str == "pose") ? ModelType::POSE : ModelType::DETECTION;
            } else {
                engine_config.type = ModelType::DETECTION;  // Default to detection
            }
            if (config["model"]["batch_size"]) {
                engine_config.batch_size = config["model"]["batch_size"].as<int>();
            } else {
                engine_config.batch_size = 1;  // Default to batch size 1
            }
            engine_config.name = "engine0";
            engine_configs_.push_back(engine_config);
        }
        
        // Load video paths from text file
        if (config["videos"]) {
            if (config["videos"]["list_file"]) {
                std::string list_file = config["videos"]["list_file"].as<std::string>();
                video_paths_.clear();
                
                // Read video paths from text file
                std::ifstream file(list_file);
                if (!file.is_open()) {
                    std::cerr << "Warning: Cannot open video list file: " << list_file << std::endl;
                } else {
                    std::string line;
                    while (std::getline(file, line)) {
                        // Trim whitespace from start
                        size_t start = line.find_first_not_of(" \t\r\n");
                        if (start != std::string::npos) {
                            line = line.substr(start);
                        } else {
                            line.clear();
                        }
                        
                        // Trim whitespace from end
                        size_t end = line.find_last_not_of(" \t\r\n");
                        if (end != std::string::npos) {
                            line = line.substr(0, end + 1);
                        }
                        
                        // Skip empty lines and comments (lines starting with #)
                        if (!line.empty() && line[0] != '#') {
                            video_paths_.push_back(line);
                        }
                    }
                    file.close();
                }
            }
            // Backward compatibility: support direct list in YAML
            else if (config["videos"].IsSequence()) {
                video_paths_.clear();
                for (const auto& video : config["videos"]) {
                    video_paths_.push_back(video.as<std::string>());
                }
            } else if (config["videos"].IsScalar()) {
                // Single video path
                video_paths_.clear();
                video_paths_.push_back(config["videos"].as<std::string>());
            }
        }
        
        // Load thread configuration
        if (config["threads"]) {
            if (config["threads"]["num_readers"]) {
                num_readers_ = config["threads"]["num_readers"].as<int>();
            }
        }
        
        // Load output directory
        if (config["output"]) {
            if (config["output"]["dir"]) {
                output_dir_ = config["output"]["dir"].as<std::string>();
            }
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing YAML config file: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error reading config file: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigParser::isValid() const {
    if (engine_configs_.empty() || video_paths_.empty() || num_readers_ <= 0) {
        return false;
    }
    
    // Check all engines have valid paths and thread counts
    for (const auto& engine : engine_configs_) {
        if (engine.path.empty() || engine.num_detectors <= 0) {
            return false;
        }
    }
    
    return true;
}

