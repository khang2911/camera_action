#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "yolo_detector.h"
#include "preprocessor.h"

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "Options:\n"
              << "  --engine ENGINE_PATH          Path to TensorRT engine file (required)\n"
              << "  --video VIDEO_PATH            Path to input video file (required)\n"
              << "  --input-width WIDTH           Input width (default: 640)\n"
              << "  --input-height HEIGHT         Input height (default: 640)\n"
              << "  --batch-size SIZE             Batch size (default: 1)\n"
              << "  --conf-threshold THRESH       Confidence threshold (default: 0.25)\n"
              << "  --nms-threshold THRESH        NMS threshold (default: 0.45)\n"
              << "  --gpu-id ID                  GPU ID (default: 0)\n"
              << "  --start-frame FRAME          Start frame index (default: 0)\n"
              << "  --end-frame FRAME            End frame index (default: -1, all frames)\n"
              << "  --max-frames COUNT           Maximum frames to process (default: -1, all)\n"
              << "  --dump-input-dir DIR         Directory to dump input tensors\n"
              << "  --dump-output-dir DIR         Directory to dump output tensors\n"
              << "  --model-type TYPE            Model type: detection or pose (default: detection)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string engine_path;
    std::string video_path;
    int input_width = 640;
    int input_height = 640;
    int batch_size = 1;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int gpu_id = 0;
    int start_frame = 0;
    int end_frame = -1;
    int max_frames = -1;
    std::string dump_input_dir;
    std::string dump_output_dir;
    ModelType model_type = ModelType::DETECTION;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--input-width" && i + 1 < argc) {
            input_width = std::stoi(argv[++i]);
        } else if (arg == "--input-height" && i + 1 < argc) {
            input_height = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--conf-threshold" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms-threshold" && i + 1 < argc) {
            nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--gpu-id" && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--start-frame" && i + 1 < argc) {
            start_frame = std::stoi(argv[++i]);
        } else if (arg == "--end-frame" && i + 1 < argc) {
            end_frame = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--dump-input-dir" && i + 1 < argc) {
            dump_input_dir = argv[++i];
        } else if (arg == "--dump-output-dir" && i + 1 < argc) {
            dump_output_dir = argv[++i];
        } else if (arg == "--model-type" && i + 1 < argc) {
            std::string type_str = argv[++i];
            if (type_str == "pose") {
                model_type = ModelType::POSE;
            } else {
                model_type = ModelType::DETECTION;
            }
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Validate required arguments
    if (engine_path.empty() || video_path.empty()) {
        std::cerr << "Error: --engine and --video are required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Check if files exist
    if (!std::filesystem::exists(engine_path)) {
        std::cerr << "Error: Engine file not found: " << engine_path << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(video_path)) {
        std::cerr << "Error: Video file not found: " << video_path << std::endl;
        return 1;
    }

    std::cout << "=== TensorRT Cross-Check Tool ===" << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Input size: " << input_width << "x" << input_height << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "GPU ID: " << gpu_id << std::endl;
    std::cout << "Model type: " << (model_type == ModelType::POSE ? "pose" : "detection") << std::endl;
    if (!dump_input_dir.empty()) {
        std::cout << "Dump input dir: " << dump_input_dir << std::endl;
    }
    if (!dump_output_dir.empty()) {
        std::cout << "Dump output dir: " << dump_output_dir << std::endl;
    }
    std::cout << std::endl;

    // Initialize detector
    YOLODetector detector(engine_path, model_type, batch_size, input_width, input_height,
                          conf_threshold, nms_threshold, gpu_id);
    
    if (!detector.initialize()) {
        std::cerr << "Error: Failed to initialize detector" << std::endl;
        return 1;
    }

    // Set dump directories
    if (!dump_input_dir.empty() || !dump_output_dir.empty()) {
        detector.setDumpDirectories(dump_input_dir, dump_output_dir, "cpp");
    }

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open video: " << video_path << std::endl;
        return 1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    std::cout << "Video info: " << total_frames << " frames, " << fps << " fps" << std::endl;

    // Set frame range
    if (end_frame < 0) {
        end_frame = total_frames - 1;
    }
    if (max_frames > 0) {
        end_frame = std::min(start_frame + max_frames - 1, end_frame);
    }
    std::cout << "Processing frames: " << start_frame << " to " << end_frame << std::endl;
    std::cout << std::endl;

    // Seek to start frame
    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);

    // Preprocessor
    Preprocessor preprocessor(input_width, input_height);

    // Process frames
    int frame_idx = start_frame;
    int batch_idx = 0;
    std::vector<cv::Mat> batch_frames;
    std::vector<std::shared_ptr<std::vector<float>>> batch_inputs;
    std::vector<int> batch_frame_indices;
    std::vector<int> batch_orig_widths;
    std::vector<int> batch_orig_heights;

    while (frame_idx <= end_frame) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Preprocess frame
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        auto preprocessed = std::make_shared<std::vector<float>>();
        preprocessor.preprocessToFloat(frame, *preprocessed);

        batch_frames.push_back(frame);
        batch_inputs.push_back(preprocessed);
        batch_frame_indices.push_back(frame_idx);
        batch_orig_widths.push_back(orig_w);
        batch_orig_heights.push_back(orig_h);

        // Process batch when full
        if (static_cast<int>(batch_frames.size()) >= batch_size) {
            std::cout << "Processing batch " << batch_idx << " (frames " 
                      << batch_frame_indices[0] << "-" << batch_frame_indices.back() << ")" << std::endl;

            // Create dummy output paths (we're only dumping tensors, not writing detection files)
            std::vector<std::string> dummy_output_paths(batch_size, "/dev/null");
            std::vector<int> dummy_frame_numbers = batch_frame_indices;
            
            // Run inference
            if (!detector.runWithPreprocessedBatch(batch_inputs, dummy_output_paths, 
                                                   dummy_frame_numbers,
                                                   batch_orig_widths, batch_orig_heights)) {
                std::cerr << "Error: Inference failed for batch " << batch_idx << std::endl;
                return 1;
            }

            std::cout << "  Batch " << batch_idx << " completed" << std::endl;

            // Clear batch
            batch_frames.clear();
            batch_inputs.clear();
            batch_frame_indices.clear();
            batch_orig_widths.clear();
            batch_orig_heights.clear();
            batch_idx++;
        }

        frame_idx++;
    }

    // Process remaining frames in batch
    if (!batch_frames.empty()) {
        std::cout << "Processing final batch " << batch_idx << " (frames " 
                  << batch_frame_indices[0] << "-" << batch_frame_indices.back() << ")" << std::endl;

        // Pad batch to batch_size if needed
        while (static_cast<int>(batch_inputs.size()) < batch_size) {
            // Duplicate the last frame to pad the batch
            batch_inputs.push_back(batch_inputs.back());
            batch_frame_indices.push_back(batch_frame_indices.back());
            batch_orig_widths.push_back(batch_orig_widths.back());
            batch_orig_heights.push_back(batch_orig_heights.back());
        }

        // Create dummy output paths
        std::vector<std::string> dummy_output_paths(batch_size, "/dev/null");
        std::vector<int> dummy_frame_numbers = batch_frame_indices;
        
        if (!detector.runWithPreprocessedBatch(batch_inputs, dummy_output_paths,
                                               dummy_frame_numbers,
                                               batch_orig_widths, batch_orig_heights)) {
            std::cerr << "Error: Inference failed for final batch" << std::endl;
            return 1;
        }

        std::cout << "  Final batch completed" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Processing Complete ===" << std::endl;
    std::cout << "Processed " << (frame_idx - start_frame) << " frames" << std::endl;
    std::cout << "Total batches: " << (batch_idx + 1) << std::endl;

    return 0;
}

