#ifndef VIDEO_CLIP_H
#define VIDEO_CLIP_H

#include <limits>
#include <string>

struct VideoClip {
    std::string path;
    std::string serial;               // Camera serial from alarm.serial/raw_alarm.serial
    std::string record_id;            // Record ID from alarm.record_id
    std::string record_date;          // Date string in YYYY-MM-DD format (derived from raw_alarm.send_at)
    double moment_time = 0.0;          // Base timestamp (seconds since epoch) for frame index zero
    double duration_seconds = 0.0;     // Total duration reported for the clip
    double start_timestamp = -std::numeric_limits<double>::infinity();
    double end_timestamp = std::numeric_limits<double>::infinity();
    bool has_time_window = false;
    
    // ROI (Region of Interest) for cropping frames before detection
    // Box format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] in normalized coordinates [0, 1]
    bool has_roi = false;
    float roi_x1 = 0.0f;  // Normalized x1 coordinate
    float roi_y1 = 0.0f;  // Normalized y1 coordinate
    float roi_x2 = 1.0f;  // Normalized x2 coordinate
    float roi_y2 = 1.0f;  // Normalized y2 coordinate
    int roi_offset_x = 0;  // Pixel offset X (for scaling detections back to original frame)
    int roi_offset_y = 0;  // Pixel offset Y (for scaling detections back to original frame)

    VideoClip() = default;
    explicit VideoClip(std::string p) : path(std::move(p)) {}
};

#endif  // VIDEO_CLIP_H

