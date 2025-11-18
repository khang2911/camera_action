#ifndef VIDEO_CLIP_H
#define VIDEO_CLIP_H

#include <limits>
#include <string>

struct VideoClip {
    std::string path;
    double moment_time = 0.0;          // Base timestamp (seconds since epoch) for frame index zero
    double duration_seconds = 0.0;     // Total duration reported for the clip
    double start_timestamp = -std::numeric_limits<double>::infinity();
    double end_timestamp = std::numeric_limits<double>::infinity();
    bool has_time_window = false;

    VideoClip() = default;
    explicit VideoClip(std::string p) : path(std::move(p)) {}
};

#endif  // VIDEO_CLIP_H

