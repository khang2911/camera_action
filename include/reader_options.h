#ifndef READER_OPTIONS_H
#define READER_OPTIONS_H

#include <cstddef>
#include <cstdint>

struct ReaderOptions {
    // FFmpeg demuxer tuning
    int ffmpeg_buffer_size = 1 << 20;          // bytes
    long long ffmpeg_probe_size = 5LL << 20;   // bytes
    long long ffmpeg_analyze_duration = 5LL * 1000000LL;  // microseconds
    int ffmpeg_read_timeout_ms = 5000;
    bool ffmpeg_fast_seek = true;
    bool ffmpeg_fast_io = true;
    
    // Decoder packet batching / scheduling
    int max_packets_per_loop = 8;
    
    // Packet prefetching / async I/O
    bool enable_prefetch = true;
    int prefetch_queue_depth = 32;
};

#endif  // READER_OPTIONS_H


