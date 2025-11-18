#include "frame_queue.h"
#include <chrono>

FrameQueue::FrameQueue(size_t max_size) : max_size_(max_size) {}

void FrameQueue::push(const FrameData& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] { return queue_.size() < max_size_; });
    queue_.push(frame);
    not_empty_.notify_one();
}

bool FrameQueue::pop(FrameData& frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (timeout_ms < 0) {
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
    } else {
        auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
    }
    
    frame = queue_.front();
    queue_.pop();
    not_full_.notify_one();
    return true;
}

bool FrameQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t FrameQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void FrameQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    not_full_.notify_all();
}

