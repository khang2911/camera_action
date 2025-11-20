#include "redis_queue.h"
#include "logger.h"
#include <chrono>
#include <thread>
#include <sstream>

RedisQueue::RedisQueue(const std::string& host, int port, const std::string& password)
    : host_(host), port_(port), password_(password), redis_client_(nullptr) {
}

RedisQueue::~RedisQueue() {
    disconnect();
}

std::string RedisQueue::buildConnectionString() const {
    std::ostringstream oss;
    oss << "tcp://";
    if (!password_.empty()) {
        oss << ":" << password_ << "@";
    }
    oss << host_ << ":" << port_;
    return oss.str();
}

bool RedisQueue::connect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::string conn_str = buildConnectionString();
        redis_client_ = std::make_unique<sw::redis::Redis>(conn_str);
        
        // Test connection with PING
        auto pong = redis_client_->ping();
        if (pong == "PONG") {
            LOG_INFO("Redis", "Connected to Redis at " + host_ + ":" + std::to_string(port_));
            return true;
        } else {
            LOG_ERROR("Redis", "Connection test failed: PING did not return PONG");
            redis_client_.reset();
            return false;
        }
    } catch (const sw::redis::Error& e) {
        LOG_ERROR("Redis", "Connection error: " + std::string(e.what()));
        redis_client_.reset();
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "Connection error: " + std::string(e.what()));
        redis_client_.reset();
        return false;
    }
}

void RedisQueue::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    redis_client_.reset();
}

bool RedisQueue::isConnected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!redis_client_) {
        return false;
    }
    
    try {
        // Test connection with PING
        auto pong = redis_client_->ping();
        return (pong == "PONG");
    } catch (...) {
        return false;
    }
}

bool RedisQueue::reconnect() {
    disconnect();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return connect();
}

void RedisQueue::handleError(const std::string& operation) {
    LOG_ERROR("Redis", operation + " error occurred");
    // Try to reconnect on error
    LOG_INFO("Redis", "Attempting to reconnect...");
    reconnect();
}

bool RedisQueue::popMessage(std::string& message, int timeout_seconds, const std::string& queue_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!redis_client_) {
        if (!reconnect()) {
            return false;
        }
    }
    
    try {
        // Use BLPOP for blocking pop with timeout
        // BLPOP returns optional<std::pair<key, value>> or throws TimeoutError
        std::vector<std::string> keys = {queue_name};
        
        // Convert timeout to chrono duration
        std::chrono::seconds timeout(timeout_seconds > 0 ? timeout_seconds : 1);
        
        // Blocking pop with timeout
        auto result = redis_client_->blpop(keys.begin(), keys.end(), timeout);
        if (result) {
            message = result->second;
            return true;
        }
        
        // Timeout occurred (no message available)
        return false;
    } catch (const sw::redis::TimeoutError&) {
        // Timeout is expected when no message is available, not an error
        return false;
    } catch (const sw::redis::Error& e) {
        LOG_ERROR("Redis", "BLPOP error: " + std::string(e.what()));
        handleError("BLPOP");
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "BLPOP exception: " + std::string(e.what()));
        handleError("BLPOP");
        return false;
    }
}

bool RedisQueue::pushMessage(const std::string& queue_name, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!redis_client_) {
        if (!reconnect()) {
            return false;
        }
    }
    
    try {
        // Use RPUSH to push message to queue
        redis_client_->rpush(queue_name, message);
        return true;
    } catch (const sw::redis::Error& e) {
        LOG_ERROR("Redis", "RPUSH error: " + std::string(e.what()));
        handleError("RPUSH");
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "RPUSH exception: " + std::string(e.what()));
        handleError("RPUSH");
        return false;
    }
}
