#include "redis_queue.h"
#include "logger.h"
#include <cstring>
#include <chrono>
#include <thread>

RedisQueue::RedisQueue(const std::string& host, int port, const std::string& password)
    : host_(host), port_(port), password_(password), context_(nullptr) {
}

RedisQueue::~RedisQueue() {
    disconnect();
}

bool RedisQueue::connect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (context_ != nullptr) {
        redisFree(context_);
        context_ = nullptr;
    }
    
    struct timeval timeout = {1, 500000};  // 1.5 seconds
    context_ = redisConnectWithTimeout(host_.c_str(), port_, timeout);
    
    if (context_ == nullptr || context_->err) {
        if (context_) {
            LOG_ERROR("Redis", "Connection error: " + std::string(context_->errstr));
            redisFree(context_);
            context_ = nullptr;
        } else {
            LOG_ERROR("Redis", "Connection error: can't allocate redis context");
        }
        return false;
    }
    
    // Authenticate if password is provided
    if (!password_.empty()) {
        redisReply* reply = (redisReply*)redisCommand(context_, "AUTH %s", password_.c_str());
        if (reply == nullptr || reply->type == REDIS_REPLY_ERROR) {
            std::string error = reply ? reply->str : "AUTH command failed";
            LOG_ERROR("Redis", "Authentication failed: " + error);
            freeReplyObject(reply);
            redisFree(context_);
            context_ = nullptr;
            return false;
        }
        freeReplyObject(reply);
    }
    
    LOG_INFO("Redis", "Connected to Redis at " + host_ + ":" + std::to_string(port_));
    return true;
}

void RedisQueue::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_ != nullptr) {
        redisFree(context_);
        context_ = nullptr;
    }
}

bool RedisQueue::reconnect() {
    disconnect();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return connect();
}

void RedisQueue::handleError(const std::string& operation) {
    if (context_ && context_->err) {
        LOG_ERROR("Redis", operation + " error: " + std::string(context_->errstr));
        if (context_->err == REDIS_ERR_IO || context_->err == REDIS_ERR_EOF) {
            // Connection lost, try to reconnect
            LOG_INFO("Redis", "Attempting to reconnect...");
            reconnect();
        }
    }
}

bool RedisQueue::popMessage(std::string& message, int timeout_seconds, const std::string& queue_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isConnected()) {
        if (!reconnect()) {
            return false;
        }
    }
    
    // Use BLPOP for blocking pop with timeout
    redisReply* reply = nullptr;
    if (timeout_seconds > 0) {
        reply = (redisReply*)redisCommand(context_, "BLPOP %s %d", queue_name.c_str(), timeout_seconds);
    } else {
        reply = (redisReply*)redisCommand(context_, "BLPOP %s 0", queue_name.c_str());
    }
    
    if (reply == nullptr) {
        handleError("BLPOP");
        return false;
    }
    
    if (reply->type == REDIS_REPLY_ERROR) {
        LOG_ERROR("Redis", "BLPOP error: " + std::string(reply->str));
        freeReplyObject(reply);
        return false;
    }
    
    if (reply->type == REDIS_REPLY_ARRAY && reply->elements >= 2) {
        // BLPOP returns [queue_name, message]
        message = std::string(reply->element[1]->str, reply->element[1]->len);
        freeReplyObject(reply);
        return true;
    }
    
    // Timeout or empty
    freeReplyObject(reply);
    return false;
}

bool RedisQueue::pushMessage(const std::string& queue_name, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isConnected()) {
        if (!reconnect()) {
            return false;
        }
    }
    
    redisReply* reply = (redisReply*)redisCommand(context_, "RPUSH %s %b", 
                                                  queue_name.c_str(), 
                                                  message.c_str(), 
                                                  message.length());
    
    if (reply == nullptr) {
        handleError("RPUSH");
        return false;
    }
    
    bool success = (reply->type != REDIS_REPLY_ERROR);
    if (!success && reply->type == REDIS_REPLY_ERROR) {
        LOG_ERROR("Redis", "RPUSH error: " + std::string(reply->str));
    }
    
    freeReplyObject(reply);
    return success;
}

