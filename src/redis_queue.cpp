#include "redis_queue.h"
#include "logger.h"
#include <chrono>
#include <thread>
#include <stdexcept>

RedisQueue::RedisQueue(const std::string& host, int port, const std::string& password)
    : host_(host), port_(port), password_(password), redis_client_(nullptr) {
}

RedisQueue::~RedisQueue() {
    disconnect();
}

bool RedisQueue::connect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        redis_client_ = std::make_unique<cpp_redis::client>();
        
        // Connect to Redis
        redis_client_->connect(host_, port_, [](const std::string& host, std::size_t port, cpp_redis::client::connect_state status) {
            if (status == cpp_redis::client::connect_state::dropped) {
                LOG_ERROR("Redis", "Client disconnected from " + host + ":" + std::to_string(port));
            }
        });
        
        if (!redis_client_->is_connected()) {
            LOG_ERROR("Redis", "Failed to connect to Redis at " + host_ + ":" + std::to_string(port_));
            redis_client_.reset();
            return false;
        }
        
        // Authenticate if password is provided
        if (!password_.empty()) {
            std::promise<bool> auth_promise;
            auto auth_future = auth_promise.get_future();
            
            redis_client_->auth(password_, [&auth_promise](cpp_redis::reply& reply) {
                if (reply.is_error()) {
                    LOG_ERROR("Redis", "Authentication failed: " + reply.as_string());
                    auth_promise.set_value(false);
                } else {
                    auth_promise.set_value(true);
                }
            });
            
            redis_client_->sync_commit();
            
            if (!auth_future.get()) {
                LOG_ERROR("Redis", "Authentication failed");
                redis_client_.reset();
                return false;
            }
        }
        
        // Test connection with PING
        std::promise<bool> ping_promise;
        auto ping_future = ping_promise.get_future();
        
        redis_client_->ping([&ping_promise](cpp_redis::reply& reply) {
            if (reply.is_string() && reply.as_string() == "PONG") {
                ping_promise.set_value(true);
            } else {
                ping_promise.set_value(false);
            }
        });
        
        redis_client_->sync_commit();
        
        if (ping_future.get()) {
            LOG_INFO("Redis", "Connected to Redis at " + host_ + ":" + std::to_string(port_));
            return true;
        } else {
            LOG_ERROR("Redis", "Connection test failed: PING did not return PONG");
            redis_client_.reset();
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "Connection error: " + std::string(e.what()));
        redis_client_.reset();
        return false;
    }
}

void RedisQueue::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (redis_client_) {
        try {
            redis_client_->disconnect();
        } catch (...) {
            // Ignore disconnect errors
        }
        redis_client_.reset();
    }
}

bool RedisQueue::isConnected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!redis_client_) {
        return false;
    }
    
    try {
        return redis_client_->is_connected();
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
    
    if (!redis_client_ || !redis_client_->is_connected()) {
        if (!reconnect()) {
            return false;
        }
    }
    
    try {
        std::promise<bool> pop_promise;
        auto pop_future = pop_promise.get_future();
        
        // Use BRPOP for blocking pop with timeout
        // BRPOP returns [queue_name, message] or timeout
        // timeout: 0 = block indefinitely, >0 = timeout in seconds
        int timeout = (timeout_seconds > 0) ? timeout_seconds : 0;
        
        redis_client_->brpop({queue_name}, timeout, [&pop_promise, &message](cpp_redis::reply& reply) {
            if (reply.is_array()) {
                const auto& array = reply.as_array();
                if (array.size() >= 2 && array[1].is_string()) {
                    // BRPOP returns [queue_name, message]
                    message = array[1].as_string();
                    pop_promise.set_value(true);
                } else {
                    pop_promise.set_value(false);
                }
            } else {
                // Timeout or error
                pop_promise.set_value(false);
            }
        });
        
        redis_client_->sync_commit();
        
        // Wait for the result (with a reasonable timeout for sync_commit)
        if (pop_future.wait_for(std::chrono::seconds(timeout > 0 ? timeout + 1 : 2)) == std::future_status::ready) {
            return pop_future.get();
        } else {
            LOG_ERROR("Redis", "BRPOP operation timed out");
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "BRPOP exception: " + std::string(e.what()));
        handleError("BRPOP");
        return false;
    }
}

bool RedisQueue::pushMessage(const std::string& queue_name, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!redis_client_ || !redis_client_->is_connected()) {
        if (!reconnect()) {
            return false;
        }
    }
    
    try {
        std::promise<bool> push_promise;
        auto push_future = push_promise.get_future();
        
        // Use RPUSH to push message to queue
        redis_client_->rpush(queue_name, {message}, [&push_promise](cpp_redis::reply& reply) {
            if (reply.is_integer()) {
                // RPUSH returns the new length of the list, so push was successful
                push_promise.set_value(true);
            } else if (reply.is_error()) {
                LOG_ERROR("Redis", "RPUSH error: " + reply.as_string());
                push_promise.set_value(false);
            } else {
                push_promise.set_value(false);
            }
        });
        
        redis_client_->sync_commit();
        
        // Wait for the result
        if (push_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
            return push_future.get();
        } else {
            LOG_ERROR("Redis", "RPUSH operation timed out");
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Redis", "RPUSH exception: " + std::string(e.what()));
        handleError("RPUSH");
        return false;
    }
}
