#ifndef REDIS_QUEUE_H
#define REDIS_QUEUE_H

#include <string>
#include <memory>
#include <mutex>

extern "C" {
#include <hiredis/hiredis.h>
}

class RedisQueue {
public:
    RedisQueue(const std::string& host, int port, const std::string& password = "");
    ~RedisQueue();
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const { return context_ != nullptr && context_->err == 0; }
    
    // Queue operations
    bool popMessage(std::string& message, int timeout_seconds = 0);  // 0 = blocking, >0 = timeout
    bool pushMessage(const std::string& queue_name, const std::string& message);
    
    // Get connection info
    std::string getHost() const { return host_; }
    int getPort() const { return port_; }
    
private:
    std::string host_;
    int port_;
    std::string password_;
    redisContext* context_;
    std::mutex mutex_;
    
    bool reconnect();
    void handleError(const std::string& operation);
};

#endif // REDIS_QUEUE_H

