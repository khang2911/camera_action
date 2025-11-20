#ifndef REDIS_QUEUE_H
#define REDIS_QUEUE_H

#include <string>
#include <memory>
#include <mutex>
#include <future>
#include <cpp_redis/cpp_redis>

class RedisQueue {
public:
    RedisQueue(const std::string& host, int port, const std::string& password = "", int db = 0);
    ~RedisQueue();
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const;
    
    // Queue operations
    bool popMessage(std::string& message, int timeout_seconds = 0, const std::string& queue_name = "input_queue");
    bool pushMessage(const std::string& queue_name, const std::string& message);
    int getQueueLength(const std::string& queue_name);
    
    // Get connection info
    std::string getHost() const { return host_; }
    int getPort() const { return port_; }
    
private:
    std::string host_;
    int port_;
    std::string password_;
    int db_;
    std::unique_ptr<cpp_redis::client> redis_client_;
    mutable std::mutex mutex_;  // mutable to allow locking in const methods
    
    bool reconnect();
    void handleError(const std::string& operation);
};

#endif // REDIS_QUEUE_H
