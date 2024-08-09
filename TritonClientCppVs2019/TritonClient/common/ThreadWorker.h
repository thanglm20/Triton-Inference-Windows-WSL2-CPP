#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

class Worker {
public:
    Worker() {}
    virtual ~Worker() { stop(); }

    void start() {
        _thread = std::thread(std::bind(&Worker::run, this));
        _thread.detach();
    }

    void stop() {
        _stop.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (_thread.joinable()) {
            _thread.join();
        }
    }

    void pause() {
        _is_paused.store(true);
    }

    void resume() {
        _is_paused.store(false);
        _cond.notify_one();
    }

    void wait() {
        {
            std::unique_lock<std::mutex> l(_mutex);
            _cond.wait(l, [this]() { return !this->_is_paused.load(); });
        }
    }

    bool is_running() {
        return !_stop.load();
    }

public:
    virtual void run() = 0;
private:
    std::thread _thread;
    std::atomic<bool> _stop{ false };
    std::condition_variable _cond;
    std::mutex _mutex;
    std::atomic<bool> _is_paused{ false };
};

