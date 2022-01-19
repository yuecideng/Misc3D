#pragma once

#include <chrono>

namespace misc3d {

/**
 * @brief Timer for duration measurement.
 * 
 */
class Timer {
public:
    void Start() { t0 = std::chrono::high_resolution_clock::now(); }
    double Stop() {
        const double timestamp = std::chrono::duration<double>(
                                     std::chrono::high_resolution_clock::now() - t0)
                                     .count();
        return timestamp;
    }

private:
    std::chrono::high_resolution_clock::time_point t0;
};

}  // namespace misc3d