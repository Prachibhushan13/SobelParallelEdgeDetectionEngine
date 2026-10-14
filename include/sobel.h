#ifndef SOBEL_H
#define SOBEL_H

#include "image_io.h"
#include <string>
#include <chrono>

/**
 * @brief High-precision timing breakdown.
 */
struct SobelTiming {
    double kernel_time_ms = 0.0;     // Pure computation / kernel execution time
    double h2d_time_ms = 0.0;        // Host to Device transfer (GPU)
    double d2h_time_ms = 0.0;        // Device to Host readback (GPU)
    double total_time_ms = 0.0;      // End-to-end execution time (excluding file I/O)
};

/**
 * @brief Execution result containing output image, timing, and metadata.
 */
struct SobelResult {
    Image output_image;
    SobelTiming timing;
    std::string mode;
    int threads_used = 1;
    bool success = false;
    std::string error_message;
};

// Sobel convolution 3x3 kernels (constant definitions)
inline constexpr int SOBEL_GX[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

inline constexpr int SOBEL_GY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

#endif // SOBEL_H
