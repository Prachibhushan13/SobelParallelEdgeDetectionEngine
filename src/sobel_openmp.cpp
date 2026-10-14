#include "sobel_openmp.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

int get_openmp_max_threads() {
    return omp_get_max_threads();
}

SobelResult sobel_openmp(const Image& input, int num_threads, const std::string& schedule_type) {
    SobelResult result;
    result.mode = "openmp";

    if (input.empty()) {
        result.success = false;
        result.error_message = "Input image is empty";
        return result;
    }

    const int max_hw_threads = omp_get_max_threads();
    int threads = (num_threads > 0) ? num_threads : max_hw_threads;
    result.threads_used = threads;

    const int width = input.width;
    const int height = input.height;

    // Allocate output image initialized to 0
    result.output_image = Image(width, height, 0);

    if (width < 3 || height < 3) {
        result.success = true;
        result.timing.kernel_time_ms = 0.0;
        result.timing.total_time_ms = 0.0;
        return result;
    }

    const uint8_t* in_data = input.data.data();
    uint8_t* out_data = result.output_image.data.data();

    auto start_time = std::chrono::steady_clock::now();

    if (schedule_type == "dynamic") {
        #pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
        for (int y = 1; y < height - 1; ++y) {
            const uint8_t* row_prev = in_data + (y - 1) * width;
            const uint8_t* row_curr = in_data + y * width;
            const uint8_t* row_next = in_data + (y + 1) * width;
            uint8_t* row_out = out_data + y * width;

            for (int x = 1; x < width - 1; ++x) {
                int gx = -static_cast<int>(row_prev[x - 1]) + static_cast<int>(row_prev[x + 1])
                         - 2 * static_cast<int>(row_curr[x - 1]) + 2 * static_cast<int>(row_curr[x + 1])
                         - static_cast<int>(row_next[x - 1]) + static_cast<int>(row_next[x + 1]);

                int gy = -static_cast<int>(row_prev[x - 1]) - 2 * static_cast<int>(row_prev[x]) - static_cast<int>(row_prev[x + 1])
                         + static_cast<int>(row_next[x - 1]) + 2 * static_cast<int>(row_next[x]) + static_cast<int>(row_next[x + 1]);

                double magnitude = std::sqrt(static_cast<double>(gx * gx + gy * gy));
                row_out[x] = static_cast<uint8_t>(std::clamp(magnitude, 0.0, 255.0));
            }
        }
    } else if (schedule_type == "guided") {
        #pragma omp parallel for num_threads(threads) schedule(guided)
        for (int y = 1; y < height - 1; ++y) {
            const uint8_t* row_prev = in_data + (y - 1) * width;
            const uint8_t* row_curr = in_data + y * width;
            const uint8_t* row_next = in_data + (y + 1) * width;
            uint8_t* row_out = out_data + y * width;

            for (int x = 1; x < width - 1; ++x) {
                int gx = -static_cast<int>(row_prev[x - 1]) + static_cast<int>(row_prev[x + 1])
                         - 2 * static_cast<int>(row_curr[x - 1]) + 2 * static_cast<int>(row_curr[x + 1])
                         - static_cast<int>(row_next[x - 1]) + static_cast<int>(row_next[x + 1]);

                int gy = -static_cast<int>(row_prev[x - 1]) - 2 * static_cast<int>(row_prev[x]) - static_cast<int>(row_prev[x + 1])
                         + static_cast<int>(row_next[x - 1]) + 2 * static_cast<int>(row_next[x]) + static_cast<int>(row_next[x + 1]);

                double magnitude = std::sqrt(static_cast<double>(gx * gx + gy * gy));
                row_out[x] = static_cast<uint8_t>(std::clamp(magnitude, 0.0, 255.0));
            }
        }
    } else {
        // Default: schedule(static) - optimal for uniform stencil computations
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int y = 1; y < height - 1; ++y) {
            const uint8_t* row_prev = in_data + (y - 1) * width;
            const uint8_t* row_curr = in_data + y * width;
            const uint8_t* row_next = in_data + (y + 1) * width;
            uint8_t* row_out = out_data + y * width;

            for (int x = 1; x < width - 1; ++x) {
                int gx = -static_cast<int>(row_prev[x - 1]) + static_cast<int>(row_prev[x + 1])
                         - 2 * static_cast<int>(row_curr[x - 1]) + 2 * static_cast<int>(row_curr[x + 1])
                         - static_cast<int>(row_next[x - 1]) + static_cast<int>(row_next[x + 1]);

                int gy = -static_cast<int>(row_prev[x - 1]) - 2 * static_cast<int>(row_prev[x]) - static_cast<int>(row_prev[x + 1])
                         + static_cast<int>(row_next[x - 1]) + 2 * static_cast<int>(row_next[x]) + static_cast<int>(row_next[x + 1]);

                double magnitude = std::sqrt(static_cast<double>(gx * gx + gy * gy));
                row_out[x] = static_cast<uint8_t>(std::clamp(magnitude, 0.0, 255.0));
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    result.timing.kernel_time_ms = duration_ms;
    result.timing.total_time_ms = duration_ms;
    result.success = true;

    return result;
}
