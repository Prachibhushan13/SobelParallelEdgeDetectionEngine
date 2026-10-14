#ifndef SOBEL_METAL_H
#define SOBEL_METAL_H

#include "sobel.h"

/**
 * @brief Checks if Apple Metal GPU compute is available on this system.
 */
bool is_metal_gpu_available();

/**
 * @brief Returns the device name of the Apple Metal GPU (e.g. "Apple M3 Pro").
 */
std::string get_metal_device_name();

/**
 * @brief Runs Sobel edge detection using Apple Silicon Metal compute pipeline.
 * Separately measures H2D transfer time, GPU kernel execution time, and D2H readback time.
 * @param input Grayscale input image
 * @return SobelResult with edge detected image, timings, and GPU device metadata
 */
SobelResult sobel_metal(const Image& input);

#endif // SOBEL_METAL_H
