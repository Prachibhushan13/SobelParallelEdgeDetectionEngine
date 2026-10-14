#ifndef SOBEL_OPENMP_H
#define SOBEL_OPENMP_H

#include "sobel.h"

/**
 * @brief Runs Sobel edge detection parallelized using OpenMP.
 * @param input Grayscale input image
 * @param num_threads Number of OpenMP threads to use (default: 0 for OpenMP auto-detect)
 * @param schedule_type Scheduling strategy ("static", "dynamic", "guided")
 * @return SobelResult with edge detected image, thread count, and execution timing
 */
SobelResult sobel_openmp(const Image& input, int num_threads = 0, const std::string& schedule_type = "static");

/**
 * @brief Returns maximum available hardware threads supported by OpenMP runtime.
 */
int get_openmp_max_threads();

#endif // SOBEL_OPENMP_H
