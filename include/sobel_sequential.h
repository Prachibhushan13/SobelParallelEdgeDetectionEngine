#ifndef SOBEL_SEQUENTIAL_H
#define SOBEL_SEQUENTIAL_H

#include "sobel.h"

/**
 * @brief Runs Sobel edge detection sequentially on CPU without parallelization.
 * @param input Grayscale input image
 * @return SobelResult with edge detected image and execution timing
 */
SobelResult sobel_sequential(const Image& input);

#endif // SOBEL_SEQUENTIAL_H
