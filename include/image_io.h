#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

/**
 * @brief Represents an 8-bit grayscale image.
 * Pixel buffer is stored in row-major order: index = y * width + x.
 */
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, uint8_t fill = 0)
        : width(w), height(h), data(w * h, fill) {}

    inline uint8_t get(int x, int y) const {
        return data[y * width + x];
    }

    inline void set(int x, int y, uint8_t val) {
        data[y * width + x] = val;
    }

    bool empty() const {
        return width <= 0 || height <= 0 || data.empty();
    }
};

/**
 * @brief Metrics for comparing two images.
 */
struct ImageDiffResult {
    bool exact_match = false;
    uint8_t max_difference = 0;
    double mean_absolute_error = 0.0;
    double mean_squared_error = 0.0;
    int differing_pixels = 0;
    int total_pixels = 0;
};

// PGM I/O Functions
bool load_pgm(const std::string& filepath, Image& image, std::string& error_msg);
bool save_pgm(const std::string& filepath, const Image& image, std::string& error_msg);

// Image verification
ImageDiffResult compare_images(const Image& img1, const Image& img2, uint8_t tolerance = 0);

// Synthetic Pattern Generators for Benchmarking and Testing
Image create_synthetic_pattern(int width, int height, const std::string& pattern_type);
Image create_gradient_ramp(int width, int height);
Image create_chessboard(int width, int height, int square_size = 32);
Image create_concentric_circles(int width, int height);
Image create_geometric_shapes(int width, int height);

#endif // IMAGE_IO_H
