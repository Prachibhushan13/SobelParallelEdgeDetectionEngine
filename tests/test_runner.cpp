#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include "image_io.h"
#include "sobel_sequential.h"
#include "sobel_openmp.h"
#include "sobel_metal.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_tests_failed++; \
            return; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "[TEST] Running " << #fn << "...\n"; \
        int initial_failed = g_tests_failed; \
        fn(); \
        if (g_tests_failed == initial_failed) { \
            std::cout << "  [PASS] " << #fn << " passed.\n"; \
            g_tests_passed++; \
        } \
    } while (0)

// Test 1: Image Parser handles valid PGM (both P5 binary and P2 ASCII) and rejects invalid files
void test_image_parser_valid_and_invalid() {
    std::string err;
    Image img;

    // Reject non-existent file
    bool ok = load_pgm("non_existent_file_12345.pgm", img, err);
    TEST_ASSERT(!ok, "Should reject non-existent file");

    // Reject invalid header (create dummy text file)
    {
        std::ofstream bad_file("test_bad_header.pgm");
        bad_file << "P9\n10 10\n255\n";
    }
    ok = load_pgm("test_bad_header.pgm", img, err);
    TEST_ASSERT(!ok, "Should reject invalid magic number P9");
    std::remove("test_bad_header.pgm");

    // Reject truncated file
    {
        std::ofstream trunc_file("test_truncated.pgm", std::ios::binary);
        trunc_file << "P5\n100 100\n255\n";
        // Write only 10 bytes instead of 10000
        char dummy[10] = {0};
        trunc_file.write(dummy, 10);
    }
    ok = load_pgm("test_truncated.pgm", img, err);
    TEST_ASSERT(!ok, "Should reject truncated P5 file");
    std::remove("test_truncated.pgm");

    // Valid P5 file roundtrip
    Image original(64, 48, 128);
    original.set(10, 10, 255);
    original.set(20, 20, 42);
    ok = save_pgm("test_valid_p5.pgm", original, err);
    TEST_ASSERT(ok, "Should successfully save valid P5");

    Image loaded;
    ok = load_pgm("test_valid_p5.pgm", loaded, err);
    TEST_ASSERT(ok, "Should successfully load valid P5");
    TEST_ASSERT(loaded.width == 64, "Loaded width must match");
    TEST_ASSERT(loaded.height == 48, "Loaded height must match");
    TEST_ASSERT(loaded.get(10, 10) == 255, "Pixel at (10, 10) must match");
    TEST_ASSERT(loaded.get(20, 20) == 42, "Pixel at (20, 20) must match");
    TEST_ASSERT(loaded.get(0, 0) == 128, "Pixel at (0, 0) must match");
    std::remove("test_valid_p5.pgm");
}

// Test 2: Boundary pixels are handled safely (must be zeroed and no out-of-bounds memory access)
void test_boundary_handling() {
    Image img(32, 32, 200); // Uniform image with bright pixels
    SobelResult res_seq = sobel_sequential(img);
    TEST_ASSERT(res_seq.success, "Sequential should succeed on 32x32 image");

    // Check boundary row 0
    for (int x = 0; x < 32; ++x) {
        TEST_ASSERT(res_seq.output_image.get(x, 0) == 0, "Top boundary pixel must be 0");
        TEST_ASSERT(res_seq.output_image.get(x, 31) == 0, "Bottom boundary pixel must be 0");
    }
    // Check boundary col 0
    for (int y = 0; y < 32; ++y) {
        TEST_ASSERT(res_seq.output_image.get(0, y) == 0, "Left boundary pixel must be 0");
        TEST_ASSERT(res_seq.output_image.get(31, y) == 0, "Right boundary pixel must be 0");
    }

    // On uniform image, interior pixels should also have gradient = 0
    for (int y = 1; y < 31; ++y) {
        for (int x = 1; x < 31; ++x) {
            TEST_ASSERT(res_seq.output_image.get(x, y) == 0, "Uniform interior pixel gradient must be 0");
        }
    }
}

// Test 3: Mathematical edge detection verification on simple vertical and horizontal steps
void test_sobel_mathematical_precision() {
    // 5x5 image with a sharp vertical step: left 2 columns = 0, right 3 columns = 100
    // [  0   0 100 100 100 ]
    // [  0   0 100 100 100 ]
    // [  0   0 100 100 100 ]
    // [  0   0 100 100 100 ]
    // [  0   0 100 100 100 ]
    Image img(5, 5, 0);
    for (int y = 0; y < 5; ++y) {
        for (int x = 2; x < 5; ++x) {
            img.set(x, y, 100);
        }
    }

    SobelResult res = sobel_sequential(img);
    TEST_ASSERT(res.success, "Sequential Sobel on step edge must succeed");

    // At pixel (x=2, y=2):
    // Gx:
    // row_prev: -1*(0) + 1*(100) = 100
    // row_curr: -2*(0) + 2*(100) = 200
    // row_next: -1*(0) + 1*(100) = 100
    // Total Gx = 400
    // Gy:
    // row_prev: -1*(0) - 2*(100) - 1*(100) = -300
    // row_curr: 0
    // row_next: 1*(0) + 2*(100) + 1*(100) = 300
    // Total Gy = 0
    // G = sqrt(400^2 + 0) = 400 -> clamped to 255
    TEST_ASSERT(res.output_image.get(2, 2) == 255, "Vertical step edge magnitude at (2,2) should clamp to 255");
}

// Test 4: Sequential and OpenMP outputs match identically across multiple thread configurations
void test_sequential_vs_openmp_equivalence() {
    Image test_img = create_synthetic_pattern(256, 256, "shapes");

    SobelResult res_seq = sobel_sequential(test_img);
    TEST_ASSERT(res_seq.success, "Sequential execution must succeed");

    std::vector<int> thread_counts = {1, 2, 4, 8};
    for (int t : thread_counts) {
        SobelResult res_omp = sobel_openmp(test_img, t, "static");
        TEST_ASSERT(res_omp.success, "OpenMP execution must succeed");

        ImageDiffResult diff = compare_images(res_seq.output_image, res_omp.output_image);
        TEST_ASSERT(diff.exact_match, "OpenMP output must match Sequential exactly with thread count " + std::to_string(t));
        TEST_ASSERT(diff.max_difference == 0, "Max difference must be 0");
        TEST_ASSERT(diff.mean_absolute_error == 0.0, "MAE must be 0.0");
    }
}

// Test 5: Metal GPU output matches sequential implementation
void test_metal_gpu_equivalence() {
    if (!is_metal_gpu_available()) {
        std::cout << "  [SKIP] Metal GPU not available on this platform.\n";
        return;
    }

    Image test_img = create_synthetic_pattern(256, 256, "chessboard");
    SobelResult res_seq = sobel_sequential(test_img);
    SobelResult res_gpu = sobel_metal(test_img);

    TEST_ASSERT(res_gpu.success, "Metal GPU execution must succeed");

    ImageDiffResult diff = compare_images(res_seq.output_image, res_gpu.output_image);
    TEST_ASSERT(diff.exact_match, "Metal GPU output must match Sequential exactly");
    TEST_ASSERT(diff.max_difference == 0, "GPU Max diff must be 0");
}

// Test 6: Repeated runs produce deterministic, consistent output
void test_reproducibility_and_consistency() {
    Image test_img = create_synthetic_pattern(128, 128, "circles");

    SobelResult r1 = sobel_openmp(test_img, 4, "static");
    SobelResult r2 = sobel_openmp(test_img, 4, "static");

    TEST_ASSERT(r1.success && r2.success, "Repeated OpenMP runs must succeed");
    ImageDiffResult diff = compare_images(r1.output_image, r2.output_image);
    TEST_ASSERT(diff.exact_match, "Repeated runs must be strictly identical");
}

// Test 7: Timing measurements are positive and reasonable
void test_benchmark_metrics_validity() {
    Image test_img = create_synthetic_pattern(512, 512, "shapes");

    SobelResult res_seq = sobel_sequential(test_img);
    SobelResult res_omp = sobel_openmp(test_img, 4);

    TEST_ASSERT(res_seq.timing.kernel_time_ms > 0.0, "Sequential time must be > 0");
    TEST_ASSERT(res_omp.timing.kernel_time_ms > 0.0, "OpenMP time must be > 0");

    double speedup = res_seq.timing.kernel_time_ms / res_omp.timing.kernel_time_ms;
    double efficiency = (speedup / 4) * 100.0;

    TEST_ASSERT(speedup > 0.0, "Speedup must be positive");
    TEST_ASSERT(efficiency > 0.0, "Efficiency must be positive");
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "      BCS702 Sobel Edge Detection Automated Test Suite\n";
    std::cout << "======================================================================\n";

    RUN_TEST(test_image_parser_valid_and_invalid);
    RUN_TEST(test_boundary_handling);
    RUN_TEST(test_sobel_mathematical_precision);
    RUN_TEST(test_sequential_vs_openmp_equivalence);
    RUN_TEST(test_metal_gpu_equivalence);
    RUN_TEST(test_reproducibility_and_consistency);
    RUN_TEST(test_benchmark_metrics_validity);

    std::cout << "======================================================================\n";
    std::cout << "Test Summary: " << g_tests_passed << " Passed, "
              << g_tests_failed << " Failed.\n";
    std::cout << "======================================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
