#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include "image_io.h"
#include "sobel_sequential.h"
#include "sobel_openmp.h"
#include "sobel_metal.h"

void print_banner() {
    std::cout << "======================================================================\n";
    std::cout << "  BCS702: Parallel Image Edge Detection Using Sobel Operator\n";
    std::cout << "  Sequential C++, OpenMP Multithreading & Apple Metal GPU Acceleration\n";
    std::cout << "======================================================================\n";
}

void print_help(const char* prog_name) {
    print_banner();
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --input, -i <path>        Path to input PGM image (P5/P2)\n";
    std::cout << "  --output, -o <path>       Path to save output edge-detected PGM image\n";
    std::cout << "  --mode, -m <mode>         Execution mode: 'seq', 'omp', 'gpu', or 'all' (default: all)\n";
    std::cout << "  --threads, -t <num>       Number of OpenMP threads (default: system max = " << get_openmp_max_threads() << ")\n";
    std::cout << "  --schedule, -s <type>     OpenMP schedule: 'static', 'dynamic', 'guided' (default: static)\n";
    std::cout << "  --generate <pattern> <w> <h> <out.pgm>\n";
    std::cout << "                            Generate synthetic image ('chessboard', 'circles', 'gradient', 'shapes')\n";
    std::cout << "  --verify                  Verify exact output equivalence between sequential and parallel\n";
    std::cout << "  --help, -h                Display this help message and exit\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " --input images/shapes_1024.pgm --output output/edges.pgm --mode all\n";
    std::cout << "  " << prog_name << " --generate chessboard 1024 1024 images/chessboard_1024.pgm\n";
    std::cout << "  " << prog_name << " --input images/chessboard_1024.pgm --mode omp --threads 4 --verify\n";
}

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path = "output/edges_output.pgm";
    std::string mode = "all";
    std::string schedule = "static";
    int threads = get_openmp_max_threads();
    bool verify = false;

    // Argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--mode" || arg == "-m") {
            if (i + 1 < argc) mode = argv[++i];
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--schedule" || arg == "-s") {
            if (i + 1 < argc) schedule = argv[++i];
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--generate") {
            if (i + 4 < argc) {
                std::string pattern = argv[++i];
                int w = std::atoi(argv[++i]);
                int h = std::atoi(argv[++i]);
                std::string gen_out = argv[++i];
                if (w <= 0 || h <= 0) {
                    std::cerr << "Error: Invalid image dimensions: " << w << "x" << h << "\n";
                    return 1;
                }
                Image gen_img = create_synthetic_pattern(w, h, pattern);
                std::string err;
                if (!save_pgm(gen_out, gen_img, err)) {
                    std::cerr << "Error saving generated image: " << err << "\n";
                    return 1;
                }
                std::cout << "[SUCCESS] Generated " << pattern << " pattern (" << w << "x" << h << ") -> " << gen_out << "\n";
                return 0;
            } else {
                std::cerr << "Error: --generate requires <pattern> <width> <height> <out.pgm>\n";
                return 1;
            }
        }
    }

    print_banner();

    // Check system status
    std::cout << "[Hardware] Logical CPU Threads Available : " << get_openmp_max_threads() << "\n";
    std::cout << "[Hardware] GPU Acceleration Device        : "
              << (is_metal_gpu_available() ? get_metal_device_name() : "None") << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    // If no input path provided, auto-create a default test image
    Image input_image;
    if (input_path.empty()) {
        std::cout << "[Info] No input image provided. Generating default 1024x1024 synthetic image...\n";
        input_image = create_synthetic_pattern(1024, 1024, "shapes");
        input_path = "images/default_shapes_1024.pgm";
        std::string err;
        save_pgm(input_path, input_image, err);
    } else {
        std::string err;
        if (!load_pgm(input_path, input_image, err)) {
            std::cerr << "[Error] Failed to load input image: " << err << "\n";
            return 1;
        }
        std::cout << "[Loaded] Input image: " << input_path << " (" << input_image.width << "x" << input_image.height << ")\n";
    }

    SobelResult res_seq, res_omp, res_gpu;

    // Sequential Execution
    if (mode == "seq" || mode == "all" || verify) {
        std::cout << "\n>>> Running Sequential C++ Sobel...\n";
        res_seq = sobel_sequential(input_image);
        if (!res_seq.success) {
            std::cerr << "[Error] Sequential execution failed: " << res_seq.error_message << "\n";
            return 1;
        }
        std::cout << "    Kernel Time: " << std::fixed << std::setprecision(4)
                  << res_seq.timing.kernel_time_ms << " ms\n";
    }

    // OpenMP Execution
    if (mode == "omp" || mode == "all" || verify) {
        std::cout << "\n>>> Running OpenMP Parallel Sobel (" << threads << " threads, schedule=" << schedule << ")...\n";
        res_omp = sobel_openmp(input_image, threads, schedule);
        if (!res_omp.success) {
            std::cerr << "[Error] OpenMP execution failed: " << res_omp.error_message << "\n";
            return 1;
        }
        std::cout << "    Kernel Time: " << std::fixed << std::setprecision(4)
                  << res_omp.timing.kernel_time_ms << " ms\n";

        if (res_seq.success) {
            double speedup = res_seq.timing.kernel_time_ms / (res_omp.timing.kernel_time_ms > 0 ? res_omp.timing.kernel_time_ms : 0.0001);
            double efficiency = (speedup / threads) * 100.0;
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
            std::cout << "    Parallel Efficiency: " << std::fixed << std::setprecision(2) << efficiency << " %\n";
        }
    }

    // Metal GPU Execution
    if (mode == "gpu" || mode == "all") {
        if (is_metal_gpu_available()) {
            std::cout << "\n>>> Running Apple Metal GPU Parallel Sobel...\n";
            res_gpu = sobel_metal(input_image);
            if (!res_gpu.success) {
                std::cerr << "    [Warning] Metal GPU execution failed: " << res_gpu.error_message << "\n";
            } else {
                std::cout << "    H2D Transfer Time : " << std::fixed << std::setprecision(4) << res_gpu.timing.h2d_time_ms << " ms\n";
                std::cout << "    GPU Kernel Time   : " << std::fixed << std::setprecision(4) << res_gpu.timing.kernel_time_ms << " ms\n";
                std::cout << "    D2H Transfer Time : " << std::fixed << std::setprecision(4) << res_gpu.timing.d2h_time_ms << " ms\n";
                std::cout << "    Total GPU Time    : " << std::fixed << std::setprecision(4) << res_gpu.timing.total_time_ms << " ms\n";

                if (res_seq.success) {
                    double speedup_kernel = res_seq.timing.kernel_time_ms / (res_gpu.timing.kernel_time_ms > 0 ? res_gpu.timing.kernel_time_ms : 0.0001);
                    double speedup_total = res_seq.timing.kernel_time_ms / (res_gpu.timing.total_time_ms > 0 ? res_gpu.timing.total_time_ms : 0.0001);
                    std::cout << "    GPU Kernel Speedup: " << std::fixed << std::setprecision(2) << speedup_kernel << "x\n";
                    std::cout << "    GPU End-to-End Speedup: " << std::fixed << std::setprecision(2) << speedup_total << "x\n";
                }
            }
        } else {
            std::cout << "\n[Info] Metal GPU not supported on this platform.\n";
        }
    }

    // Verification
    if (verify || mode == "all") {
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << ">>> Verification & Equivalence Check:\n";
        if (res_seq.success && res_omp.success) {
            ImageDiffResult diff_omp = compare_images(res_seq.output_image, res_omp.output_image);
            std::cout << "    [Sequential vs OpenMP] Exact Match: "
                      << (diff_omp.exact_match ? "YES (Identical 100%)" : "NO")
                      << " | Max Diff: " << (int)diff_omp.max_difference
                      << " | MAE: " << diff_omp.mean_absolute_error << "\n";
        }
        if (res_seq.success && res_gpu.success) {
            ImageDiffResult diff_gpu = compare_images(res_seq.output_image, res_gpu.output_image);
            std::cout << "    [Sequential vs Metal GPU] Exact Match: "
                      << (diff_gpu.exact_match ? "YES (Identical 100%)" : "NO")
                      << " | Max Diff: " << (int)diff_gpu.max_difference
                      << " | MAE: " << diff_gpu.mean_absolute_error << "\n";
        }
    }

    // Save final output image
    const Image* to_save = nullptr;
    if (res_omp.success) to_save = &res_omp.output_image;
    else if (res_gpu.success) to_save = &res_gpu.output_image;
    else if (res_seq.success) to_save = &res_seq.output_image;

    if (to_save) {
        std::string err;
        if (save_pgm(output_path, *to_save, err)) {
            std::cout << "\n[Saved] Edge detected output written to: " << output_path << "\n";
        } else {
            std::cerr << "[Error] Failed to save output image: " << err << "\n";
        }
    }

    std::cout << "======================================================================\n";
    return 0;
}
