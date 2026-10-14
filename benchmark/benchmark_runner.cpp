#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <cmath>
#include "image_io.h"
#include "sobel_sequential.h"
#include "sobel_openmp.h"
#include "sobel_metal.h"

struct BenchmarkConfig {
    std::vector<int> image_sizes = {512, 1024, 2048};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    int repetitions = 5;
    int warmup_runs = 2;
    std::string output_csv = "results/benchmark_results.csv";
};

struct BenchmarkRecord {
    int image_width = 0;
    int image_height = 0;
    std::string implementation;
    int threads = 1;
    double avg_kernel_time_ms = 0.0;
    double stddev_kernel_time_ms = 0.0;
    double avg_h2d_ms = 0.0;
    double avg_d2h_ms = 0.0;
    double avg_total_ms = 0.0;
    double speedup = 1.0;
    double efficiency_percent = 100.0;
    bool verified_match = true;
};

static void compute_stats(const std::vector<double>& times, double& avg, double& stddev) {
    if (times.empty()) {
        avg = 0.0;
        stddev = 0.0;
        return;
    }
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    avg = sum / times.size();

    double accum = 0.0;
    for (double d : times) {
        accum += (d - avg) * (d - avg);
    }
    stddev = std::sqrt(accum / times.size());
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    BenchmarkConfig config;
    int max_hw_threads = get_openmp_max_threads();
    if (std::find(config.thread_counts.begin(), config.thread_counts.end(), max_hw_threads) == config.thread_counts.end()) {
        config.thread_counts.push_back(max_hw_threads);
        std::sort(config.thread_counts.begin(), config.thread_counts.end());
    }

    std::cout << "======================================================================\n";
    std::cout << "         BCS702 Sobel Operator Comprehensive Benchmarking Suite\n";
    std::cout << "======================================================================\n";
    std::cout << "Repetitions per test: " << config.repetitions << " (Warmups: " << config.warmup_runs << ")\n";
    std::cout << "Image Resolutions   : 512x512, 1024x1024, 2048x2048\n";
    std::cout << "Thread Counts       : ";
    for (int t : config.thread_counts) std::cout << t << " ";
    std::cout << "\nGPU Target          : " << (is_metal_gpu_available() ? get_metal_device_name() : "None") << "\n";
    std::cout << "Output CSV File     : " << config.output_csv << "\n";
    std::cout << "----------------------------------------------------------------------\n\n";

    std::vector<BenchmarkRecord> all_records;

    for (int size : config.image_sizes) {
        std::cout << ">>> Benchmarking Image Resolution: " << size << " x " << size << " ("
                  << (size * size) / 1000000.0 << " Megapixels)...\n";

        // Generate synthetic test image for exact reproducibility
        Image test_img = create_synthetic_pattern(size, size, "shapes");

        // Save reference images for inspection
        std::string sample_in_path = "images/benchmark_input_" + std::to_string(size) + ".pgm";
        std::string err;
        save_pgm(sample_in_path, test_img, err);

        // -------------------------------------------------------------
        // 1. Sequential Baseline
        // -------------------------------------------------------------
        std::cout << "  [1/3] Benchmarking Sequential CPU Baseline...\n";
        for (int w = 0; w < config.warmup_runs; ++w) {
            sobel_sequential(test_img);
        }

        std::vector<double> seq_times;
        SobelResult seq_ref;
        for (int r = 0; r < config.repetitions; ++r) {
            SobelResult res = sobel_sequential(test_img);
            seq_times.push_back(res.timing.kernel_time_ms);
            if (r == 0) seq_ref = res;
        }

        double seq_avg = 0.0, seq_stddev = 0.0;
        compute_stats(seq_times, seq_avg, seq_stddev);

        BenchmarkRecord seq_rec;
        seq_rec.image_width = size;
        seq_rec.image_height = size;
        seq_rec.implementation = "Sequential";
        seq_rec.threads = 1;
        seq_rec.avg_kernel_time_ms = seq_avg;
        seq_rec.stddev_kernel_time_ms = seq_stddev;
        seq_rec.avg_total_ms = seq_avg;
        seq_rec.speedup = 1.0;
        seq_rec.efficiency_percent = 100.0;
        seq_rec.verified_match = true;
        all_records.push_back(seq_rec);

        std::cout << "        Avg Time: " << std::fixed << std::setprecision(3) << seq_avg
                  << " ms (+/- " << seq_stddev << " ms)\n";

        // -------------------------------------------------------------
        // 2. OpenMP Parallel Across Thread Counts
        // -------------------------------------------------------------
        std::cout << "  [2/3] Benchmarking OpenMP Multithreading...\n";
        for (int threads : config.thread_counts) {
            // Warmup
            for (int w = 0; w < config.warmup_runs; ++w) {
                sobel_openmp(test_img, threads, "static");
            }

            std::vector<double> omp_times;
            SobelResult omp_last;
            for (int r = 0; r < config.repetitions; ++r) {
                SobelResult res = sobel_openmp(test_img, threads, "static");
                omp_times.push_back(res.timing.kernel_time_ms);
                if (r == 0) omp_last = res;
            }

            double omp_avg = 0.0, omp_stddev = 0.0;
            compute_stats(omp_times, omp_avg, omp_stddev);

            double speedup = seq_avg / (omp_avg > 0 ? omp_avg : 0.0001);
            double efficiency = (speedup / threads) * 100.0;

            ImageDiffResult diff = compare_images(seq_ref.output_image, omp_last.output_image);

            BenchmarkRecord omp_rec;
            omp_rec.image_width = size;
            omp_rec.image_height = size;
            omp_rec.implementation = "OpenMP";
            omp_rec.threads = threads;
            omp_rec.avg_kernel_time_ms = omp_avg;
            omp_rec.stddev_kernel_time_ms = omp_stddev;
            omp_rec.avg_total_ms = omp_avg;
            omp_rec.speedup = speedup;
            omp_rec.efficiency_percent = efficiency;
            omp_rec.verified_match = diff.exact_match;
            all_records.push_back(omp_rec);

            std::cout << "        Threads: " << std::setw(2) << threads
                      << " | Avg: " << std::setw(7) << std::fixed << std::setprecision(3) << omp_avg << " ms"
                      << " | Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x"
                      << " | Efficiency: " << std::setw(5) << std::fixed << std::setprecision(1) << efficiency << "%"
                      << " | Verified: " << (diff.exact_match ? "PASS" : "FAIL") << "\n";
        }

        // -------------------------------------------------------------
        // 3. Apple Metal GPU Acceleration
        // -------------------------------------------------------------
        if (is_metal_gpu_available()) {
            std::cout << "  [3/3] Benchmarking Apple Metal GPU Acceleration...\n";
            // Warmup
            for (int w = 0; w < config.warmup_runs; ++w) {
                sobel_metal(test_img);
            }

            std::vector<double> gpu_kernel_times;
            std::vector<double> gpu_h2d_times;
            std::vector<double> gpu_d2h_times;
            std::vector<double> gpu_total_times;
            SobelResult gpu_last;

            for (int r = 0; r < config.repetitions; ++r) {
                SobelResult res = sobel_metal(test_img);
                gpu_kernel_times.push_back(res.timing.kernel_time_ms);
                gpu_h2d_times.push_back(res.timing.h2d_time_ms);
                gpu_d2h_times.push_back(res.timing.d2h_time_ms);
                gpu_total_times.push_back(res.timing.total_time_ms);
                if (r == 0) gpu_last = res;
            }

            double k_avg, k_stddev, h2d_avg, h2d_stddev, d2h_avg, d2h_stddev, tot_avg, tot_stddev;
            compute_stats(gpu_kernel_times, k_avg, k_stddev);
            compute_stats(gpu_h2d_times, h2d_avg, h2d_stddev);
            compute_stats(gpu_d2h_times, d2h_avg, d2h_stddev);
            compute_stats(gpu_total_times, tot_avg, tot_stddev);

            double gpu_k_speedup = seq_avg / (k_avg > 0 ? k_avg : 0.0001);
            double gpu_tot_speedup = seq_avg / (tot_avg > 0 ? tot_avg : 0.0001);

            ImageDiffResult diff = compare_images(seq_ref.output_image, gpu_last.output_image);

            BenchmarkRecord gpu_rec;
            gpu_rec.image_width = size;
            gpu_rec.image_height = size;
            gpu_rec.implementation = "Metal_GPU";
            gpu_rec.threads = 1; // GPU kernel
            gpu_rec.avg_kernel_time_ms = k_avg;
            gpu_rec.stddev_kernel_time_ms = k_stddev;
            gpu_rec.avg_h2d_ms = h2d_avg;
            gpu_rec.avg_d2h_ms = d2h_avg;
            gpu_rec.avg_total_ms = tot_avg;
            gpu_rec.speedup = gpu_k_speedup;
            gpu_rec.efficiency_percent = 0.0; // N/A for GPU
            gpu_rec.verified_match = diff.exact_match;
            all_records.push_back(gpu_rec);

            std::cout << "        Kernel Avg: " << std::fixed << std::setprecision(3) << k_avg << " ms"
                      << " (Speedup: " << std::fixed << std::setprecision(2) << gpu_k_speedup << "x)\n"
                      << "        H2D: " << h2d_avg << " ms | D2H: " << d2h_avg << " ms | Total End-to-End: "
                      << tot_avg << " ms (Speedup: " << gpu_tot_speedup << "x)\n"
                      << "        Verified: " << (diff.exact_match ? "PASS" : "FAIL") << "\n";
        }
        std::cout << "\n";
    }

    // -------------------------------------------------------------
    // Save to CSV
    // -------------------------------------------------------------
    std::ofstream csv(config.output_csv);
    if (!csv.is_open()) {
        std::cerr << "Error writing to " << config.output_csv << "\n";
        return 1;
    }

    csv << "ImageSize,Width,Height,Implementation,Threads,KernelTime_ms,StdDev_ms,H2D_ms,D2H_ms,TotalTime_ms,Speedup,Efficiency_pct,Verified\n";
    for (const auto& r : all_records) {
        csv << r.image_width << "x" << r.image_height << ","
            << r.image_width << ","
            << r.image_height << ","
            << r.implementation << ","
            << r.threads << ","
            << std::fixed << std::setprecision(4) << r.avg_kernel_time_ms << ","
            << std::fixed << std::setprecision(4) << r.stddev_kernel_time_ms << ","
            << std::fixed << std::setprecision(4) << r.avg_h2d_ms << ","
            << std::fixed << std::setprecision(4) << r.avg_d2h_ms << ","
            << std::fixed << std::setprecision(4) << r.avg_total_ms << ","
            << std::fixed << std::setprecision(3) << r.speedup << ","
            << std::fixed << std::setprecision(2) << r.efficiency_percent << ","
            << (r.verified_match ? "TRUE" : "FALSE") << "\n";
    }
    csv.close();

    std::cout << "======================================================================\n";
    std::cout << "[SUCCESS] Benchmark completed! Results written to: " << config.output_csv << "\n";
    std::cout << "======================================================================\n";

    return 0;
}
