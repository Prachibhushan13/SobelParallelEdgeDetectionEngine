# Performance Analysis of Parallel Image Edge Detection Using the Sobel Operator with Sequential C++ and OpenMP

**Course:** VTU 7th-Semester Parallel Computing (BCS702)  
**Target Platform:** macOS (Apple Silicon M3 Pro & Intel x86_64) & Linux  
**Language Standards:** C++17, OpenMP 5.1, Apple Metal Shading Language, Python 3  

---

## 📌 Executive Summary

This repository delivers an academic and industrial-grade parallel computing project analyzing 2D spatial convolution using the **Sobel Edge Detection Operator**. The project compares:
1. **Sequential C++ Baseline:** Pure C++17 single-threaded execution using high-resolution `std::chrono` timers.
2. **OpenMP Multithreading:** Shared-memory row-wise domain decomposition supporting configurable thread counts ($p \in \{1, 2, 4, 8, 11\}$) and scheduling strategies (`static`, `dynamic`, `guided`).
3. **Apple Metal GPU Acceleration:** Native GPU compute pipeline written in Objective-C++ and Metal Shading Language (MSL) executing on Apple Silicon's integrated GPU.
4. **Automated Test Runner:** Rigorous test suite validating input parsers, boundary conditions, mathematical precision, and bit-exact equivalence ($MAE = 0.0$).
5. **Interactive Demonstration Dashboard:** A modern, local web dashboard (`web/server.py` + `web/index.html`) featuring real-time image processing, thread sliders, and live performance metrics.
6. **Academic Documentation:** Complete 19-section academic project report (`report/report_template.md`) and a 25-question viva-voce examination guide (`report/viva_questions_answers.md`).

---

## 🏗️ Repository Architecture

```
pcassign/
├── Makefile                      # Primary build automation (all, test, bench, plot, dashboard)
├── CMakeLists.txt                # Cross-platform CMake 3.16+ configuration
├── README.md                     # Comprehensive project documentation
├── include/
│   ├── image_io.h                # PGM image read/write (P5/P2), synthetic generators, diff metrics
│   ├── sobel.h                   # Common data structures (SobelResult, SobelTiming, Kernels)
│   ├── sobel_sequential.h       # Sequential C++ edge detector header
│   ├── sobel_openmp.h           # OpenMP parallel edge detector header
│   └── sobel_metal.h            # Apple Silicon Metal GPU acceleration header
├── src/
│   ├── image_io.cpp              # Robust PGM parsing and synthetic pattern generation
│   ├── sobel_sequential.cpp      # Single-threaded baseline with contiguous pointer math
│   ├── sobel_openmp.cpp          # OpenMP work-sharing construct implementation
│   ├── sobel_metal.mm           # Objective-C++ Metal compute pipeline and runtime shader compilation
│   └── main.cpp                  # Feature-complete command line interface (CLI)
├── tests/
│   └── test_runner.cpp           # Automated unit and integration test suite (7 test suites)
├── benchmark/
│   ├── benchmark_runner.cpp      # Automated benchmark harness with warmups and repetitions
│   └── plot_results.py           # Publication-quality plotting using Python/Matplotlib
├── web/
│   ├── server.py                 # Lightweight REST API server (zero external web frameworks)
│   └── index.html                # Modern glassmorphic interactive demonstration UI
├── images/                       # Generated and test grayscale images (512, 1024, 2048)
├── output/                       # Output edge-detected PGM images
├── results/                      # Benchmark CSV data and publication figures
└── report/
    ├── report_template.md        # Full 19-section VTU BCS702 academic report template
    └── viva_questions_answers.md # 4-member student breakdown & top 25 viva questions
```

---

## ⚡ Quickstart Commands

### 1. Build the Entire Project
```bash
make
```
*Builds `bin/sobel_app`, `bin/test_runner`, and `bin/benchmark_runner`.*

### 2. Run Automated Verification Tests
```bash
make test
```
*Executes all 7 test suites, verifying 100% mathematical equivalence across Sequential, OpenMP, and GPU.*

### 3. Run Edge Detection via CLI
```bash
# Generate a synthetic 1024x1024 test image with geometric shapes
./bin/sobel_app --generate shapes 1024 1024 images/demo.pgm

# Run all implementations (Sequential + OpenMP + Metal GPU) with 8 threads and equivalence check
./bin/sobel_app --input images/demo.pgm --output output/edges.pgm --mode all --threads 8 --verify
```

### 4. Run Comprehensive Benchmarks & Plot Results
```bash
# Run benchmark across 512x512, 1024x1024, 2048x2048 and thread counts
make bench

# Generate 5 publication-grade figures in results/
make plot
```

### 5. Launch Interactive Web Demonstration Dashboard
```bash
make dashboard
```
*Open [http://localhost:8080](http://localhost:8080) in your web browser.*

---

## 📊 Empirical Benchmark Results (Apple M3 Pro)

The following measurements were collected empirically on an **Apple M3 Pro** (11 hardware CPU cores, Apple Metal GPU):

| Image Size | Implementation | Threads ($p$) | Kernel Time ($\mu \pm \sigma$) | Speedup $S(p)$ | Efficiency $E(p)$ | Equivalence |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **512 &times; 512** | Sequential | 1 | $0.210\text{ ms}$ | $1.00\times$ | $100.0\%$ | Baseline |
| (0.26 MP) | OpenMP | 4 | $0.199\text{ ms}$ | $1.06\times$ | $26.4\%$ | Exact Match |
| | OpenMP | 8 | $0.166\text{ ms}$ | $1.27\times$ | $15.8\%$ | Exact Match |
| | Metal GPU (Kernel) | -- | $0.329\text{ ms}$ | $0.64\times$ | -- | Exact Match |
| **1024 &times; 1024**| Sequential | 1 | $0.691\text{ ms}$ | $1.00\times$ | $100.0\%$ | Baseline |
| (1.05 MP) | OpenMP | 4 | $0.467\text{ ms}$ | $1.48\times$ | $37.0\%$ | Exact Match |
| | OpenMP | 8 | $0.390\text{ ms}$ | $1.77\times$ | $22.2\%$ | Exact Match |
| | OpenMP | 11 | $0.374\text{ ms}$ | $1.85\times$ | $16.8\%$ | Exact Match |
| | Metal GPU (Kernel) | -- | $0.360\text{ ms}$ | **$1.92\times$** | -- | Exact Match |
| **2048 &times; 2048**| Sequential | 1 | $2.361\text{ ms}$ | $1.00\times$ | $100.0\%$ | Baseline |
| (4.19 MP) | OpenMP | 4 | $1.768\text{ ms}$ | $1.34\times$ | $33.4\%$ | Exact Match |
| | OpenMP | 8 | $1.470\text{ ms}$ | $1.61\times$ | $20.1\%$ | Exact Match |
| | OpenMP | 11 | $1.345\text{ ms}$ | $1.76\times$ | $16.0\%$ | Exact Match |
| | Metal GPU (Kernel) | -- | $1.022\text{ ms}$ | **$2.31\times$** | -- | Exact Match |
| | Metal GPU (Total) | -- | $1.560\text{ ms}$ | **$1.52\times$** | -- | Exact Match |

### Key Academic Findings:
1. **GPU Dispatch Overhead on Small Images:** At $512 \times 512$, CPU sequential execution is faster than GPU ($0.210\text{ ms}$ vs $0.329\text{ ms}$) because command buffer encoding and GPU submission latency dominate.
2. **GPU Scaling on Large Images:** At $2048 \times 2048$ (4.19 Megapixels), the Metal GPU kernel completes in $1.022\text{ ms}$, achieving a **$2.31\times$ speedup** over sequential CPU.
3. **OpenMP Scaling:** OpenMP scales from $6.305\text{ ms}$ (1 thread) down to $1.345\text{ ms}$ (11 threads), yielding a **$4.69\times$ intra-runtime speedup**.

---

## 📈 Generated Publication Figures

| Figure | Filename | Description |
|:---:|:---|:---|
| **Fig 1** | `results/fig1_execution_time_vs_size.png` | Execution Time vs Image Dimension ($512$ to $2048$) |
| **Fig 2** | `results/fig2_execution_time_vs_threads.png` | Execution Time vs Thread Count across resolutions |
| **Fig 3** | `results/fig3_speedup_vs_threads.png` | Speedup $S(p)$ vs Threads compared to Amdahl's Law |
| **Fig 4** | `results/fig4_efficiency_vs_threads.png` | Parallel Efficiency $E(p)$ vs Threads |
| **Summary**| `results/benchmark_summary.png` | 2&times;2 Comprehensive Academic Dashboard |

---

## 👥 Four-Member Team Work Division

- **Student 1:** Sobel Convolution Mathematics, PGM Image I/O, Sequential C++ Baseline.
- **Student 2:** OpenMP Parallelization, False Sharing Elimination, Scheduling Strategies (`static`, `dynamic`, `guided`).
- **Student 3:** Apple Metal GPU Compute Pipeline, MSL Kernel, Automated Test Suite (`tests/test_runner.cpp`).
- **Student 4:** Automated Benchmarking Suite, Matplotlib Visualization, Interactive Web Dashboard, Academic Report.

*See [`report/viva_questions_answers.md`](report/viva_questions_answers.md) for detailed role definitions, presentation sequence, and top 25 viva questions with model answers.*

---

## 📜 Academic Coursework Compliance
- **Course:** Visvesvaraya Technological University (VTU) B.E. 7th Semester BCS702.
- **Academic Report:** Available in [`report/report_template.md`](report/report_template.md).
- **Viva Voce Preparation:** Available in [`report/viva_questions_answers.md`](report/viva_questions_answers.md).
