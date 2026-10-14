# VTU BCS702: Parallel Computing — Viva Voce Preparation & Team Demonstration Guide

**Project Title:** Performance Analysis of Parallel Image Edge Detection Using the Sobel Operator with Sequential C++ and OpenMP  
**Course Code:** BCS702 (7th Semester B.E. Computer Science & Engineering)

---

## 1. Four-Member Team Division of Work

| Student | Assigned Role | Primary Technical Responsibilities | Key Files Owned |
|:---|:---|:---|:---|
| **Student 1** | **Sobel Algorithm & Sequential C++ Baseline** | • Mathematical formulation of Sobel horizontal ($G_x$) and vertical ($G_y$) kernels.<br>• Implementation of pure C++17 sequential baseline.<br>• Zero-dependency PGM image parser (P5 binary & P2 ASCII) and boundary zero-padding.<br>• Accurate baseline timing using `std::chrono::steady_clock`. | `include/image_io.h`<br>`src/image_io.cpp`<br>`include/sobel_sequential.h`<br>`src/sobel_sequential.cpp` |
| **Student 2** | **OpenMP Parallelization & Thread Optimization** | • OpenMP shared-memory architecture and work-sharing loop construct design.<br>• Row-wise domain decomposition to eliminate false sharing.<br>• Implementation of static, dynamic, and guided scheduling policies.<br>• Thread scalability across 1, 2, 4, 8, and 11 hardware threads.<br>• Synchronisation analysis and proving race-condition-free execution. | `include/sobel_openmp.h`<br>`src/sobel_openmp.cpp`<br>`Makefile` (OpenMP flags) |
| **Student 3** | **Apple Metal GPU Compute & Automated Verification** | • Hardware environment investigation (Apple M3 Pro ARM64 architecture).<br>• Investigation of GPU options (Metal vs. CUDA vs. OpenCL on Apple Silicon).<br>• Implementation of native Apple Metal compute pipeline in Objective-C++.<br>• Dynamic runtime compilation of Metal Shading Language (MSL) 2D kernel.<br>• Separation of H2D, kernel, and D2H timings.<br>• Automated test suite verifying parser safety, boundary pixels, and bit-exact equivalence ($MAE = 0.0$). | `include/sobel_metal.h`<br>`src/sobel_metal.mm`<br>`tests/test_runner.cpp`<br>`CMakeLists.txt` |
| **Student 4** | **Benchmarking, Visualization, Web Dashboard & Report** | • Automated benchmark harness with warmups, repetitions, and standard deviation.<br>• CSV dataset generation across resolutions ($512\times 512$ to $2048\times 2048$).<br>• Python Matplotlib publication plotting (Time vs Size, Time vs Threads, Speedup, Efficiency).<br>• Mathematical Amdahl's Law modeling and heterogeneous core analysis.<br>• Interactive web demonstration dashboard (`web/server.py`, `web/index.html`).<br>• Academic project report compilation according to VTU BCS702 guidelines. | `benchmark/benchmark_runner.cpp`<br>`benchmark/plot_results.py`<br>`web/server.py`<br>`web/index.html`<br>`report/report_template.md` |

---

## 2. Five-Minute Live Demonstration Sequence for Examiners

When presenting to the external and internal viva examiners, execute the demonstration in the following order:

### **Step 1: Introduction & Hardware Overview (Student 1 & Student 3)**
- State the project objective: Parallelizing 2D spatial convolution using OpenMP and hardware acceleration.
- Display system architecture:
  ```bash
  sysctl -n machdep.cpu.brand_string hw.ncpu
  ```
  *Show: Apple M3 Pro, 11 Hardware Cores (Heterogeneous: P-cores + E-cores), Unified Memory.*

### **Step 2: Automated Correctness Verification (Student 3)**
- Run the automated test runner to prove all implementations are mathematically identical and bug-free:
  ```bash
  make test
  ```
  *Highlight: 7 test suites passing, zero failures, exact equivalence ($MAE = 0.0$).*

### **Step 3: CLI Demonstration (Student 1 & Student 2)**
- Run a live edge detection on a $1024 \times 1024$ image comparing Sequential, OpenMP, and Metal GPU:
  ```bash
  ./bin/sobel_app --generate shapes 1024 1024 images/demo.pgm
  ./bin/sobel_app --input images/demo.pgm --output output/edges_demo.pgm --mode all --threads 8 --verify
  ```
  *Show the speedup output, thread efficiency, and pixel verification.*

### **Step 4: Interactive Web Dashboard Demonstration (Student 4)**
- Launch the demonstration dashboard:
  ```bash
  make dashboard
  ```
- Open `http://localhost:8080` in the browser:
  1. Switch between synthetic patterns (Shapes, Chessboard, Circles).
  2. Slide the OpenMP thread slider from 1 to 11 threads in real-time.
  3. Show the side-by-side original and edge-detected outputs.
  4. Switch to the **Benchmark Analysis & Plots** tab to display the publication graphs and raw CSV data.

### **Step 5: Theoretical Discussion (All Students)**
- Conclude by discussing Amdahl's Law, memory bandwidth limits, and why the GPU requires large image sizes to overcome command queue launch overhead.

---

## 3. Top 25 Likely Viva-Voce Questions with Academic Answers

### Q1. What is the Sobel Operator, and what mathematical operation does it perform?
**Answer:** The Sobel operator is a discrete differentiation operator used in image processing to compute an approximation of the gradient of image intensity. It performs a 2D spatial convolution using two $3 \times 3$ integer kernels ($G_x$ for horizontal gradient and $G_y$ for vertical gradient) to detect edges where pixel luminance exhibits sharp spatial discontinuities.

### Q2. What are the horizontal and vertical Sobel kernels?
**Answer:**
$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
$$
$G_x$ computes the central difference in the horizontal direction smoothed by a triangular filter $[1, 2, 1]^T$ vertically to suppress high-frequency noise. $G_y$ computes vertical difference smoothed horizontally.

### Q3. How is the final edge magnitude calculated?
**Answer:** The total gradient magnitude is:
$$G(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$
The result is clamped to the range $[0, 255]$ and cast to an 8-bit unsigned integer (`uint8_t`).

### Q4. Why is Sobel edge detection considered "embarrassingly parallel"?
**Answer:** In the Sobel algorithm, each output pixel $I_{out}(x, y)$ depends solely on a localized $3 \times 3$ neighborhood in the input image. Because the input buffer is strictly read-only and each output pixel is written to a distinct memory location in the destination buffer, there are no loop-carried data dependencies, no race conditions, and no need for locks or atomic operations.

### Q5. How are the boundary pixels of the image handled?
**Answer:** For boundary pixels ($x=0$, $x=W-1$, $y=0$, $y=H-1$), a complete $3 \times 3$ neighborhood does not exist within the image bounds. In our implementation, boundary pixels are initialized to zero (black borders). This eliminates out-of-bounds memory accesses and guarantees identical border values across sequential, OpenMP, and GPU versions.

### Q6. Which OpenMP directive was used to parallelize the algorithm?
**Answer:**
```cpp
#pragma omp parallel for num_threads(threads) schedule(static)
for (int y = 1; y < height - 1; ++y) { ... }
```
This work-sharing directive spawns a team of threads and distributes the outer loop iterations (rows of the image) among them.

### Q7. Why did you parallelize the outer row loop (`y`) instead of the inner column loop (`x`)?
**Answer:**
1. **Cache Locality:** Images are stored in row-major order in contiguous memory. Distributing rows gives each thread contiguous blocks of memory, maximizing L1/L2 cache prefetching.
2. **False Sharing Elimination:** If adjacent threads wrote to adjacent columns, they would write to the same 64-byte cache line, causing cache invalidation traffic (false sharing). Row-wise chunking separates thread write addresses by hundreds or thousands of bytes.
3. **Loop Overhead:** Parallelizing the outer loop creates the thread team once per image, whereas parallelizing the inner loop would incur thread fork-join overhead $H$ times.

### Q8. What is the difference between `schedule(static)`, `schedule(dynamic)`, and `schedule(guided)`?
**Answer:**
- **Static:** Loop iterations are partitioned into equal chunks before execution and assigned round-robin to threads. It has minimal runtime overhead and is optimal for uniform computational workloads like Sobel convolution.
- **Dynamic:** Iterations are divided into chunks and assigned to threads dynamically from a shared task queue as threads become idle. Useful when workloads vary across pixels (e.g. Mandelbrot sets), but adds scheduling lock overhead.
- **Guided:** Starts with large chunks to minimize overhead and exponentially decreases chunk size toward the end to achieve load balancing.

### Q9. Why is OpenMP with 1 thread slower than pure sequential C++?
**Answer:** Pure sequential C++ executes raw, uninterrupted assembly with no thread runtime library. OpenMP with 1 thread still initializes the LLVM OpenMP runtime library, constructs thread descriptors, evaluates team structures, and checks thread synchronization barriers upon exiting the parallel region. This runtime overhead accounts for the latency difference.

### Q10. What is Speedup, and how is it calculated?
**Answer:** Speedup $S(p)$ is the ratio of sequential execution time to parallel execution time using $p$ threads:
$$S(p) = \frac{T_{sequential}}{T_{parallel}(p)}$$

### Q11. What is Parallel Efficiency?
**Answer:** Parallel Efficiency $E(p)$ measures the percentage of CPU capacity effectively utilized by parallelization:
$$E(p) = \frac{S(p)}{p} \times 100\% = \frac{T_{sequential}}{p \cdot T_{parallel}(p)} \times 100\%$$
For ideal linear speedup, $S(p) = p$ and $E(p) = 100\%$.

### Q12. State Amdahl's Law and explain its significance.
**Answer:** Amdahl's Law defines the theoretical maximum speedup of a program when parallelized:
$$S(p) = \frac{1}{(1 - f) + \frac{f}{p}}$$
where $f$ is the parallelizable fraction of the workload. Even if $f = 95\%$, the theoretical speedup can never exceed $\frac{1}{1 - 0.95} = 20\times$, regardless of how many hundreds of processors are added.

### Q13. Why does the measured Speedup not scale linearly with thread count?
**Answer:**
1. **Memory Bandwidth Saturation:** Sobel has low arithmetic intensity ($\sim 1.67\text{ FLOPs/byte}$). Multiple cores simultaneously request memory across the shared bus, saturating memory bandwidth.
2. **Synchronization Barriers:** OpenMP threads synchronize at the implicit barrier at the end of the `#pragma omp parallel for` block.
3. **Hardware Core Heterogeneity:** On Apple Silicon, some cores are High-Performance (P-cores) while others are Efficiency (E-cores). Slower E-cores create a straggler effect.

### Q14. What is the architecture of the Apple M3 Pro chip used in this benchmark?
**Answer:** The Apple M3 Pro is an ARM64 System-on-Chip (SoC) featuring:
- 11 total CPU cores: 5 High-Performance P-cores (~4.0 GHz) + 6 Energy-Efficient E-cores (~2.8 GHz).
- Integrated Apple Metal GPU with unified memory.
- Unified Memory Architecture (UMA): CPU and GPU share the same physical high-bandwidth LPDDR5 memory bus.

### Q15. Why can't NVIDIA CUDA be used on this development machine?
**Answer:** CUDA is a proprietary parallel computing platform developed by NVIDIA that requires physical NVIDIA GPU hardware and an NVIDIA CUDA driver. Apple Silicon Macs do not contain NVIDIA GPUs, and macOS does not support NVIDIA drivers. On Apple Silicon, the native GPU acceleration API is **Apple Metal Compute**.

### Q16. How does the Apple Metal GPU implementation work?
**Answer:**
1. The GPU kernel is written in Metal Shading Language (MSL) and compiled into an `MTLComputePipelineState` at runtime.
2. Memory buffers (`MTLBuffer`) are allocated in shared unified memory.
3. A 2D grid of threads is dispatched in threadgroups of $16 \times 16$ (256 threads per threadgroup).
4. The command buffer is committed to the GPU command queue and synchronized via `[commandBuffer waitUntilCompleted]`.

### Q17. Why is the Metal GPU slower than the CPU for a $512 \times 512$ image?
**Answer:** On a $512 \times 512$ image, sequential CPU execution takes only $0.21\text{ ms}$. Creating and encoding a Metal command buffer, submitting it to the GPU driver queue, and waiting for GPU synchronization takes $\sim 0.3\text{ ms}$ of fixed launch latency. The GPU dispatch overhead exceeds the total computation time.

### Q18. At what resolution does the GPU outperform the CPU?
**Answer:** At $1024 \times 1024$ and $2048 \times 2048$. On the $2048 \times 2048$ image (4.19 Megapixels), the GPU kernel completes in $1.022\text{ ms}$ compared to $2.361\text{ ms}$ for sequential CPU ($2.31\times$ kernel speedup), because the workload is large enough to saturate the GPU's execution units and amortize driver launch latency.

### Q19. Did you separate memory transfer time from GPU kernel execution time?
**Answer:** Yes. In `sobel_metal.mm`:
- Host-to-Device (H2D) allocation/copy: $0.27\text{ ms}$ (at 2048x2048)
- GPU Kernel execution: $1.022\text{ ms}$
- Device-to-Host (D2H) readback: $0.26\text{ ms}$
- Total end-to-end GPU time: $1.560\text{ ms}$ ($1.52\times$ speedup including all staging).

### Q20. How did you verify correctness and equivalence between implementations?
**Answer:** We implemented `compare_images()` in `src/image_io.cpp`. It compares every corresponding pixel $(x, y)$ between the sequential baseline and the parallel output, calculating:
- Maximum Absolute Difference
- Mean Absolute Error (MAE): $\frac{1}{N} \sum |I_1 - I_2|$
- Mean Squared Error (MSE)
All OpenMP (1, 2, 4, 8, 11 threads) and Metal GPU runs yielded a maximum difference of $0$ and $MAE = 0.0$ (100% bit-exact match).

### Q21. Why did you choose the PGM format instead of PNG or JPEG?
**Answer:** PGM (Portable GrayMap, P5 format) is a standard, uncompressed raster format with a simple ASCII header followed by raw 8-bit binary pixel bytes. This eliminates external compression/decompression library dependencies (like `libpng` or `libjpeg`), ensuring image I/O is transparent, deterministic, and does not contaminate CPU kernel benchmarks.

### Q22. How did you prevent file I/O time from polluting benchmark measurements?
**Answer:** File loading (`load_pgm`) and file writing (`save_pgm`) are performed entirely outside the timing block. The high-resolution timer (`std::chrono::steady_clock`) is started immediately before the convolution loops and stopped immediately after, measuring strictly the mathematical kernel computation.

### Q23. What are warmup runs, and why are they necessary in benchmarking?
**Answer:** Warmup runs execute the kernel once or twice without recording times before the benchmark begins. This ensures that:
- Operating system memory pages are mapped and committed (no cold page faults).
- CPU instruction and data caches are primed.
- Dynamic GPU shader compilation is completed and cached in memory.

### Q24. How would you further optimize this implementation in high-performance production code?
**Answer:**
1. **SIMD Vectorization:** Use ARM NEON vector intrinsics (or x86 AVX-512) to process 16 or 32 pixels per vector instruction.
2. **2D Cache Tiling:** Partition the image into $32 \times 32$ or $64 \times 64$ tiles to keep input pixels in L1 cache during multi-pass operations.
3. **Integer Arithmetic Approximations:** Approximate magnitude with the $L_1$ norm $G \approx |G_x| + |G_y|$ to replace costly floating-point square roots.

### Q25. What is the primary academic takeaway from this project?
**Answer:** Parallel edge detection demonstrates that while data-parallel stencil computations have high theoretical parallelism, practical speedup is bounded by hardware realities: memory bus bandwidth, thread management overhead, and heterogeneous core clock speeds. Furthermore, GPU acceleration provides substantial speedups only when workload dimensions are large enough to amortize driver dispatch latency.
