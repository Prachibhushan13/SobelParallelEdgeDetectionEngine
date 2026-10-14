#!/usr/bin/env python3
"""
BCS702 - Parallel Image Edge Detection Using Sobel Operator
Benchmark Plotting Script
Reads results/benchmark_results.csv and generates academic-grade publication figures:
  1. Execution Time vs Image Size (Megapixels / Resolution)
  2. Execution Time vs Thread Count (for 512, 1024, 2048)
  3. Speedup vs Thread Count compared with Linear Ideal Speedup & Amdahl's Law
  4. Parallel Efficiency vs Thread Count
  5. GPU vs Multicore CPU Performance Breakdown
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_academic_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'lines.linewidth': 2.2,
        'lines.markersize': 8,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

def plot_benchmarks(csv_path="results/benchmark_results.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    set_academic_style()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # -------------------------------------------------------------
    # Plot 1: Execution Time vs Image Size
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    sizes = sorted(df['Width'].unique())

    # Sequential line
    seq_df = df[df['Implementation'] == 'Sequential'].sort_values('Width')
    plt.plot(seq_df['Width'], seq_df['KernelTime_ms'], marker='o', label='Sequential C++ (1 thread)', color='#d62728', linewidth=2.5)

    # OpenMP lines for selected thread counts (e.g. 2, 4, 8)
    for idx, t in enumerate([2, 4, 8]):
        t_df = df[(df['Implementation'] == 'OpenMP') & (df['Threads'] == t)].sort_values('Width')
        if not t_df.empty:
            plt.plot(t_df['Width'], t_df['KernelTime_ms'], marker='s', label=f'OpenMP ({t} threads)', color=colors[idx])

    # GPU line if present
    gpu_df = df[df['Implementation'] == 'Metal_GPU'].sort_values('Width')
    if not gpu_df.empty:
        plt.plot(gpu_df['Width'], gpu_df['KernelTime_ms'], marker='^', label='Apple Metal GPU (Kernel)', color='#2ca02c', linestyle='--')

    plt.title('Figure 1: Execution Time vs Image Dimension (W x H)', fontweight='bold')
    plt.xlabel('Image Width / Height (pixels)')
    plt.ylabel('Execution Time (ms)')
    plt.xticks(sizes, [f"{s}x{s}" for s in sizes])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_execution_time_vs_size.png'), dpi=300)
    plt.close()
    print("Generated: fig1_execution_time_vs_size.png")

    # -------------------------------------------------------------
    # Plot 2: Execution Time vs Thread Count
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    omp_df = df[df['Implementation'] == 'OpenMP']

    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            plt.plot(s_df['Threads'], s_df['KernelTime_ms'], marker='o', label=f'Size {size}x{size}', color=colors[idx])

    plt.title('Figure 2: Execution Time vs OpenMP Thread Count', fontweight='bold')
    plt.xlabel('Number of Threads (p)')
    plt.ylabel('Kernel Execution Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_execution_time_vs_threads.png'), dpi=300)
    plt.close()
    print("Generated: fig2_execution_time_vs_threads.png")

    # -------------------------------------------------------------
    # Plot 3: Speedup vs Thread Count (with Ideal Linear Speedup)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    threads_list = sorted(omp_df['Threads'].unique())
    plt.plot(threads_list, threads_list, 'k--', label='Ideal Linear Speedup S(p)=p', alpha=0.7)

    # Amdahl's Law model for f = 0.95
    f = 0.95
    amdahl_p = np.array(threads_list)
    amdahl_s = 1.0 / ((1.0 - f) + f / amdahl_p)
    plt.plot(amdahl_p, amdahl_s, 'r:', label=f"Amdahl's Law (f = {f})", alpha=0.8)

    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            plt.plot(s_df['Threads'], s_df['Speedup'], marker='s', label=f'Measured {size}x{size}', color=colors[idx])

    plt.title("Figure 3: Speedup S(p) vs Thread Count", fontweight='bold')
    plt.xlabel('Number of Threads (p)')
    plt.ylabel('Speedup Factor S(p) = T_seq / T_par')
    plt.xticks(threads_list)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_speedup_vs_threads.png'), dpi=300)
    plt.close()
    print("Generated: fig3_speedup_vs_threads.png")

    # -------------------------------------------------------------
    # Plot 4: Parallel Efficiency vs Thread Count
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.axhline(y=100.0, color='k', linestyle='--', label='Ideal Efficiency (100%)', alpha=0.7)

    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            plt.plot(s_df['Threads'], s_df['Efficiency_pct'], marker='^', label=f'{size}x{size}', color=colors[idx])

    plt.title("Figure 4: Parallel Efficiency E(p) vs Thread Count", fontweight='bold')
    plt.xlabel('Number of Threads (p)')
    plt.ylabel('Parallel Efficiency E(p) (%)')
    plt.ylim(0, 115)
    plt.xticks(threads_list)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_efficiency_vs_threads.png'), dpi=300)
    plt.close()
    print("Generated: fig4_efficiency_vs_threads.png")

    # -------------------------------------------------------------
    # Plot 5: 2x2 Comprehensive Academic Summary Dashboard
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0, 0): Time vs Size
    ax = axes[0, 0]
    ax.plot(seq_df['Width'], seq_df['KernelTime_ms'], marker='o', label='Sequential', color='#d62728')
    for idx, t in enumerate([2, 4, 8]):
        t_df = df[(df['Implementation'] == 'OpenMP') & (df['Threads'] == t)].sort_values('Width')
        if not t_df.empty:
            ax.plot(t_df['Width'], t_df['KernelTime_ms'], marker='s', label=f'OpenMP ({t}T)', color=colors[idx])
    if not gpu_df.empty:
        ax.plot(gpu_df['Width'], gpu_df['KernelTime_ms'], marker='^', label='Metal GPU (Kernel)', color='#2ca02c', linestyle='--')
    ax.set_title('(a) Execution Time vs Image Size')
    ax.set_xlabel('Resolution (Width)')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(sizes)
    ax.grid(True)
    ax.legend(fontsize=8)

    # (0, 1): Time vs Threads
    ax = axes[0, 1]
    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            ax.plot(s_df['Threads'], s_df['KernelTime_ms'], marker='o', label=f'{size}x{size}', color=colors[idx])
    ax.set_title('(b) OpenMP Time vs Threads')
    ax.set_xlabel('Threads (p)')
    ax.set_ylabel('Time (ms)')
    ax.grid(True)
    ax.legend(fontsize=8)

    # (1, 0): Speedup
    ax = axes[1, 0]
    ax.plot(threads_list, threads_list, 'k--', label='Linear S(p)=p', alpha=0.6)
    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            ax.plot(s_df['Threads'], s_df['Speedup'], marker='s', label=f'{size}x{size}', color=colors[idx])
    ax.set_title('(c) Speedup S(p) vs Thread Count')
    ax.set_xlabel('Threads (p)')
    ax.set_ylabel('Speedup Factor')
    ax.set_xticks(threads_list)
    ax.grid(True)
    ax.legend(fontsize=8)

    # (1, 1): Efficiency
    ax = axes[1, 1]
    ax.axhline(y=100.0, color='k', linestyle='--', label='100% Ideal', alpha=0.6)
    for idx, size in enumerate(sizes):
        s_df = omp_df[omp_df['Width'] == size].sort_values('Threads')
        if not s_df.empty:
            ax.plot(s_df['Threads'], s_df['Efficiency_pct'], marker='^', label=f'{size}x{size}', color=colors[idx])
    ax.set_title('(d) Parallel Efficiency E(p) vs Threads')
    ax.set_xlabel('Threads (p)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_ylim(0, 115)
    ax.set_xticks(threads_list)
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.suptitle('VTU BCS702: Parallel Sobel Performance Benchmark Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_summary.png'), dpi=300)
    plt.close()
    print("Generated: benchmark_summary.png")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
    plot_benchmarks(csv_file)
