#!/usr/bin/env python3
"""
BCS702 - Sobel Parallel Edge Detection Demonstration Web Server
Serves a high-fidelity local interactive demonstration dashboard.
Communicates with the C++ compiled binaries (`bin/sobel_app` and `bin/benchmark_runner`).
Zero external web frameworks required (uses standard library http.server + Python 3).
"""

import os
import sys
import json
import base64
import subprocess
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_APP = os.path.join(BASE_DIR, "bin", "sobel_app")
BIN_BENCH = os.path.join(BASE_DIR, "bin", "benchmark_runner")
RESULTS_CSV = os.path.join(BASE_DIR, "results", "benchmark_results.csv")
WEB_DIR = os.path.join(BASE_DIR, "web")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def read_pgm_to_base64_png(pgm_path):
    """Reads a binary P5 PGM file and converts it to a base64 encoded PNG for browser rendering."""
    if not os.path.exists(pgm_path):
        return None
    try:
        with open(pgm_path, 'rb') as f:
            header = f.readline().decode().strip()
            if header not in ['P5', 'P2']:
                return None
            
            # Read width, height, maxval skipping comments
            tokens = []
            while len(tokens) < 3:
                line = f.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if line_str.startswith('#'):
                    continue
                tokens.extend(line_str.split())

            width, height, maxval = int(tokens[0]), int(tokens[1]), int(tokens[2])
            
            if header == 'P5':
                data = f.read(width * height)
                arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            else:
                data = f.read().decode().split()
                arr = np.array([int(x) for x in data], dtype=np.uint8).reshape((height, width))

            # Save to temporary PNG in memory
            import io
            buf = io.BytesIO()
            plt.imsave(buf, arr, cmap='gray', format='png')
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error converting PGM to PNG: {e}")
        return None

class DashboardHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/csv":
            if os.path.exists(RESULTS_CSV):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=benchmark_results.csv")
                self.end_headers()
                with open(RESULTS_CSV, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "CSV results not found")
            return

        elif path.startswith("/results/"):
            filename = os.path.basename(path)
            file_path = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(file_path):
                mime, _ = mimetypes.guess_type(file_path)
                self.send_response(200)
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
            return

        elif path == "/api/benchmark_data":
            if os.path.exists(RESULTS_CSV):
                import csv
                rows = []
                with open(RESULTS_CSV, "r") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "data": rows}).encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "No benchmark data found"}).encode())
            return

        self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            params = json.loads(body.decode('utf-8'))

            pattern = params.get("pattern", "shapes")
            width = int(params.get("width", 1024))
            height = int(params.get("height", 1024))
            threads = int(params.get("threads", 8))
            mode = params.get("mode", "all")
            schedule = params.get("schedule", "static")

            in_file = os.path.join(IMAGES_DIR, f"web_input_{pattern}_{width}.pgm")
            out_file = os.path.join(OUTPUT_DIR, f"web_edges_{pattern}_{width}.pgm")

            # Generate synthetic pattern
            gen_cmd = [BIN_APP, "--generate", pattern, str(width), str(height), in_file]
            subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Run Sobel edge detection
            run_cmd = [BIN_APP, "--input", in_file, "--output", out_file, "--mode", mode,
                       "--threads", str(threads), "--schedule", schedule, "--verify"]
            proc = subprocess.run(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Convert images to base64 for browser display
            input_b64 = read_pgm_to_base64_png(in_file)
            output_b64 = read_pgm_to_base64_png(out_file)

            # Parse console output
            output_lines = proc.stdout.split("\n")
            seq_time = 0.0
            omp_time = 0.0
            gpu_time = 0.0
            gpu_kernel = 0.0
            speedup = 1.0
            efficiency = 100.0
            exact_match = True

            for line in output_lines:
                if "Sequential C++ Sobel" in line:
                    pass
                elif "Kernel Time:" in line and seq_time == 0.0:
                    try:
                        seq_time = float(line.split(":")[1].replace("ms", "").strip())
                    except: pass
                elif "OpenMP Parallel Sobel" in line:
                    pass
                elif "Speedup:" in line and "GPU" not in line:
                    try:
                        speedup = float(line.split(":")[1].replace("x", "").strip())
                    except: pass
                elif "Parallel Efficiency:" in line:
                    try:
                        efficiency = float(line.split(":")[1].replace("%", "").strip())
                    except: pass
                elif "Kernel Time:" in line and omp_time == 0.0 and seq_time > 0.0:
                    try:
                        omp_time = float(line.split(":")[1].replace("ms", "").strip())
                    except: pass
                elif "GPU Kernel Time" in line:
                    try:
                        gpu_kernel = float(line.split(":")[1].replace("ms", "").strip())
                    except: pass
                elif "Total GPU Time" in line:
                    try:
                        gpu_time = float(line.split(":")[1].replace("ms", "").strip())
                    except: pass

            res_payload = {
                "success": True,
                "input_image": input_b64,
                "output_image": output_b64,
                "seq_time_ms": seq_time,
                "omp_time_ms": omp_time,
                "gpu_kernel_ms": gpu_kernel,
                "gpu_total_ms": gpu_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "threads": threads,
                "width": width,
                "height": height,
                "pattern": pattern,
                "raw_stdout": proc.stdout
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload).encode())
            return

        elif path == "/api/run_benchmark":
            # Run benchmark runner and update plots
            subprocess.run([BIN_BENCH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "benchmark", "plot_results.py")],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode())
            return

        self.send_error(404, "Endpoint not found")

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"======================================================================")
    print(f"  BCS702 Interactive Demonstration Dashboard Server Running!")
    print(f"  URL: http://localhost:{port}")
    print(f"======================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demonstration server...")
        httpd.server_close()

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(p)
