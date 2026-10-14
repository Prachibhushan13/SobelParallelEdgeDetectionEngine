CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Iinclude -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
LDFLAGS ?= -L/opt/homebrew/opt/libomp/lib -lomp -framework Metal -framework Foundation
PYTHON ?= /opt/anaconda3/bin/python3

BUILD_DIR = build
BIN_DIR = bin

TARGET_APP = $(BIN_DIR)/sobel_app
TARGET_TEST = $(BIN_DIR)/test_runner
TARGET_BENCH = $(BIN_DIR)/benchmark_runner

SRCS_CORE = src/image_io.cpp src/sobel_sequential.cpp src/sobel_openmp.cpp
OBJC_SRCS = src/sobel_metal.mm

OBJS_CORE = $(BUILD_DIR)/image_io.o $(BUILD_DIR)/sobel_sequential.o $(BUILD_DIR)/sobel_openmp.o $(BUILD_DIR)/sobel_metal.o

all: dirs $(TARGET_APP) $(TARGET_TEST) $(TARGET_BENCH)
	@echo "============================================================"
	@echo "Build complete! All targets built successfully:"
	@echo "  1. Main Application  : $(TARGET_APP)"
	@echo "  2. Test Runner       : $(TARGET_TEST)"
	@echo "  3. Benchmark Runner  : $(TARGET_BENCH)"
	@echo "============================================================"

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) images output results report

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/sobel_metal.o: src/sobel_metal.mm
	$(CXX) -std=c++17 -O3 -Iinclude -c $< -o $@

$(TARGET_APP): $(OBJS_CORE) $(BUILD_DIR)/main.o
	$(CXX) $(OBJS_CORE) $(BUILD_DIR)/main.o $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_TEST): $(OBJS_CORE) $(BUILD_DIR)/test_runner.o
	$(CXX) $(OBJS_CORE) $(BUILD_DIR)/test_runner.o $(LDFLAGS) -o $@

$(BUILD_DIR)/test_runner.o: tests/test_runner.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_BENCH): $(OBJS_CORE) $(BUILD_DIR)/benchmark_runner.o
	$(CXX) $(OBJS_CORE) $(BUILD_DIR)/benchmark_runner.o $(LDFLAGS) -o $@

$(BUILD_DIR)/benchmark_runner.o: benchmark/benchmark_runner.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TARGET_TEST)
	@echo ">>> Running automated test suite..."
	./$(TARGET_TEST)

bench: $(TARGET_BENCH)
	@echo ">>> Running benchmark suite..."
	./$(TARGET_BENCH)

plot:
	@echo ">>> Generating publication plots from benchmark results..."
	$(PYTHON) benchmark/plot_results.py

dashboard:
	@echo ">>> Launching interactive demonstration dashboard..."
	$(PYTHON) web/server.py

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all dirs test bench plot dashboard clean
