cmake_minimum_required(VERSION 3.16)
project(BCS702_Sobel_Parallel_Edge_Detection LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Apple Silicon Homebrew OpenMP detection
if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
        include_directories("/opt/homebrew/opt/libomp/include")
        link_directories("/opt/homebrew/opt/libomp/lib")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
        set(OpenMP_CXX_LIBRARIES "-lomp")
    endif()
endif()

# Fallback standard find_package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(OPENMP_LIBS OpenMP::OpenMP_CXX)
else()
    set(OPENMP_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

# Metal Framework on Apple platforms
if(APPLE)
    find_library(METAL_LIB Metal REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)
    set(GPU_SOURCES src/sobel_metal.mm)
    set(GPU_LIBS ${METAL_LIB} ${FOUNDATION_LIB})
    add_compile_definitions(HAVE_METAL=1)
else()
    set(GPU_SOURCES "")
    set(GPU_LIBS "")
endif()

include_directories(include)

# Common library
add_library(sobel_core STATIC
    src/image_io.cpp
    src/sobel_sequential.cpp
    src/sobel_openmp.cpp
    ${GPU_SOURCES}
)
target_link_libraries(sobel_core PUBLIC ${OPENMP_LIBS} ${GPU_LIBS})

# 1. Main CLI Executable
add_executable(sobel_app src/main.cpp)
target_link_libraries(sobel_app PRIVATE sobel_core)

# 2. Automated Test Runner Executable
add_executable(test_runner tests/test_runner.cpp)
target_link_libraries(test_runner PRIVATE sobel_core)

# 3. Dedicated Benchmarking Runner Executable
add_executable(benchmark_runner benchmark/benchmark_runner.cpp)
target_link_libraries(benchmark_runner PRIVATE sobel_core)
