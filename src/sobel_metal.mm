#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "sobel_metal.h"
#include <chrono>
#include <cstring>
#include <iostream>

static NSString* const kSobelMetalShaderSource = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void sobel_edge_kernel(\n"
"    device const uchar* in_data      [[buffer(0)]],\n"
"    device uchar*       out_data     [[buffer(1)]],\n"
"    constant int&       width        [[buffer(2)]],\n"
"    constant int&       height       [[buffer(3)]],\n"
"    uint2               gid          [[thread_position_in_grid]])\n"
"{\n"
"    int x = int(gid.x);\n"
"    int y = int(gid.y);\n"
"\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    // Zero boundary border pixels\n"
"    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {\n"
"        out_data[y * width + x] = 0;\n"
"        return;\n"
"    }\n"
"\n"
"    // 3x3 Neighborhood access\n"
"    int p00 = in_data[(y - 1) * width + (x - 1)];\n"
"    int p01 = in_data[(y - 1) * width + x];\n"
"    int p02 = in_data[(y - 1) * width + (x + 1)];\n"
"\n"
"    int p10 = in_data[y * width + (x - 1)];\n"
"    int p12 = in_data[y * width + (x + 1)];\n"
"\n"
"    int p20 = in_data[(y + 1) * width + (x - 1)];\n"
"    int p21 = in_data[(y + 1) * width + x];\n"
"    int p22 = in_data[(y + 1) * width + (x + 1)];\n"
"\n"
"    // Horizontal Sobel gradient Gx\n"
"    int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;\n"
"\n"
"    // Vertical Sobel gradient Gy\n"
"    int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;\n"
"\n"
"    // Gradient magnitude\n"
"    float mag = sqrt(float(gx * gx + gy * gy));\n"
"    out_data[y * width + x] = static_cast<uchar>(clamp(mag, 0.0f, 255.0f));\n"
"}\n";

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLComputePipelineState> g_pipeline_state = nil;

static bool initialize_metal_pipeline(std::string& error_msg) {
    if (g_pipeline_state != nil) return true;

    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) {
            error_msg = "No Metal-compatible GPU found on this system";
            return false;
        }

        g_command_queue = [g_metal_device newCommandQueue];
        if (!g_command_queue) {
            error_msg = "Failed to create Metal command queue";
            return false;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [g_metal_device newLibraryWithSource:kSobelMetalShaderSource
                                                              options:nil
                                                                error:&error];
        if (!library) {
            error_msg = std::string("Metal shader compilation failed: ") + [[error localizedDescription] UTF8String];
            return false;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"sobel_edge_kernel"];
        if (!kernelFunction) {
            error_msg = "Kernel function 'sobel_edge_kernel' not found in Metal library";
            return false;
        }

        g_pipeline_state = [g_metal_device newComputePipelineStateWithFunction:kernelFunction
                                                                         error:&error];
        if (!g_pipeline_state) {
            error_msg = std::string("Failed to build compute pipeline: ") + [[error localizedDescription] UTF8String];
            return false;
        }
    }
    return true;
}

bool is_metal_gpu_available() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return (dev != nil);
    }
}

std::string get_metal_device_name() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev) {
            return [[dev name] UTF8String];
        }
        return "None";
    }
}

SobelResult sobel_metal(const Image& input) {
    SobelResult result;
    result.mode = "metal_gpu";
    result.threads_used = 1; // GPU execution

    if (input.empty()) {
        result.success = false;
        result.error_message = "Input image is empty";
        return result;
    }

    std::string init_err;
    if (!initialize_metal_pipeline(init_err)) {
        result.success = false;
        result.error_message = init_err;
        return result;
    }

    const int width = input.width;
    const int height = input.height;
    const size_t byte_count = static_cast<size_t>(width * height);

    result.output_image = Image(width, height, 0);

    @autoreleasepool {
        // Step 1: Host to Device Buffer Allocation & Data Transfer (H2D)
        auto t0 = std::chrono::steady_clock::now();

        id<MTLBuffer> inBuffer = [g_metal_device newBufferWithBytes:input.data.data()
                                                             length:byte_count
                                                            options:MTLResourceStorageModeShared];

        id<MTLBuffer> outBuffer = [g_metal_device newBufferWithLength:byte_count
                                                              options:MTLResourceStorageModeShared];

        id<MTLBuffer> widthBuffer = [g_metal_device newBufferWithBytes:&width
                                                                length:sizeof(int)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> heightBuffer = [g_metal_device newBufferWithBytes:&height
                                                                 length:sizeof(int)
                                                                options:MTLResourceStorageModeShared];

        auto t1 = std::chrono::steady_clock::now();
        double h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Step 2: GPU Kernel Dispatch & Execution
        id<MTLCommandBuffer> commandBuffer = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_pipeline_state];
        [encoder setBuffer:inBuffer offset:0 atIndex:0];
        [encoder setBuffer:outBuffer offset:0 atIndex:1];
        [encoder setBuffer:widthBuffer offset:0 atIndex:2];
        [encoder setBuffer:heightBuffer offset:0 atIndex:3];

        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake((width + 15) / 16 * 16, (height + 15) / 16 * 16, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        auto kernel_start = std::chrono::steady_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto kernel_end = std::chrono::steady_clock::now();

        double kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

        // Step 3: Device to Host Readback (D2H)
        auto t2 = std::chrono::steady_clock::now();
        std::memcpy(result.output_image.data.data(), [outBuffer contents], byte_count);
        auto t3 = std::chrono::steady_clock::now();

        double d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        result.timing.h2d_time_ms = h2d_ms;
        result.timing.kernel_time_ms = kernel_ms;
        result.timing.d2h_time_ms = d2h_ms;
        result.timing.total_time_ms = h2d_ms + kernel_ms + d2h_ms;
        result.success = true;
    }

    return result;
}
