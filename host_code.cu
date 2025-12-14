#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "image_utils.h"

#define BLOCK_DIM 32 // 32x32 threads per block
// Simple timer for CPU measurements
typedef struct {
    clock_t start_time;
} Timer;

void timer_start(Timer* t) {
    t->start_time = clock();
}

double timer_stop(Timer* t) {
    clock_t end_time = clock();
    double elapsed = ((double)(end_time - t->start_time)) / CLOCKS_PER_SEC * 1000.0;
    return elapsed;
}

#define CUDA_CHECK(call)                                                          \
do {                                                                              \
    cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error in file '%s' at line %d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
} while (0)

// Forward declarations of CUDA kernels (implemented in cuda_kernels.cu)
extern __global__ void kernel_warmup();
extern __global__ void kernel_grayscale(const RgbPixel* d_input_rgb, GrayPixel* d_intermediate_gray, int width, int height);
extern __global__ void kernel_blur_naive(const GrayPixel* d_intermediate_gray, GrayPixel* d_output, int width, int height, const float* d_gaussian_kernel);
extern __global__ void kernel_blur_shared(const GrayPixel* d_intermediate_gray, GrayPixel* d_output, int width, int height, const float* d_gaussian_kernel);
extern __global__ void kernel_fused_shared(const RgbPixel* d_input_rgb, GrayPixel* d_output, int width, int height, const float* d_gaussian_kernel);
extern __global__ void kernel_fused_cg(const RgbPixel* d_input_rgb, GrayPixel* d_output, int width, int height, const float* d_gaussian_kernel);
extern __global__ void kernel_persistent_simple(const RgbPixel* d_input_all, GrayPixel* d_output_all, int width, int height, int num_images, const float* d_gaussian_kernel);
extern __global__ void kernel_persistent_optimized(const RgbPixel* d_input_all, GrayPixel* d_output_all, int width, int height, int num_images, const float* d_gaussian_kernel);

// Helper to time kernel execution (call this after launching kernels)
float time_and_record(cudaEvent_t* start, cudaEvent_t* stop) {
    CUDA_CHECK(cudaEventRecord(*stop, 0));
    CUDA_CHECK(cudaEventSynchronize(*stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, *start, *stop));
    CUDA_CHECK(cudaEventDestroy(*start));
    CUDA_CHECK(cudaEventDestroy(*stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    return milliseconds;
}

// Helper to create timing events and record start
void start_timing(cudaEvent_t* start, cudaEvent_t* stop) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventCreate(start));
    CUDA_CHECK(cudaEventCreate(stop));
    CUDA_CHECK(cudaEventRecord(*start, 0));
}

// Helper to write output image
void write_output_image(const char* filename, GrayPixel* d_output, int width, int height) {
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t gray_bytes = width * height * sizeof(GrayPixel);
    GrayPixel* h_out = (GrayPixel*)malloc(gray_bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_output, gray_bytes, cudaMemcpyDeviceToHost));
    write_ppm(filename, h_out, width, height);
    free(h_out);
}

// --- CPU-side versions of kernels ---
void cpu_grayscale(const RgbPixel* input_rgb, GrayPixel* output_gray, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            RgbPixel p = input_rgb[index];
            unsigned char gray_value = (unsigned char)(0.299f * p.r + 0.587f * p.g + 0.114f * p.b);
            output_gray[index].gray = gray_value;
        }
    }
}

void cpu_blur(const GrayPixel* input_gray, GrayPixel* output_gray, int width, int height, const float* kernel) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int kernel_idx = 0;
            
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    // Clamp to edge
                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;
                    
                    float pixel_value = input_gray[ny * width + nx].gray;
                    float kernel_weight = kernel[kernel_idx++];
                    sum += pixel_value * kernel_weight;
                }
            }
            
            output_gray[y * width + x].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
        }
    }
}

int main(int argc, char** argv) {
    // Load image
    RgbPixel* h_input_rgb = NULL;
    int width, height;
    
    printf("\n=== Reading the image ===\n");
    if (!read_ppm("input_image.ppm", &h_input_rgb, &width, &height)) {
        printf("\n--- ERROR! Unable to read the image ---\n");
        return 1;
    }
    
    size_t rgb_bytes = width * height * sizeof(RgbPixel);
    size_t gray_bytes = width * height * sizeof(GrayPixel);
    
    // --- Run CPU version first (10 iterations) ---
    printf("\n=== CPU VERSION (10 iterations) ===\n");
    
    GrayPixel* h_intermediate_gray = (GrayPixel*)malloc(gray_bytes);
    GrayPixel* h_output_gray = (GrayPixel*)malloc(gray_bytes);
    
    const int NUM_ITERATIONS = 10;
    double cpu_total_time = 0.0;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        Timer t;
        timer_start(&t);
        
        cpu_grayscale(h_input_rgb, h_intermediate_gray, width, height);
        cpu_blur(h_intermediate_gray, h_output_gray, width, height, GAUSSIAN_KERNEL);
        
        double elapsed = timer_stop(&t);
        cpu_total_time += elapsed;
    }
    
    double cpu_avg_time = cpu_total_time / NUM_ITERATIONS;
    printf("CPU Average Time: %.2f ms\n", cpu_avg_time);
    
    write_ppm("output_cpu.ppm", h_output_gray, width, height);
    
    // Print GPU information
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\n=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Cooperative launch supported: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    
    // Allocate device memory
    RgbPixel* d_input_rgb = NULL;
    GrayPixel* d_intermediate_gray = NULL;
    GrayPixel* d_output = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_intermediate_gray, gray_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, gray_bytes));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input_rgb, h_input_rgb, rgb_bytes, cudaMemcpyHostToDevice));
    
    // Copy Gaussian kernel to device
    float* d_gaussian_kernel = NULL;
    size_t kernel_bytes = KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_gaussian_kernel, kernel_bytes));
    CUDA_CHECK(cudaMemcpy(d_gaussian_kernel, GAUSSIAN_KERNEL, kernel_bytes, cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions (reused for all kernels)
    dim3 threads = {BLOCK_DIM, BLOCK_DIM, 1};
    dim3 grid = {(width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1};
    
    // Calculate shared memory size (reused for all kernels that need it)
    // You can change this if you want to experiment with different tile sizes
    // You can set a different shared_mem_size in Part 4 Optimized
    int tile_width = BLOCK_DIM + 2 * KERNEL_RADIUS;
    size_t shared_mem_size = tile_width * tile_width * sizeof(unsigned char);
    
    // Calculate optimal grid size for cooperative kernel (maximum occupancy) - 1D layout
    // Check this if you are going for cooperative groups implementation of Part 3
    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel_fused_cg, BLOCK_DIM * BLOCK_DIM, 0);
    int numBlocks = numBlocksPerSm * prop.multiProcessorCount;
    dim3 grid_cg = {(unsigned int)numBlocks, 1, 1};
    dim3 threads_cg = {BLOCK_DIM * BLOCK_DIM, 1, 1};
    printf("\nCooperative kernel grid configuration:\n");
    printf("  Blocks per SM: %d\n", numBlocksPerSm);
    printf("  Total blocks: %d\n", numBlocks);
    printf("  Threads per block: %d\n", BLOCK_DIM * BLOCK_DIM);
    
    printf("\n=== GPU VERSIONS (10 iterations each) ===\n");
    
    // Execute GPU parts with averaging
    float part1_total = 0.0f, part2_total = 0.0f, part3_shared_total = 0.0f, part3_cg_total = 0.0f;
    cudaEvent_t start, stop;
    
    // --- Part 1: Naive (Global Memory) ---
    printf("\n--- Running Part 1: Naive (Global Memory) ---\n");
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        kernel_grayscale<<<grid, threads>>>(d_input_rgb, d_intermediate_gray, width, height);
        kernel_blur_naive<<<grid, threads>>>(d_intermediate_gray, d_output, width, height, d_gaussian_kernel);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            write_output_image("output_part1.ppm", d_output, width, height);
        }

        if (i != 0) { // the first iteration is a warm-up
            part1_total += time;
        }
        CUDA_CHECK(cudaMemset(d_intermediate_gray, 0, gray_bytes));
        CUDA_CHECK(cudaMemset(d_output, 0, gray_bytes));
    }
    // host_code.cu, around Line 233
    // --- Part 2: Optimized Blur (Shared Memory) ---
    printf("\n--- Running Part 2: Optimized Blur (Shared Memory) ---\n");
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        kernel_grayscale<<<grid, threads>>>(d_input_rgb, d_intermediate_gray, width, height);
        kernel_blur_shared<<<grid, threads, shared_mem_size>>>(d_intermediate_gray, d_output, width, height, d_gaussian_kernel);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            write_output_image("output_part2.ppm", d_output, width, height);
        }
        
        if (i != 0) { // the first iteration is a warm-up
            part2_total += time;
        }
        CUDA_CHECK(cudaMemset(d_intermediate_gray, 0, gray_bytes));
        CUDA_CHECK(cudaMemset(d_output, 0, gray_bytes));
    }
    
    // --- Part 3a: Kernel Fusion (Shared Memory) ---
    printf("\n--- Running Part 3a: Kernel Fusion (Shared Memory) ---\n");
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        kernel_fused_shared<<<grid, threads, shared_mem_size>>>(d_input_rgb, d_output, width, height, d_gaussian_kernel);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            write_output_image("output_part3_shared.ppm", d_output, width, height);
        }
        
        if (i != 0) { // the first iteration is a warm-up
            part3_shared_total += time;
        }
        CUDA_CHECK(cudaMemset(d_output, 0, gray_bytes));
    }
    
    // --- Part 3b: Kernel Fusion (Cooperative Groups) ---
    printf("\n--- Running Part 3b: Kernel Fusion (Cooperative Groups) ---\n");
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        void* kernelArgs[] = {
            (void*)&d_input_rgb,
            (void*)&d_output,
            (void*)&width,
            (void*)&height,
            (void*)&d_gaussian_kernel
        };
        cudaLaunchCooperativeKernel((void*)kernel_fused_cg, grid_cg, threads_cg, kernelArgs);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            write_output_image("output_part3_cg.ppm", d_output, width, height);
        }
        
        if (i != 0) { // the first iteration is a warm-up
            part3_cg_total += time;
        }
        CUDA_CHECK(cudaMemset(d_output, 0, gray_bytes));
    }
    
    // --- Part 4: Persistent Kernel (100 images) ---
    printf("\n--- Running Part 4: Persistent Kernel (100 images) ---\n");
    const int NUM_IMAGES = 100;
    
    // Allocate memory for 100 copies of the input image
    size_t batch_rgb_bytes = rgb_bytes * NUM_IMAGES;
    size_t batch_gray_bytes = gray_bytes * NUM_IMAGES;
    
    RgbPixel* d_input_batch = NULL;
    GrayPixel* d_output_batch = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input_batch, batch_rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output_batch, batch_gray_bytes));
    
    // Copy the single input image 100 times
    for (int i = 0; i < NUM_IMAGES; i++) {
        CUDA_CHECK(cudaMemcpy(d_input_batch + (i * width * height), d_input_rgb, rgb_bytes, cudaMemcpyDeviceToDevice));
    }
    
    // Benchmark Part 3 run 100 times for comparison
    float part3_batch_total = 0.0f;
    printf("Running Part 3x100: Part 3 run 100 times (for comparison)\n");
    
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        for (int j = 0; j < NUM_IMAGES; j++) {
            const RgbPixel* d_input_offset = d_input_batch + (j * width * height);
            GrayPixel* d_output_offset = d_output_batch + (j * width * height);
            
            kernel_fused_shared<<<grid, threads, shared_mem_size>>>(d_input_offset, d_output_offset, width, height, d_gaussian_kernel);
        }
        
        float time = time_and_record(&start, &stop);
        if (i != 0) { // the first iteration is a warm-up
            part3_batch_total += time;
        }
        
        CUDA_CHECK(cudaMemset(d_output_batch, 0, batch_gray_bytes));
    }
    
    // Benchmark Part 4a (Simple persistent kernel)
    float part4_simple_total = 0.0f;
    printf("Running Part 4a: Persistent Simple\n");
    
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);

        /* NOTE: if you are using a Cooperative Kernel from Part 3 Version 2, 
           ensure proper setup with cudaLaunchCooperativeKernel
           and corresponding parameter passing logic */ 
        kernel_persistent_simple<<<grid, threads, shared_mem_size>>>(d_input_batch, d_output_batch, width, height, NUM_IMAGES, d_gaussian_kernel);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            // Write last image from batch
            write_output_image("output_part4a.ppm", d_output_batch + ((NUM_IMAGES - 1) * width * height), width, height);
        }
        
        if (i != 0) { // the first iteration is a warm-up
            part4_simple_total += time;
        }
        CUDA_CHECK(cudaMemset(d_output_batch, 0, batch_gray_bytes));
    }
    
    // Benchmark Part 4b (Optimized persistent kernel with shared Gaussian)
    float part4_optimized_total = 0.0f;
    printf("Running Part 4b: Persistent Optimized\n");
    
    // Calculate shared memory size including Gaussian kernel
    size_t gaussian_mem_size = KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);
    size_t total_shared_mem = ((gaussian_mem_size + sizeof(unsigned char) - 1) / sizeof(unsigned char)) + shared_mem_size;
    
    for (int i = 0; i <= NUM_ITERATIONS; i++) {
        start_timing(&start, &stop);
        
        /* NOTE: if you are using a Cooperative Kernel from Part 3 Version 2, 
           ensure proper setup with cudaLaunchCooperativeKernel
           and corresponding parameter passing logic */ 
        kernel_persistent_optimized<<<grid, threads, total_shared_mem>>>(d_input_batch, d_output_batch, width, height, NUM_IMAGES, d_gaussian_kernel);
        float time = time_and_record(&start, &stop);
        
        if (i == 0) {
            // Write last image from batch
            write_output_image("output_part4b.ppm", d_output_batch + ((NUM_IMAGES - 1) * width * height), width, height);
        }
        
        if (i != 0) { // the first iteration is a warm-up
            part4_optimized_total += time;
        }
        CUDA_CHECK(cudaMemset(d_output_batch, 0, batch_gray_bytes));
    }
    
    CUDA_CHECK(cudaFree(d_input_batch));
    CUDA_CHECK(cudaFree(d_output_batch));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input_rgb));
    CUDA_CHECK(cudaFree(d_intermediate_gray));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gaussian_kernel));
    free(h_input_rgb);
    free(h_intermediate_gray);
    free(h_output_gray);
    
    printf("\n=== SUMMARY (Average of %d iterations) ===\n", NUM_ITERATIONS);
    printf("CPU:                                  %.4f ms (single image)\n", cpu_avg_time);
    
    /* Uncomment each of theses lines to see the performance of your implementations */ 
    // printf("Part 1 (Naive):                       %.4f ms (single image)\n", part1_total / NUM_ITERATIONS);
    // printf("Part 2 (Shared Memory Blur):          %.4f ms (single image)\n", part2_total / NUM_ITERATIONS);
    // printf("Part 3a (Fused Shared):               %.4f ms (single image)\n", part3_shared_total / NUM_ITERATIONS);
    // printf("Part 3b (Fused Cooperative Groups):   %.4f ms (single image)\n", part3_cg_total / NUM_ITERATIONS);
    // printf("Part 3x100 (compare with Part 4):     %.4f ms\n", part3_batch_total / NUM_ITERATIONS);
    // printf("Part 4a (Persistent Simple, 100 img): %.4f ms\n", part4_simple_total / NUM_ITERATIONS);
    // printf("Part 4b (Persistent Optimized, 100):  %.4f ms\n", part4_optimized_total / NUM_ITERATIONS);
    // printf("\nSpeedups vs Part 3x100:\n");
    // printf("  Part 4a (Simple):    %.2fx\n", (part3_batch_total / NUM_ITERATIONS) / (part4_simple_total / NUM_ITERATIONS));
    // printf("  Part 4b (Optimized): %.2fx\n", (part3_batch_total / NUM_ITERATIONS) / (part4_optimized_total / NUM_ITERATIONS));
    printf("\nProgram finished successfully.\n");
    return 0;
}
