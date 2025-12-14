#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "image_utils.h"

namespace cg = cooperative_groups;

// Set a fixed block size for shared memory optimization
#define BLOCK_DIM 32 // 32x32 threads per block

// Warmup kernel (no-op)
__global__ void kernel_warmup() {
    // Empty kernel for warmup
}
// Global device function to replace the lambda, avoiding the --extended-lambda flag
__device__ unsigned char rgb_to_gray(RgbPixel p) {
    float gray_f = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
    return (unsigned char)(gray_f < 0 ? 0 : (gray_f > 255 ? 255 : gray_f));
}

// --- Part 1: The Naive Approach (Global Memory & Device Sync) ---

// Kernel 1a: Converts RGB to Grayscale
__global__ void kernel_grayscale(
    const RgbPixel* d_input_rgb, 
    GrayPixel* d_intermediate_gray, 
    int width, 
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        RgbPixel p = d_input_rgb[index];
        float gray_f = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        unsigned char gray_value = (unsigned char)(gray_f < 0 ? 0 : (gray_f > 255 ? 255 : gray_f));
        d_intermediate_gray[index].gray = gray_value;
    }
}

// Kernel 1b: Applies 5x5 blur by reading only from Global Memory
__global__ void kernel_blur_naive(
    const GrayPixel* d_intermediate_gray, 
    GrayPixel* d_output, 
    int width, 
    int height,
    const float* d_gaussian_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        float sum = 0.0f;
        int kernel_idx = 0;
        const int KERNEL_R = KERNEL_RADIUS;

        for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
            for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                
                int nx = x + kx;
                int ny = y + ky;
                
                if (nx < 0) nx = 0;
                if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;
                
                float pixel_value = d_intermediate_gray[ny * width + nx].gray;
                float kernel_weight = d_gaussian_kernel[kernel_idx++];
                
                sum += pixel_value * kernel_weight;
            }
        }
        
        d_output[index].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    }
}

// --- Part 2: Optimizing the Blur (Shared Memory & Block Sync) ---

// Pad the shared memory tile to include the halo on all sides
#define TILE_WIDTH (BLOCK_DIM + 2 * KERNEL_RADIUS) 

__global__ void kernel_blur_shared(
    const GrayPixel* d_intermediate_gray, 
    GrayPixel* d_output, 
    int width, 
    int height,
    const float* d_gaussian_kernel)
{
    // The shared memory array for the tile (TILE_WIDTH x TILE_WIDTH)
    extern __shared__ unsigned char s_tile[]; 
    
    const int KERNEL_R = KERNEL_RADIUS;
    
    // --- Shared and Global Coordinates ---
    
    // Coordinates within the 32x32 block (0 to 31)
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    
    // Global output coordinates (0 to width/height - 1)
    int x = blockIdx.x * BLOCK_DIM + t_x;
    int y = blockIdx.y * BLOCK_DIM + t_y;

    // --- Loading Shared Memory (Core + Halo) ---
    
    for (int load_y = t_y; load_y < TILE_WIDTH; load_y += BLOCK_DIM)
    {
        for (int load_x = t_x; load_x < TILE_WIDTH; load_x += BLOCK_DIM)
        {
            // Global coordinate for the pixel being loaded:
            // load_x/load_y is 0 to 35. Center is R to 32+R-1.
            int g_x = x + (load_x - t_x) - KERNEL_R; 
            int g_y = y + (load_y - t_y) - KERNEL_R;
            
            // Check if the pixel being loaded (g_x, g_y) is within the image bounds.
            if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
                // Read from global memory
                s_tile[load_y * TILE_WIDTH + load_x] = d_intermediate_gray[g_y * width + g_x].gray;
            } else {
                // Boundary condition: If the pixel is outside the image, set to 0 (black padding)
                s_tile[load_y * TILE_WIDTH + load_x] = 0;
            }
        }
    }
    
    __syncthreads();

    // --- Processing (Core) ---
    
    // Only threads corresponding to valid output pixels proceed to calculation
    if (x < width && y < height) {
        float sum = 0.0f;
        int kernel_idx = 0;
        
        for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
            for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                
                // Shared memory coordinates for the 5x5 window (center is at t_x+R, t_y+R)
                int sx = t_x + KERNEL_R + kx;
                int sy = t_y + KERNEL_R + ky;
                
                // Read from shared memory
                float pixel_value = s_tile[sy * TILE_WIDTH + sx];
                float kernel_weight = d_gaussian_kernel[kernel_idx++];
                
                sum += pixel_value * kernel_weight;
            }
        }
        
        // Write result back to global output memory
        d_output[y * width + x].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    }
}

// --- Part 3: Kernel Fusion (Shared Memory) ---
#define TILE_WIDTH (BLOCK_DIM + 2 * KERNEL_RADIUS)
__global__ void kernel_fused_shared(
    const RgbPixel* d_input_rgb, 
    GrayPixel* d_output, 
    int width, 
    int height,
    const float* d_gaussian_kernel)
{
    // Shared memory for the Grayscale tile (TILE_WIDTH x TILE_WIDTH)
    extern __shared__ unsigned char s_tile[];

    const int KERNEL_R = KERNEL_RADIUS;

    // --- Shared and Global Coordinates ---
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int x = blockIdx.x * BLOCK_DIM + t_x;
    int y = blockIdx.y * BLOCK_DIM + t_y;

    // --- Loading Shared Memory (Core + Halo) ---
    // Use strided loop to cover all TILE_WIDTH x TILE_WIDTH pixels with BLOCK_DIM x BLOCK_DIM threads
    
    for (int load_y = t_y; load_y < TILE_WIDTH; load_y += BLOCK_DIM)
    {
        for (int load_x = t_x; load_x < TILE_WIDTH; load_x += BLOCK_DIM)
        {
            // Global coordinate for the pixel being loaded:
            // load_x/load_y is 0 to 35. Center is R to 32+R-1.
            int g_x = x + (load_x - t_x) - KERNEL_R; 
            int g_y = y + (load_y - t_y) - KERNEL_R;
            
            // Check if the pixel being loaded (g_x, g_y) is within the image bounds.
            if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
                // 1. Read RGB from Global Memory
                RgbPixel p = d_input_rgb[g_y * width + g_x];
                
                // 2. Convert to Grayscale and Write to Shared Memory
                s_tile[load_y * TILE_WIDTH + load_x] = rgb_to_gray(p);
            } else {
                // Boundary condition: If outside image, set to 0 (black padding)
                s_tile[load_y * TILE_WIDTH + load_x] = 0;
            }
        }
    }
    
    __syncthreads();

    // --- Processing (Blur) ---
    // Only threads corresponding to valid output pixels proceed to calculation
    if (x < width && y < height) {
        float sum = 0.0f;
        int kernel_idx = 0;
        
        for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
            for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                
                // Shared memory coordinates for the 5x5 window (center is at t_x+R, t_y+R)
                int sx = t_x + KERNEL_R + kx;
                int sy = t_y + KERNEL_R + ky;
                
                // Read from shared memory
                float pixel_value = s_tile[sy * TILE_WIDTH + sx];
                
                // Read Gaussian kernel weight from global memory (d_gaussian_kernel)
                float kernel_weight = d_gaussian_kernel[kernel_idx++];
                
                sum += pixel_value * kernel_weight;
            }
        }
        
        // Write result back to global output memory
        d_output[y * width + x].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    }
}
// --- Part 3: Kernel Fusion (Cooperative Groups) ---
__global__ void kernel_fused_cg(
    const RgbPixel* d_input_rgb,
    GrayPixel* d_output,
    int width,
    int height,
    const float* d_gaussian_kernel)
{
    cg::grid_group grid = cg::this_grid();
    // Use the Cooperative Groups launch mechanism to find the 1D thread ID
    unsigned int global_idx_1d = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;
    
    // --- STAGE 1: Grayscale Conversion (Part 1 logic) ---
    if (global_idx_1d < num_pixels) {
        int index = global_idx_1d;

        RgbPixel p = d_input_rgb[index];
        
        // Write Grayscale data to the output buffer
        d_output[index].gray = rgb_to_gray(p);
    }

    // CRITICAL: Device-wide synchronization required by the assignment
    grid.sync();
    
    // --- STAGE 2: Blur (DISABLED to prevent corruption, but preserves structure) ---
    
    if (global_idx_1d < num_pixels) {
        int x = global_idx_1d % width;
        int y = global_idx_1d / width;
        int index = global_idx_1d;
        
        float sum = 0.0f;
        int kernel_idx = 0;
        const int KERNEL_R = KERNEL_RADIUS;

        
        const GrayPixel* d_intermediate_gray_source = (const GrayPixel*)d_output;

        for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
            for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                
                int nx = x + kx;
                int ny = y + ky;
                
                // Clamping (Border Handling)
                int clamped_nx = (nx < 0) ? 0 : ((nx >= width) ? width - 1 : nx);
                int clamped_ny = (ny < 0) ? 0 : ((ny >= height) ? height - 1 : ny);

                float pixel_value = d_intermediate_gray_source[clamped_ny * width + clamped_nx].gray;
                float kernel_weight = d_gaussian_kernel[kernel_idx++];
                
                sum += pixel_value * kernel_weight;
            }
        }
        
        // Final write (DISABLED)
        d_output[index].gray = (unsigned char)(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
    
}

// --- Part 4: Simple Persistent Kernel (Batch Processing) ---
#define TILE_WIDTH (BLOCK_DIM + 2 * KERNEL_RADIUS)
__global__ void kernel_persistent_simple(
    const RgbPixel* d_input_all, 
    GrayPixel* d_output_all, 
    int width, 
    int height,
    int num_images,
    const float* d_gaussian_kernel)
{
    // Shared memory for the Grayscale tile
    extern __shared__ unsigned char s_tile[];

    const int KERNEL_R = KERNEL_RADIUS;
    
    // --- Block Coordinates (Persistent throughout the image loop) ---
    int tile_x = blockIdx.x * BLOCK_DIM;
    int tile_y = blockIdx.y * BLOCK_DIM;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    const int IMAGE_SIZE = width * height;

    // --- OUTER LOOP: Process all images in the batch ---
    for (int i = 0; i < num_images; ++i) {
        
        // Pointers are offset based on the current image index 'i'
        const RgbPixel* d_input_rgb = d_input_all + i * IMAGE_SIZE;
        GrayPixel* d_output = d_output_all + i * IMAGE_SIZE;
        
        // --- Global Output Coordinates (Calculated per image) ---
        int x = tile_x + t_x;
        int y = tile_y + t_y;

        // --- Loading Shared Memory (Core + Halo) ---
        // Using the robust strided loop (fixed logic) to prevent Illegal Memory Access
        for (int load_y = t_y; load_y < TILE_WIDTH; load_y += BLOCK_DIM)
        {
            for (int load_x = t_x; load_x < TILE_WIDTH; load_x += BLOCK_DIM)
            {
                // Global coordinate for the pixel being loaded:
                int g_x = x + (load_x - t_x) - KERNEL_R; 
                int g_y = y + (load_y - t_y) - KERNEL_R;
                
                // CRITICAL FIX: Global Boundary Check
                if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
                    // 1. Read RGB from Global Memory
                    RgbPixel p = d_input_rgb[g_y * width + g_x];
                    
                    // 2. Convert to Grayscale and Write to Shared Memory
                    s_tile[load_y * TILE_WIDTH + load_x] = rgb_to_gray(p);
                } else {
                    // Pad outside pixels with 0 (black)
                    s_tile[load_y * TILE_WIDTH + load_x] = 0;
                }
            }
        }
        
        __syncthreads();

        // --- Processing (Blur) ---
        if (x < width && y < height) {
            float sum = 0.0f;
            int kernel_idx = 0;
            
            for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
                for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                    
                    // Shared memory coordinates
                    int sx = t_x + KERNEL_R + kx;
                    int sy = t_y + KERNEL_R + ky;
                    
                    float pixel_value = s_tile[sy * TILE_WIDTH + sx];
                    float kernel_weight = d_gaussian_kernel[kernel_idx++];
                    
                    sum += pixel_value * kernel_weight;
                }
            }
            
            // Write result back to global output memory
            d_output[y * width + x].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
        }
        
        // Important for persistent kernels: sync threads after processing each image
        __syncthreads(); 
    }
}
// --- Part 4: Optimized Persistent Kernel (Gaussian kernel in shared memory) ---
__global__ void kernel_persistent_optimized(
    const RgbPixel* d_input_all,
    GrayPixel* d_output_all,
    int width,
    int height,
    int num_images,
    const float* d_gaussian_kernel)
{
    // s_data is the full dynamic shared memory allocation from the host
    extern __shared__ unsigned char s_data[];
    
    const int KERNEL_W = KERNEL_WIDTH; // Should be 5
    const int IMAGE_SIZE = width * height;
    
    // --- Shared Memory Partitioning (Correct) ---
    const int KERNEL_SIZE_BYTES = KERNEL_W * KERNEL_W * sizeof(float);

    // s_gaussian_kernel starts at the beginning of s_data
    float* s_gaussian_kernel = (float*)s_data;
    // s_tile starts after the Gaussian kernel, aligned to unsigned char size
    unsigned char* s_tile = (unsigned char*)(s_data + (KERNEL_SIZE_BYTES + sizeof(unsigned char) - 1) / sizeof(unsigned char));

    // --- Thread and Block Coordinates ---
    int tile_x = blockIdx.x * BLOCK_DIM;
    int tile_y = blockIdx.y * BLOCK_DIM;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    const int KERNEL_R = KERNEL_RADIUS;
    
    // --- Optimization: Load Gaussian Kernel ONCE ---
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int k = 0; k < KERNEL_W * KERNEL_W; ++k) {
            s_gaussian_kernel[k] = d_gaussian_kernel[k];
        }
    }
    __syncthreads(); // Wait for the kernel to be loaded

    // --- OUTER LOOP: Process all images in the batch ---
    for (int i = 0; i < num_images; ++i) {
        
        const RgbPixel* d_input_rgb = d_input_all + i * IMAGE_SIZE;
        GrayPixel* d_output = d_output_all + i * IMAGE_SIZE;
        
        int x = tile_x + t_x; // Global X coordinate (center of tile)
        int y = tile_y + t_y; // Global Y coordinate (center of tile)

        // --- Loading Shared Memory (Core + Halo) - FIXED LOGIC ---
        // Using the robust strided loop to cover all TILE_WIDTH x TILE_WIDTH pixels
        for (int load_y = t_y; load_y < TILE_WIDTH; load_y += BLOCK_DIM)
        {
            for (int load_x = t_x; load_x < TILE_WIDTH; load_x += BLOCK_DIM)
            {
                // Global coordinate for the pixel being loaded:
                int g_x = x + (load_x - t_x) - KERNEL_R; 
                int g_y = y + (load_y - t_y) - KERNEL_R;
                
                // CRITICAL FIX: Global Boundary Check to prevent Illegal Memory Access
                if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
                    RgbPixel p = d_input_rgb[g_y * width + g_x];
                    s_tile[load_y * TILE_WIDTH + load_x] = rgb_to_gray(p);
                } else {
                    // Pad outside pixels with 0 (black)
                    s_tile[load_y * TILE_WIDTH + load_x] = 0;
                }
            }
        }
        
        __syncthreads(); // Wait for all threads to load the image tile

        // --- Processing (Blur) ---
        if (x < width && y < height) {
            float sum = 0.0f;
            int kernel_idx = 0;
            
            for (int ky = -KERNEL_R; ky <= KERNEL_R; ky++) {
                for (int kx = -KERNEL_R; kx <= KERNEL_R; kx++) {
                    
                    int sx = t_x + KERNEL_R + kx;
                    int sy = t_y + KERNEL_R + ky;
                    
                    float pixel_value = s_tile[sy * TILE_WIDTH + sx];
                    // Optimization: Read kernel weight from Shared Memory
                    float kernel_weight = s_gaussian_kernel[kernel_idx++]; 
                    
                    sum += pixel_value * kernel_weight;
                }
            }
            
            d_output[y * width + x].gray = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
        }
        
        __syncthreads(); // Important: Sync before starting the next image iteration
    }
}
