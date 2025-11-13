/**
 * Lesson 6: Shared Memory - The Secret to 100x Speedups
 * =====================================================
 *
 * The Journey to Here:
 * -------------------
 * Week 1: You learned that global memory is SLOW (200-800 cycles)
 * Now: Discover shared memory - 100x faster (2-4 cycles)!
 *
 * The Problem We're Solving:
 * -------------------------
 * Global memory bandwidth is the #1 bottleneck in GPU computing.
 * Your RTX 2050: 112 GB/s sounds fast, but with 2048 cores, that's
 * only 56 MB/s per core! Meanwhile, each core can process data at
 * multiple GB/s. How do we feed these hungry cores?
 *
 * The Answer: Memory Hierarchy
 * ---------------------------
 * Registers → Shared Memory → L1/L2 Cache → Global Memory
 * (1 cycle)   (2-4 cycles)    (20-200)      (200-800)
 *
 * What We'll Build:
 * ----------------
 * 1. Understand shared memory from transistor level up
 * 2. See why matrix transpose is the perfect example
 * 3. Build naive version (often SLOWER than CPU!)
 * 4. Add shared memory (10x+ speedup)
 * 5. Handle bank conflicts (another 2x)
 * 6. Master the most important optimization in CUDA
 *
 * Real-World Impact:
 * -----------------
 * Every fast GPU algorithm uses shared memory:
 * - Matrix multiplication: 95% of deep learning
 * - Convolution: Image processing, CNNs
 * - Reduction: Sum, max, min operations
 * - FFT: Signal processing
 * - Stencils: Scientific computing
 *
 * Compile: nvcc -O3 -arch=sm_86 -o lesson06 lesson06-shared-memory.cu
 * Run: ./lesson06
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

/**
 * FIRST PRINCIPLES: What is Shared Memory?
 * ---------------------------------------
 * 
 * Think of GPU architecture like a restaurant:
 * 
 * Global Memory = Main warehouse (far away, huge capacity)
 * Shared Memory = Kitchen pantry (nearby, limited space)
 * Registers = Chef's workstation (immediate access, tiny)
 * 
 * Smart chefs (threads) bring ingredients from warehouse to
 * pantry ONCE, then all chefs in that kitchen (block) can
 * use them repeatedly!
 * 
 * Hardware Reality:
 * - Located ON the SM (Streaming Multiprocessor)
 * - 48 KB per SM on your RTX 2050
 * - Accessible by all threads in a block
 * - As fast as registers for conflict-free access
 * - Divided into 32 banks (we'll see why)
 */

/**
 * THE PROBLEM: Matrix Transpose
 * -----------------------------
 * 
 * Why transpose? It's the PERFECT teaching example:
 * 
 * Input:          Output:
 * 1 2 3 4         1 5 9 13
 * 5 6 7 8   -->   2 6 10 14
 * 9 10 11 12      3 7 11 15
 * 13 14 15 16     4 8 12 16
 * 
 * Problem: Coalesced reads become scattered writes!
 * - Threads 0-3 read row 0: [1,2,3,4] (coalesced ✓)
 * - But write to: [0,4,8,12] (scattered ✗)
 * 
 * This DESTROYS performance. Shared memory fixes it!
 */

// Configuration - these matter!
#define TILE_DIM 32     // Tile dimension (32x32 = 1024 threads)
#define BLOCK_ROWS 8    // Threads per block in y-dimension

// Helper to get time
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * STEP 1: CPU Reference Implementation
 * ------------------------------------
 * This is our baseline for correctness
 */
void transposeCPU(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

/**
 * STEP 2: Naive GPU - Direct Translation
 * --------------------------------------
 * Warning: This is often SLOWER than CPU!
 * Why? Terrible memory access patterns.
 */
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (xIndex < width && yIndex < height) {
        int index_in = yIndex * width + xIndex;
        int index_out = xIndex * height + yIndex;
        output[index_out] = input[index_in];
    }
}

/**
 * STEP 3: Coalesced Transpose (Still Naive)
 * -----------------------------------------
 * Let's fix the write pattern but still use global memory
 */
__global__ void transposeCoalesced(float *input, float *output, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (xIndex < width && yIndex < height) {
        int index_in = yIndex * width + xIndex;
        // Transpose during write by swapping block indices
        int xIndex_out = blockIdx.y * TILE_DIM + threadIdx.x;
        int yIndex_out = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_out = yIndex_out * height + xIndex_out;
        
        if (xIndex_out < height && yIndex_out < width) {
            output[index_out] = input[index_in];
        }
    }
}

/**
 * STEP 4: Shared Memory Magic!
 * ---------------------------
 * This is where we see massive speedup
 */
__global__ void transposeShared(float *input, float *output, int width, int height) {
    // Allocate shared memory tile
    // +1 to avoid bank conflicts (we'll explain!)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    // Calculate input indices
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Collaborative loading: Each thread loads one element
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = input[yIndex * width + xIndex];
    }
    
    // CRITICAL: Wait for all threads to finish loading
    __syncthreads();
    
    // Calculate output indices (transposed)
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Collaborative storing: Read from transposed tile
    if (xIndex < height && yIndex < width) {
        output[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * STEP 5: Understanding Bank Conflicts
 * ------------------------------------
 * Shared memory is divided into 32 banks (like 32 separate memories).
 * If threads in a warp access different banks → full speed!
 * If threads access the same bank → serialized access (SLOW!)
 * 
 * Memory layout for 32-bit values:
 * Bank 0: addresses 0, 32, 64, 96...
 * Bank 1: addresses 1, 33, 65, 97...
 * ...
 * Bank 31: addresses 31, 63, 95...
 */
__global__ void demonstrateBankConflicts(float *output) {
    __shared__ float shared[32][32];    // Potential conflicts!
    __shared__ float shared_padded[32][33];  // Conflict-free!
    
    int tid = threadIdx.x;
    
    // Scenario 1: No bank conflicts (stride-1 access)
    shared[0][tid] = tid;  // Threads 0-31 access banks 0-31
    
    // Scenario 2: 2-way bank conflict (stride-2)
    shared[tid][0] = tid;  // Even threads hit even banks, odd hit odd
    
    // Scenario 3: 32-way bank conflict (worst case!)
    shared[tid][0] = tid;  // All threads hit bank 0!
    
    // Solution: Padding eliminates conflicts
    shared_padded[tid][0] = tid;  // Now threads hit different banks
    
    __syncthreads();
    
    // Just to use the values (prevent optimization)
    if (tid == 0) {
        output[0] = shared[0][0] + shared_padded[0][0];
    }
}

/**
 * STEP 6: Optimized with Unrolling
 * --------------------------------
 * Process multiple elements per thread for better efficiency
 */
__global__ void transposeOptimized(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Each thread loads TILE_DIM/BLOCK_ROWS elements
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = 
                input[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block offset
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = 
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * Verification function
 */
bool verifyTranspose(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float expected = input[y * width + x];
            float actual = output[x * height + y];
            if (fabsf(expected - actual) > 1e-5) {
                printf("Mismatch at (%d,%d): expected %f, got %f\n",
                       x, y, expected, actual);
                return false;
            }
        }
    }
    return true;
}

/**
 * Measure memory bandwidth
 */
void reportBandwidth(const char* name, int width, int height, double time_ms) {
    double gb = 2.0 * width * height * sizeof(float) / 1e9;  // Read + write
    double bandwidth = gb / (time_ms / 1000.0);
    printf("%-25s: %7.2f ms, %7.2f GB/s", name, time_ms, bandwidth);
    
    // Show percentage of peak bandwidth (112 GB/s for RTX 2050)
    double peak = 112.0;
    printf(" (%5.1f%% of peak)\n", 100.0 * bandwidth / peak);
}

/**
 * Main experiment
 */
int main() {
    printf("========================================================\n");
    printf("LESSON 6: Shared Memory - The Secret to GPU Speed\n");
    printf("========================================================\n\n");
    
    // Test different matrix sizes
    int sizes[] = {1024, 2048, 4096};
    const char* size_names[] = {"1K x 1K", "2K x 2K", "4K x 4K"};
    
    for (int test = 0; test < 3; test++) {
        int size = sizes[test];
        size_t bytes = size * size * sizeof(float);
        
        printf("\n🔬 EXPERIMENT %d: Matrix Transpose %s\n", test + 1, size_names[test]);
        printf("================================================\n");
        printf("Matrix size: %d x %d = %.2f million elements\n", 
               size, size, size * size / 1e6);
        printf("Memory size: %.2f MB\n\n", bytes / (1024.0 * 1024.0));
        
        // Allocate host memory
        float *h_input = (float*)malloc(bytes);
        float *h_output_cpu = (float*)malloc(bytes);
        float *h_output_gpu = (float*)malloc(bytes);
        
        // Initialize with pattern
        for (int i = 0; i < size * size; i++) {
            h_input[i] = (float)i;
        }
        
        // CPU reference
        printf("CPU Implementation\n");
        printf("------------------\n");
        double cpu_start = getTime();
        transposeCPU(h_input, h_output_cpu, size, size);
        double cpu_time = (getTime() - cpu_start) * 1000.0;
        reportBandwidth("CPU", size, size, cpu_time);
        
        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
        
        // Configure kernel launch
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
        dim3 dimGrid((size + TILE_DIM - 1) / TILE_DIM,
                     (size + TILE_DIM - 1) / TILE_DIM);
        
        printf("\nGPU Implementations\n");
        printf("-------------------\n");
        printf("Block: %dx%d, Grid: %dx%d\n\n", 
               dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        
        // Warmup
        transposeNaive<<<dimGrid, dimBlock>>>(d_input, d_output, size, size);
        cudaDeviceSynchronize();
        
        // Test each implementation
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 1. Naive
        cudaEventRecord(start);
        transposeNaive<<<dimGrid, dimBlock>>>(d_input, d_output, size, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float naive_time;
        cudaEventElapsedTime(&naive_time, start, stop);
        
        cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
        bool naive_correct = verifyTranspose(h_input, h_output_gpu, size, size);
        reportBandwidth("GPU Naive", size, size, naive_time);
        if (!naive_correct) printf("  ❌ Incorrect results!\n");
        
        // 2. Coalesced
        cudaEventRecord(start);
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_input, d_output, size, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float coalesced_time;
        cudaEventElapsedTime(&coalesced_time, start, stop);
        reportBandwidth("GPU Coalesced", size, size, coalesced_time);
        
        // 3. Shared Memory
        cudaEventRecord(start);
        transposeShared<<<dimGrid, dimBlock>>>(d_input, d_output, size, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float shared_time;
        cudaEventElapsedTime(&shared_time, start, stop);
        
        cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
        bool shared_correct = verifyTranspose(h_input, h_output_gpu, size, size);
        reportBandwidth("GPU Shared Memory", size, size, shared_time);
        if (!shared_correct) printf("  ❌ Incorrect results!\n");
        
        // 4. Optimized
        cudaEventRecord(start);
        transposeOptimized<<<dimGrid, dimBlock>>>(d_input, d_output, size, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float optimized_time;
        cudaEventElapsedTime(&optimized_time, start, stop);
        reportBandwidth("GPU Optimized", size, size, optimized_time);
        
        // Speedup analysis
        printf("\nSpeedup Analysis\n");
        printf("----------------\n");
        printf("vs CPU:           Naive: %5.1fx, Shared: %5.1fx, Optimized: %5.1fx\n",
               cpu_time / naive_time, cpu_time / shared_time, cpu_time / optimized_time);
        printf("vs Naive GPU:     Shared: %5.1fx, Optimized: %5.1fx\n",
               naive_time / shared_time, naive_time / optimized_time);
        
        // Cleanup
        free(h_input);
        free(h_output_cpu);
        free(h_output_gpu);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Educational: Bank conflict demonstration
    printf("\n\n📚 BANK CONFLICTS EXPLAINED\n");
    printf("===========================\n\n");
    
    printf("Shared memory has 32 banks (like 32 memory chips).\n");
    printf("Best case: Each thread accesses a different bank.\n");
    printf("Worst case: All threads access the same bank (32x slower!).\n\n");
    
    printf("Example layouts:\n");
    printf("float shared[32][32]:    Column access = 32-way conflict!\n");
    printf("float shared[32][33]:    Column access = No conflicts!\n");
    printf("                        (padding shifts each row by 1 bank)\n\n");
    
    // Key insights
    printf("🔑 KEY INSIGHTS\n");
    printf("===============\n\n");
    
    printf("1. Global memory latency: 200-800 cycles\n");
    printf("   Shared memory latency: 2-4 cycles (100x faster!)\n\n");
    
    printf("2. Shared memory per SM: 48 KB (your RTX 2050)\n");
    printf("   Must be divided among active blocks\n\n");
    
    printf("3. Bank conflicts can destroy performance\n");
    printf("   Solution: Padding or access pattern changes\n\n");
    
    printf("4. __syncthreads() is MANDATORY\n");
    printf("   Ensures all threads finished loading before any read\n\n");
    
    printf("5. Occupancy matters: Too much shared memory = fewer blocks\n");
    printf("   Balance is key!\n\n");
    
    printf("✅ Congratulations! You've mastered GPU memory hierarchy!\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Bank Conflict Analysis:
 *    For array float s[32][32]:
 *    - Which accesses cause conflicts: s[i][0] or s[0][i]?
 *    - How many way conflict for s[i][i]?
 *    - Why does s[32][33] fix column access?
 *
 * 2. Memory Calculation:
 *    You have 48KB shared memory per SM.
 *    - How many 32x32 float tiles fit?
 *    - If you need 3 tiles per block, max blocks per SM?
 *    - How does this affect occupancy?
 *
 * 3. Synchronization:
 *    What happens if you remove __syncthreads()?
 *    Try it and explain the results.
 *
 * CODING EXERCISES:
 * 4. Different Tile Sizes:
 *    Modify code for 16x16 and 64x64 tiles.
 *    Which performs best? Why?
 *
 * 5. Rectangular Matrices:
 *    Handle non-square matrices (e.g., 1024x2048).
 *    What changes are needed?
 *
 * 6. Double Buffering:
 *    Use two tiles to overlap loading and computing.
 *    (Advanced technique for more overlap)
 *
 * 7. 3D Transpose:
 *    Implement transpose for 3D arrays.
 *    How do you tile in 3D?
 *
 * OPTIMIZATION CHALLENGES:
 * 8. Matrix Multiplication:
 *    Use shared memory for matrix multiply.
 *    This is THE classic GPU algorithm!
 *    Target: 80% of cuBLAS performance
 *
 * 9. Convolution:
 *    Implement 2D convolution with shared memory.
 *    Handle halo regions correctly.
 *
 * 10. Reduction:
 *     Sum all elements using shared memory.
 *     No bank conflicts allowed!
 *
 * ANALYSIS EXERCISES:
 * 11. Measure Bank Conflicts:
 *     Use Nsight Compute to measure:
 *     - Shared memory throughput
 *     - Bank conflict rate
 *     - Warp stall reasons
 *
 * 12. Occupancy Impact:
 *     Plot performance vs shared memory usage.
 *     Find the sweet spot.
 *
 * 13. Compare with L1 Cache:
 *     Some GPUs let you configure L1/shared split.
 *     When to use more L1 vs more shared?
 *
 * ADVANCED PROJECTS:
 * 14. Tiled Matrix Operations:
 *     Implement:
 *     - Tiled matrix addition
 *     - Tiled matrix multiply
 *     - Tiled convolution
 *     Compare performance
 *
 * 15. Stencil Computation:
 *     5-point stencil with shared memory.
 *     Handle ghost cells efficiently.
 *
 * 16. Real Application:
 *     Apply shared memory to image blur project.
 *     What's the speedup?
 *
 * DEEP UNDERSTANDING:
 * 17. Why 32 Banks?
 *     Research: Related to warp size?
 *     What if we had 64 banks?
 *
 * 18. Dynamic Shared Memory:
 *     Learn to allocate shared memory at runtime:
 *     extern __shared__ float dynamic[];
 *     When is this useful?
 *
 * MENTAL MODELS:
 * 
 * Model 1: The Warehouse
 * - Global memory = Remote warehouse
 * - Shared memory = Local storage room
 * - Registers = Your desk
 * - Move inventory to storage room once, use many times
 * 
 * Model 2: The Cache Hierarchy  
 * - Like CPU cache but MANUAL
 * - You control what goes in
 * - You control when to load/store
 * - Great power, great responsibility
 * 
 * Model 3: The Neighborhood
 * - Block = Neighborhood
 * - Threads = Neighbors  
 * - Shared memory = Community garden
 * - Everyone helps load, everyone can use
 * 
 * Master shared memory and you master GPU programming!
 */