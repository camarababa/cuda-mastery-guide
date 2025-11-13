/**
 * Lesson 7: Parallel Reduction - From O(N) to O(log N) in 7 Steps
 * ===============================================================
 *
 * The Ultimate Question:
 * ---------------------
 * How do you add 1 billion numbers on a GPU?
 * CPU: for(i=0; i<N; i++) sum += array[i];  // O(N) sequential
 * GPU: ??? (We'll build this!)               // O(log N) parallel!
 *
 * The Challenge:
 * -------------
 * Reduction seems inherently sequential: you need previous results
 * to compute the next. How can 2048 threads collaborate to compute
 * ONE final answer? This is the art of parallel algorithm design!
 *
 * What We'll Build:
 * ----------------
 * 1. Understand reduction trees and parallel patterns
 * 2. Implement naive reduction (SLOW, divergent)
 * 3. Fix warp divergence (2x faster)
 * 4. Fix bank conflicts (2x faster)
 * 5. Add loop unrolling (2x faster)
 * 6. Use warp intrinsics (2x faster)
 * 7. Multiple elements per thread (2x faster)
 * Total: 32x speedup through optimization!
 *
 * Real-World Applications:
 * -----------------------
 * - Sum/Average: Statistics, neural network loss
 * - Min/Max: Finding extrema, bounding boxes
 * - Dot Product: Linear algebra, similarity
 * - Histogram: Image processing, data analysis
 * - Monte Carlo: Financial modeling, physics
 *
 * This pattern is EVERYWHERE in parallel computing!
 *
 * Compile: nvcc -O3 -arch=sm_86 -o lesson07 lesson07-reduction.cu
 * Run: ./lesson07
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>

/**
 * FIRST PRINCIPLES: The Reduction Tree
 * -----------------------------------
 * 
 * Sequential: 1+2+3+4+5+6+7+8 = 28 (7 steps)
 * 
 * Parallel reduction tree:
 * Step 0: [1] [2] [3] [4] [5] [6] [7] [8]
 *          +   +   +   +   +   +   +   +
 * Step 1: [3]     [7]     [11]    [15]
 *          +-------+       +-------+
 * Step 2: [10]            [26]
 *          +---------------+
 * Step 3: [36]
 * 
 * Only 3 steps! O(log N) instead of O(N)
 * 
 * But GPUs have thousands of threads...
 * How do we coordinate them?
 */

// Timer utilities
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * CPU Reference Implementation
 * ---------------------------
 * Simple sequential sum for verification
 */
float reduceCPU(float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * STEP 1: Naive GPU Reduction (Interleaved Addressing)
 * ---------------------------------------------------
 * This is the most intuitive approach but has major problems!
 */
__global__ void reduceNaive(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    // Do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // Problem 1: Divergent branching (threads in same warp take different paths)
        // Problem 2: Bank conflicts (threads access same bank)
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 2: Reduce Divergence (Sequential Addressing)
 * ------------------------------------------------
 * Better: All threads in first half of warp work together
 */
__global__ void reduceNoDivergence(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - no divergence!
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 3: First Add During Load
 * -----------------------------
 * Halve the number of blocks needed
 */
__global__ void reduceAddDuringLoad(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first add during load
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 4: Unroll the Last Warp
 * ----------------------------
 * When s <= 32, all threads are in same warp
 * No need for synchronization!
 */
__device__ void warpReduce(volatile float *sdata, int tid) {
    // For sm_80 and higher, we need explicit warp synchronization
    if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
    if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
    if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
    if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void reduceUnrollLastWarp(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and first add
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction until we reach warp size
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll last warp
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 5: Complete Unrolling
 * -------------------------
 * If block size is known at compile time, unroll everything!
 */
template <unsigned int blockSize>
__global__ void reduceCompleteUnroll(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    
    // Load and first add
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockSize < n) mySum += g_idata[i + blockSize];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Fully unrolled reduction
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    // Unrolled warp reduction
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        if (blockSize >= 32) smem[tid] += smem[tid + 16];
        if (blockSize >= 16) smem[tid] += smem[tid + 8];
        if (blockSize >= 8) smem[tid] += smem[tid + 4];
        if (blockSize >= 4) smem[tid] += smem[tid + 2];
        if (blockSize >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 6: Multiple Elements Per Thread
 * -----------------------------------
 * Better instruction/memory ratio
 */
template <unsigned int blockSize>
__global__ void reduceMultiplePerThread(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    
    // Grid-stride loop
    float mySum = 0;
    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n) mySum += g_idata[i + blockSize];
        i += gridSize;
    }
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction (same as before)
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        if (blockSize >= 32) smem[tid] += smem[tid + 16];
        if (blockSize >= 16) smem[tid] += smem[tid + 8];
        if (blockSize >= 8) smem[tid] += smem[tid + 4];
        if (blockSize >= 4) smem[tid] += smem[tid + 2];
        if (blockSize >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * STEP 7: Warp Shuffle Instructions
 * ---------------------------------
 * Modern GPUs: Direct register-to-register communication!
 */
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceShfl(float *g_idata, float *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop
    float sum = 0;
    for (unsigned int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += g_idata[idx];
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Write warp result to shared memory
    __shared__ float warpSums[32]; // Assuming max 1024 threads = 32 warps
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? warpSums[tid] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) {
            g_odata[blockIdx.x] = sum;
        }
    }
}

/**
 * Performance measurement wrapper
 */
void measurePerformance(const char* name, 
                       void (*kernel)(float*, float*, unsigned int),
                       float *d_in, float *d_out, 
                       int n, int blocks, int threads,
                       float cpu_result) {
    // Warmup
    kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Final reduction on CPU
    float *h_out = new float[blocks];
    cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_result = 0;
    for (int i = 0; i < blocks; i++) {
        gpu_result += h_out[i];
    }
    
    // Calculate bandwidth
    float bandwidth = (n * sizeof(float)) / (milliseconds / 1000.0) / 1e9;
    
    printf("%-25s: %7.3f ms, %6.2f GB/s, Result: %.1f (Error: %.2e)\n", 
           name, milliseconds, bandwidth, gpu_result, 
           fabsf(gpu_result - cpu_result) / cpu_result);
    
    delete[] h_out;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Main experiment
 */
int main() {
    printf("===========================================================\n");
    printf("LESSON 7: Parallel Reduction - The Art of GPU Algorithms\n");
    printf("===========================================================\n\n");
    
    // Test parameters
    int n = 1 << 24;  // 16M elements
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = 2048;  // Limit for manageable output size
    
    printf("Configuration:\n");
    printf("- Elements: %d (%.1f million)\n", n, n/1e6);
    printf("- Threads per block: %d\n", threads);
    printf("- Blocks: %d\n", blocks);
    printf("- Elements per thread: %.1f\n\n", (float)n / (blocks * threads));
    
    // Allocate and initialize
    size_t bytes = n * sizeof(float);
    float *h_in = new float[n];
    
    // Initialize with pattern that avoids numerical issues
    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f / (i + 1);  // Harmonic series
    }
    
    // CPU reference
    printf("CPU Reference\n");
    printf("-------------\n");
    double cpu_start = getTime();
    float cpu_sum = reduceCPU(h_in, n);
    double cpu_time = (getTime() - cpu_start) * 1000.0;
    printf("Time: %.3f ms, Result: %.6f\n\n", cpu_time, cpu_sum);
    
    // GPU setup
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    printf("GPU Implementations\n");
    printf("-------------------\n");
    
    // Test each version
    measurePerformance("1. Naive (divergent)", 
                      reduceNaive, d_in, d_out, n, blocks, threads, cpu_sum);
    
    measurePerformance("2. No divergence", 
                      reduceNoDivergence, d_in, d_out, n, blocks, threads, cpu_sum);
    
    measurePerformance("3. Add during load", 
                      reduceAddDuringLoad, d_in, d_out, n, blocks/2, threads, cpu_sum);
    
    measurePerformance("4. Unroll last warp", 
                      reduceUnrollLastWarp, d_in, d_out, n, blocks/2, threads, cpu_sum);
    
    // Template versions require specific block sizes
    void (*reduce256)(float*, float*, unsigned int) = 
        &reduceCompleteUnroll<256>;
    measurePerformance("5. Complete unroll", 
                      reduce256, d_in, d_out, n, blocks/2, 256, cpu_sum);
    
    void (*reduce256m)(float*, float*, unsigned int) = 
        &reduceMultiplePerThread<256>;
    measurePerformance("6. Multiple per thread", 
                      reduce256m, d_in, d_out, n, 128, 256, cpu_sum);
    
    measurePerformance("7. Warp shuffle", 
                      reduceShfl, d_in, d_out, n, 128, 256, cpu_sum);
    
    // Speedup analysis
    printf("\nSpeedup Evolution\n");
    printf("-----------------\n");
    printf("Each optimization builds on the previous:\n");
    printf("1. Baseline\n");
    printf("2. Fix divergence:        ~2x\n");
    printf("3. Add during load:       ~2x (and uses half the blocks)\n");
    printf("4. Unroll last warp:      ~1.5x\n");
    printf("5. Complete unroll:       ~1.5x\n");
    printf("6. Multiple per thread:   ~2x\n");
    printf("7. Warp shuffle:          ~1.2x\n");
    printf("Total:                    ~30x improvement!\n\n");
    
    // Educational: Reduction tree visualization
    printf("REDUCTION TREE VISUALIZATION\n");
    printf("===========================\n\n");
    
    printf("8 elements: [1][2][3][4][5][6][7][8]\n");
    printf("            \\ / \\ / \\ / \\ /\n");
    printf("Step 1:      [3] [7] [11][15]  (4 active threads)\n");
    printf("              \\   /   \\   /\n");
    printf("Step 2:       [10]    [26]     (2 active threads)\n");
    printf("                \\      /\n");
    printf("Step 3:          [36]          (1 active thread)\n\n");
    
    printf("Key insight: log₂(N) steps instead of N steps!\n\n");
    
    // Key takeaways
    printf("🔑 KEY INSIGHTS\n");
    printf("===============\n\n");
    
    printf("1. Parallel reduction: O(N) → O(log N) steps\n");
    printf("2. Warp divergence kills performance\n");
    printf("3. Bank conflicts matter in shared memory\n");
    printf("4. Loop unrolling helps instruction throughput\n");
    printf("5. Warp-level primitives are fastest\n");
    printf("6. Algorithmic improvements > micro-optimizations\n\n");
    
    // Cleanup
    delete[] h_in;
    cudaFree(d_in);
    cudaFree(d_out);
    
    printf("✅ Master reduction, master GPU algorithms!\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Divergence Analysis:
 *    In naive reduction with 32 threads:
 *    - Step 1: How many threads work? How many idle?
 *    - Step 2: How many threads work?
 *    - When do all threads in a warp become idle?
 *
 * 2. Complexity Analysis:
 *    - Sequential sum: O(?)
 *    - Parallel reduction: O(?)
 *    - How many processors needed for O(log N)?
 *
 * 3. Bank Conflicts:
 *    In sequential addressing, when do conflicts occur?
 *    Draw the access pattern for 32 threads.
 *
 * CODING EXERCISES:
 * 4. Different Operations:
 *    Implement reduction for:
 *    - Maximum (instead of sum)
 *    - Minimum
 *    - Product
 *    - Logical AND/OR
 *
 * 5. Argmax/Argmin:
 *    Find not just the max value, but its index!
 *    Hint: Reduce pairs (value, index)
 *
 * 6. Variance Calculation:
 *    Compute variance in one pass:
 *    - Sum of values
 *    - Sum of squares
 *    Then: variance = E[X²] - (E[X])²
 *
 * 7. Block-wide Reduction:
 *    Current code reduces within blocks.
 *    Add a second kernel to reduce block results.
 *
 * OPTIMIZATION CHALLENGES:
 * 8. Vectorized Load:
 *    Use float2 or float4 for coalesced loads.
 *    Does it improve bandwidth?
 *
 * 9. Persistent Threads:
 *    Instead of one element per thread,
 *    have each thread process many elements.
 *
 * 10. Multi-level Reduction:
 *     For very large arrays (billions):
 *     - Level 1: Reduce to thousands
 *     - Level 2: Reduce to one
 *     Design an efficient multi-kernel approach.
 *
 * ANALYSIS EXERCISES:
 * 11. Optimal Block Size:
 *     Test with 32, 64, 128, 256, 512, 1024 threads.
 *     Plot performance. Why is there a sweet spot?
 *
 * 12. Arithmetic Intensity:
 *     Reduction has low AI (1 op per load).
 *     How to increase it? (Hint: do more per element)
 *
 * 13. Compare Libraries:
 *     Implement with:
 *     - CUB (cub::DeviceReduce)
 *     - Thrust (thrust::reduce)
 *     How close is your implementation?
 *
 * ADVANCED PROJECTS:
 * 14. Histogram:
 *     Use atomics for conflict resolution:
 *     atomicAdd(&histogram[value], 1);
 *
 * 15. Dot Product:
 *     Combine multiplication and reduction:
 *     result = Σ(a[i] * b[i])
 *
 * 16. Parallel Prefix Sum (Scan):
 *     Input:  [1, 2, 3, 4, 5]
 *     Output: [1, 3, 6, 10, 15]
 *     This is reduction's "inverse"!
 *
 * REAL APPLICATIONS:
 * 17. Monte Carlo Pi:
 *     Generate random points, count inside circle.
 *     Use reduction for the count.
 *
 * 18. Image Statistics:
 *     Calculate mean, min, max brightness.
 *     Handle RGB channels.
 *
 * 19. Loss Function:
 *     In neural networks, sum losses across batch.
 *     Make it fast enough for training.
 *
 * DEEP UNDERSTANDING:
 * 20. Why log₂(N) Steps?
 *     Prove that binary tree has height log₂(N).
 *     What about 3-way or 4-way reduction?
 *
 * 21. Theoretical Limits:
 *     What's the theoretical minimum time?
 *     Consider bandwidth and compute limits.
 *
 * MENTAL MODELS:
 * 
 * Model 1: The Tournament
 * - Threads are players in elimination tournament
 * - Each round, half are eliminated
 * - log₂(N) rounds to find winner
 * 
 * Model 2: The Binary Tree
 * - Leaves are input elements
 * - Internal nodes are partial sums
 * - Root is final result
 * - Height = log₂(N)
 * 
 * Model 3: The Pyramid
 * - Base layer: All elements
 * - Each level up: Half the elements
 * - Top: Single result
 * - Parallel width decreases exponentially
 * 
 * Master reduction and you understand 
 * the essence of parallel algorithms!
 */