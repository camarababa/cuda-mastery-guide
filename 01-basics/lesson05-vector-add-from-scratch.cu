/**
 * Lesson 5: The Complete Journey - Building a Real GPU Application
 * ================================================================
 *
 * The Challenge:
 * -------------
 * You need to add two vectors with 100 million elements. On CPU, this
 * takes seconds. Can we make it 100x faster? Let's build a complete
 * GPU solution from scratch, learning every optimization along the way.
 *
 * What We'll Build:
 * ----------------
 * 1. Start with CPU baseline (understand the problem)
 * 2. Port to naive GPU (often SLOWER!)
 * 3. Analyze performance bottlenecks
 * 4. Apply optimizations one by one
 * 5. Achieve massive speedup
 * 6. Understand when GPU wins (and when it doesn't)
 *
 * This is Your First Complete GPU Application:
 * -------------------------------------------
 * - Memory allocation strategies
 * - Kernel optimization
 * - Performance measurement
 * - Real-world considerations
 * 
 * By the end, you'll understand why GPUs can be 100x faster...
 * and why sometimes they're not.
 *
 * Compile: nvcc -O3 -arch=sm_86 -o lesson05 lesson05-vector-add-from-scratch.cu
 * Run: ./lesson05
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/**
 * FIRST PRINCIPLES: What is Vector Addition?
 * -----------------------------------------
 * Given: a[] and b[] with n elements each
 * Compute: c[i] = a[i] + b[i] for all i
 * 
 * CPU Approach: for(i=0; i<n; i++) c[i] = a[i] + b[i]
 * GPU Approach: Thread i computes c[i] = a[i] + b[i]
 * 
 * Sounds simple? The devil is in the details...
 */

// Configuration based on your GPU
#define THREADS_PER_BLOCK 256  // Good default for most GPUs
#define ELEMENTS_PER_THREAD 1  // We'll experiment with this

/**
 * STEP 1: CPU Baseline Implementation
 * -----------------------------------
 * This is our reference. Everything is measured against this.
 */
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// More compute-intensive version to show GPU advantages
void vectorAddCPU_Complex(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        // Simulate more complex computation
        float val_a = a[i];
        float val_b = b[i];
        c[i] = sqrtf(val_a * val_a + val_b * val_b) + sinf(val_a) * cosf(val_b);
    }
}

/**
 * STEP 2: Naive GPU Implementation
 * --------------------------------
 * Direct port of CPU code. Often disappointing!
 */
__global__ void vectorAddGPU_Naive(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * STEP 3: Optimized GPU - Coalesced Memory Access
 * -----------------------------------------------
 * Ensure threads in a warp access consecutive memory
 */
__global__ void vectorAddGPU_Coalesced(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Same as naive, but we'll ensure proper alignment and access pattern
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * STEP 4: Multiple Elements Per Thread
 * ------------------------------------
 * Amortize instruction overhead
 */
__global__ void vectorAddGPU_MultiplePerThread(float *a, float *b, float *c, int n, int elementsPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

/**
 * STEP 5: Complex Computation GPU
 * -------------------------------
 * Where GPU really shines
 */
__global__ void vectorAddGPU_Complex(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val_a = a[idx];
        float val_b = b[idx];
        c[idx] = sqrtf(val_a * val_a + val_b * val_b) + sinf(val_a) * cosf(val_b);
    }
}

/**
 * Performance Timer
 */
class Timer {
    double start_time;
public:
    void start() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start_time = ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
    }
    
    double elapsed() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double end_time = ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
        return end_time - start_time;
    }
};

/**
 * Initialize vectors with reproducible random data
 */
void initializeVectors(float *a, float *b, int n) {
    srand(1234);  // Fixed seed for reproducibility
    for (int i = 0; i < n; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * Verify results match
 */
bool verifyResults(float *cpu, float *gpu, int n, const char* name) {
    float epsilon = 1e-5;  // Tolerance for floating point comparison
    
    for (int i = 0; i < n; i++) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            printf("❌ MISMATCH in %s at index %d: CPU=%f, GPU=%f\n", 
                   name, i, cpu[i], gpu[i]);
            return false;
        }
    }
    
    printf("✅ %s results match!\n", name);
    return true;
}

/**
 * Calculate and display bandwidth
 */
void reportBandwidth(const char* name, int n, double time_ms) {
    // 3 arrays × 4 bytes × n elements
    double bytes = 3.0 * sizeof(float) * n;
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = gb / (time_ms / 1000.0);
    
    printf("%s: %.2f GB/s effective bandwidth\n", name, bandwidth);
}

/**
 * MAIN EXPERIMENT
 */
int main() {
    printf("===========================================================\n");
    printf("LESSON 5: Building Complete GPU Application From Scratch\n");
    printf("===========================================================\n\n");
    
    // Test multiple problem sizes
    int sizes[] = {100000, 1000000, 10000000, 100000000};
    const char* size_labels[] = {"100K", "1M", "10M", "100M"};
    
    for (int test = 0; test < 4; test++) {
        int n = sizes[test];
        size_t size = n * sizeof(float);
        
        printf("\n🔬 EXPERIMENT %d: Vector Addition with %s elements\n", test+1, size_labels[test]);
        printf("================================================\n");
        printf("Data size: %.2f MB per vector × 3 = %.2f MB total\n\n",
               size / (1024.0 * 1024.0), 
               3 * size / (1024.0 * 1024.0));
        
        Timer timer;
        
        // Allocate host memory
        float *h_a = (float*)malloc(size);
        float *h_b = (float*)malloc(size);
        float *h_c_cpu = (float*)malloc(size);
        float *h_c_gpu = (float*)malloc(size);
        
        // Initialize
        printf("Initializing vectors...\n");
        initializeVectors(h_a, h_b, n);
        
        // ====================================
        // PART 1: CPU Baseline
        // ====================================
        printf("\n📊 CPU BASELINE\n");
        printf("---------------\n");
        
        // Simple addition
        timer.start();
        vectorAddCPU(h_a, h_b, h_c_cpu, n);
        double cpu_simple_time = timer.elapsed();
        
        printf("Simple addition: %.3f ms\n", cpu_simple_time);
        reportBandwidth("CPU Simple", n, cpu_simple_time);
        
        // Complex computation
        timer.start();
        vectorAddCPU_Complex(h_a, h_b, h_c_cpu, n);
        double cpu_complex_time = timer.elapsed();
        
        printf("Complex computation: %.3f ms\n", cpu_complex_time);
        printf("Complex/Simple ratio: %.2fx slower\n", cpu_complex_time / cpu_simple_time);
        
        // ====================================
        // PART 2: GPU Implementations
        // ====================================
        printf("\n📊 GPU IMPLEMENTATIONS\n");
        printf("--------------------\n");
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        
        // Copy data to device
        timer.start();
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
        double transfer_to_time = timer.elapsed();
        
        // Calculate grid dimensions
        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        printf("Grid configuration: %d blocks × %d threads = %d total threads\n",
               blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);
        printf("Memory transfer to GPU: %.3f ms\n", transfer_to_time);
        
        // Test 1: Naive GPU
        timer.start();
        vectorAddGPU_Naive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        double gpu_naive_time = timer.elapsed();
        
        timer.start();
        cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
        double transfer_from_time = timer.elapsed();
        
        double total_naive_time = transfer_to_time + gpu_naive_time + transfer_from_time;
        
        printf("\n1. Naive GPU:\n");
        printf("   Kernel execution: %.3f ms\n", gpu_naive_time);
        printf("   Memory transfer from GPU: %.3f ms\n", transfer_from_time);
        printf("   Total time: %.3f ms\n", total_naive_time);
        printf("   Speedup vs CPU: %.2fx\n", cpu_simple_time / total_naive_time);
        reportBandwidth("GPU Naive", n, gpu_naive_time);
        
        // Verify correctness
        vectorAddCPU(h_a, h_b, h_c_cpu, n);  // Recompute CPU result
        verifyResults(h_c_cpu, h_c_gpu, n, "Naive GPU");
        
        // Test 2: Multiple elements per thread
        timer.start();
        vectorAddGPU_MultiplePerThread<<<blocksPerGrid/4, threadsPerBlock>>>(d_a, d_b, d_c, n, 4);
        cudaDeviceSynchronize();
        double gpu_multiple_time = timer.elapsed();
        
        printf("\n2. Multiple Elements Per Thread:\n");
        printf("   Kernel execution: %.3f ms\n", gpu_multiple_time);
        printf("   Speedup vs naive: %.2fx\n", gpu_naive_time / gpu_multiple_time);
        
        // Test 3: Complex computation
        timer.start();
        vectorAddGPU_Complex<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        double gpu_complex_time = timer.elapsed();
        
        printf("\n3. Complex Computation GPU:\n");
        printf("   Kernel execution: %.3f ms\n", gpu_complex_time);
        printf("   Speedup vs CPU complex: %.2fx\n", cpu_complex_time / gpu_complex_time);
        printf("   GPU advantage for complex ops: Clear winner!\n");
        
        // ====================================
        // PART 3: Analysis
        // ====================================
        printf("\n📊 PERFORMANCE ANALYSIS\n");
        printf("---------------------\n");
        
        // Calculate arithmetic intensity
        double flops_simple = n;  // n additions
        double flops_complex = n * 10;  // Estimate for complex computation
        double bytes_transferred = 3.0 * n * sizeof(float);
        
        printf("\nArithmetic Intensity:\n");
        printf("Simple:  %.3f FLOP/byte\n", flops_simple / bytes_transferred);
        printf("Complex: %.3f FLOP/byte\n", flops_complex / bytes_transferred);
        printf("\nConclusion: Simple addition is MEMORY BOUND\n");
        printf("           Complex computation is COMPUTE BOUND\n");
        
        // Memory bandwidth analysis
        double peak_bandwidth = 112.0;  // GB/s for RTX 2050
        double achieved_bandwidth = (bytes_transferred / 1e9) / (gpu_naive_time / 1000.0);
        
        printf("\nMemory Bandwidth:\n");
        printf("Peak theoretical: %.1f GB/s\n", peak_bandwidth);
        printf("Achieved:         %.1f GB/s (%.1f%% of peak)\n", 
               achieved_bandwidth, 100.0 * achieved_bandwidth / peak_bandwidth);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c_cpu);
        free(h_c_gpu);
    }
    
    // ====================================
    // PART 4: When Does GPU Win?
    // ====================================
    printf("\n\n🎯 KEY INSIGHTS: When Does GPU Win?\n");
    printf("===================================\n\n");
    
    printf("GPU WINS when:\n");
    printf("✅ Large data (millions of elements)\n");
    printf("✅ Complex computations (high arithmetic intensity)\n");
    printf("✅ Parallel algorithms (no dependencies)\n");
    printf("✅ Data reuse (process same data multiple times)\n\n");
    
    printf("GPU LOSES when:\n");
    printf("❌ Small data (overhead > benefit)\n");
    printf("❌ Simple operations (memory bound)\n");
    printf("❌ Sequential algorithms\n");
    printf("❌ Constant CPU↔GPU transfers\n\n");
    
    printf("OPTIMIZATION STRATEGIES:\n");
    printf("1. Minimize memory transfers\n");
    printf("2. Maximize arithmetic intensity\n");
    printf("3. Ensure coalesced memory access\n");
    printf("4. Use appropriate block size\n");
    printf("5. Consider unified memory for complex access patterns\n\n");
    
    // ====================================
    // PART 5: Real-World Considerations
    // ====================================
    printf("🏭 PRODUCTION CONSIDERATIONS\n");
    printf("===========================\n\n");
    
    printf("Error Handling:\n");
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("✅ No CUDA errors detected\n");
    }
    
    printf("\nMemory Management:\n");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU memory: %.1f MB free / %.1f MB total\n", 
           free_mem / (1024.0 * 1024.0), 
           total_mem / (1024.0 * 1024.0));
    
    printf("\nYour Journey:\n");
    printf("Week 1 ✅ - You can now build GPU applications!\n");
    printf("Next: Week 2 - Advanced optimizations\n");
    printf("      Week 3 - Complex algorithms\n");
    printf("      Week 4 - Production techniques\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * PERFORMANCE EXPERIMENTS:
 * 1. Vary Block Size:
 *    Test with 32, 64, 128, 256, 512, 1024 threads per block
 *    Plot performance vs block size. What's optimal?
 *
 * 2. Memory vs Compute Bound:
 *    Implement these operations and measure speedup:
 *    - c[i] = a[i] + b[i] (memory bound)
 *    - c[i] = sqrt(a[i]) + sqrt(b[i]) (balanced)
 *    - c[i] = sin(a[i]) * cos(b[i]) + tan(a[i]*b[i]) (compute bound)
 *
 * 3. Transfer Overhead:
 *    Measure time for:
 *    - Allocation only
 *    - Transfer only
 *    - Compute only
 *    Create a pie chart of where time goes
 *
 * BUILD COMPLETE APPLICATIONS:
 * 4. Vector Dot Product:
 *    result = Σ(a[i] * b[i])
 *    This requires reduction - preview of Week 3!
 *
 * 5. SAXPY Operation:
 *    y[i] = a * x[i] + y[i]
 *    where 'a' is a scalar. Common in linear algebra.
 *
 * 6. Distance Calculation:
 *    Given points (x1[i], y1[i]) and (x2[i], y2[i])
 *    Calculate distances[i] = sqrt((x2-x1)² + (y2-y1)²)
 *
 * OPTIMIZATION CHALLENGES:
 * 7. Fused Operations:
 *    Instead of:
 *    - Kernel 1: c[i] = a[i] + b[i]
 *    - Kernel 2: d[i] = c[i] * 2
 *    Fuse into one kernel. Measure improvement.
 *
 * 8. Pinned Memory:
 *    Use cudaMallocHost() for pinned memory.
 *    How much does transfer speed improve?
 *
 * 9. Streams (Preview):
 *    Split work into 4 chunks.
 *    Overlap transfer and compute.
 *    (This is Week 4 material!)
 *
 * ANALYSIS EXERCISES:
 * 10. Roofline Model:
 *     Plot your kernels on a roofline model:
 *     - X-axis: Arithmetic intensity (FLOP/byte)
 *     - Y-axis: Performance (GFLOP/s)
 *     - Identify memory vs compute bound
 *
 * 11. Weak vs Strong Scaling:
 *     Strong: Fix problem size, vary GPU resources
 *     Weak: Scale problem size with resources
 *     Which scaling does vector addition exhibit?
 *
 * 12. Energy Efficiency:
 *     Estimate energy usage:
 *     - CPU: ~100W for X seconds
 *     - GPU: ~50W for Y seconds
 *     Which is more energy efficient?
 *
 * DEBUGGING CHALLENGES:
 * 13. Wrong Results:
 *     Kernel gives wrong answers for n=1000 but not n=1024.
 *     Why? (Hint: think about block/grid calculation)
 *
 * 14. Performance Mystery:
 *     Same kernel is 2x slower with n=1000001 vs n=1000000.
 *     Why? (Hint: memory alignment)
 *
 * THINK DEEPER:
 * 15. Algorithm Choice:
 *     For adding 1 billion numbers, would you:
 *     a) Use vector addition to sum pairs recursively?
 *     b) Use a reduction algorithm?
 *     c) Something else?
 *     Design the approach.
 *
 * 16. Production Pipeline:
 *     Design a complete image processing pipeline:
 *     - Load image from disk
 *     - Apply 5 filters in sequence
 *     - Save result
 *     How do you minimize transfers?
 *
 * FINAL PROJECT IDEAS:
 * 17. Benchmark Suite:
 *     Create a tool that automatically:
 *     - Tests different array sizes
 *     - Tests different operations
 *     - Generates performance report
 *     - Recommends CPU vs GPU
 *
 * 18. Vector Math Library:
 *     Implement:
 *     - Addition, subtraction, multiplication
 *     - Dot product, cross product
 *     - Magnitude, normalization
 *     - All optimized for GPU
 *
 * MENTAL MODEL:
 * You've completed the journey from CPU to GPU:
 * 
 * CPU: One worker doing tasks sequentially
 *      Good for: complex logic, small data
 * 
 * GPU: 2048 workers doing simple tasks in parallel
 *      Good for: simple operations, massive data
 * 
 * The Art: Knowing when to use which tool!
 * 
 * Congratulations! You now understand GPU computing! 🎉
 */