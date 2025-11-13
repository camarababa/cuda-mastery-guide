/**
 * Lesson 3: From Theory to Practice - Processing Real Data in Parallel
 * =====================================================================
 *
 * The Problem We're Solving:
 * -------------------------
 * You have 1 million numbers to process. On CPU, you process them one by one.
 * On GPU, you can process thousands simultaneously. But HOW do we coordinate
 * 2048 threads to work on 1 million elements without chaos?
 *
 * What We'll Build:
 * ----------------
 * 1. Understand the thread-to-data mapping pattern
 * 2. Build array processing from first principles
 * 3. Master bounds checking (critical for correctness)
 * 4. See memory allocation strategies
 * 5. Measure actual speedup vs CPU
 * 6. Debug common mistakes before they happen
 *
 * Real-World Application:
 * -----------------------
 * This pattern is used EVERYWHERE:
 * - Image filters (each thread → one pixel)
 * - Audio processing (each thread → one sample)
 * - Scientific simulations (each thread → one particle)
 * - Machine learning (each thread → one neuron)
 *
 * Compile: nvcc -o lesson03 lesson03-array-operation.cu
 * Run: ./lesson03
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

/**
 * FIRST PRINCIPLES: The Fundamental Pattern
 * -----------------------------------------
 * Question: How do 1000 workers process 1 million items?
 * Answer: Each worker takes items 1000 apart.
 * 
 * Worker 0: items 0, 1000, 2000, 3000, ...
 * Worker 1: items 1, 1001, 2001, 3001, ...
 * 
 * But GPUs use a simpler pattern:
 * Thread 0 → Element 0
 * Thread 1 → Element 1
 * ...
 * Thread N → Element N
 * 
 * If we have more elements than threads, we launch multiple "waves"
 */

/**
 * Let's build up the concept step by step
 */

// Step 1: The most basic pattern - one thread per element
__global__ void demonstrateMapping(int *debug_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread writes its ID to prove the mapping works
    if (idx < n) {
        debug_array[idx] = idx;
        
        // First few threads report what they're doing
        if (idx < 5) {
            printf("Thread %d (Block %d, Local %d) → Element %d\n",
                   idx, blockIdx.x, threadIdx.x, idx);
        }
    }
}

// Step 2: Actual computation - multiply by 2
__global__ void multiplyByTwo(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // THE CRITICAL PATTERN: Always check bounds!
    if (idx < n) {
        int old_value = arr[idx];
        arr[idx] = old_value * 2;
        
        // Debug: First thread of each block reports
        if (threadIdx.x == 0 && blockIdx.x < 3) {
            printf("Block %d processing elements %d-%d\n",
                   blockIdx.x, 
                   idx, 
                   idx + blockDim.x - 1);
        }
    }
}

// Step 3: Why bounds checking matters
__global__ void demonstrateBoundsImportance(int *arr, int n, int *violation_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // WITHOUT bounds check (DON'T DO THIS IN REAL CODE!)
    // This shows what happens if we forget
    if (idx >= n && idx < n + 32) {  // Threads that SHOULDN'T access array
        atomicAdd(violation_count, 1);  // Count violations
        // In real code, this would corrupt memory!
    }
    
    // WITH bounds check (ALWAYS DO THIS!)
    if (idx < n) {
        arr[idx] = arr[idx] * 3;  // Different operation to distinguish
    }
}

// Step 4: Different patterns for different algorithms
__global__ void processWithStride(int *arr, int n) {
    // Alternative pattern: Each thread handles multiple elements
    // Thread 0: elements 0, gridDim.x*blockDim.x, 2*gridDim.x*blockDim.x, ...
    // This is useful when n >> number of threads
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Total threads in grid
    
    for (int idx = tid; idx < n; idx += stride) {
        arr[idx] = arr[idx] + 10;
        
        // Show the pattern (first thread only)
        if (tid == 0 && idx < stride * 3) {
            printf("Thread 0 processing element %d\n", idx);
        }
    }
}

// Helper: Initialize array with pattern
void initializeArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i;  // Simple pattern: 0, 1, 2, 3, ...
    }
}

// Helper: Verify results
bool verifyResults(int *arr, int n, int expected_multiplier) {
    for (int i = 0; i < n; i++) {
        if (arr[i] != i * expected_multiplier) {
            printf("ERROR at index %d: expected %d, got %d\n",
                   i, i * expected_multiplier, arr[i]);
            return false;
        }
    }
    return true;
}

// Helper: CPU version for comparison
void cpuMultiply(int *arr, int n, int multiplier) {
    for (int i = 0; i < n; i++) {
        arr[i] = arr[i] * multiplier;
    }
}

// Helper: Measure time in milliseconds
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("========================================================\n");
    printf("LESSON 3: From Theory to Practice - Array Processing\n");
    printf("========================================================\n\n");
    
    // PART 1: Understanding Thread-to-Data Mapping
    printf("PART 1: How Do Threads Map to Data?\n");
    printf("-----------------------------------\n");
    
    int demo_size = 32;
    int *demo_array;
    cudaMallocManaged(&demo_array, demo_size * sizeof(int));
    cudaMemset(demo_array, 0, demo_size * sizeof(int));
    
    printf("Launching 2 blocks × 16 threads for %d elements:\n\n", demo_size);
    demonstrateMapping<<<2, 16>>>(demo_array, demo_size);
    cudaDeviceSynchronize();
    
    printf("\nArray after mapping: [");
    for (int i = 0; i < demo_size && i < 10; i++) {
        printf("%d", demo_array[i]);
        if (i < 9) printf(", ");
    }
    printf(", ...]\n");
    printf("✓ Each thread wrote its global ID to its element!\n\n");
    
    // PART 2: Real Computation
    printf("PART 2: Actual Array Processing\n");
    printf("-------------------------------\n");
    
    int n = 100;  // Modest size for clear output
    int *arr;
    cudaMallocManaged(&arr, n * sizeof(int));
    
    // Initialize
    initializeArray(arr, n);
    printf("Processing %d elements: multiply each by 2\n\n", n);
    
    // Calculate grid
    int threadsPerBlock = 32;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Grid config: %d blocks × %d threads = %d total threads\n", 
           blocks, threadsPerBlock, blocks * threadsPerBlock);
    printf("Elements: %d (wasting %d threads)\n\n", 
           n, blocks * threadsPerBlock - n);
    
    multiplyByTwo<<<blocks, threadsPerBlock>>>(arr, n);
    cudaDeviceSynchronize();
    
    // Verify
    if (verifyResults(arr, n, 2)) {
        printf("\n✓ SUCCESS! All elements correctly doubled.\n\n");
    }
    
    // PART 3: The Importance of Bounds Checking
    printf("PART 3: Why Bounds Checking is CRITICAL\n");
    printf("---------------------------------------\n");
    
    int *violation_counter;
    cudaMallocManaged(&violation_counter, sizeof(int));
    *violation_counter = 0;
    
    // Reset array
    initializeArray(arr, n);
    
    // Launch with intentionally too many threads
    int extra_blocks = 2;  // Extra blocks beyond what we need
    printf("Launching %d blocks (need %d) to show bounds checking...\n",
           blocks + extra_blocks, blocks);
    
    demonstrateBoundsImportance<<<blocks + extra_blocks, threadsPerBlock>>>
        (arr, n, violation_counter);
    cudaDeviceSynchronize();
    
    printf("Out-of-bounds access attempts caught: %d\n", *violation_counter);
    printf("These threads correctly skipped processing!\n\n");
    
    // PART 4: Performance Comparison
    printf("PART 4: GPU vs CPU Performance\n");
    printf("------------------------------\n");
    
    // Test with larger array
    int perf_n = 10000000;  // 10 million elements
    int *cpu_arr, *gpu_arr;
    
    // Allocate arrays
    cpu_arr = (int*)malloc(perf_n * sizeof(int));
    cudaMallocManaged(&gpu_arr, perf_n * sizeof(int));
    
    // Initialize both
    for (int i = 0; i < perf_n; i++) {
        cpu_arr[i] = i;
        gpu_arr[i] = i;
    }
    
    printf("Processing %d elements (%.1f MB)...\n\n", 
           perf_n, perf_n * sizeof(int) / (1024.0 * 1024.0));
    
    // CPU timing
    double cpu_start = getTime();
    cpuMultiply(cpu_arr, perf_n, 2);
    double cpu_time = getTime() - cpu_start;
    printf("CPU Time: %.3f ms\n", cpu_time);
    
    // GPU timing
    blocks = (perf_n + threadsPerBlock - 1) / threadsPerBlock;
    
    double gpu_start = getTime();
    multiplyByTwo<<<blocks, threadsPerBlock>>>(gpu_arr, perf_n);
    cudaDeviceSynchronize();
    double gpu_time = getTime() - gpu_start;
    printf("GPU Time: %.3f ms (including memory management)\n", gpu_time);
    
    printf("\nSpeedup: %.2fx\n\n", cpu_time / gpu_time);
    
    if (gpu_time > cpu_time) {
        printf("NOTE: GPU is slower for this simple operation because:\n");
        printf("- Memory transfer overhead\n");
        printf("- Simple computation (not enough work per thread)\n");
        printf("- We'll see better speedups with complex operations\n\n");
    }
    
    // PART 5: Alternative Access Patterns
    printf("PART 5: Grid-Stride Loops (Advanced Pattern)\n");
    printf("--------------------------------------------\n");
    printf("What if we have 1 billion elements but only 1 million threads?\n");
    printf("Answer: Each thread processes multiple elements!\n\n");
    
    int stride_n = 100;
    int *stride_arr;
    cudaMallocManaged(&stride_arr, stride_n * sizeof(int));
    initializeArray(stride_arr, stride_n);
    
    // Launch fewer threads than elements
    processWithStride<<<2, 16>>>(stride_arr, stride_n);
    cudaDeviceSynchronize();
    
    printf("\nThis pattern scales to ANY size without changing grid!\n\n");
    
    // PART 6: Memory Allocation Strategies
    printf("PART 6: Memory Allocation Options\n");
    printf("---------------------------------\n");
    printf("1. cudaMallocManaged (Unified Memory):\n");
    printf("   + Easy to use (CPU & GPU access)\n");
    printf("   - Slower (automatic transfers)\n");
    printf("   - Use for: Prototyping, small data\n\n");
    
    printf("2. cudaMalloc + cudaMemcpy (Explicit):\n");
    printf("   + Full control & best performance\n");
    printf("   - More complex code\n");
    printf("   - Use for: Production, large data\n\n");
    
    // Key Insights
    printf("KEY INSIGHTS FROM THIS LESSON\n");
    printf("=============================\n");
    printf("1. The pattern: idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. ALWAYS check bounds: if (idx < n)\n");
    printf("3. Each thread → one element (usually)\n");
    printf("4. Grid size = (n + blockSize - 1) / blockSize\n");
    printf("5. Simple ops may be memory-bound (not compute-bound)\n");
    printf("6. This pattern works for ANY array size\n\n");
    
    // Cleanup
    cudaFree(demo_array);
    cudaFree(arr);
    cudaFree(violation_counter);
    cudaFree(gpu_arr);
    cudaFree(stride_arr);
    free(cpu_arr);
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Thread Mapping:
 *    With 3 blocks of 32 threads processing 80 elements:
 *    - Which thread processes element 47?
 *    - Which elements does block 1 handle?
 *    - How many threads are wasted?
 *
 * 2. Grid Calculation:
 *    Calculate blocks needed for:
 *    - 1000 elements, 128 threads/block
 *    - 1000 elements, 256 threads/block
 *    - Why might you choose one over the other?
 *
 * CODING EXERCISES:
 * 3. Create kernels that:
 *    - Add 10 to each element
 *    - Square each element
 *    - Set even indices to 0, odd to 1
 *
 * 4. Multi-Operation:
 *    Create a kernel that does: arr[i] = (arr[i] * 3) + 7
 *    Launch it and verify results.
 *
 * 5. Reverse Array:
 *    Create a kernel where thread i writes to position n-1-i
 *    (Hint: You'll need two arrays or careful synchronization)
 *
 * EXPERIMENT EXERCISES:
 * 6. Performance Testing:
 *    Time these operations for n=10,000,000:
 *    - Multiply by constant
 *    - Square each element  
 *    - Compute sin(arr[i])
 *    Which has best GPU speedup? Why?
 *
 * 7. Block Size Comparison:
 *    Process same array with:
 *    - 32 threads/block
 *    - 128 threads/block
 *    - 512 threads/block
 *    - 1024 threads/block
 *    Plot the performance. What's optimal?
 *
 * 8. Memory Access Patterns:
 *    Compare performance of:
 *    - arr[idx] = arr[idx] * 2 (same location read/write)
 *    - out[idx] = in[idx] * 2 (different arrays)
 *    Why might they differ?
 *
 * BUILD THIS:
 * 9. Vector Operations:
 *    Implement these vector operations:
 *    - Vector addition: c[i] = a[i] + b[i]
 *    - Dot product: sum of a[i] * b[i]
 *    - Euclidean norm: sqrt(sum of a[i]²)
 *
 * 10. Image Filter (Simplified):
 *     Treat array as 1D image. Each thread:
 *     - Reads its pixel and neighbors
 *     - Computes average (blur)
 *     - Writes result
 *
 * 11. Parallel Histogram:
 *     Count occurrences of values 0-255 in array
 *     (Challenging: Requires atomic operations)
 *
 * DEBUGGING CHALLENGES:
 * 12. Find the Bug:
 *     ```cuda
 *     __global__ void buggy(int *arr, int n) {
 *         int idx = threadIdx.x;  // BUG! Missing blockIdx
 *         arr[idx] = idx * 2;     // No bounds check!
 *     }
 *     ```
 *     Fix both issues.
 *
 * 13. Race Condition:
 *     Try this:
 *     ```cuda
 *     __global__ void race(int *counter) {
 *         (*counter)++;  // All threads increment same location!
 *     }
 *     ```
 *     What happens? How would you fix it?
 *
 * ADVANCED THINKING:
 * 14. When does GPU win over CPU?
 *     - Simple operations (add, multiply)?
 *     - Complex operations (sqrt, sin, exp)?
 *     - Large arrays vs small arrays?
 *     Create experiments to find the crossover points.
 *
 * 15. Memory Bandwidth:
 *     Calculate theoretical bandwidth:
 *     - 10M integers × 4 bytes × 2 (read+write) / time
 *     Compare to your GPU's spec (112 GB/s for RTX 2050)
 *     What percentage are you achieving?
 *
 * MENTAL MODEL:
 * Think of array processing like a massive assembly line:
 * - Each worker (thread) stands at one position
 * - The array moves past on a conveyor belt  
 * - Each worker performs their operation on their item
 * - All workers work simultaneously
 * - Bounds check = worker checks if item exists before touching it
 * 
 * This mental model helps you understand:
 * - Why we need one thread per element
 * - Why bounds checking matters
 * - How parallelism gives us speedup
 */