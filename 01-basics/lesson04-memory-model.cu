/**
 * Lesson 4: Mastering GPU Memory - The Foundation of Performance
 * ==============================================================
 *
 * The Problem We're Solving:
 * -------------------------
 * Your CPU and GPU have SEPARATE memory spaces. It's like having two
 * computers that need to share data. How do we efficiently move millions
 * of numbers between them? One wrong choice and your GPU code becomes
 * SLOWER than CPU!
 *
 * What We'll Build:
 * ----------------
 * 1. Understand WHY CPU and GPU memory are separate
 * 2. Master explicit memory management (full control)
 * 3. Learn unified memory (automatic but slower)
 * 4. Measure transfer overhead (the hidden cost)
 * 5. Build a mental model of data movement
 * 6. Learn optimization strategies used in production
 *
 * Real-World Impact:
 * -----------------
 * Memory management is THE #1 factor in GPU performance:
 * - Wrong: 10x slower than CPU
 * - Right: 100x faster than CPU
 * The difference? Understanding this lesson.
 *
 * Compile: nvcc -o lesson04 lesson04-memory-model.cu
 * Run: ./lesson04
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>

/**
 * FIRST PRINCIPLES: Why Are CPU and GPU Memory Separate?
 * ------------------------------------------------------
 * 
 * my Computer Architecture:
 * 
 * [CPU] ←PCIe→ [GPU]
 *   ↓           ↓
 * [RAM]      [VRAM]
 * (16GB)     (4GB)
 * 
 * CPU Memory (RAM):
 * - Connected to CPU via memory controller
 * - ~50 GB/s bandwidth to CPU
 * - Can't be directly accessed by GPU
 * 
 * GPU Memory (VRAM):
 * - Connected to GPU via wide bus
 * - ~112 GB/s bandwidth to GPU (your RTX 2050)
 * - Can't be directly accessed by CPU
 * 
 * PCIe Bridge:
 * - Connects CPU and GPU
 * - ~8-16 GB/s transfer speed
 * - This is the BOTTLENECK!
 */

// Simple kernel for our experiments
__global__ void addConstant(float *arr, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] += value;
    }
}

// More compute-intensive kernel to show when GPU wins
__global__ void computeIntensive(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = arr[idx];
        // Simulate complex computation
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        arr[idx] = val;
    }
}

// Helper: Get time in milliseconds
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Helper: Initialize array with pattern
void initializeArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)i / n;  // Values 0.0 to 1.0
    }
}

/**
 * METHOD 1: Explicit Memory Management
 * ------------------------------------
 * YOU control exactly when and what to transfer.
 * More code, but maximum performance.
 */
void demonstrateExplicitMemory() {
    printf("\nMETHOD 1: Explicit Memory Management\n");
    printf("====================================\n\n");
    
    int n = 1000000;  // 1 million elements
    size_t size = n * sizeof(float);
    printf("Working with %d elements (%.2f MB)\n\n", n, size / (1024.0 * 1024.0));
    
    // Step-by-step timing
    double start, end;
    
    // Step 1: Allocate CPU memory
    printf("Step 1: CPU Memory Allocation\n");
    start = getTime();
    float *h_array = (float*)malloc(size);
    end = getTime();
    printf("  Allocated %.2f MB on CPU in %.3f ms\n", 
           size / (1024.0 * 1024.0), end - start);
    printf("  Address: %p (in system RAM)\n\n", h_array);
    
    // Step 2: Initialize on CPU
    printf("Step 2: Initialize Data on CPU\n");
    start = getTime();
    initializeArray(h_array, n);
    end = getTime();
    printf("  Initialized in %.3f ms\n\n", end - start);
    
    // Step 3: Allocate GPU memory
    printf("Step 3: GPU Memory Allocation\n");
    float *d_array;
    start = getTime();
    cudaMalloc(&d_array, size);
    end = getTime();
    printf("  Allocated %.2f MB on GPU in %.3f ms\n", 
           size / (1024.0 * 1024.0), end - start);
    printf("  Address: %p (in GPU VRAM)\n\n", d_array);
    
    // Step 4: Transfer CPU → GPU
    printf("Step 4: Transfer CPU → GPU\n");
    start = getTime();
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    end = getTime();
    double h2d_time = end - start;
    printf("  Transferred %.2f MB in %.3f ms\n", 
           size / (1024.0 * 1024.0), h2d_time);
    printf("  Bandwidth: %.2f GB/s\n\n", 
           (size / (1024.0 * 1024.0 * 1024.0)) / (h2d_time / 1000.0));
    
    // Step 5: Launch kernel
    printf("Step 5: GPU Computation\n");
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    start = getTime();
    addConstant<<<blocks, threadsPerBlock>>>(d_array, 1.0f, n);
    cudaDeviceSynchronize();
    end = getTime();
    double kernel_time = end - start;
    printf("  Kernel executed in %.3f ms\n", kernel_time);
    printf("  Processing rate: %.2f million elements/sec\n\n", 
           n / (kernel_time * 1000.0));
    
    // Step 6: Transfer GPU → CPU
    printf("Step 6: Transfer GPU → CPU\n");
    start = getTime();
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    end = getTime();
    double d2h_time = end - start;
    printf("  Transferred %.2f MB in %.3f ms\n", 
           size / (1024.0 * 1024.0), d2h_time);
    printf("  Bandwidth: %.2f GB/s\n\n", 
           (size / (1024.0 * 1024.0 * 1024.0)) / (d2h_time / 1000.0));
    
    // Step 7: Analyze overhead
    printf("Performance Analysis:\n");
    printf("  Memory transfers: %.3f ms (%.1f%%)\n", 
           h2d_time + d2h_time, 
           100.0 * (h2d_time + d2h_time) / (h2d_time + d2h_time + kernel_time));
    printf("  Computation:      %.3f ms (%.1f%%)\n", 
           kernel_time,
           100.0 * kernel_time / (h2d_time + d2h_time + kernel_time));
    printf("  Total:            %.3f ms\n\n", 
           h2d_time + d2h_time + kernel_time);
    
    if ((h2d_time + d2h_time) > kernel_time) {
        printf("⚠️  WARNING: Transfer time > Compute time!\n");
        printf("   This kernel is MEMORY BOUND.\n\n");
    }
    
    // Cleanup
    cudaFree(d_array);
    free(h_array);
}

/**
 * METHOD 2: Unified Memory
 * ------------------------
 * CUDA manages transfers automatically.
 * Less code, but less control.
 */
void demonstrateUnifiedMemory() {
    printf("\nMETHOD 2: Unified Memory (Managed)\n");
    printf("==================================\n\n");
    
    int n = 1000000;
    size_t size = n * sizeof(float);
    double start, end;
    
    // Step 1: Allocate unified memory
    printf("Step 1: Unified Memory Allocation\n");
    float *managed_array;
    start = getTime();
    cudaMallocManaged(&managed_array, size);
    end = getTime();
    printf("  Allocated %.2f MB in %.3f ms\n", 
           size / (1024.0 * 1024.0), end - start);
    printf("  Address: %p (accessible from BOTH CPU and GPU!)\n\n", 
           managed_array);
    
    // Step 2: Initialize on CPU
    printf("Step 2: Initialize on CPU\n");
    start = getTime();
    initializeArray(managed_array, n);
    end = getTime();
    printf("  Initialized in %.3f ms\n");
    printf("  (Data is currently on CPU side)\n\n");
    
    // Step 3: Launch kernel (automatic migration)
    printf("Step 3: GPU Computation\n");
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    start = getTime();
    addConstant<<<blocks, threadsPerBlock>>>(managed_array, 1.0f, n);
    cudaDeviceSynchronize();
    end = getTime();
    printf("  Kernel + auto-migration in %.3f ms\n", end - start);
    printf("  (CUDA automatically moved data CPU→GPU)\n\n");
    
    // Step 4: Access from CPU (another migration)
    printf("Step 4: Access from CPU\n");
    start = getTime();
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        sum += managed_array[i];  // This triggers GPU→CPU migration
    }
    end = getTime();
    printf("  Accessed first 10 elements in %.3f ms\n", end - start);
    printf("  (CUDA automatically moved data GPU→CPU)\n");
    printf("  Sum of first 10: %.3f\n\n", sum);
    
    // Cleanup
    cudaFree(managed_array);
    
    printf("Unified Memory Summary:\n");
    printf("  ✓ Simpler code (no explicit copies)\n");
    printf("  ✓ Automatic migration\n");
    printf("  ✗ Less control over timing\n");
    printf("  ✗ Can have hidden performance costs\n\n");
}

/**
 * COMPARISON: Which Method Wins?
 */
void compareMemoryMethods() {
    printf("\nPERFORMANCE COMPARISON\n");
    printf("=====================\n\n");
    
    // Test different scenarios
    int sizes[] = {10000, 100000, 1000000, 10000000};
    const char* labels[] = {"10K", "100K", "1M", "10M"};
    
    printf("Testing simple operation (add constant):\n");
    printf("Size    | Explicit (ms) | Unified (ms) | Winner\n");
    printf("--------|---------------|--------------|--------\n");
    
    for (int i = 0; i < 4; i++) {
        int n = sizes[i];
        size_t size = n * sizeof(float);
        
        // Test explicit
        float *h_arr = (float*)malloc(size);
        float *d_arr;
        cudaMalloc(&d_arr, size);
        initializeArray(h_arr, n);
        
        double explicit_start = getTime();
        cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
        addConstant<<<(n+255)/256, 256>>>(d_arr, 1.0f, n);
        cudaDeviceSynchronize();
        cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
        double explicit_time = getTime() - explicit_start;
        
        cudaFree(d_arr);
        free(h_arr);
        
        // Test unified
        float *u_arr;
        cudaMallocManaged(&u_arr, size);
        initializeArray(u_arr, n);
        
        double unified_start = getTime();
        addConstant<<<(n+255)/256, 256>>>(u_arr, 1.0f, n);
        cudaDeviceSynchronize();
        float dummy = u_arr[0]; // Force migration back
        double unified_time = getTime() - unified_start;
        
        cudaFree(u_arr);
        
        // Print results
        printf("%-8s| %13.3f | %12.3f | %s\n",
               labels[i], explicit_time, unified_time,
               explicit_time < unified_time ? "Explicit" : "Unified");
    }
    
    printf("\nKey Insights:\n");
    printf("- For small data: Unified memory often wins (less overhead)\n");
    printf("- For large data: Explicit usually wins (better control)\n");
    printf("- For repeated access: Explicit wins (no repeated migrations)\n\n");
}

/**
 * ADVANCED PATTERNS
 */
void demonstrateAdvancedPatterns() {
    printf("\nADVANCED MEMORY PATTERNS\n");
    printf("========================\n\n");
    
    // Pattern 1: Pinned Memory
    printf("Pattern 1: Pinned (Page-Locked) Memory\n");
    printf("-------------------------------------\n");
    
    int n = 10000000;  // 10M elements
    size_t size = n * sizeof(float);
    
    // Regular memory
    float *regular = (float*)malloc(size);
    float *d_regular;
    cudaMalloc(&d_regular, size);
    initializeArray(regular, n);
    
    double start = getTime();
    cudaMemcpy(d_regular, regular, size, cudaMemcpyHostToDevice);
    double regular_time = getTime() - start;
    
    // Pinned memory
    float *pinned;
    cudaMallocHost(&pinned, size);  // Pinned allocation
    float *d_pinned;
    cudaMalloc(&d_pinned, size);
    initializeArray(pinned, n);
    
    start = getTime();
    cudaMemcpy(d_pinned, pinned, size, cudaMemcpyHostToDevice);
    double pinned_time = getTime() - start;
    
    printf("Regular memory transfer: %.3f ms (%.2f GB/s)\n",
           regular_time, (size/1e9) / (regular_time/1000));
    printf("Pinned memory transfer:  %.3f ms (%.2f GB/s)\n",
           pinned_time, (size/1e9) / (pinned_time/1000));
    printf("Speedup: %.2fx\n\n", regular_time / pinned_time);
    
    // Cleanup
    free(regular);
    cudaFreeHost(pinned);
    cudaFree(d_regular);
    cudaFree(d_pinned);
    
    // Pattern 2: Memory Access Patterns
    printf("Pattern 2: Minimizing Transfers\n");
    printf("-------------------------------\n");
    printf("Bad:  CPU→GPU, compute, GPU→CPU (repeat)\n");
    printf("Good: CPU→GPU, compute many times, GPU→CPU\n\n");
    
    // Pattern 3: Concurrent Operations (preview)
    printf("Pattern 3: Overlap Transfer & Compute (Advanced)\n");
    printf("-----------------------------------------------\n");
    printf("Stream 0: [Transfer A] → [Compute A] → [Transfer A back]\n");
    printf("Stream 1:           [Transfer B] → [Compute B] → [Transfer B back]\n");
    printf("Result: Hide transfer latency! (We'll learn in Week 4)\n\n");
}

/**
 * MENTAL MODEL: The Restaurant Analogy
 */
void explainMentalModel() {
    printf("\nMENTAL MODEL: The Restaurant Kitchen\n");
    printf("===================================\n\n");
    
    printf("Think of GPU computing like a restaurant:\n\n");
    
    printf("CPU = Main Kitchen\n");
    printf("- Where ingredients are stored (RAM)\n");
    printf("- Where final plating happens\n");
    printf("- Good for complex, sequential recipes\n\n");
    
    printf("GPU = Specialized Prep Station\n");
    printf("- Has its own workspace (VRAM)\n");
    printf("- Many cooks working in parallel\n");
    printf("- Great for repetitive tasks (chopping 1000 onions)\n\n");
    
    printf("The Challenge:\n");
    printf("- Moving ingredients between kitchens takes time\n");
    printf("- The delivery corridor (PCIe) is narrow\n");
    printf("- Smart chefs minimize trips!\n\n");
    
    printf("Strategies:\n");
    printf("1. Explicit: You plan every ingredient transfer\n");
    printf("2. Unified: Ingredients magically appear (but slower)\n");
    printf("3. Best: Send ingredients once, do all prep, send back\n\n");
}

int main() {
    printf("========================================================\n");
    printf("LESSON 4: Mastering GPU Memory - The Foundation\n");
    printf("========================================================\n\n");
    
    // Check device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Your GPU: %s\n", prop.name);
    printf("- Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("- Memory bandwidth: ~112 GB/s\n");
    printf("- PCIe bandwidth: ~8-16 GB/s (the bottleneck!)\n\n");
    
    // Run demonstrations
    demonstrateExplicitMemory();
    demonstrateUnifiedMemory();
    compareMemoryMethods();
    demonstrateAdvancedPatterns();
    explainMentalModel();
    
    // Key takeaways
    printf("\nKEY TAKEAWAYS\n");
    printf("=============\n");
    printf("1. CPU and GPU have SEPARATE memory spaces\n");
    printf("2. Transfers are EXPENSIVE (PCIe bottleneck)\n");
    printf("3. Minimize transfers for best performance\n");
    printf("4. Explicit = control, Unified = convenience\n");
    printf("5. Pinned memory can double transfer speed\n");
    printf("6. Memory bandwidth often limits performance\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Memory Locations:
 *    Draw a diagram showing where each pointer points:
 *    - malloc() pointer: ?
 *    - cudaMalloc() pointer: ?
 *    - cudaMallocManaged() pointer: ?
 *
 * 2. Transfer Calculations:
 *    Your PCIe is 8 GB/s. How long to transfer:
 *    - 100 MB?
 *    - 1 GB?
 *    - 4 GB (entire GPU memory)?
 *
 * 3. Break-Even Analysis:
 *    If transfer takes 10ms and CPU compute takes 100ms,
 *    how fast must GPU compute be to break even?
 *
 * CODING EXERCISES:
 * 4. Implement Both Ways:
 *    Create vector addition using:
 *    - Explicit memory
 *    - Unified memory
 *    Time both. Which wins?
 *
 * 5. Multi-Kernel Pipeline:
 *    Run 3 kernels in sequence on same data:
 *    - add 1
 *    - multiply by 2  
 *    - square
 *    Compare: 3 round trips vs 1 round trip
 *
 * 6. Memory Bandwidth Test:
 *    Create a kernel that just copies data:
 *    out[i] = in[i]
 *    Measure GB/s. Compare to theoretical max.
 *
 * EXPERIMENT EXERCISES:
 * 7. Find the Crossover Point:
 *    At what array size does GPU become faster than CPU?
 *    Test: 1K, 10K, 100K, 1M, 10M elements
 *    Plot the results.
 *
 * 8. Unified Memory Page Faults:
 *    Access unified memory alternately from CPU and GPU.
 *    Time each access. See the migration cost?
 *
 * 9. Allocation Performance:
 *    Time allocating many small arrays vs one big array.
 *    Why is there a difference?
 *
 * BUILD THIS:
 * 10. Memory Pool Manager:
 *     Instead of allocating each time, pre-allocate a big
 *     chunk and sub-allocate from it.
 *
 * 11. Zero-Copy Experiment:
 *     Use cudaHostAlloc with cudaHostAllocMapped.
 *     GPU can access CPU memory directly (slowly!)
 *     When might this be useful?
 *
 * 12. Bandwidth Monitor:
 *     Create a tool that measures and reports:
 *     - H2D bandwidth
 *     - D2H bandwidth
 *     - Kernel memory bandwidth
 *
 * DEBUGGING CHALLENGES:
 * 13. Find the Bug:
 *     ```cuda
 *     float *h_data = (float*)malloc(n * sizeof(float));
 *     float *d_data;
 *     cudaMalloc(&d_data, n); // BUG: forgot sizeof(float)!
 *     ```
 *
 * 14. Memory Leak Hunt:
 *     Write a program that allocates in a loop but
 *     "forgets" to free. Watch GPU memory with nvidia-smi.
 *
 * ADVANCED THINKING:
 * 15. When is Unified Memory Better?
 *     Think of scenarios where UM wins:
 *     - Sparse access patterns?
 *     - Complex data structures?
 *     - Unknown access patterns?
 *
 * 16. Memory Hierarchy Design:
 *     You're building a image filter. Design the memory flow:
 *     - Where to store input image?
 *     - Where to store filter weights?
 *     - Where to store output?
 *     - How to minimize transfers?
 *
 * MENTAL MODELS:
 * Model 1: The Two-Computer Model
 * - CPU = Computer 1 with RAM
 * - GPU = Computer 2 with VRAM
 * - PCIe = Network cable between them
 * - Goal: Minimize network traffic
 * 
 * Model 2: The Factory Model
 * - CPU memory = Warehouse
 * - GPU memory = Factory floor
 * - PCIe = Delivery trucks
 * - Trucks are slow, keep materials on factory floor!
 * 
 * Model 3: The Cache Hierarchy
 * - Registers (fastest, smallest)
 * - Shared Memory
 * - L1/L2 Cache
 * - Global Memory
 * - System RAM (through PCIe)
 * - Disk (slowest, largest)
 * 
 * Understanding these models helps you write fast code!
 */