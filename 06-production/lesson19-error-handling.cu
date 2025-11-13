/*
 * Lesson 19: Error Handling & Debugging
 * Building Robust GPU Applications
 *
 * "It's not working" → "Here's exactly what's wrong and how to fix it"
 * This lesson transforms you from hoping code works to knowing it works.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <execinfo.h>  // For stack traces
#include <signal.h>
#include <unistd.h>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Errors Are Different on GPU
// =====================================================

/*
 * CPU vs GPU ERROR HANDLING:
 * 
 * CPU:
 * - Exceptions thrown immediately
 * - Stack traces available
 * - Debugger can stop at error
 * 
 * GPU:
 * - Async execution (errors appear later)
 * - No exceptions in kernels
 * - Silent failures common
 * - Race conditions hard to debug
 * 
 * This lesson teaches you to build bulletproof GPU code.
 */

// =====================================================
// PART 2: PROPER ERROR CHECKING
// =====================================================

// Basic error checking macro (DON'T use in production)
#define CUDA_CHECK_BASIC(call) \
    if ((call) != cudaSuccess) { \
        printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    }

// Better error checking with message
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Production-grade error checking with context
#define CUDA_CHECK_CONTEXT(call, context) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at %s:%d\n", \
                    context, __FILE__, __LINE__); \
            fprintf(stderr, "Error: %s (%d)\n", \
                    cudaGetErrorString(err), err); \
            fprintf(stderr, "Description: %s\n", \
                    cudaGetErrorName(err)); \
            exit(1); \
        } \
    } while(0)

// Check last error (for kernel launches)
#define CUDA_CHECK_LAST_ERROR(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error: %s\n", msg); \
            fprintf(stderr, "Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// =====================================================
// PART 3: DEBUGGING TECHNIQUES
// =====================================================

// Helper class for automatic error checking
class CudaErrorChecker {
private:
    const char* file;
    int line;
    const char* function;
    
public:
    CudaErrorChecker(const char* f, int l, const char* func) 
        : file(f), line(l), function(func) {}
    
    ~CudaErrorChecker() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "\n=== CUDA Error Detected ===\n");
            fprintf(stderr, "Function: %s\n", function);
            fprintf(stderr, "Location: %s:%d\n", file, line);
            fprintf(stderr, "Error: %s (%d)\n", 
                    cudaGetErrorString(err), err);
            fprintf(stderr, "========================\n");
            abort();
        }
    }
};

#define CUDA_ERROR_SCOPE() \
    CudaErrorChecker _error_checker(__FILE__, __LINE__, __FUNCTION__)

// Debug kernel - intentionally buggy
__global__ void buggyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bug 1: No bounds checking
    data[idx] = idx;  // Will crash if idx >= n
    
    // Bug 2: Shared memory bank conflicts
    __shared__ float shared[32];
    shared[threadIdx.x % 32] = data[idx];  // All threads hit same bank
    
    // Bug 3: Race condition
    if (threadIdx.x < 32) {
        data[0] += 1.0f;  // Multiple threads write to same location
    }
    
    // Bug 4: Warp divergence
    if (idx % 2 == 0) {
        for (int i = 0; i < idx; i++) {  // Very divergent loop
            data[idx] *= 2.0f;
        }
    }
}

// Fixed version
__global__ void fixedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Fix 1: Bounds checking
    if (idx < n) {
        data[idx] = idx;
    }
    
    // Fix 2: Avoid bank conflicts
    __shared__ float shared[33];  // Padding
    if (idx < n) {
        shared[threadIdx.x] = data[idx];
    }
    __syncthreads();
    
    // Fix 3: Atomic operation for race condition
    if (threadIdx.x < 32 && idx < n) {
        atomicAdd(&data[0], 1.0f);
    }
    
    // Fix 4: Minimize divergence
    if (idx < n && idx % 2 == 0) {
        int iterations = min(idx, 10);  // Cap iterations
        for (int i = 0; i < iterations; i++) {
            data[idx] *= 2.0f;
        }
    }
}

// =====================================================
// PART 4: MEMORY ERROR DETECTION
// =====================================================

// Common memory errors
void demonstrateMemoryErrors() {
    printf("\n=== Memory Error Examples ===\n");
    
    // Error 1: Allocation failure
    {
        size_t huge_size = 100ULL * 1024 * 1024 * 1024;  // 100 GB
        void *ptr;
        cudaError_t err = cudaMalloc(&ptr, huge_size);
        if (err != cudaSuccess) {
            printf("Expected allocation failure: %s\n", 
                   cudaGetErrorString(err));
        }
    }
    
    // Error 2: Invalid pointer (commented out to avoid crash)
    /*
    {
        float *bad_ptr = (float*)0x12345678;
        cudaMemset(bad_ptr, 0, 100);  // Will fail
    }
    */
    
    // Error 3: Out of bounds access detection
    {
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, 100 * sizeof(float)));
        
        // Launch kernel that accesses out of bounds
        // Run with cuda-memcheck to detect
        buggyKernel<<<1, 200>>>(d_data, 100);  // 200 threads, 100 elements
        
        // Synchronize to catch error
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel error detected: %s\n", cudaGetErrorString(err));
        }
        
        cudaFree(d_data);
    }
}

// =====================================================
// PART 5: KERNEL DEBUGGING WITH PRINTF
// =====================================================

__global__ void debugWithPrintf(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread 0 prints header
    if (idx == 0) {
        printf("=== Kernel Debug Info ===\n");
        printf("Grid: %d blocks\n", gridDim.x);
        printf("Block: %d threads\n", blockDim.x);
    }
    
    // All threads print their info (limit output)
    if (idx < 5) {
        printf("Thread %d: Block %d, Thread %d, Processing element %d\n",
               idx, blockIdx.x, threadIdx.x, idx);
    }
    
    // Conditional debugging
    if (idx < n) {
        int old_value = data[idx];
        data[idx] = idx * idx;
        
        // Print suspicious values
        if (data[idx] > 1000000) {
            printf("WARNING: Large value at idx %d: %d\n", 
                   idx, data[idx]);
        }
    }
    
    // Thread 0 prints footer
    if (idx == 0) {
        printf("===================\n");
    }
}

// =====================================================
// PART 6: ASSERT IN KERNELS
// =====================================================

__global__ void kernelWithAsserts(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Assert preconditions
    assert(n > 0);
    assert(data != nullptr);
    
    if (idx < n) {
        float old_val = data[idx];
        
        // Assert valid input range
        assert(old_val >= 0.0f && old_val <= 1.0f);
        
        // Do computation
        data[idx] = sqrtf(old_val);
        
        // Assert postcondition
        assert(!isnan(data[idx]));
        assert(data[idx] >= 0.0f);
    }
}

// =====================================================
// PART 7: CUDA-GDB DEBUGGING HELPERS
// =====================================================

// Kernel to demonstrate cuda-gdb
__global__ void debuggableKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Set breakpoint here in cuda-gdb:
    // (cuda-gdb) break debuggableKernel
    // (cuda-gdb) run
    // (cuda-gdb) info cuda threads
    // (cuda-gdb) cuda thread 100
    // (cuda-gdb) print idx
    
    if (idx < n) {
        // Complex computation for debugging
        int temp = data[idx];
        temp = temp * 2;
        temp = temp + blockIdx.x;
        temp = temp - threadIdx.x;
        data[idx] = temp;
        
        // Can examine variables at each step
    }
}

// =====================================================
// PART 8: ERROR RECOVERY STRATEGIES
// =====================================================

class CudaContext {
private:
    bool initialized;
    cudaStream_t stream;
    size_t allocated_memory;
    std::vector<void*> allocations;
    
public:
    CudaContext() : initialized(false), allocated_memory(0) {}
    
    bool initialize() {
        // Reset device
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to reset device: %s\n", 
                    cudaGetErrorString(err));
            return false;
        }
        
        // Create stream
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to create stream: %s\n", 
                    cudaGetErrorString(err));
            return false;
        }
        
        initialized = true;
        return true;
    }
    
    void* allocate(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        
        if (err == cudaSuccess) {
            allocations.push_back(ptr);
            allocated_memory += size;
            return ptr;
        } else if (err == cudaErrorMemoryAllocation) {
            // Try to free some memory and retry
            fprintf(stderr, "Out of memory, attempting cleanup...\n");
            cleanup();
            
            err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess) {
                allocations.push_back(ptr);
                allocated_memory += size;
                return ptr;
            }
        }
        
        fprintf(stderr, "Allocation failed: %s\n", 
                cudaGetErrorString(err));
        return nullptr;
    }
    
    void cleanup() {
        for (void* ptr : allocations) {
            cudaFree(ptr);
        }
        allocations.clear();
        allocated_memory = 0;
    }
    
    ~CudaContext() {
        cleanup();
        if (initialized) {
            cudaStreamDestroy(stream);
            cudaDeviceReset();
        }
    }
};

// =====================================================
// PART 9: COMPREHENSIVE ERROR HANDLING
// =====================================================

// Signal handler for debugging
void signalHandler(int sig) {
    fprintf(stderr, "\n=== SIGNAL %d CAUGHT ===\n", sig);
    
    // Print stack trace
    void *array[20];
    size_t size = backtrace(array, 20);
    fprintf(stderr, "Stack trace:\n");
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    // Print CUDA error if any
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Last CUDA error: %s\n", 
                cudaGetErrorString(err));
    }
    
    exit(1);
}

// Comprehensive test function
void runComprehensiveTest() {
    printf("\n=== Comprehensive Error Handling Test ===\n");
    
    // Install signal handler
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);
    
    // Test 1: Proper error checking
    {
        CUDA_ERROR_SCOPE();
        
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, 1000 * sizeof(float)));
        
        // Launch kernel
        debugWithPrintf<<<10, 100>>>((int*)d_data, 1000);
        CUDA_CHECK_LAST_ERROR("debugWithPrintf");
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_data));
        printf("✓ Basic error checking passed\n");
    }
    
    // Test 2: Assert handling
    {
        float *d_data, *h_data;
        int n = 100;
        
        h_data = new float[n];
        for (int i = 0; i < n; i++) {
            h_data[i] = (float)i / n;  // Values in [0, 1]
        }
        
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        kernelWithAsserts<<<1, n>>>(d_data, n);
        cudaError_t err = cudaDeviceSynchronize();
        
        if (err == cudaSuccess) {
            printf("✓ Assert test passed\n");
        } else {
            printf("✗ Assert triggered: %s\n", cudaGetErrorString(err));
        }
        
        CUDA_CHECK(cudaFree(d_data));
        delete[] h_data;
    }
    
    // Test 3: Memory error detection
    demonstrateMemoryErrors();
    
    // Test 4: Error recovery
    {
        CudaContext ctx;
        if (ctx.initialize()) {
            // Try to allocate increasingly large amounts
            for (size_t size = 1024; size < 10ULL * 1024 * 1024 * 1024; 
                 size *= 2) {
                void* ptr = ctx.allocate(size);
                if (ptr) {
                    printf("✓ Allocated %zu MB\n", size / 1024 / 1024);
                } else {
                    printf("✗ Failed at %zu MB\n", size / 1024 / 1024);
                    break;
                }
            }
        }
    }
}

// =====================================================
// PART 10: MAIN - DEBUGGING WORKFLOW
// =====================================================

int main(int argc, char **argv) {
    printf("==================================================\n");
    printf("ERROR HANDLING & DEBUGGING\n");
    printf("==================================================\n");
    
    // Device query
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Debug support: %s\n", 
           prop.kernelExecTimeoutEnabled ? "Yes (timeout enabled)" : "Yes");
    printf("\n");
    
    // Enable debugging features
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024));
    
    // Run tests
    runComprehensiveTest();
    
    printf("\n==================================================\n");
    printf("DEBUGGING WORKFLOW\n");
    printf("==================================================\n");
    printf("1. ALWAYS check CUDA API calls\n");
    printf("2. Use cuda-memcheck for memory errors:\n");
    printf("   cuda-memcheck ./your_program\n");
    printf("3. Use cuda-gdb for interactive debugging:\n");
    printf("   cuda-gdb ./your_program\n");
    printf("4. Enable asserts in debug builds:\n");
    printf("   nvcc -G -g your_code.cu\n");
    printf("5. Use compute-sanitizer for race conditions:\n");
    printf("   compute-sanitizer --tool racecheck ./your_program\n");
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. GPU errors are often silent - always check\n");
    printf("2. Synchronize before checking kernel errors\n");
    printf("3. Use tools - don't debug blind\n");
    printf("4. Build error recovery into production code\n");
    printf("5. Test error paths, not just success paths\n");
    printf("6. Printf debugging works but has limitations\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why are GPU errors asynchronous?
 * 2. What's the cost of error checking?
 * 3. When to use assert vs error codes?
 * 4. How does cuda-memcheck work?
 * 5. Why is debugging harder on GPU?
 *
 * === Implementation ===
 * 6. Build custom error handling framework
 * 7. Create memory leak detector
 * 8. Implement kernel timeout handler
 * 9. Build error reporting system
 * 10. Create debug visualization tools
 *
 * === Debugging Practice ===
 * 11. Debug race condition in reduction
 * 12. Find memory corruption in matrix multiply
 * 13. Fix deadlock in synchronization
 * 14. Debug numerical instability
 * 15. Find performance regression
 *
 * === Advanced ===
 * 16. Implement checkpoint/restart
 * 17. Build fault-tolerant algorithms
 * 18. Create production monitoring
 * 19. Implement error injection testing
 * 20. Build automated debugging tools
 *
 * === Production ===
 * 21. Design error handling policy
 * 22. Implement telemetry collection
 * 23. Create debugging container
 * 24. Build CI/CD with GPU tests
 * 25. Implement graceful degradation
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Trust but Verify"
 *    - Never assume operations succeed
 *    - Check every API call
 *    - Validate kernel results
 *
 * 2. "Defense in Depth"
 *    - API error checking (first line)
 *    - Kernel assertions (second line)
 *    - Result validation (third line)
 *
 * 3. "Fail Fast, Fail Loud"
 *    - Detect errors immediately
 *    - Report clearly
 *    - Don't hide failures
 *
 * 4. Debugging Hierarchy:
 *    - Printf (quick and dirty)
 *    - cuda-gdb (interactive)
 *    - NSight (visual)
 *    - Custom tools (specialized)
 */
