/*
 * Lesson 9: Atomic Operations & Synchronization
 * Beyond __syncthreads() - Building Thread-Safe Algorithms
 *
 * This lesson teaches you how to handle race conditions and build
 * correct parallel algorithms when threads need to update shared data.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <chrono>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Atomics?
// =====================================================

/*
 * THE PROBLEM:
 * 
 * Imagine 1000 people trying to update the same bank account simultaneously.
 * Without proper synchronization, you get race conditions:
 * 
 * Thread 1: Read balance ($100)     Thread 2: Read balance ($100)
 * Thread 1: Add $10 ($110)          Thread 2: Add $20 ($120)  
 * Thread 1: Write $110              Thread 2: Write $120
 * 
 * Result: $120 (lost $10!)
 * 
 * ATOMIC OPERATIONS guarantee that read-modify-write happens as one
 * indivisible operation. No other thread can interfere.
 * 
 * Real-world analogy: It's like having a lock on the bank vault.
 * Only one person can access it at a time.
 */

// Simple timer for measurements
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
public:
    Timer() : start(Clock::now()) {}
    float elapsed() {
        return std::chrono::duration<float, std::milli>(
            Clock::now() - start).count();
    }
};

// =====================================================
// PART 2: DEMONSTRATE THE RACE CONDITION
// =====================================================

// Naive increment - INCORRECT due to race conditions
__global__ void incrementNaive(int *counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // This is NOT atomic!
        // Read-modify-write can be interrupted
        *counter = *counter + 1;
    }
}

// Atomic increment - CORRECT
__global__ void incrementAtomic(int *counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // atomicAdd guarantees thread-safe increment
        atomicAdd(counter, 1);
    }
}

// Demonstrate various atomic operations
__global__ void demonstrateAtomics(int *results) {
    int tid = threadIdx.x;
    
    // Initialize shared memory
    __shared__ int shared_counter;
    __shared__ int shared_max;
    __shared__ int shared_min;
    __shared__ unsigned int shared_bits;
    
    if (tid == 0) {
        shared_counter = 0;
        shared_max = 0;
        shared_min = 1000000;
        shared_bits = 0;
    }
    __syncthreads();
    
    // Each thread contributes
    int my_value = tid * tid;  // Some computation
    
    // 1. Atomic Add
    atomicAdd(&shared_counter, my_value);
    
    // 2. Atomic Max
    atomicMax(&shared_max, my_value);
    
    // 3. Atomic Min  
    atomicMin(&shared_min, my_value);
    
    // 4. Atomic OR (set bits)
    atomicOr(&shared_bits, 1 << (tid % 32));
    
    __syncthreads();
    
    // Thread 0 reports results
    if (tid == 0) {
        results[0] = shared_counter;
        results[1] = shared_max;
        results[2] = shared_min;
        results[3] = shared_bits;
    }
}

// =====================================================
// PART 3: HISTOGRAM - CLASSIC ATOMIC USE CASE
// =====================================================

// CPU histogram for comparison
void histogramCPU(unsigned char *data, int *hist, int n, int nbins) {
    for (int i = 0; i < nbins; i++) {
        hist[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        int bin = data[i] % nbins;
        hist[bin]++;
    }
}

// GPU histogram with atomic operations
__global__ void histogramGPUAtomic(unsigned char *data, int *hist, int n, int nbins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int bin = data[tid] % nbins;
        atomicAdd(&hist[bin], 1);
    }
}

// Optimized histogram with shared memory
__global__ void histogramGPUShared(unsigned char *data, int *hist, int n, int nbins) {
    extern __shared__ int shared_hist[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    
    // Initialize shared histogram
    for (int i = lid; i < nbins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Accumulate in shared memory (less contention)
    if (tid < n) {
        int bin = data[tid] % nbins;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Write shared results to global
    for (int i = lid; i < nbins; i += blockDim.x) {
        atomicAdd(&hist[i], shared_hist[i]);
    }
}

// =====================================================
// PART 4: CUSTOM ATOMIC WITH CAS
// =====================================================

// Atomic float add using Compare-And-Swap (CAS)
// Note: atomicAdd supports float natively since compute capability 2.0
// This shows how to build custom atomics
__device__ float atomicAddCustom(float *address, float val) {
    unsigned int *address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, 
                       __float_as_uint(__uint_as_float(assumed) + val));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

// Example: Atomic moving average
__global__ void atomicMovingAverage(float *sum, int *count, float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        atomicAdd(sum, data[tid]);
        atomicAdd(count, 1);
    }
    
    // Note: The average is sum/count, computed on host
    // Computing it here would create race conditions!
}

// =====================================================
// PART 5: LOCK-FREE DATA STRUCTURES
// =====================================================

// Simple lock-free stack using CAS
struct Node {
    int data;
    int next;  // Index to next node (-1 for null)
};

__global__ void lockFreeStackPush(Node *nodes, int *head, int *node_counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Allocate a node
        int my_node = atomicAdd(node_counter, 1);
        nodes[my_node].data = tid;
        
        // Push onto stack
        int old_head;
        do {
            old_head = *head;
            nodes[my_node].next = old_head;
        } while (atomicCAS(head, old_head, my_node) != old_head);
    }
}

// =====================================================
// PART 6: WARP VOTE FUNCTIONS
// =====================================================

__global__ void demonstrateVoting() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Simulate some condition
    bool my_vote = (lane_id % 2 == 0);  // Even lanes vote yes
    
    // All threads in warp participate
    unsigned int mask = __activemask();
    
    // Check if any thread in warp votes yes
    bool any_yes = __any_sync(mask, my_vote);
    
    // Check if all threads in warp vote yes  
    bool all_yes = __all_sync(mask, my_vote);
    
    // Count votes
    int yes_votes = __popc(__ballot_sync(mask, my_vote));
    
    if (lane_id == 0) {
        printf("Warp %d: any_yes=%d, all_yes=%d, yes_votes=%d\n",
               warp_id, any_yes, all_yes, yes_votes);
    }
}

// =====================================================
// PART 7: ADVANCED - MEMORY ORDERING
// =====================================================

// Demonstrate memory fence
__global__ void producerConsumer(int *data, int *flag) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        // Producer
        data[0] = 42;  // Write data
        __threadfence();  // Ensure write is visible
        atomicExch(flag, 1);  // Signal ready
    } else if (tid == 1) {
        // Consumer
        while (atomicAdd(flag, 0) == 0);  // Spin wait
        __threadfence();  // Ensure we see latest data
        int value = data[0];  // Read data
        printf("Consumer read: %d\n", value);
    }
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE TESTING
// =====================================================

int main() {
    printf("==================================================\n");
    printf("ATOMIC OPERATIONS & SYNCHRONIZATION\n");
    printf("==================================================\n\n");
    
    // Test 1: Demonstrate race condition
    printf("PART 1: Race Condition Demo\n");
    printf("--------------------------\n");
    
    const int N = 1000000;
    int *d_counter;
    int h_counter;
    
    cudaMalloc(&d_counter, sizeof(int));
    
    // Naive (incorrect) increment
    cudaMemset(d_counter, 0, sizeof(int));
    incrementNaive<<<(N + 255) / 256, 256>>>(d_counter, N);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Naive increment result: %d (expected %d) - RACE CONDITION!\n", h_counter, N);
    
    // Atomic (correct) increment
    cudaMemset(d_counter, 0, sizeof(int));
    incrementAtomic<<<(N + 255) / 256, 256>>>(d_counter, N);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Atomic increment result: %d (expected %d) - CORRECT!\n\n", h_counter, N);
    
    // Test 2: Various atomic operations
    printf("PART 2: Atomic Operations Demo\n");
    printf("-----------------------------\n");
    
    int *d_results;
    int h_results[4];
    cudaMalloc(&d_results, 4 * sizeof(int));
    
    demonstrateAtomics<<<1, 32>>>(d_results);
    cudaMemcpy(h_results, d_results, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum of tid² for 32 threads: %d\n", h_results[0]);
    printf("Max of tid²: %d\n", h_results[1]);
    printf("Min of tid²: %d\n", h_results[2]);
    printf("OR of bits: 0x%08X\n\n", h_results[3]);
    
    // Test 3: Histogram
    printf("PART 3: Histogram Performance\n");
    printf("----------------------------\n");
    
    const int DATA_SIZE = 10000000;
    const int NBINS = 256;
    
    unsigned char *h_data = new unsigned char[DATA_SIZE];
    int *h_hist_cpu = new int[NBINS];
    int *h_hist_gpu = new int[NBINS];
    
    // Generate random data
    for (int i = 0; i < DATA_SIZE; i++) {
        h_data[i] = rand() % 256;
    }
    
    unsigned char *d_data;
    int *d_hist;
    cudaMalloc(&d_data, DATA_SIZE);
    cudaMalloc(&d_hist, NBINS * sizeof(int));
    cudaMemcpy(d_data, h_data, DATA_SIZE, cudaMemcpyHostToDevice);
    
    // CPU histogram
    Timer cpu_timer;
    histogramCPU(h_data, h_hist_cpu, DATA_SIZE, NBINS);
    float cpu_time = cpu_timer.elapsed();
    printf("CPU time: %.2f ms\n", cpu_time);
    
    // GPU atomic histogram
    cudaMemset(d_hist, 0, NBINS * sizeof(int));
    Timer gpu_timer1;
    histogramGPUAtomic<<<(DATA_SIZE + 255) / 256, 256>>>(d_data, d_hist, DATA_SIZE, NBINS);
    cudaDeviceSynchronize();
    float gpu_time1 = gpu_timer1.elapsed();
    cudaMemcpy(h_hist_gpu, d_hist, NBINS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU atomic time: %.2f ms (speedup: %.2fx)\n", gpu_time1, cpu_time / gpu_time1);
    
    // GPU shared memory histogram
    cudaMemset(d_hist, 0, NBINS * sizeof(int));
    Timer gpu_timer2;
    int blocks = (DATA_SIZE + 255) / 256;
    histogramGPUShared<<<blocks, 256, NBINS * sizeof(int)>>>(d_data, d_hist, DATA_SIZE, NBINS);
    cudaDeviceSynchronize();
    float gpu_time2 = gpu_timer2.elapsed();
    printf("GPU shared memory time: %.2f ms (speedup: %.2fx)\n", gpu_time2, cpu_time / gpu_time2);
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < NBINS; i++) {
        if (h_hist_cpu[i] != h_hist_gpu[i]) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Test 4: Warp voting
    printf("PART 4: Warp Vote Functions\n");
    printf("--------------------------\n");
    demonstrateVoting<<<1, 64>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 5: Producer-Consumer
    printf("PART 5: Producer-Consumer Pattern\n");
    printf("--------------------------------\n");
    int *d_data, *d_flag;
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    
    producerConsumer<<<1, 32>>>(d_data, d_flag);
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_counter);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaFree(d_flag);
    delete[] h_data;
    delete[] h_hist_cpu;
    delete[] h_hist_gpu;
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Race conditions are real - always use atomics for shared data\n");
    printf("2. Atomics have overhead - minimize contention\n");
    printf("3. Shared memory atomics are faster than global\n");
    printf("4. CAS enables building custom atomic operations\n");
    printf("5. Warp vote functions enable efficient consensus\n");
    printf("6. Memory fences ensure ordering for complex patterns\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why do naive increments lose updates? Draw the timeline.
 * 2. Calculate the theoretical max contention for 256 threads
 * 3. When are atomics necessary vs __syncthreads()?
 * 4. What's the performance impact of atomic contention?
 * 5. How do warp vote functions differ from atomics?
 *
 * === Coding ===
 * 6. Implement atomic float max using CAS
 * 7. Build a parallel unique counter (count distinct values)
 * 8. Create a lock-free queue using atomics
 * 9. Implement parallel stream compaction
 * 10. Build a simple spinlock using atomicCAS
 *
 * === Optimization ===
 * 11. Reduce atomic contention with privatization
 * 12. Implement histogram with multiple arrays (less contention)
 * 13. Use warp-level primitives to reduce atomics
 * 14. Compare performance: atomics vs reduction
 * 15. Profile atomic operations with NSight
 *
 * === Advanced ===
 * 16. Implement Peterson's algorithm for 2 threads
 * 17. Build a work-stealing queue
 * 18. Create atomic operations for complex numbers
 * 19. Implement parallel radix sort with atomics
 * 20. Build a GPU memory allocator
 *
 * === Research ===
 * 21. Study relaxed memory ordering models
 * 22. Implement lock-free B-tree operations
 * 23. Create wait-free algorithms
 * 24. Build Software Transactional Memory (STM)
 * 25. Design new atomic primitives for your use case
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Atomics as Locks"
 *    - Each atomic operation briefly locks the memory location
 *    - Other threads wait their turn
 *    - More contention = more waiting = slower
 *
 * 2. "Bank Teller Analogy"
 *    - Multiple customers (threads) updating accounts (memory)
 *    - Need exclusive access to prevent errors
 *    - Can have multiple tellers (reduce contention)
 *
 * 3. "Thundering Herd"
 *    - All threads hitting same memory location
 *    - Creates serialization bottleneck
 *    - Solution: Spread the work (privatization)
 *
 * 4. Hardware Reality:
 *    - Atomics go through L2 cache (not L1)
 *    - Memory controller serializes conflicting atomics
 *    - Shared memory atomics stay within SM (faster)
 */
