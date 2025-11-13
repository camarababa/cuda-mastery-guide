/*
 * Lesson 11: Warp-Level Primitives
 * Programming at the Warp Level for Maximum Efficiency
 *
 * This lesson unlocks the power of warp-synchronous programming,
 * where 32 threads work together as a single unit.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =====================================================
// PART 1: FIRST PRINCIPLES - What is a Warp?
// =====================================================

/*
 * FUNDAMENTAL TRUTH:
 * GPUs don't execute threads individually. They execute WARPS.
 * 
 * A warp = 32 threads that execute in lockstep (SIMT)
 * 
 * Think of it like this:
 * - Orchestra: Each musician (thread) plays their part
 * - But they all follow the same conductor (instruction pointer)
 * - They play the same note (instruction) at the same time
 * 
 * This lesson teaches you to leverage this hardware reality
 * for massive performance gains.
 */

// Timer for performance measurements
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
// PART 2: WARP SHUFFLE - REGISTER-TO-REGISTER COMMUNICATION
// =====================================================

// Traditional approach: through shared memory
__global__ void sumPairsSharedMemory(float *data, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Sum pairs
    if (tid % 2 == 0 && tid + 1 < blockDim.x) {
        sdata[tid] += sdata[tid + 1];
    }
    __syncthreads();
    
    // Write back
    if (tid % 2 == 0 && idx < n) {
        data[idx] = sdata[tid];
    }
}

// Modern approach: warp shuffle (no shared memory!)
__global__ void sumPairsWarpShuffle(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;
    
    // Direct register-to-register communication!
    // Get value from lane+1 (next thread in warp)
    float neighbor = __shfl_down_sync(0xffffffff, val, 1);
    
    // Even lanes sum with their neighbor
    if ((threadIdx.x % 2 == 0) && (idx < n)) {
        data[idx] = val + neighbor;
    }
}

// Demonstrate all shuffle variants
__global__ void demonstrateShuffles() {
    int lane = threadIdx.x % 32;
    int value = lane;  // Each thread's initial value is its lane ID
    
    if (threadIdx.x < 32) {  // First warp only
        printf("Initial: Lane %2d has value %2d\n", lane, value);
        
        // 1. Shuffle down (shift right)
        int down = __shfl_down_sync(0xffffffff, value, 2);
        if (lane < 30) printf("Shuffle down by 2: Lane %2d got %2d from lane %2d\n", 
                             lane, down, lane + 2);
        
        // 2. Shuffle up (shift left)
        int up = __shfl_up_sync(0xffffffff, value, 2);
        if (lane >= 2) printf("Shuffle up by 2: Lane %2d got %2d from lane %2d\n", 
                             lane, up, lane - 2);
        
        // 3. Shuffle xor (butterfly pattern)
        int xor_val = __shfl_xor_sync(0xffffffff, value, 1);
        printf("Shuffle xor 1: Lane %2d swapped with lane %2d, got %2d\n", 
               lane, lane ^ 1, xor_val);
        
        // 4. Broadcast (all get same value)
        int broadcast = __shfl_sync(0xffffffff, value, 0);  // Broadcast from lane 0
        if (lane == 31) printf("Broadcast: All lanes got value %d from lane 0\n", broadcast);
    }
}

// =====================================================
// PART 3: WARP REDUCTION - THE ULTIMATE OPTIMIZATION
// =====================================================

// Warp reduction using shuffle
__inline__ __device__ float warpReduceSum(float val) {
    // Each iteration halves the active threads
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Full block reduction leveraging warp primitives
__global__ void blockReduceOptimized(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;  // Grid-stride loop
    
    // Load and add during load (2 elements per thread)
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    // Warp-level reduction (no shared memory needed!)
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    sum = warpReduceSum(sum);
    
    // Write warp sums to shared memory
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction of warp sums
    if (tid < 32) {
        sum = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
        sum = warpReduceSum(sum);
        
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// =====================================================
// PART 4: WARP VOTE FUNCTIONS - CONSENSUS BUILDING
// =====================================================

__global__ void demonstrateVoting(int *results) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Simulate different conditions
    bool condition1 = (lane < 16);           // Half true
    bool condition2 = (lane % 2 == 0);       // Even lanes
    bool condition3 = (lane == 0);           // Only first
    bool condition4 = true;                  // All true
    
    unsigned mask = 0xffffffff;  // All lanes participate
    
    // Store results for first warp
    if (warp_id == 0) {
        // Any: Is at least one thread true?
        if (lane == 0) {
            results[0] = __any_sync(mask, condition1);
            results[1] = __any_sync(mask, condition3);
        }
        
        // All: Are all threads true?
        if (lane == 1) {
            results[2] = __all_sync(mask, condition4);
            results[3] = __all_sync(mask, condition1);
        }
        
        // Ballot: Get bitmask of which threads are true
        unsigned ballot1 = __ballot_sync(mask, condition2);
        if (lane == 2) {
            results[4] = __popc(ballot1);  // Count set bits
        }
        
        // Match any: Find threads with same value
        unsigned match = __match_any_sync(mask, lane / 8);
        if (lane == 3) {
            results[5] = __popc(match);  // Threads with same value
        }
    }
}

// =====================================================
// PART 5: COOPERATIVE GROUPS - FLEXIBLE THREAD GROUPS
// =====================================================

__global__ void demonstrateCooperativeGroups() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int tid = block.thread_rank();
    int warp_id = tid / 32;
    int lane = warp.thread_rank();
    
    // Create subgroups
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(warp);
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(warp);
    
    // Synchronize at different levels
    if (warp_id == 0) {
        // Warp-level sum
        int value = lane;
        for (int i = warp.size() / 2; i > 0; i /= 2) {
            value += warp.shfl_down(value, i);
        }
        
        if (lane == 0) {
            printf("Warp %d sum: %d\n", warp_id, value);
        }
        
        // Tile4 reduction (groups of 4)
        int tile4_value = lane;
        for (int i = tile4.size() / 2; i > 0; i /= 2) {
            tile4_value += tile4.shfl_down(tile4_value, i);
        }
        
        if (tile4.thread_rank() == 0) {
            printf("Tile4 group %d sum: %d\n", tid / 4, tile4_value);
        }
    }
}

// =====================================================
// PART 6: ADVANCED WARP TECHNIQUES
// =====================================================

// Warp-wide prefix sum (scan)
__device__ int warpPrefixSum(int val) {
    int lane = threadIdx.x % 32;
    
    // Kogge-Stone parallel prefix sum
    for (int i = 1; i < 32; i *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) val += n;
    }
    
    return val;
}

// Warp histogram - no atomics needed!
__device__ void warpHistogram(int value, int *hist) {
    int lane = threadIdx.x % 32;
    
    // Each thread checks if others have same value
    unsigned mask = __match_any_sync(0xffffffff, value);
    
    // Leader thread (lowest lane) accumulates count
    int leader = __ffs(mask) - 1;  // Find first set bit
    if (lane == leader) {
        int count = __popc(mask);  // Count matching threads
        hist[value] += count;      // Single write, no atomics!
    }
}

// =====================================================
// PART 7: PERFORMANCE COMPARISON
// =====================================================

__global__ void reduceSumNaive(float *data, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Naive reduction with shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE TESTING
// =====================================================

int main() {
    printf("==================================================\n");
    printf("WARP-LEVEL PRIMITIVES\n");
    printf("==================================================\n\n");
    
    // Test 1: Demonstrate shuffle operations
    printf("PART 1: Shuffle Operations Demo\n");
    printf("------------------------------\n");
    demonstrateShuffles<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    // Test 2: Performance comparison - sum pairs
    printf("PART 2: Sum Pairs Performance\n");
    printf("----------------------------\n");
    
    const int N = 10000000;
    float *d_data;
    float *h_data = new float[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i % 100;
    }
    
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Shared memory version
    Timer timer1;
    sumPairsSharedMemory<<<blocks, threads, threads * sizeof(float)>>>(d_data, N);
    cudaDeviceSynchronize();
    float time1 = timer1.elapsed();
    printf("Shared memory version: %.2f ms\n", time1);
    
    // Reset data
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warp shuffle version
    Timer timer2;
    sumPairsWarpShuffle<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();
    float time2 = timer2.elapsed();
    printf("Warp shuffle version: %.2f ms\n", time2);
    printf("Speedup: %.2fx\n\n", time1 / time2);
    
    // Test 3: Block reduction performance
    printf("PART 3: Reduction Performance\n");
    printf("----------------------------\n");
    
    float *d_output;
    blocks = 1024;
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    // Naive reduction
    Timer timer3;
    reduceSumNaive<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_output, N);
    cudaDeviceSynchronize();
    float time3 = timer3.elapsed();
    printf("Naive reduction: %.2f ms\n", time3);
    
    // Warp-optimized reduction
    Timer timer4;
    blockReduceOptimized<<<blocks, threads, 32 * sizeof(float)>>>(d_data, d_output, N);
    cudaDeviceSynchronize();
    float time4 = timer4.elapsed();
    printf("Warp-optimized reduction: %.2f ms\n", time4);
    printf("Speedup: %.2fx\n\n", time3 / time4);
    
    // Test 4: Voting functions
    printf("PART 4: Warp Vote Functions\n");
    printf("--------------------------\n");
    
    int *d_results, h_results[6];
    cudaMalloc(&d_results, 6 * sizeof(int));
    demonstrateVoting<<<1, 64>>>(d_results);
    cudaMemcpy(h_results, d_results, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("__any_sync(lane < 16): %s\n", h_results[0] ? "true" : "false");
    printf("__any_sync(lane == 0): %s\n", h_results[1] ? "true" : "false");
    printf("__all_sync(true): %s\n", h_results[2] ? "true" : "false");
    printf("__all_sync(lane < 16): %s\n", h_results[3] ? "true" : "false");
    printf("Even lanes count: %d\n", h_results[4]);
    printf("Threads with same lane/8: %d\n\n", h_results[5]);
    
    // Test 5: Cooperative groups
    printf("PART 5: Cooperative Groups\n");
    printf("-------------------------\n");
    demonstrateCooperativeGroups<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // Performance analysis
    printf("\n==================================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("==================================================\n");
    
    size_t shared_mem_bandwidth = threads * sizeof(float) * blocks / (time1 / 1000.0f) / 1e9;
    size_t register_bandwidth = N * sizeof(float) / (time2 / 1000.0f) / 1e9;
    
    printf("Shared memory bandwidth: %.2f GB/s\n", shared_mem_bandwidth);
    printf("Register shuffle bandwidth: %.2f GB/s (no memory!)\n", register_bandwidth);
    printf("Reduction GFLOPS: %.2f\n", (N / 1e9) / (time4 / 1000.0f));
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_results);
    delete[] h_data;
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Warp shuffles are 10-100x faster than shared memory\n");
    printf("2. No __syncthreads() needed within a warp\n");
    printf("3. Warp primitives enable register-only algorithms\n");
    printf("4. Vote functions enable efficient predicate evaluation\n");
    printf("5. Modern GPUs are designed for warp-level programming\n");
    printf("6. Cooperative groups provide flexible synchronization\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why is warp size exactly 32? (Hint: hardware design)
 * 2. Calculate shuffle bandwidth vs shared memory bandwidth
 * 3. When does warp divergence occur? How to avoid it?
 * 4. Why don't warp primitives need __syncthreads()?
 * 5. How do masks work in shuffle operations?
 *
 * === Coding ===
 * 6. Implement warp-wide minimum using shuffles
 * 7. Create parallel prefix sum for a full block
 * 8. Build warp-level merge sort
 * 9. Implement broadcast within tile<8>
 * 10. Create warp-wide unique count
 *
 * === Optimization ===
 * 11. Optimize matrix transpose with shuffle
 * 12. Implement register-only 1D convolution
 * 13. Create multi-warp producer-consumer queue
 * 14. Build warp-aggregated atomics (reduce contention)
 * 15. Profile shuffle vs shared memory bandwidth
 *
 * === Advanced ===
 * 16. Implement butterfly FFT with shuffles
 * 17. Create warp-wide hash table
 * 18. Build GPU-wide barrier using voting
 * 19. Implement work stealing with warp primitives
 * 20. Create custom reduction for complex numbers
 *
 * === Research ===
 * 21. Study independent thread scheduling (Volta+)
 * 22. Implement nanosleep using voting
 * 23. Create persistent warp specialization
 * 24. Build warp-synchronous RNG
 * 25. Design new algorithms assuming warp=64
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Dance Troupe"
 *    - 32 dancers moving in perfect sync
 *    - Can pass items (shuffle) without stopping
 *    - Vote on next move (voting functions)
 *    - Split into subgroups (tiles) for complex routines
 *
 * 2. "Highway Lanes"
 *    - 32 lanes of traffic moving together
 *    - Cars can quickly exchange passengers (shuffle)
 *    - No traffic lights needed (no sync)
 *    - All lanes see same signs (broadcast)
 *
 * 3. Hardware Reality:
 *    - Warp = SIMD unit in hardware
 *    - Shuffle = crossbar network between registers
 *    - Vote = reduction tree in hardware
 *    - All happen in 1-2 cycles!
 */
