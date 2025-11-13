/*
 * Lesson 12: Parallel Scan (Prefix Sum)
 * Fundamental Parallel Algorithm with Broad Applications
 *
 * Scan is to parallel computing what loops are to sequential computing.
 * Master this, and you unlock a universe of parallel algorithms.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <chrono>

// =====================================================
// PART 1: FIRST PRINCIPLES - What is Scan?
// =====================================================

/*
 * SCAN (PREFIX SUM):
 * Given: [a₀, a₁, a₂, a₃, ...]
 * 
 * Inclusive Scan: [a₀, a₀+a₁, a₀+a₁+a₂, a₀+a₁+a₂+a₃, ...]
 * Exclusive Scan: [0, a₀, a₀+a₁, a₀+a₁+a₂, ...]
 * 
 * Sequential: O(N) - trivial
 * Parallel: O(log N) - mind-blowing!
 * 
 * WHY IT MATTERS:
 * Scan is a building block for:
 * - Stream compaction (remove nulls)
 * - Radix sort (fastest GPU sort)
 * - Tree operations
 * - Polynomial evaluation
 * - String comparison
 * - And much more...
 * 
 * It's the "secret sauce" of parallel algorithms!
 */

// Timer for measurements
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
// PART 2: SEQUENTIAL BASELINE
// =====================================================

void scanCPU(int *output, int *input, int n, bool inclusive = true) {
    if (n <= 0) return;
    
    if (inclusive) {
        output[0] = input[0];
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i];
        }
    } else {
        output[0] = 0;
        for (int i = 1; i < n; i++) {
            output[i] = output[i-1] + input[i-1];
        }
    }
}

// =====================================================
// PART 3: NAIVE PARALLEL SCAN (INEFFICIENT)
// =====================================================

__global__ void scanNaive(int *output, int *input, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Naive approach: O(N log N) work!
    // Each thread adds all previous elements
    int sum = 0;
    for (int i = 0; i <= tid; i++) {
        sum += sdata[i];
    }
    
    // Write result
    if (idx < n) {
        output[idx] = sum;
    }
}

// =====================================================
// PART 4: WORK-EFFICIENT PARALLEL SCAN
// =====================================================

/*
 * Blelloch Algorithm (1990)
 * Two phases:
 * 1. Up-sweep (reduce)
 * 2. Down-sweep (scan)
 * 
 * Total work: O(N) - optimal!
 * Depth: O(log N)
 */

__global__ void scanWorkEfficient(int *output, int *input, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // UP-SWEEP (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element
    if (tid == blockDim.x - 1) {
        sdata[tid] = 0;
    }
    __syncthreads();
    
    // DOWN-SWEEP phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }
    
    // Write results (exclusive scan)
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// =====================================================
// PART 5: BANK CONFLICT FREE VERSION
// =====================================================

#define CONFLICT_FREE_OFFSET(n) ((n) >> 5 + (n) >> 10)

__global__ void scanBankConflictFree(int *output, int *input, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load with padding to avoid bank conflicts
    int ai = tid;
    int bi = tid + blockDim.x / 2;
    int offset_a = CONFLICT_FREE_OFFSET(ai);
    int offset_b = CONFLICT_FREE_OFFSET(bi);
    
    sdata[ai + offset_a] = (idx < n) ? input[idx] : 0;
    sdata[bi + offset_b] = (idx + blockDim.x / 2 < n) ? input[idx + blockDim.x / 2] : 0;
    
    int offset = 1;
    
    // UP-SWEEP phase
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        int index = blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1);
        sdata[index] = 0;
    }
    
    // DOWN-SWEEP phase
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results
    if (idx < n) output[idx] = sdata[ai + offset_a];
    if (idx + blockDim.x / 2 < n) output[idx + blockDim.x / 2] = sdata[bi + offset_b];
}

// =====================================================
// PART 6: WARP-LEVEL SCAN
// =====================================================

__device__ __forceinline__ int warpScan(int val) {
    // Kogge-Stone algorithm using shuffle
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val;
}

__global__ void scanWarpLevel(int *output, int *input, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load value
    int val = (idx < n) ? input[idx] : 0;
    
    // Warp-level scan
    int warp_sum = warpScan(val);
    
    // Last lane of each warp writes total
    __shared__ int warp_totals[32];
    if (lane == 31) {
        warp_totals[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Scan warp totals (single warp)
    if (tid < 32) {
        int total = (tid < blockDim.x / 32) ? warp_totals[tid] : 0;
        total = warpScan(total);
        warp_totals[tid] = total;
    }
    __syncthreads();
    
    // Add scanned base to each warp's results
    if (warp_id > 0) {
        warp_sum += warp_totals[warp_id - 1];
    }
    
    // Write result
    if (idx < n) {
        output[idx] = warp_sum;
    }
}

// =====================================================
// PART 7: APPLICATIONS - STREAM COMPACTION
// =====================================================

// Mark valid elements
__global__ void markValid(int *flags, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Keep non-zero elements
        flags[idx] = (data[idx] != 0) ? 1 : 0;
    }
}

// Scatter valid elements using scan results
__global__ void scatter(int *output, int *input, int *addresses, int *flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) {
        output[addresses[idx] - 1] = input[idx];  // -1 for exclusive scan
    }
}

// =====================================================
// PART 8: LARGE ARRAY SCAN
// =====================================================

// Scan large arrays that don't fit in one block
void scanLarge(int *output, int *input, int n) {
    const int BLOCK_SIZE = 512;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate temporary arrays
    int *blockSums;
    cudaMalloc(&blockSums, numBlocks * sizeof(int));
    
    // Phase 1: Scan each block
    scanWorkEfficient<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(output, input, n);
    
    // Extract block sums
    // ... (implementation details)
    
    // Phase 2: Scan block sums
    if (numBlocks > 1) {
        scanLarge(blockSums, blockSums, numBlocks);
    }
    
    // Phase 3: Add block offsets
    // ... (implementation details)
    
    cudaFree(blockSums);
}

// =====================================================
// PART 9: MAIN - COMPREHENSIVE TESTING
// =====================================================

void printArray(const char *name, int *arr, int n, int limit = 20) {
    printf("%s: ", name);
    for (int i = 0; i < std::min(n, limit); i++) {
        printf("%d ", arr[i]);
    }
    if (n > limit) printf("...");
    printf("\n");
}

int main() {
    printf("==================================================\n");
    printf("PARALLEL SCAN (PREFIX SUM)\n");
    printf("==================================================\n\n");
    
    // Test with small array first
    const int SMALL_N = 16;
    int h_small[SMALL_N] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    int h_result[SMALL_N];
    int *d_input, *d_output;
    
    cudaMalloc(&d_input, SMALL_N * sizeof(int));
    cudaMalloc(&d_output, SMALL_N * sizeof(int));
    
    printf("PART 1: Correctness Test (N=%d)\n", SMALL_N);
    printf("--------------------------------\n");
    printArray("Input", h_small, SMALL_N);
    
    // CPU reference
    scanCPU(h_result, h_small, SMALL_N, false);  // exclusive
    printArray("CPU exclusive scan", h_result, SMALL_N);
    
    // GPU work-efficient
    cudaMemcpy(d_input, h_small, SMALL_N * sizeof(int), cudaMemcpyHostToDevice);
    scanWorkEfficient<<<1, SMALL_N, SMALL_N * sizeof(int)>>>(d_output, d_input, SMALL_N);
    cudaMemcpy(h_result, d_output, SMALL_N * sizeof(int), cudaMemcpyDeviceToHost);
    printArray("GPU work-efficient", h_result, SMALL_N);
    printf("\n");
    
    // Performance test with large array
    const int N = 10000000;
    int *h_input = new int[N];
    int *h_output_cpu = new int[N];
    int *h_output_gpu = new int[N];
    
    // Initialize with random data
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 10;
    }
    
    printf("PART 2: Performance Test (N=%d)\n", N);
    printf("--------------------------------\n");
    
    // CPU scan
    Timer cpu_timer;
    scanCPU(h_output_cpu, h_input, N, true);
    float cpu_time = cpu_timer.elapsed();
    printf("CPU inclusive scan: %.2f ms\n", cpu_time);
    
    // GPU setup
    int *d_large_input, *d_large_output;
    cudaMalloc(&d_large_input, N * sizeof(int));
    cudaMalloc(&d_large_output, N * sizeof(int));
    cudaMemcpy(d_large_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 512;
    int blocks = (N + threads - 1) / threads;
    
    // Naive GPU scan (for comparison)
    Timer gpu_timer1;
    // Note: Only run on small subset due to O(N²) complexity
    if (N <= 10000) {
        scanNaive<<<blocks, threads, threads * sizeof(int)>>>(d_large_output, d_large_input, N);
        cudaDeviceSynchronize();
        float gpu_time1 = gpu_timer1.elapsed();
        printf("GPU naive scan: %.2f ms (DON'T USE!)\n", gpu_time1);
    }
    
    // Work-efficient GPU scan
    Timer gpu_timer2;
    scanWorkEfficient<<<blocks, threads, threads * sizeof(int)>>>(d_large_output, d_large_input, N);
    cudaDeviceSynchronize();
    float gpu_time2 = gpu_timer2.elapsed();
    printf("GPU work-efficient: %.2f ms (speedup: %.2fx)\n", 
           gpu_time2, cpu_time / gpu_time2);
    
    // Warp-level scan
    Timer gpu_timer3;
    scanWarpLevel<<<blocks, threads>>>(d_large_output, d_large_input, N);
    cudaDeviceSynchronize();
    float gpu_time3 = gpu_timer3.elapsed();
    printf("GPU warp-level: %.2f ms (speedup: %.2fx)\n", 
           gpu_time3, cpu_time / gpu_time3);
    
    // Verify correctness (spot check)
    cudaMemcpy(h_output_gpu, d_large_output, 100 * sizeof(int), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 1; i < 100; i++) {
        if (h_output_gpu[i] != h_output_gpu[i-1] + h_input[i]) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Test 3: Stream Compaction Application
    printf("PART 3: Stream Compaction Application\n");
    printf("------------------------------------\n");
    
    // Create sparse data
    int *h_sparse = new int[1000];
    int non_zeros = 0;
    for (int i = 0; i < 1000; i++) {
        h_sparse[i] = (rand() % 10 == 0) ? 0 : i + 1;  // 90% non-zero
        if (h_sparse[i] != 0) non_zeros++;
    }
    printf("Original array: %d elements (%d non-zero)\n", 1000, non_zeros);
    
    int *d_sparse, *d_flags, *d_addresses, *d_compacted;
    cudaMalloc(&d_sparse, 1000 * sizeof(int));
    cudaMalloc(&d_flags, 1000 * sizeof(int));
    cudaMalloc(&d_addresses, 1000 * sizeof(int));
    cudaMalloc(&d_compacted, 1000 * sizeof(int));
    
    cudaMemcpy(d_sparse, h_sparse, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Stream compaction steps
    Timer compact_timer;
    markValid<<<(1000 + 255) / 256, 256>>>(d_flags, d_sparse, 1000);
    scanWorkEfficient<<<2, 512, 512 * sizeof(int)>>>(d_addresses, d_flags, 1000);
    scatter<<<(1000 + 255) / 256, 256>>>(d_compacted, d_sparse, d_addresses, d_flags, 1000);
    cudaDeviceSynchronize();
    float compact_time = compact_timer.elapsed();
    
    // Get compacted size
    int compact_size;
    cudaMemcpy(&compact_size, d_addresses + 999, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Stream compaction: %.2f ms\n", compact_time);
    printf("Compacted array: %d elements (removed %d zeros)\n\n", 
           compact_size, 1000 - compact_size);
    
    // Performance analysis
    printf("==================================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("==================================================\n");
    
    float work_efficiency = (float)(N) / (gpu_time2 * 1e6);  // elements per microsecond
    printf("Work efficiency: %.2f Gelements/s\n", work_efficiency);
    printf("Memory bandwidth: %.2f GB/s\n", 
           (2 * N * sizeof(int)) / (gpu_time2 * 1e6));  // Read + Write
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_sparse;
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_large_input);
    cudaFree(d_large_output);
    cudaFree(d_sparse);
    cudaFree(d_flags);
    cudaFree(d_addresses);
    cudaFree(d_compacted);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Scan is O(log N) parallel steps, O(N) work\n");
    printf("2. Work-efficient algorithm matches sequential work\n");
    printf("3. Bank conflicts can significantly impact performance\n");
    printf("4. Warp-level primitives provide best performance\n");
    printf("5. Scan enables countless parallel algorithms\n");
    printf("6. Stream compaction is killer app for scan\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Trace through work-efficient scan for array of 8 elements
 * 2. Why is exclusive scan often more useful than inclusive?
 * 3. Calculate theoretical speedup for N=1M elements
 * 4. How many barriers needed for scan of size 1024?
 * 5. Why is scan called "prefix sum"?
 *
 * === Coding ===
 * 6. Implement inclusive scan (modify exclusive version)
 * 7. Create segmented scan (multiple scans in one array)
 * 8. Build scan for other operations (max, min, multiply)
 * 9. Implement double-buffer scan (no temp arrays)
 * 10. Create scan with arbitrary associative operator
 *
 * === Optimization ===
 * 11. Optimize for different block sizes (256, 512, 1024)
 * 12. Implement multi-block scan efficiently
 * 13. Use CUB library and compare performance
 * 14. Create specialized scan for small arrays
 * 15. Optimize bank conflicts for different architectures
 *
 * === Applications ===
 * 16. Implement radix sort using scan
 * 17. Build quicksort partition using scan
 * 18. Create sparse matrix-vector multiply with scan
 * 19. Implement line-of-sight algorithm
 * 20. Build tree traversal using scan
 *
 * === Advanced ===
 * 21. Implement work-stealing queue with scan
 * 22. Create parallel BFS using scan
 * 23. Build adaptive quadrature integration
 * 24. Implement parallel polynomial evaluation
 * 25. Design new parallel algorithm using scan
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Domino Chain"
 *    - Each domino (element) affects all following dominoes
 *    - Parallel: Set up multiple chains simultaneously
 *    - Merge chains at boundaries
 *
 * 2. "River System"
 *    - Tributaries (partial sums) flow into larger rivers
 *    - Up-sweep: Combine tributaries
 *    - Down-sweep: Distribute total flow
 *
 * 3. "Binary Tree"
 *    - Up-sweep builds tree bottom-up
 *    - Down-sweep propagates root value down
 *    - Each level is one parallel step
 *
 * 4. Why Scan is Fundamental:
 *    - Converts "sequential" problems to parallel
 *    - Building block for complex algorithms
 *    - Optimal work and depth
 */
