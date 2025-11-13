/*
 * Lesson 17: Profiling & Systematic Optimization
 * From Guesswork to Science - Profile-Guided Optimization
 *
 * "Premature optimization is the root of all evil" - Knuth
 * "But mature optimization requires measurement" - This lesson
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Profile?
// =====================================================

/*
 * THE OPTIMIZATION WORKFLOW:
 * 
 * 1. MEASURE - Find the bottleneck
 * 2. ANALYZE - Understand why it's slow  
 * 3. OPTIMIZE - Apply targeted fix
 * 4. VERIFY - Ensure improvement
 * 5. REPEAT - Until fast enough
 * 
 * Without profiling, you're optimizing blind.
 * With profiling, you see exactly where time is spent.
 * 
 * Real-world analogy:
 * Doctor uses X-ray/MRI before surgery
 * Mechanic uses diagnostics before repair
 * We use profilers before optimization
 */

// Built-in CUDA events for timing
class CudaTimer {
    cudaEvent_t start, stop;
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    float elapsed() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// =====================================================
// PART 2: METRICS THAT MATTER
// =====================================================

void printDeviceProperties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device Properties ===\n");
    printf("Device: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Registers/block: %d\n", prop.regsPerBlock);
    printf("Shared mem/block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Memory clock: %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Peak bandwidth: %.0f GB/s\n\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
}

// Calculate theoretical limits
struct PerformanceLimits {
    float peak_gflops;
    float peak_bandwidth_gbps;
    float arithmetic_intensity_threshold;
    
    PerformanceLimits() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // Assume 2 FLOPs per clock per core
        peak_gflops = prop.multiProcessorCount * prop.clockRate / 1000.0f * 
                     128 * 2;  // 128 cores/SM * 2 ops
        
        peak_bandwidth_gbps = 2.0 * prop.memoryClockRate * 
                             (prop.memoryBusWidth / 8) / 1e6;
        
        arithmetic_intensity_threshold = peak_gflops / peak_bandwidth_gbps;
    }
    
    void print() {
        printf("=== Theoretical Limits ===\n");
        printf("Peak compute: %.0f GFLOPS\n", peak_gflops);
        printf("Peak bandwidth: %.0f GB/s\n", peak_bandwidth_gbps);
        printf("Balance point: %.1f FLOP/byte\n\n", arithmetic_intensity_threshold);
    }
};

// =====================================================
// PART 3: IDENTIFYING BOTTLENECKS - Case Study
// =====================================================

// Version 1: Naive matrix multiplication (terrible!)
__global__ void matmulNaive(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Version 2: Coalesced memory access
__global__ void matmulCoalesced(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // B is now accessed in column-major for coalescing
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * N + col] = sum;
    }
}

// Version 3: Tiled with shared memory
template<int TILE_SIZE>
__global__ void matmulTiled(float *C, float *A, float *B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        if (row < N && t * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// =====================================================
// PART 4: PROFILING WITH NVTX
// =====================================================

void profileMatrixMultiplication(int N) {
    size_t bytes = N * N * sizeof(float);
    
    // Allocate memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // NVTX markers for profiling
    nvtxRangePush("Memory Transfer H2D");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    nvtxRangePop();
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    // Profile different versions
    printf("Matrix size: %d x %d\n\n", N, N);
    
    // Version 1: Naive
    nvtxRangePush("Naive MatMul");
    CudaTimer timer1;
    matmulNaive<<<grid, block>>>(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();
    float time1 = timer1.elapsed();
    nvtxRangePop();
    
    printf("Naive: %.2f ms\n", time1);
    float gflops1 = (2.0f * N * N * N) / (time1 * 1e6);
    float bandwidth1 = (3 * bytes) / (time1 * 1e6);  // Read A, B, Write C
    printf("  Performance: %.2f GFLOPS\n", gflops1);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth1);
    
    // Version 2: Coalesced
    nvtxRangePush("Coalesced MatMul");
    CudaTimer timer2;
    matmulCoalesced<<<grid, block>>>(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();
    float time2 = timer2.elapsed();
    nvtxRangePop();
    
    printf("\nCoalesced: %.2f ms (%.2fx speedup)\n", time2, time1/time2);
    float gflops2 = (2.0f * N * N * N) / (time2 * 1e6);
    printf("  Performance: %.2f GFLOPS\n", gflops2);
    
    // Version 3: Tiled
    nvtxRangePush("Tiled MatMul");
    CudaTimer timer3;
    matmulTiled<16><<<grid, block>>>(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();
    float time3 = timer3.elapsed();
    nvtxRangePop();
    
    printf("\nTiled: %.2f ms (%.2fx speedup)\n", time3, time1/time3);
    float gflops3 = (2.0f * N * N * N) / (time3 * 1e6);
    printf("  Performance: %.2f GFLOPS\n", gflops3);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// =====================================================
// PART 5: OCCUPANCY ANALYSIS
// =====================================================

void analyzeOccupancy() {
    printf("\n=== Occupancy Analysis ===\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Example kernel configuration
    int threadsPerBlock = 256;
    int registersPerThread = 32;
    int sharedMemPerBlock = 4096;
    
    // Calculate occupancy limits
    int maxBlocksPerSM_threads = prop.maxThreadsPerMultiProcessor / threadsPerBlock;
    int maxBlocksPerSM_regs = prop.regsPerMultiprocessor / 
                              (registersPerThread * threadsPerBlock);
    int maxBlocksPerSM_smem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;
    
    int maxBlocksPerSM = std::min({maxBlocksPerSM_threads, 
                                  maxBlocksPerSM_regs, 
                                  maxBlocksPerSM_smem});
    
    float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) / 
                     prop.maxThreadsPerMultiProcessor * 100;
    
    printf("Configuration:\n");
    printf("  Threads/block: %d\n", threadsPerBlock);
    printf("  Registers/thread: %d\n", registersPerThread);
    printf("  Shared mem/block: %d bytes\n", sharedMemPerBlock);
    
    printf("\nLimiting factors:\n");
    printf("  Max blocks (threads): %d\n", maxBlocksPerSM_threads);
    printf("  Max blocks (registers): %d\n", maxBlocksPerSM_regs);
    printf("  Max blocks (shared mem): %d\n", maxBlocksPerSM_smem);
    
    printf("\nResult:\n");
    printf("  Blocks per SM: %d\n", maxBlocksPerSM);
    printf("  Occupancy: %.1f%%\n", occupancy);
    
    if (maxBlocksPerSM == maxBlocksPerSM_regs) {
        printf("  LIMITED BY: Registers\n");
    } else if (maxBlocksPerSM == maxBlocksPerSM_smem) {
        printf("  LIMITED BY: Shared memory\n");
    } else {
        printf("  LIMITED BY: Thread count\n");
    }
}

// =====================================================
// PART 6: ROOFLINE MODEL
// =====================================================

void rooflineAnalysis(float achieved_gflops, float achieved_bandwidth_gbps, 
                     float arithmetic_intensity) {
    printf("\n=== Roofline Analysis ===\n");
    
    PerformanceLimits limits;
    
    float compute_bound = limits.peak_gflops;
    float memory_bound = limits.peak_bandwidth_gbps * arithmetic_intensity;
    float roofline = std::min(compute_bound, memory_bound);
    
    printf("Arithmetic intensity: %.2f FLOP/byte\n", arithmetic_intensity);
    printf("Achieved: %.0f GFLOPS, %.0f GB/s\n", 
           achieved_gflops, achieved_bandwidth_gbps);
    printf("Roofline: %.0f GFLOPS\n", roofline);
    printf("Efficiency: %.1f%%\n", (achieved_gflops / roofline) * 100);
    
    if (arithmetic_intensity < limits.arithmetic_intensity_threshold) {
        printf("Status: MEMORY BOUND\n");
        printf("Suggestion: Increase arithmetic intensity or optimize memory access\n");
    } else {
        printf("Status: COMPUTE BOUND\n");
        printf("Suggestion: Optimize computation or increase parallelism\n");
    }
}

// =====================================================
// PART 7: OPTIMIZATION WORKFLOW DEMONSTRATION
// =====================================================

// Example: Optimize a reduction kernel step by step
template<int STEP>
__global__ void reductionStep(float *data, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Step 1: Basic reduction
    if (STEP == 1) {
        sdata[tid] = (idx < n) ? data[idx] : 0.0f;
        __syncthreads();
        
        for (int s = 1; s < blockDim.x; s *= 2) {
            if (tid % (2 * s) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
    }
    
    // Step 2: Remove divergence
    else if (STEP == 2) {
        sdata[tid] = (idx < n) ? data[idx] : 0.0f;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
    }
    
    // Step 3: Multiple elements per thread
    else if (STEP == 3) {
        float sum = (idx < n) ? data[idx] : 0.0f;
        if (idx + blockDim.x < n) sum += data[idx + blockDim.x];
        sdata[tid] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
    }
    
    // Step 4: Unroll last warp
    else if (STEP == 4) {
        float sum = (idx < n) ? data[idx] : 0.0f;
        if (idx + blockDim.x < n) sum += data[idx + blockDim.x];
        sdata[tid] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        
        // Unroll last warp
        if (tid < 32) {
            volatile float *smem = sdata;
            smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// =====================================================
// PART 8: MAIN - SYSTEMATIC OPTIMIZATION
// =====================================================

int main() {
    printf("==================================================\n");
    printf("PROFILING & SYSTEMATIC OPTIMIZATION\n");
    printf("==================================================\n\n");
    
    // Show device capabilities
    printDeviceProperties();
    
    // Show theoretical limits
    PerformanceLimits limits;
    limits.print();
    
    // Profile matrix multiplication versions
    printf("=== Matrix Multiplication Case Study ===\n");
    profileMatrixMultiplication(512);
    
    // Occupancy analysis
    analyzeOccupancy();
    
    // Demonstrate optimization workflow
    printf("\n\n=== Optimization Workflow Demo ===\n");
    printf("Optimizing reduction kernel step by step...\n\n");
    
    const int N = 10000000;
    float *d_data, *d_output;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, 1024 * sizeof(float));
    
    // Initialize data
    float *h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = 1024;
    
    // Profile each optimization step
    float times[5];
    const char* names[5] = {
        "Baseline", 
        "Step 1: Basic", 
        "Step 2: No divergence", 
        "Step 3: Multiple elements", 
        "Step 4: Unroll warp"
    };
    
    // Baseline (for reference)
    CudaTimer timer0;
    for (int i = 0; i < 100; i++) {
        reductionStep<1><<<blocks, threads, threads * sizeof(float)>>>(
            d_data, d_output, N);
    }
    cudaDeviceSynchronize();
    times[0] = timer0.elapsed() / 100;
    
    // Step 1-4
    for (int step = 1; step <= 4; step++) {
        CudaTimer timer;
        for (int i = 0; i < 100; i++) {
            if (step == 1) 
                reductionStep<1><<<blocks, threads, threads * sizeof(float)>>>(
                    d_data, d_output, N);
            else if (step == 2)
                reductionStep<2><<<blocks, threads, threads * sizeof(float)>>>(
                    d_data, d_output, N);
            else if (step == 3)
                reductionStep<3><<<blocks, threads, threads * sizeof(float)>>>(
                    d_data, d_output, N);
            else
                reductionStep<4><<<blocks, threads, threads * sizeof(float)>>>(
                    d_data, d_output, N);
        }
        cudaDeviceSynchronize();
        times[step] = timer.elapsed() / 100;
    }
    
    // Show results
    for (int i = 0; i < 5; i++) {
        printf("%s: %.3f ms", names[i], times[i]);
        if (i > 0) printf(" (%.2fx)", times[0] / times[i]);
        printf("\n");
    }
    
    // Roofline analysis for final kernel
    float flops = N;  // N additions
    float bytes = N * sizeof(float) + blocks * sizeof(float);  // Read + write
    float ai = flops / bytes;
    float achieved_gflops = (flops / 1e9) / (times[4] / 1000);
    float achieved_bandwidth = (bytes / 1e9) / (times[4] / 1000);
    
    rooflineAnalysis(achieved_gflops, achieved_bandwidth, ai);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    delete[] h_data;
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Always profile before optimizing\n");
    printf("2. Focus on the bottleneck (compute vs memory)\n");
    printf("3. Use metrics to guide optimization\n");
    printf("4. Verify improvements at each step\n");
    printf("5. Know your hardware limits\n");
    printf("6. Use tools: nsys, ncu, Nsight Compute\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile -o report ./your_program\n");
    printf("ncu --set full -o profile ./your_program\n");
    printf("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./your_program\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Calculate roofline for your specific GPU
 * 2. Why does occupancy not always correlate with performance?
 * 3. List all types of memory bandwidth (L1, L2, global)
 * 4. What's the difference between nsys and ncu?
 * 5. How do you identify false sharing?
 *
 * === Profiling Practice ===
 * 6. Profile your previous projects with ncu
 * 7. Create custom NVTX ranges for your code
 * 8. Use --metrics to find cache hit rates
 * 9. Compare different block sizes systematically
 * 10. Profile memory allocation overhead
 *
 * === Optimization ===
 * 11. Optimize a memory-bound kernel to compute-bound
 * 12. Reduce register pressure without losing performance
 * 13. Find optimal tile size for your GPU
 * 14. Minimize kernel launch overhead
 * 15. Profile and optimize atomic contention
 *
 * === Advanced Analysis ===
 * 16. Build automated performance regression tests
 * 17. Create performance model for your kernel
 * 18. Use PC sampling to find hotspots
 * 19. Analyze instruction mix (INT vs FP)
 * 20. Profile multi-kernel dependencies
 *
 * === Real-World ===
 * 21. Profile actual ML training workload
 * 22. Optimize end-to-end application latency
 * 23. Find performance cliffs (sudden slowdowns)
 * 24. Profile with different input sizes
 * 25. Create performance dashboard
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Doctor's Diagnosis"
 *    - Symptoms: Slow performance
 *    - Tests: Profiler metrics
 *    - Diagnosis: Bottleneck identification
 *    - Treatment: Targeted optimization
 *
 * 2. "Performance Triangle"
 *    - Corners: Compute, Memory, Latency
 *    - You can only optimize two
 *    - Profile tells you which matters
 *
 * 3. "Optimization Ladder"
 *    - Each rung: Different optimization
 *    - Climb systematically
 *    - Measure at each step
 *    - Stop when "fast enough"
 *
 * 4. Tools Hierarchy:
 *    - nsys: System-wide timeline
 *    - ncu: Kernel deep-dive
 *    - Nsight Compute: Interactive analysis
 *    - Custom timers: Quick checks
 */
