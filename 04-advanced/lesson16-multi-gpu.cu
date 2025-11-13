/*
 * Lesson 16: Multi-GPU Programming  
 * Scaling Beyond Single GPU
 *
 * One GPU is fast. Multiple GPUs are REALLY fast.
 * Learn to harness the power of GPU clusters!
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Multi-GPU?
// =====================================================

/*
 * SCALING LIMITS:
 * 
 * Single GPU limits:
 * - Memory: 24-80GB max
 * - Compute: ~20 TFLOPS
 * - Problems are getting bigger!
 * 
 * Multi-GPU solutions:
 * - 8x A100 = 640GB memory, 160 TFLOPS
 * - DGX systems: 8-16 GPUs
 * - Supercomputers: 1000s of GPUs
 * 
 * CHALLENGES:
 * - Data distribution
 * - Communication overhead  
 * - Load balancing
 * - Synchronization
 * 
 * This lesson teaches you to overcome these!
 */

// Timer
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
// PART 2: GPU DISCOVERY & MANAGEMENT
// =====================================================

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("=== Multi-GPU System Information ===\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SMs: %d\n", prop.multiProcessorCount);
        printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
        
        // Check P2P access to other GPUs
        printf("  P2P access to: ");
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                if (canAccess) printf("%d ", j);
            }
        }
        printf("\n");
    }
    printf("\n");
}

// Enable peer access between GPUs
void enablePeerAccess(int deviceCount) {
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                if (canAccess) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err == cudaSuccess) {
                        printf("Enabled P2P: GPU %d -> GPU %d\n", i, j);
                    }
                }
            }
        }
    }
}

// =====================================================
// PART 3: DATA PARALLELISM - SPLIT ACROSS GPUS
// =====================================================

// Simple vector addition kernel
__global__ void vectorAdd(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Multi-GPU vector addition
void multiGPUVectorAdd(float *h_a, float *h_b, float *h_c, int n) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("\n=== Data Parallel Vector Addition ===\n");
    printf("Vector size: %d elements\n", n);
    printf("Using %d GPUs\n", deviceCount);
    
    // Calculate work per GPU
    int elementsPerGPU = n / deviceCount;
    int remainder = n % deviceCount;
    
    // Allocate device pointers for each GPU
    std::vector<float*> d_a(deviceCount);
    std::vector<float*> d_b(deviceCount);
    std::vector<float*> d_c(deviceCount);
    std::vector<cudaStream_t> streams(deviceCount);
    
    Timer timer;
    
    // Setup each GPU
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        
        // Calculate this GPU's workload
        int offset = i * elementsPerGPU + std::min(i, remainder);
        int elements = elementsPerGPU + (i < remainder ? 1 : 0);
        
        // Allocate memory
        cudaMalloc(&d_a[i], elements * sizeof(float));
        cudaMalloc(&d_b[i], elements * sizeof(float));
        cudaMalloc(&d_c[i], elements * sizeof(float));
        
        // Create stream
        cudaStreamCreate(&streams[i]);
        
        // Copy data asynchronously
        cudaMemcpyAsync(d_a[i], h_a + offset, elements * sizeof(float),
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, elements * sizeof(float),
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel
        int threads = 256;
        int blocks = (elements + threads - 1) / threads;
        vectorAdd<<<blocks, threads, 0, streams[i]>>>(d_c[i], d_a[i], d_b[i], elements);
        
        // Copy result back
        cudaMemcpyAsync(h_c + offset, d_c[i], elements * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    float elapsed = timer.elapsed();
    printf("Multi-GPU time: %.2f ms\n", elapsed);
    printf("Throughput: %.2f GB/s\n", 
           (3 * n * sizeof(float)) / (elapsed * 1e6));
    
    // Cleanup
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaStreamDestroy(streams[i]);
    }
}

// =====================================================
// PART 4: PEER-TO-PEER COMMUNICATION
// =====================================================

// Halo exchange pattern (common in stencil computations)
__global__ void stencilComputation(float *output, float *input, 
                                  int width, int height, int halo) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + halo;
    int y = blockIdx.y * blockDim.y + threadIdx.y + halo;
    
    if (x < width - halo && y < height - halo) {
        // 5-point stencil
        float sum = input[y * width + x] * 4.0f;
        sum -= input[(y-1) * width + x];
        sum -= input[(y+1) * width + x];
        sum -= input[y * width + (x-1)];
        sum -= input[y * width + (x+1)];
        
        output[y * width + x] = sum;
    }
}

// Multi-GPU stencil with P2P halo exchange
void multiGPUStencil(int width, int total_height, int iterations) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("\n=== Multi-GPU Stencil (P2P) ===\n");
    printf("Grid: %d x %d\n", width, total_height);
    printf("Iterations: %d\n", iterations);
    
    const int halo = 1;
    int rows_per_gpu = total_height / deviceCount;
    
    // Allocate memory on each GPU
    std::vector<float*> d_input(deviceCount);
    std::vector<float*> d_output(deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        int height = rows_per_gpu + 2 * halo;  // Include halo regions
        cudaMalloc(&d_input[i], width * height * sizeof(float));
        cudaMalloc(&d_output[i], width * height * sizeof(float));
        
        // Initialize
        cudaMemset(d_input[i], 0, width * height * sizeof(float));
    }
    
    Timer timer;
    
    // Iterative computation
    for (int iter = 0; iter < iterations; iter++) {
        // Step 1: Compute interior (overlap with communication)
        for (int i = 0; i < deviceCount; i++) {
            cudaSetDevice(i);
            
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                     (rows_per_gpu + block.y - 1) / block.y);
            
            stencilComputation<<<grid, block>>>(
                d_output[i], d_input[i], width, rows_per_gpu + 2*halo, halo);
        }
        
        // Step 2: Exchange halos with P2P
        for (int i = 0; i < deviceCount - 1; i++) {
            // Copy bottom row of GPU i to top halo of GPU i+1
            cudaMemcpyPeerAsync(
                d_input[i+1],                    // dst
                i+1,                             // dst device
                d_output[i] + (rows_per_gpu * width),  // src (bottom row)
                i,                               // src device
                width * sizeof(float)            // size
            );
            
            // Copy top row of GPU i+1 to bottom halo of GPU i
            cudaMemcpyPeerAsync(
                d_input[i] + ((rows_per_gpu + halo) * width),  // dst
                i,                               // dst device
                d_output[i+1] + (halo * width),  // src (top row)
                i+1,                             // src device
                width * sizeof(float)            // size
            );
        }
        
        // Synchronize
        for (int i = 0; i < deviceCount; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        
        // Swap buffers
        std::swap(d_input, d_output);
    }
    
    float elapsed = timer.elapsed();
    printf("Multi-GPU stencil time: %.2f ms\n", elapsed);
    printf("Throughput: %.2f Gcells/s\n", 
           (width * total_height * iterations) / (elapsed * 1e6));
    
    // Cleanup
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }
}

// =====================================================
// PART 5: UNIFIED MEMORY MULTI-GPU
// =====================================================

// Matrix multiplication using Unified Memory
__global__ void matrixMul(float *C, float *A, float *B, int N) {
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

void unifiedMemoryMultiGPU(int N) {
    printf("\n=== Unified Memory Multi-GPU ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    
    size_t bytes = N * N * sizeof(float);
    
    // Allocate Unified Memory
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    // Hints for data placement
    for (int i = 0; i < deviceCount; i++) {
        cudaMemAdvise(A, bytes, cudaMemAdviseSetReadMostly, i);
        cudaMemAdvise(B, bytes, cudaMemAdviseSetReadMostly, i);
    }
    
    Timer timer;
    
    // Launch on multiple GPUs
    std::vector<cudaStream_t> streams(deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        
        // Each GPU computes part of C
        int rowsPerGPU = N / deviceCount;
        int startRow = i * rowsPerGPU;
        
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (rowsPerGPU + block.y - 1) / block.y);
        
        matrixMul<<<grid, block, 0, streams[i]>>>(
            C + startRow * N, A + startRow * N, B, N);
    }
    
    // Synchronize
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    float elapsed = timer.elapsed();
    printf("Unified Memory time: %.2f ms\n", elapsed);
    
    // Cleanup
    for (int i = 0; i < deviceCount; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

// =====================================================
// PART 6: NCCL - NVIDIA COLLECTIVE COMMUNICATIONS
// =====================================================

// Note: NCCL requires linking with -lnccl
// This is a simplified example structure

/*
#include <nccl.h>

void ncclExample(int size) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    // Initialize NCCL
    ncclComm_t comms[deviceCount];
    ncclCommInitAll(comms, deviceCount, nullptr);
    
    // Allocate memory on each GPU
    float** d_data = new float*[deviceCount];
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], size * sizeof(float));
    }
    
    // AllReduce example
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        ncclAllReduce(d_data[i], d_data[i], size, ncclFloat,
                     ncclSum, comms[i], nullptr);
    }
    
    // Cleanup
    for (int i = 0; i < deviceCount; i++) {
        ncclCommDestroy(comms[i]);
        cudaFree(d_data[i]);
    }
    delete[] d_data;
}
*/

// =====================================================
// PART 7: MULTI-GPU PATTERNS
// =====================================================

// Pattern 1: Pipeline parallel
void pipelinePattern() {
    printf("\n=== Pipeline Pattern ===\n");
    printf("GPU 0 -> GPU 1 -> GPU 2 -> ...\n");
    printf("Each GPU processes different stage\n");
}

// Pattern 2: Data parallel
void dataParallelPattern() {
    printf("\n=== Data Parallel Pattern ===\n");
    printf("Split data across GPUs\n");
    printf("Same operation on each chunk\n");
}

// Pattern 3: Model parallel
void modelParallelPattern() {
    printf("\n=== Model Parallel Pattern ===\n");
    printf("Split model layers across GPUs\n");
    printf("Forward/backward through GPUs\n");
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE DEMO
// =====================================================

int main() {
    printf("==================================================\n");
    printf("MULTI-GPU PROGRAMMING\n");
    printf("==================================================\n\n");
    
    // Check system
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount < 2) {
        printf("This system has only %d GPU(s).\n", deviceCount);
        printf("Multi-GPU examples require at least 2 GPUs.\n");
        printf("Showing what would be done with multiple GPUs...\n\n");
    }
    
    // Print GPU information
    printGPUInfo();
    
    if (deviceCount >= 2) {
        // Enable peer access
        enablePeerAccess(deviceCount);
        
        // Test 1: Data parallel vector addition
        int n = 10000000;
        float *h_a = new float[n];
        float *h_b = new float[n];
        float *h_c = new float[n];
        
        for (int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i * 2;
        }
        
        multiGPUVectorAdd(h_a, h_b, h_c, n);
        
        // Verify
        bool correct = true;
        for (int i = 0; i < 100; i++) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                correct = false;
                break;
            }
        }
        printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
        
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        
        // Test 2: Stencil with P2P
        multiGPUStencil(1024, 1024, 10);
        
        // Test 3: Unified Memory
        unifiedMemoryMultiGPU(1024);
    }
    
    // Show patterns
    printf("\n=== Multi-GPU Patterns ===\n");
    pipelinePattern();
    dataParallelPattern();
    modelParallelPattern();
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Multiple GPUs multiply compute & memory\n");
    printf("2. Communication is the bottleneck\n");
    printf("3. P2P avoids host memory (fast!)\n");
    printf("4. Overlap computation with communication\n");
    printf("5. NCCL provides optimized collectives\n");
    printf("6. Choose right pattern for your problem\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Calculate PCIe bandwidth limits
 * 2. When does multi-GPU beat single GPU?
 * 3. Compare P2P vs host-staged transfers
 * 4. How does NVLink improve performance?
 * 5. What's the optimal GPU count?
 *
 * === Implementation ===
 * 6. Implement multi-GPU reduction
 * 7. Create distributed matrix transpose
 * 8. Build multi-GPU sort
 * 9. Implement broadcast operation
 * 10. Create GPU-aware MPI wrapper
 *
 * === Optimization ===
 * 11. Overlap computation and communication
 * 12. Minimize synchronization points
 * 13. Balance load across GPUs
 * 14. Optimize data placement
 * 15. Profile multi-GPU bottlenecks
 *
 * === Advanced ===
 * 16. Implement model parallel training
 * 17. Create multi-GPU FFT
 * 18. Build distributed graph algorithms
 * 19. Implement checkpointing
 * 20. Create fault-tolerant computation
 *
 * === Production ===
 * 21. Handle GPU failures gracefully
 * 22. Dynamic GPU allocation
 * 23. Multi-node GPU clusters
 * 24. Container orchestration (K8s)
 * 25. Cloud GPU deployment
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Orchestra"
 *    - Each GPU is a section
 *    - Need coordination (conductor)
 *    - Beautiful when synchronized
 *
 * 2. "Highway System"
 *    - GPUs are cities
 *    - PCIe/NVLink are highways
 *    - Traffic (data) must flow efficiently
 *
 * 3. "Divide and Conquer"
 *    - Split problem across GPUs
 *    - Minimize communication
 *    - Maximize parallelism
 *
 * 4. Communication Patterns:
 *    - P2P: Direct GPU-GPU
 *    - Collective: All-to-all
 *    - Staged: Through host
 *    - Unified: Automatic migration
 */
