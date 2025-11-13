/*
 * Lesson 13: Tensor Cores & Mixed Precision
 * Leveraging Modern GPU Hardware for AI/HPC
 *
 * Tensor Cores provide up to 8x speedup for matrix operations.
 * This lesson teaches you to harness this incredible power.
 */

#include <cuda_runtime.h>
#include <mma.h>  // CUDA Matrix Multiply Accumulate
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace nvcuda;

// =====================================================
// PART 1: FIRST PRINCIPLES - What are Tensor Cores?
// =====================================================

/*
 * REVOLUTIONARY HARDWARE:
 * 
 * Traditional CUDA Core: 1 FMA per clock (a * b + c)
 * Tensor Core: 64 FMAs per clock (4x4x4 matrix multiply)
 * 
 * That's 64x more work in the same time!
 * 
 * THE CATCH:
 * - Only works on matrix multiply
 * - Specific data types (FP16, TF32, INT8, etc.)
 * - Must use special APIs
 * 
 * Real-world impact:
 * - GPT training: Months → Days
 * - Scientific simulations: 5-10x faster
 * - Real-time AI inference: Now possible
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
// PART 2: UNDERSTANDING PRECISION
// =====================================================

void demonstratePrecision() {
    printf("=== Precision Comparison ===\n");
    
    // FP32 (single precision)
    float fp32_val = 1.0f / 3.0f;
    printf("FP32: %.9f (32 bits: 1 sign, 8 exp, 23 mantissa)\n", fp32_val);
    
    // FP16 (half precision)
    __half fp16_val = __float2half(1.0f / 3.0f);
    printf("FP16: %.9f (16 bits: 1 sign, 5 exp, 10 mantissa)\n", 
           __half2float(fp16_val));
    
    // TF32 (TensorFloat-32) - Ampere and newer
    // 19 bits: 1 sign, 8 exp, 10 mantissa (range of FP32, precision of FP16)
    printf("TF32: Range of FP32, precision of FP16\n");
    
    // BF16 (Brain Float 16)
    // 16 bits: 1 sign, 8 exp, 7 mantissa (same range as FP32)
    printf("BF16: Same exponent as FP32, reduced mantissa\n\n");
    
    // Demonstrate precision loss
    float large = 1e6f;
    float small = 1.0f;
    
    printf("Precision test: 1e6 + 1.0\n");
    printf("FP32: %.1f\n", large + small);
    printf("FP16: %.1f (precision lost!)\n", 
           __half2float(__hadd(__float2half(large), __float2half(small))));
}

// =====================================================
// PART 3: BASIC MATRIX MULTIPLY - CPU vs GPU vs TENSOR
// =====================================================

// CPU baseline
void matmulCPU(float *C, const float *A, const float *B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Standard CUDA kernel
__global__ void matmulCUDA(float *C, const float *A, const float *B, 
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =====================================================
// PART 4: WMMA API - WARP MATRIX MULTIPLY ACCUMULATE
// =====================================================

// Tensor Core matrix multiply using WMMA
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void matmulWMMA(half *C, const half *A, const half *B, 
                          int M, int N, int K) {
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
                   wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Tensor Core operation
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, 
                               wmma::mem_row_major);
    }
}

// =====================================================
// PART 5: MIXED PRECISION TRAINING
// =====================================================

// Mixed precision GEMM: C = alpha * A * B + beta * C
// A, B in FP16, C in FP32 for accumulation
__global__ void mixedPrecisionGEMM(float *C, const half *A, const half *B,
                                  float alpha, float beta,
                                  int M, int N, int K) {
    const int TILE_SIZE = 16;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A and B tiles, converting to FP32
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = __half2float(A[row * K + t * TILE_SIZE + tx]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = __half2float(B[(t * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute in FP32
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with scaling
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// =====================================================
// PART 6: LOSS SCALING FOR NUMERICAL STABILITY
// =====================================================

// Gradient scaling to prevent underflow in FP16
__global__ void scaleGradients(half *gradients, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float grad = __half2float(gradients[idx]);
        grad *= scale;
        
        // Clamp to FP16 range
        grad = fminf(fmaxf(grad, -65504.0f), 65504.0f);
        
        gradients[idx] = __float2half(grad);
    }
}

// Check for overflow/underflow
__global__ void checkGradients(const half *gradients, int n, bool *has_inf_nan) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = __half2float(gradients[idx]);
        if (isinf(val) || isnan(val)) {
            *has_inf_nan = true;
        }
    }
}

// =====================================================
// PART 7: AUTOMATIC MIXED PRECISION SIMULATION
// =====================================================

class AMPSimulator {
private:
    float loss_scale;
    int scale_growth_interval;
    int step_count;
    float scale_growth_factor;
    float scale_backoff_factor;
    
public:
    AMPSimulator() : loss_scale(1024.0f), scale_growth_interval(2000),
                    step_count(0), scale_growth_factor(2.0f),
                    scale_backoff_factor(0.5f) {}
    
    void step(bool has_inf_nan) {
        if (has_inf_nan) {
            // Reduce scale if overflow detected
            loss_scale *= scale_backoff_factor;
            step_count = 0;
            printf("Overflow detected! Reducing scale to %.1f\n", loss_scale);
        } else {
            step_count++;
            if (step_count >= scale_growth_interval) {
                // Increase scale if no overflow for a while
                loss_scale *= scale_growth_factor;
                step_count = 0;
                printf("Increasing scale to %.1f\n", loss_scale);
            }
        }
    }
    
    float get_scale() { return loss_scale; }
};

// =====================================================
// PART 8: MAIN - COMPREHENSIVE COMPARISON
// =====================================================

int main() {
    printf("==================================================\n");
    printf("TENSOR CORES & MIXED PRECISION\n");
    printf("==================================================\n\n");
    
    // Check GPU capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major >= 7) {
        printf("✓ Tensor Cores supported!\n");
    } else {
        printf("✗ Tensor Cores NOT supported (need CC 7.0+)\n");
    }
    printf("\n");
    
    // Demonstrate precision
    demonstratePrecision();
    
    // Matrix dimensions (must be multiples of 16 for Tensor Cores)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("=== Matrix Multiplication %dx%dx%d ===\n", M, N, K);
    
    // Allocate memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_cpu = new float[M * N];
    float *h_C_gpu = new float[M * N];
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // CPU baseline (small matrix only)
    if (M <= 128) {
        Timer cpu_timer;
        matmulCPU(h_C_cpu, h_A, h_B, M, N, K);
        float cpu_time = cpu_timer.elapsed();
        printf("CPU time: %.2f ms\n", cpu_time);
    }
    
    // Allocate device memory
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    
    cudaMalloc(&d_A_fp32, size_A);
    cudaMalloc(&d_B_fp32, size_B);
    cudaMalloc(&d_C_fp32, size_C);
    cudaMalloc(&d_A_fp16, M * K * sizeof(half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(half));
    cudaMalloc(&d_C_fp16, M * N * sizeof(half));
    
    // Copy data
    cudaMemcpy(d_A_fp32, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Convert to FP16
    half *h_A_fp16 = new half[M * K];
    half *h_B_fp16 = new half[K * N];
    for (int i = 0; i < M * K; i++) h_A_fp16[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_fp16[i] = __float2half(h_B[i]);
    
    cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Standard CUDA kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    Timer cuda_timer;
    for (int i = 0; i < 10; i++) {
        matmulCUDA<<<grid, block>>>(d_C_fp32, d_A_fp32, d_B_fp32, M, N, K);
    }
    cudaDeviceSynchronize();
    float cuda_time = cuda_timer.elapsed() / 10;
    printf("CUDA FP32 time: %.2f ms\n", cuda_time);
    
    // Tensor Core kernel (if supported)
    if (prop.major >= 7) {
        const int WMMA_M = 16;
        const int WMMA_N = 16;
        const int WMMA_K = 16;
        
        dim3 tc_grid((N + WMMA_N - 1) / WMMA_N, 
                     (M + WMMA_M - 1) / WMMA_M);
        dim3 tc_block(32, 1);  // One warp per block
        
        Timer tensor_timer;
        for (int i = 0; i < 10; i++) {
            matmulWMMA<WMMA_M, WMMA_N, WMMA_K><<<tc_grid, tc_block>>>(
                d_C_fp16, d_A_fp16, d_B_fp16, M, N, K);
        }
        cudaDeviceSynchronize();
        float tensor_time = tensor_timer.elapsed() / 10;
        printf("Tensor Core time: %.2f ms (%.2fx speedup!)\n", 
               tensor_time, cuda_time / tensor_time);
    }
    
    // Mixed precision kernel
    Timer mixed_timer;
    for (int i = 0; i < 10; i++) {
        mixedPrecisionGEMM<<<grid, block>>>(
            d_C_fp32, d_A_fp16, d_B_fp16, 1.0f, 0.0f, M, N, K);
    }
    cudaDeviceSynchronize();
    float mixed_time = mixed_timer.elapsed() / 10;
    printf("Mixed precision time: %.2f ms\n", mixed_time);
    
    // cuBLAS comparison
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable tensor cores in cuBLAS
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    float alpha = 1.0f, beta = 0.0f;
    Timer cublas_timer;
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha, d_B_fp32, N, d_A_fp32, K,
                   &beta, d_C_fp32, N);
    }
    cudaDeviceSynchronize();
    float cublas_time = cublas_timer.elapsed() / 10;
    printf("cuBLAS FP32 (with TC) time: %.2f ms\n", cublas_time);
    
    // Performance analysis
    printf("\n=== Performance Analysis ===\n");
    float flops = 2.0f * M * N * K;  // 2 ops per MAC
    printf("Total FLOPs: %.2f GFLOPs\n", flops / 1e9);
    printf("CUDA FP32: %.2f TFLOPS\n", flops / cuda_time / 1e9);
    if (prop.major >= 7) {
        printf("Tensor Core: %.2f TFLOPS\n", flops / tensor_time / 1e9);
    }
    printf("cuBLAS: %.2f TFLOPS\n", flops / cublas_time / 1e9);
    
    // Demonstrate loss scaling
    printf("\n=== Automatic Mixed Precision Demo ===\n");
    AMPSimulator amp;
    
    // Simulate training steps
    for (int step = 0; step < 10; step++) {
        bool overflow = (step == 3 || step == 7);  // Simulate overflow
        amp.step(overflow);
    }
    
    // Cleanup
    cublasDestroy(handle);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    delete[] h_A_fp16;
    delete[] h_B_fp16;
    
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    cudaFree(d_C_fp32);
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp16);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Tensor Cores provide 4-8x speedup for matrix ops\n");
    printf("2. Mixed precision training: FP16 compute, FP32 accumulate\n");
    printf("3. Loss scaling prevents gradient underflow\n");
    printf("4. Not all operations benefit from Tensor Cores\n");
    printf("5. Memory bandwidth often still the bottleneck\n");
    printf("6. Use cuBLAS/cuDNN when possible - they use TC automatically\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why 16x16x16 tiles for WMMA? (hardware design)
 * 2. Calculate theoretical TFLOPS for your GPU
 * 3. When does FP16 precision loss matter?
 * 4. Why accumulate in FP32 even with FP16 inputs?
 * 5. Compare tensor core efficiency vs occupancy
 *
 * === Implementation ===
 * 6. Implement INT8 quantized GEMM
 * 7. Create fused bias+ReLU with tensor cores
 * 8. Build batch matrix multiply with TC
 * 9. Implement convolution using tensor cores
 * 10. Create mixed precision layer norm
 *
 * === Optimization ===
 * 11. Overlap compute and data movement
 * 12. Optimize for different M, N, K sizes
 * 13. Compare different tile sizes
 * 14. Implement double buffering with TC
 * 15. Profile tensor core utilization
 *
 * === Advanced ===
 * 16. Build transformer attention with TC
 * 17. Implement Flash Attention with FP16
 * 18. Create custom CUTLASS kernel
 * 19. Mix TF32 and FP16 operations
 * 20. Implement quantization-aware training
 *
 * === Research ===
 * 21. Explore FP8 on Hopper GPUs
 * 22. Implement block-sparse operations
 * 23. Create structured sparsity patterns
 * 24. Build neural architecture search
 * 25. Design new mixed-precision strategies
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Assembly Line Upgrade"
 *    - Regular cores: One product at a time
 *    - Tensor cores: 64 products at once
 *    - Must reorganize factory (code) to use
 *
 * 2. "Precision vs Speed Trade-off"
 *    - FP32: Full precision, slower
 *    - FP16: Half precision, 2x faster
 *    - Mixed: Smart compromise
 *
 * 3. "The Tensor Core Recipe"
 *    - Ingredients: 16x16 matrices
 *    - Operation: D = A×B + C
 *    - Result: Massive acceleration
 *
 * 4. Hardware Evolution:
 *    - Volta (V100): First tensor cores
 *    - Turing (T4): INT8/INT4
 *    - Ampere (A100): TF32, sparsity
 *    - Hopper (H100): FP8, transformer engine
 */
