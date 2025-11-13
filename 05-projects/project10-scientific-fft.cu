/*
 * Specialized Track: Scientific Computing
 * Fast Fourier Transform (FFT) on GPU
 *
 * The FFT is fundamental to signal processing, physics simulations,
 * and many scientific applications. Let's build it from scratch!
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why FFT?
// =====================================================

/*
 * THE DISCRETE FOURIER TRANSFORM (DFT):
 * 
 * Naive DFT: O(N²) operations
 * FFT: O(N log N) operations
 * 
 * For N=1024:
 * - DFT: 1,048,576 operations
 * - FFT: 10,240 operations (100x faster!)
 * 
 * Applications:
 * - Signal processing (audio, radar)
 * - Image processing (filtering, compression)
 * - Physics simulations (solving PDEs)
 * - Polynomial multiplication
 * 
 * GPU advantages:
 * - Massive parallelism in butterfly operations
 * - High memory bandwidth
 * - Perfect for batched transforms
 */

using Complex = std::complex<float>;
const float PI = 3.14159265358979323846f;

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
// PART 2: NAIVE DFT (for comparison)
// =====================================================

__global__ void naiveDFT(float2* output, const float2* input, int N, int inverse) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < N) {
        float2 sum = make_float2(0.0f, 0.0f);
        float sign = inverse ? 1.0f : -1.0f;
        
        for (int n = 0; n < N; n++) {
            float angle = sign * 2.0f * PI * k * n / N;
            float cos_angle = cosf(angle);
            float sin_angle = sinf(angle);
            
            // Complex multiplication: (a + bi) * (c + di)
            sum.x += input[n].x * cos_angle - input[n].y * sin_angle;
            sum.y += input[n].x * sin_angle + input[n].y * cos_angle;
        }
        
        if (inverse) {
            sum.x /= N;
            sum.y /= N;
        }
        
        output[k] = sum;
    }
}

// =====================================================
// PART 3: COOLEY-TUKEY FFT (Radix-2)
// =====================================================

// Bit reversal permutation
__global__ void bitReverse(float2* output, const float2* input, int N, int log2N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // Reverse bits
        int reversed = 0;
        int temp = tid;
        
        for (int i = 0; i < log2N; i++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        output[reversed] = input[tid];
    }
}

// FFT butterfly operation
__device__ void butterfly(float2& a, float2& b, float2 w) {
    float2 t;
    // t = b * w
    t.x = b.x * w.x - b.y * w.y;
    t.y = b.x * w.y + b.y * w.x;
    
    // b = a - t
    b.x = a.x - t.x;
    b.y = a.y - t.y;
    
    // a = a + t
    a.x = a.x + t.x;
    a.y = a.y + t.y;
}

// Radix-2 FFT kernel
__global__ void fftRadix2(float2* data, int N, int log2N, int inverse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one butterfly per stage
    float sign = inverse ? 1.0f : -1.0f;
    
    for (int stage = 0; stage < log2N; stage++) {
        int butterflySize = 1 << (stage + 1);
        int butterflyGrp = N / butterflySize;
        int numButterfly = butterflySize / 2;
        
        if (tid < N / 2) {
            int grp = tid / numButterfly;
            int pos = tid % numButterfly;
            
            int ia = grp * butterflySize + pos;
            int ib = ia + numButterfly;
            
            float angle = sign * 2.0f * PI * pos / butterflySize;
            float2 w = make_float2(cosf(angle), sinf(angle));
            
            float2 a = data[ia];
            float2 b = data[ib];
            
            butterfly(a, b, w);
            
            data[ia] = a;
            data[ib] = b;
        }
        __syncthreads();
    }
    
    // Scale for inverse transform
    if (inverse && tid < N) {
        data[tid].x /= N;
        data[tid].y /= N;
    }
}

// =====================================================
// PART 4: OPTIMIZED FFT WITH SHARED MEMORY
// =====================================================

template<int RADIX>
__global__ void fftShared(float2* data, int N, int stage) {
    extern __shared__ float2 shared[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int butterflySize = 1 << (stage + 1);
    int numButterfly = butterflySize / 2;
    
    // Load data to shared memory
    int offset = bid * blockDim.x * 2;
    if (offset + tid < N) {
        shared[tid] = data[offset + tid];
    }
    if (offset + tid + blockDim.x < N) {
        shared[tid + blockDim.x] = data[offset + tid + blockDim.x];
    }
    __syncthreads();
    
    // Perform butterflies in shared memory
    int grp = tid / numButterfly;
    int pos = tid % numButterfly;
    
    if (tid < blockDim.x / 2) {
        int ia = grp * butterflySize + pos;
        int ib = ia + numButterfly;
        
        if (ia < blockDim.x * 2 && ib < blockDim.x * 2) {
            float angle = -2.0f * PI * pos / butterflySize;
            float2 w = make_float2(cosf(angle), sinf(angle));
            
            butterfly(shared[ia], shared[ib], w);
        }
    }
    __syncthreads();
    
    // Write back to global memory
    if (offset + tid < N) {
        data[offset + tid] = shared[tid];
    }
    if (offset + tid + blockDim.x < N) {
        data[offset + tid + blockDim.x] = shared[tid + blockDim.x];
    }
}

// =====================================================
// PART 5: BATCHED FFT
// =====================================================

__global__ void batchedFFT(float2* data, int N, int batchSize, int log2N) {
    int tid = threadIdx.x;
    int batch = blockIdx.y;
    
    if (batch < batchSize) {
        float2* batchData = data + batch * N;
        
        // Simplified: each block handles one complete FFT
        // In practice, use more sophisticated mapping
        for (int stage = 0; stage < log2N; stage++) {
            int butterflySize = 1 << (stage + 1);
            int numButterfly = butterflySize / 2;
            
            for (int butterfly = tid; butterfly < N / 2; butterfly += blockDim.x) {
                int grp = butterfly / numButterfly;
                int pos = butterfly % numButterfly;
                
                int ia = grp * butterflySize + pos;
                int ib = ia + numButterfly;
                
                float angle = -2.0f * PI * pos / butterflySize;
                float2 w = make_float2(cosf(angle), sinf(angle));
                
                float2 a = batchData[ia];
                float2 b = batchData[ib];
                
                butterfly(a, b, w);
                
                batchData[ia] = a;
                batchData[ib] = b;
            }
            __syncthreads();
        }
    }
}

// =====================================================
// PART 6: 2D FFT
// =====================================================

// Row-wise FFT
__global__ void fft2D_rows(float2* data, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height) {
        // Simplified: call 1D FFT on each row
        // In practice, use optimized implementation
    }
}

// Column-wise FFT (with transpose)
__global__ void fft2D_cols_transpose(float2* output, const float2* input, 
                                    int width, int height) {
    __shared__ float2 tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load tile
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Transpose within tile
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// =====================================================
// PART 7: APPLICATIONS
// =====================================================

// Convolution using FFT
void convolutionFFT(float2* result, const float2* signal, const float2* kernel, 
                   int N, cufftHandle plan) {
    // 1. FFT both signal and kernel
    cufftExecC2C(plan, (cufftComplex*)signal, (cufftComplex*)signal, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex*)kernel, (cufftComplex*)kernel, CUFFT_FORWARD);
    
    // 2. Pointwise multiply in frequency domain
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // multiplyComplex<<<blocks, threads>>>(result, signal, kernel, N);
    
    // 3. Inverse FFT
    cufftExecC2C(plan, (cufftComplex*)result, (cufftComplex*)result, CUFFT_INVERSE);
}

// =====================================================
// PART 8: PERFORMANCE COMPARISON
// =====================================================

void compareFFTImplementations(int N) {
    printf("\n=== FFT Performance Comparison (N=%d) ===\n", N);
    
    // Allocate memory
    size_t size = N * sizeof(float2);
    float2 *h_input = new float2[N];
    float2 *h_output = new float2[N];
    
    // Initialize with test signal
    for (int i = 0; i < N; i++) {
        float t = (float)i / N;
        h_input[i].x = sinf(2 * PI * 4 * t) + 0.5f * sinf(2 * PI * 8 * t);
        h_input[i].y = 0.0f;
    }
    
    float2 *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Test 1: Naive DFT (only for small N)
    if (N <= 1024) {
        cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
        Timer timer1;
        naiveDFT<<<(N + 255) / 256, 256>>>(d_output, d_input, N, 0);
        cudaDeviceSynchronize();
        float time1 = timer1.elapsed();
        printf("Naive DFT: %.2f ms\n", time1);
    }
    
    // Test 2: Radix-2 FFT
    int log2N = (int)log2f((float)N);
    if ((1 << log2N) == N) {  // Power of 2
        cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
        
        // Bit reversal
        float2 *d_temp;
        cudaMalloc(&d_temp, size);
        bitReverse<<<(N + 255) / 256, 256>>>(d_temp, d_output, N, log2N);
        
        Timer timer2;
        fftRadix2<<<(N/2 + 255) / 256, 256>>>(d_temp, N, log2N, 0);
        cudaDeviceSynchronize();
        float time2 = timer2.elapsed();
        printf("Radix-2 FFT: %.2f ms\n", time2);
        
        cudaFree(d_temp);
    }
    
    // Test 3: cuFFT
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    
    cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
    Timer timer3;
    for (int i = 0; i < 10; i++) {
        cufftExecC2C(plan, (cufftComplex*)d_output, (cufftComplex*)d_output, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    float time3 = timer3.elapsed() / 10;
    printf("cuFFT: %.2f ms\n", time3);
    
    cufftDestroy(plan);
    
    // Calculate GFLOPS (5N log2(N) for complex FFT)
    double flops = 5.0 * N * log2f(N);
    printf("\nPerformance (GFLOPS):\n");
    if (N <= 1024) printf("Naive DFT: %.2f\n", flops / 1e6 / timer1.elapsed());
    if ((1 << log2N) == N) printf("Radix-2: %.2f\n", flops / 1e6 / timer2.elapsed());
    printf("cuFFT: %.2f\n", flops / 1e6 / time3);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

// =====================================================
// PART 9: MAIN - COMPREHENSIVE DEMO
// =====================================================

int main() {
    printf("==================================================\n");
    printf("FAST FOURIER TRANSFORM ON GPU\n");
    printf("==================================================\n");
    
    // Test different sizes
    int sizes[] = {256, 1024, 4096, 16384, 65536, 1048576};
    
    for (int size : sizes) {
        compareFFTImplementations(size);
    }
    
    // Demonstrate 2D FFT
    printf("\n=== 2D FFT Demo ===\n");
    int width = 512, height = 512;
    printf("2D FFT size: %dx%d\n", width, height);
    
    float2 *d_data;
    cudaMalloc(&d_data, width * height * sizeof(float2));
    
    cufftHandle plan2d;
    cufftPlan2d(&plan2d, height, width, CUFFT_C2C);
    
    Timer timer2d;
    cufftExecC2C(plan2d, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    printf("2D FFT time: %.2f ms\n", timer2d.elapsed());
    
    cufftDestroy(plan2d);
    cudaFree(d_data);
    
    // Demonstrate batched FFT
    printf("\n=== Batched FFT Demo ===\n");
    int batchSize = 1000;
    int fftSize = 1024;
    
    float2 *d_batched;
    cudaMalloc(&d_batched, batchSize * fftSize * sizeof(float2));
    
    cufftHandle planBatch;
    cufftPlan1d(&planBatch, fftSize, CUFFT_C2C, batchSize);
    
    Timer timerBatch;
    cufftExecC2C(planBatch, (cufftComplex*)d_batched, 
                (cufftComplex*)d_batched, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    float batchTime = timerBatch.elapsed();
    printf("Batched FFT: %d transforms of size %d in %.2f ms\n", 
           batchSize, fftSize, batchTime);
    printf("Throughput: %.2f transforms/ms\n", batchSize / batchTime);
    
    cufftDestroy(planBatch);
    cudaFree(d_batched);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. FFT reduces O(N²) to O(N log N)\n");
    printf("2. GPU excels at batched transforms\n");
    printf("3. cuFFT highly optimized for all sizes\n");
    printf("4. Shared memory crucial for small FFTs\n");
    printf("5. Applications: signal processing, physics\n");
    printf("6. 2D FFT = row FFT + transpose + column FFT\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Derive butterfly operation from DFT formula
 * 2. Why is bit reversal needed?
 * 3. Calculate memory bandwidth requirements
 * 4. When does FFT beat direct convolution?
 * 5. Compare different radix algorithms
 *
 * === Implementation ===
 * 6. Implement mixed-radix FFT
 * 7. Create real-to-complex FFT
 * 8. Build Bluestein's algorithm
 * 9. Implement 3D FFT
 * 10. Create FFT-based filtering
 *
 * === Optimization ===
 * 11. Optimize for different GPU architectures
 * 12. Implement cache-friendly transpose
 * 13. Fuse operations (FFT + multiply)
 * 14. Minimize memory transfers
 * 15. Auto-tune for different sizes
 *
 * === Applications ===
 * 16. Audio spectrum analyzer
 * 17. Image compression (JPEG-like)
 * 18. Solve Poisson equation
 * 19. Polynomial multiplication
 * 20. Cross-correlation for pattern matching
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Divide and Conquer"
 *    - Split into even/odd
 *    - Solve smaller problems
 *    - Combine with butterflies
 *
 * 2. "Butterfly Network"
 *    - Each stage connects pairs
 *    - Pattern determines connections
 *    - Log N stages total
 *
 * 3. "Frequency Domain"
 *    - Time ↔ Frequency
 *    - Convolution → Multiplication
 *    - Often simpler in frequency
 */
