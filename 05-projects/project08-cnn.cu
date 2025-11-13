/*
 * PROJECT 8: CONVOLUTIONAL NEURAL NETWORK
 * Deep Learning Primitives from Scratch
 *
 * Build the fundamental operations that power computer vision.
 * Understand how frameworks like PyTorch and TensorFlow work under the hood.
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why CNNs?
// =====================================================

/*
 * CONVOLUTIONAL NEURAL NETWORKS:
 * 
 * Key insights:
 * - Spatial locality: nearby pixels are related
 * - Weight sharing: same filter across image
 * - Translation invariance: detect features anywhere
 * 
 * Operations we'll implement:
 * 1. Convolution (the heart of CNNs)
 * 2. Pooling (downsampling)
 * 3. Activation functions (ReLU, etc.)
 * 4. Batch normalization
 * 5. Fully connected layers
 * 
 * This is what happens when you call conv2d() in PyTorch!
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
// PART 2: DATA STRUCTURES
// =====================================================

// 4D Tensor (NCHW format: batch, channels, height, width)
struct Tensor4D {
    float* data;
    int N, C, H, W;
    size_t size;
    
    Tensor4D(int n, int c, int h, int w) : N(n), C(c), H(h), W(w) {
        size = N * C * H * W * sizeof(float);
        cudaMalloc(&data, size);
    }
    
    ~Tensor4D() {
        cudaFree(data);
    }
    
    void setZero() {
        cudaMemset(data, 0, size);
    }
    
    __device__ float& at(int n, int c, int h, int w) {
        return data[n * C * H * W + c * H * W + h * W + w];
    }
};

// Convolution parameters
struct ConvParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    
    int getOutputSize(int input_size) const {
        return (input_size + 2 * padding - kernel_size) / stride + 1;
    }
};

// =====================================================
// PART 3: BASIC CONVOLUTION
// =====================================================

// Naive convolution (direct implementation)
__global__ void conv2dNaive(
    float* output, const float* input, const float* kernel, const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {
    
    // Output position
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = threadIdx.x;
    
    if (n < N && c_out < C_out && h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
        
        // Convolution operation
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    // Input position
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    
                    // Boundary check
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Input: [N, C_in, H_in, W_in]
                        int input_idx = n * C_in * H_in * W_in + 
                                      c_in * H_in * W_in + 
                                      h_in * W_in + w_in;
                        
                        // Kernel: [C_out, C_in, K, K]
                        int kernel_idx = c_out * C_in * K * K + 
                                       c_in * K * K + 
                                       kh * K + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        // Add bias and write output
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // Output: [N, C_out, H_out, W_out]
        int output_idx = n * C_out * H_out * W_out + 
                       c_out * H_out * W_out + 
                       h_out * W_out + w_out;
        output[output_idx] = sum;
    }
}

// =====================================================
// PART 4: IM2COL OPTIMIZATION
// =====================================================

// Im2col: Transform input for efficient GEMM
__global__ void im2col(
    float* col, const float* input,
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int H_out, int W_out) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = C * K * K * H_out * W_out;
    
    if (tid < total_elements) {
        // Decode position in col matrix
        int w_out = tid % W_out;
        int h_out = (tid / W_out) % H_out;
        int kw = (tid / (W_out * H_out)) % K;
        int kh = (tid / (W_out * H_out * K)) % K;
        int c = tid / (W_out * H_out * K * K);
        
        // Calculate input position
        int h_in = h_out * stride - padding + kh;
        int w_in = w_out * stride - padding + kw;
        
        // col: [C*K*K, H_out*W_out]
        int col_idx = tid;
        
        // Boundary check and copy
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int input_idx = c * H * W + h_in * W + w_in;
            col[col_idx] = input[input_idx];
        } else {
            col[col_idx] = 0.0f;  // Padding
        }
    }
}

// Convolution using im2col + GEMM
void conv2dIm2col(
    float* output, const float* input, const float* kernel, const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding,
    cublasHandle_t cublas_handle) {
    
    // Allocate col buffer
    int col_height = C_in * K * K;
    int col_width = H_out * W_out;
    float* d_col;
    cudaMalloc(&d_col, col_height * col_width * sizeof(float));
    
    // Process each batch
    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * C_in * H_in * W_in;
        float* output_n = output + n * C_out * H_out * W_out;
        
        // Im2col transform
        int threads = 256;
        int blocks = (col_height * col_width + threads - 1) / threads;
        im2col<<<blocks, threads>>>(
            d_col, input_n, 1, C_in, H_in, W_in, 
            K, stride, padding, H_out, W_out);
        
        // GEMM: output = kernel * col
        // kernel: [C_out, C_in * K * K]
        // col: [C_in * K * K, H_out * W_out]
        // output: [C_out, H_out * W_out]
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   H_out * W_out, C_out, col_height,
                   &alpha, d_col, H_out * W_out,
                   kernel, col_height,
                   &beta, output_n, H_out * W_out);
        
        // Add bias if provided
        if (bias != nullptr) {
            // TODO: Add bias kernel
        }
    }
    
    cudaFree(d_col);
}

// =====================================================
// PART 5: OPTIMIZED TILED CONVOLUTION
// =====================================================

#define TILE_WIDTH 16

template<int KERNEL_SIZE>
__global__ void conv2dTiled(
    float* output, const float* input, const float* kernel, const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int stride, int padding) {
    
    // Shared memory for input tile and kernel
    __shared__ float tile_input[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
    __shared__ float tile_kernel[KERNEL_SIZE][KERNEL_SIZE];
    
    // Output position
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_WIDTH + tx;
    int out_y = blockIdx.y * TILE_WIDTH + ty;
    int out_c = blockIdx.z % C_out;
    int batch = blockIdx.z / C_out;
    
    float sum = 0.0f;
    
    // Process each input channel
    for (int in_c = 0; in_c < C_in; in_c++) {
        // Load kernel tile
        if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
            tile_kernel[ty][tx] = kernel[out_c * C_in * KERNEL_SIZE * KERNEL_SIZE +
                                        in_c * KERNEL_SIZE * KERNEL_SIZE +
                                        ty * KERNEL_SIZE + tx];
        }
        
        // Load input tile (with halo)
        int in_x = out_x * stride - padding + tx;
        int in_y = out_y * stride - padding + ty;
        
        if (in_x >= 0 && in_x < W_in && in_y >= 0 && in_y < H_in) {
            tile_input[ty][tx] = input[batch * C_in * H_in * W_in +
                                      in_c * H_in * W_in +
                                      in_y * W_in + in_x];
        } else {
            tile_input[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute convolution for this tile
        if (out_x < W_out && out_y < H_out) {
            #pragma unroll
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                #pragma unroll
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    sum += tile_input[ty * stride + ky][tx * stride + kx] * 
                           tile_kernel[ky][kx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (out_x < W_out && out_y < H_out) {
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        output[batch * C_out * H_out * W_out +
               out_c * H_out * W_out +
               out_y * W_out + out_x] = sum;
    }
}

// =====================================================
// PART 6: POOLING LAYERS
// =====================================================

// Max pooling
__global__ void maxPool2d(
    float* output, const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pool_size, int stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * C * H_out * W_out;
    
    if (idx < total_output) {
        // Decode output position
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        
        // Find max in pooling window
        float max_val = -INFINITY;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_out * stride + ph;
                int w_in = w_out * stride + pw;
                
                if (h_in < H_in && w_in < W_in) {
                    int input_idx = n * C * H_in * W_in +
                                  c * H_in * W_in +
                                  h_in * W_in + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        
        output[idx] = max_val;
    }
}

// Average pooling
__global__ void avgPool2d(
    float* output, const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pool_size, int stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * C * H_out * W_out;
    
    if (idx < total_output) {
        // Decode output position
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        
        // Compute average in pooling window
        float sum = 0.0f;
        int count = 0;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_out * stride + ph;
                int w_in = w_out * stride + pw;
                
                if (h_in < H_in && w_in < W_in) {
                    int input_idx = n * C * H_in * W_in +
                                  c * H_in * W_in +
                                  h_in * W_in + w_in;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        output[idx] = (count > 0) ? sum / count : 0.0f;
    }
}

// =====================================================
// PART 7: ACTIVATION FUNCTIONS
// =====================================================

// ReLU activation
__global__ void relu(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Leaky ReLU
__global__ void leakyRelu(float* output, const float* input, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = (val > 0) ? val : alpha * val;
    }
}

// Sigmoid
__global__ void sigmoid(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh
__global__ void tanh_activation(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

// =====================================================
// PART 8: BATCH NORMALIZATION
// =====================================================

__global__ void batchNorm2d(
    float* output, const float* input, 
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    int N, int C, int H, int W,
    float eps, bool training) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        // Decode position
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        
        float val = input[idx];
        
        // Normalize
        float mean = running_mean[c];
        float var = running_var[c];
        float normalized = (val - mean) / sqrtf(var + eps);
        
        // Scale and shift
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// =====================================================
// PART 9: SIMPLE CNN ARCHITECTURE
// =====================================================

class SimpleCNN {
private:
    // Layer parameters
    ConvParams conv1_params = {1, 32, 3, 1, 1};    // Conv1: 1->32 channels
    ConvParams conv2_params = {32, 64, 3, 1, 1};   // Conv2: 32->64 channels
    
    // Weights
    float *conv1_weight, *conv1_bias;
    float *conv2_weight, *conv2_bias;
    float *fc_weight, *fc_bias;
    
    // Buffers
    Tensor4D *conv1_output, *relu1_output;
    Tensor4D *pool1_output;
    Tensor4D *conv2_output, *relu2_output;
    Tensor4D *pool2_output;
    
    cublasHandle_t cublas_handle;
    
public:
    SimpleCNN() {
        cublasCreate(&cublas_handle);
        
        // Allocate weights (simplified initialization)
        cudaMalloc(&conv1_weight, 32 * 1 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv1_bias, 32 * sizeof(float));
        cudaMalloc(&conv2_weight, 64 * 32 * 3 * 3 * sizeof(float));
        cudaMalloc(&conv2_bias, 64 * sizeof(float));
        
        // Initialize weights (Xavier/He initialization in practice)
        initializeWeights();
    }
    
    ~SimpleCNN() {
        cudaFree(conv1_weight);
        cudaFree(conv1_bias);
        cudaFree(conv2_weight);
        cudaFree(conv2_bias);
        cudaFree(fc_weight);
        cudaFree(fc_bias);
        
        cublasDestroy(cublas_handle);
    }
    
    void initializeWeights() {
        // Simplified - in practice use proper initialization
        cudaMemset(conv1_bias, 0, 32 * sizeof(float));
        cudaMemset(conv2_bias, 0, 64 * sizeof(float));
    }
    
    void forward(float* input, int batch_size, int height, int width) {
        // Layer 1: Conv -> ReLU -> Pool
        int h1 = conv1_params.getOutputSize(height);
        int w1 = conv1_params.getOutputSize(width);
        
        conv1_output = new Tensor4D(batch_size, 32, h1, w1);
        dim3 grid1((w1 + 15) / 16, (h1 + 15) / 16, 32 * batch_size);
        dim3 block1(16, 16);
        
        conv2dNaive<<<grid1, block1>>>(
            conv1_output->data, input, conv1_weight, conv1_bias,
            batch_size, 1, height, width,
            32, h1, w1, 3, 1, 1);
        
        // ReLU
        relu1_output = new Tensor4D(batch_size, 32, h1, w1);
        int total1 = batch_size * 32 * h1 * w1;
        relu<<<(total1 + 255) / 256, 256>>>(
            relu1_output->data, conv1_output->data, total1);
        
        // Continue with more layers...
        
        printf("Forward pass completed\n");
    }
};

// =====================================================
// PART 10: MAIN - PERFORMANCE COMPARISON
// =====================================================

void compareConvolutionMethods() {
    printf("\n=== Convolution Performance Comparison ===\n");
    
    // Test parameters
    int N = 1;      // Batch size
    int C_in = 3;   // Input channels (RGB)
    int H = 224;    // Height
    int W = 224;    // Width
    int C_out = 64; // Output channels
    int K = 3;      // Kernel size
    int stride = 1;
    int padding = 1;
    
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    
    printf("Input: %dx%dx%dx%d\n", N, C_in, H, W);
    printf("Output: %dx%dx%dx%d\n", N, C_out, H_out, W_out);
    printf("Kernel: %dx%dx%dx%d\n", C_out, C_in, K, K);
    
    // Allocate memory
    float *d_input, *d_output, *d_kernel, *d_bias;
    size_t input_size = N * C_in * H * W * sizeof(float);
    size_t output_size = N * C_out * H_out * W_out * sizeof(float);
    size_t kernel_size = C_out * C_in * K * K * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_bias, C_out * sizeof(float));
    
    // Initialize with random values
    cudaMemset(d_bias, 0, C_out * sizeof(float));
    
    // Method 1: Naive convolution
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, C_out * N);
    dim3 block(16, 16);
    
    Timer timer1;
    for (int i = 0; i < 10; i++) {
        conv2dNaive<<<grid, block>>>(
            d_output, d_input, d_kernel, d_bias,
            N, C_in, H, W, C_out, H_out, W_out,
            K, stride, padding);
    }
    cudaDeviceSynchronize();
    float time1 = timer1.elapsed() / 10;
    printf("\nNaive convolution: %.2f ms\n", time1);
    
    // Method 2: Im2col + GEMM
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    Timer timer2;
    for (int i = 0; i < 10; i++) {
        conv2dIm2col(d_output, d_input, d_kernel, d_bias,
                    N, C_in, H, W, C_out, H_out, W_out,
                    K, stride, padding, cublas_handle);
    }
    cudaDeviceSynchronize();
    float time2 = timer2.elapsed() / 10;
    printf("Im2col + GEMM: %.2f ms (%.2fx speedup)\n", 
           time2, time1 / time2);
    
    // Calculate GFLOPS
    double flops = 2.0 * N * C_out * H_out * W_out * C_in * K * K;
    printf("\nPerformance:\n");
    printf("Naive: %.2f GFLOPS\n", flops / time1 / 1e6);
    printf("Im2col: %.2f GFLOPS\n", flops / time2 / 1e6);
    
    // Cleanup
    cublasDestroy(cublas_handle);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
}

int main() {
    printf("==================================================\n");
    printf("CONVOLUTIONAL NEURAL NETWORK\n");
    printf("==================================================\n");
    
    // Check cuDNN availability
    printf("\n=== Environment Check ===\n");
    int cudnn_version = cudnnGetVersion();
    printf("cuDNN version: %d.%d.%d\n",
           cudnn_version / 1000,
           (cudnn_version % 1000) / 100,
           cudnn_version % 100);
    
    // Compare convolution implementations
    compareConvolutionMethods();
    
    // Test other layers
    printf("\n=== Testing Other Layers ===\n");
    
    // Test pooling
    int test_size = 1000;
    float *d_test_input, *d_test_output;
    cudaMalloc(&d_test_input, test_size * sizeof(float));
    cudaMalloc(&d_test_output, test_size * sizeof(float));
    
    // Initialize test data
    std::vector<float> h_test(test_size);
    for (int i = 0; i < test_size; i++) {
        h_test[i] = (float)(rand() % 100) / 10.0f - 5.0f;
    }
    cudaMemcpy(d_test_input, h_test.data(), test_size * sizeof(float),
              cudaMemcpyHostToDevice);
    
    // Test activations
    Timer act_timer;
    relu<<<(test_size + 255) / 256, 256>>>(d_test_output, d_test_input, test_size);
    cudaDeviceSynchronize();
    printf("ReLU activation: %.3f ms for %d elements\n", 
           act_timer.elapsed(), test_size);
    
    // Simple CNN demo
    printf("\n=== Simple CNN Demo ===\n");
    SimpleCNN model;
    
    // Dummy input (batch_size=1, 28x28 grayscale)
    float* d_dummy_input;
    cudaMalloc(&d_dummy_input, 1 * 1 * 28 * 28 * sizeof(float));
    model.forward(d_dummy_input, 1, 28, 28);
    
    // Cleanup
    cudaFree(d_test_input);
    cudaFree(d_test_output);
    cudaFree(d_dummy_input);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Im2col transforms convolution to matrix multiply\n");
    printf("2. Tiling essential for shared memory efficiency\n");
    printf("3. Tensor layout (NCHW vs NHWC) affects performance\n");
    printf("4. Fused operations reduce memory bandwidth\n");
    printf("5. cuDNN provides highly optimized implementations\n");
    printf("6. Understanding primitives helps debug/optimize\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why does im2col improve performance?
 * 2. Calculate memory usage for different approaches
 * 3. When to use NHWC vs NCHW layout?
 * 4. How does kernel size affect efficiency?
 * 5. Compare with cuDNN performance
 *
 * === Implementation ===
 * 6. Add depthwise separable convolution
 * 7. Implement Winograd convolution
 * 8. Create dilated convolution
 * 9. Build transposed convolution
 * 10. Add group convolution
 *
 * === Optimization ===
 * 11. Fuse conv+batch_norm+relu
 * 12. Implement FFT-based convolution
 * 13. Optimize for different architectures
 * 14. Add INT8 quantized operations
 * 15. Create autotuning framework
 *
 * === Advanced ===
 * 16. Build full ResNet model
 * 17. Implement attention layers
 * 18. Create custom loss functions
 * 19. Add automatic differentiation
 * 20. Build distributed training
 *
 * === Applications ===
 * 21. Image classification pipeline
 * 22. Object detection (YOLO-style)
 * 23. Semantic segmentation
 * 24. Style transfer
 * 25. Real-time video processing
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Sliding Window"
 *    - Kernel slides across input
 *    - Each position = one output
 *    - Parallel windows = parallel threads
 *
 * 2. "Transform and Conquer"
 *    - Im2col: reshape problem
 *    - Convolution → Matrix multiply
 *    - Use optimized BLAS
 *
 * 3. "Memory Hierarchy"
 *    - Global: Input/Output tensors
 *    - Shared: Tiles of data
 *    - Registers: Accumulation
 *
 * 4. CNN Building Blocks:
 *    - Conv: Feature extraction
 *    - Pool: Dimensionality reduction
 *    - Activation: Non-linearity
 *    - Batch Norm: Stable training
 */
