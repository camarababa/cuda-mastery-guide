/*
 * PROJECT 3: GPU-ACCELERATED ATTENTION MECHANISM
 * Building the Heart of Transformers from Scratch
 *
 * "Attention is all you need" - but first, let's understand WHY.
 * This project implements scaled dot-product attention, the foundation
 * of ChatGPT, BERT, and all modern LLMs.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>

// ============================================
// PART 1: FIRST PRINCIPLES - What is Attention?
// ============================================

/*
 * THE PROBLEM ATTENTION SOLVES:
 * 
 * Traditional RNNs process sequences step-by-step (sequential, slow).
 * We want to look at ALL positions simultaneously (parallel, fast).
 * 
 * ATTENTION MECHANISM:
 * For each position, compute how much to "attend" to every other position.
 * 
 * Real-world analogy:
 * Reading this sentence, your brain doesn't process each word in isolation.
 * It relates words to each other: "this" refers to "sentence", etc.
 * 
 * Mathematical formulation:
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 * 
 * Where:
 * - Q (Query): What information am I looking for?
 * - K (Key): What information do I have?
 * - V (Value): What is the actual content?
 * - Scale by sqrt(d_k) to prevent gradient vanishing
 */

// Configuration
const int BATCH_SIZE = 8;           // Number of sequences
const int SEQ_LENGTH = 64;          // Sequence length (tokens)
const int HEAD_DIM = 64;            // Dimension per attention head
const int NUM_HEADS = 12;           // Multi-head attention
const int HIDDEN_DIM = NUM_HEADS * HEAD_DIM;  // Total dimension (768 for BERT-base)

// Performance timer
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

// ============================================
// PART 2: CPU BASELINE - Understanding the Math
// ============================================

// CPU matrix multiplication C = A @ B^T
void matmulCPU(float* A, float* B, float* C, int M, int N, int K) {
    // C[M,N] = A[M,K] @ B[N,K]^T
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];  // B is already transposed in memory
            }
            C[m * N + n] = sum;
        }
    }
}

// CPU softmax (numerically stable)
void softmaxCPU(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Find max for numerical stability
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            max_val = fmaxf(max_val, input[i * cols + j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum;
        }
    }
}

// CPU attention implementation
void attentionCPU(float* Q, float* K, float* V, float* output,
                 int batch_size, int num_heads, int seq_length, int head_dim) {
    float scale = 1.0f / sqrtf(head_dim);
    
    // Temporary buffers
    std::vector<float> scores(seq_length * seq_length);
    std::vector<float> attention_weights(seq_length * seq_length);
    
    // Process each batch and head
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Get pointers to this head's QKV
            int head_offset = (b * num_heads + h) * seq_length * head_dim;
            float* q_head = Q + head_offset;
            float* k_head = K + head_offset;
            float* v_head = V + head_offset;
            float* out_head = output + head_offset;
            
            // Step 1: Compute attention scores (Q @ K^T)
            matmulCPU(q_head, k_head, scores.data(), seq_length, seq_length, head_dim);
            
            // Step 2: Scale scores
            for (int i = 0; i < seq_length * seq_length; i++) {
                scores[i] *= scale;
            }
            
            // Step 3: Apply softmax
            softmaxCPU(scores.data(), attention_weights.data(), seq_length, seq_length);
            
            // Step 4: Apply attention to values (attention_weights @ V)
            for (int i = 0; i < seq_length; i++) {
                for (int j = 0; j < head_dim; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < seq_length; k++) {
                        sum += attention_weights[i * seq_length + k] * 
                               v_head[k * head_dim + j];
                    }
                    out_head[i * head_dim + j] = sum;
                }
            }
        }
    }
}

// ============================================
// PART 3: GPU IMPLEMENTATION - Naive Approach
// ============================================

// GPU matrix multiplication for attention scores
__global__ void computeScoresNaive(float* Q, float* K, float* scores,
                                  int batch_size, int num_heads, int seq_length, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_length * seq_length;
    
    if (idx < total_elements) {
        // Decode indices
        int b = idx / (num_heads * seq_length * seq_length);
        int h = (idx % (num_heads * seq_length * seq_length)) / (seq_length * seq_length);
        int i = (idx % (seq_length * seq_length)) / seq_length;
        int j = idx % seq_length;
        
        // Compute dot product for position (i,j)
        float sum = 0.0f;
        int qk_offset = (b * num_heads + h) * seq_length * head_dim;
        
        for (int k = 0; k < head_dim; k++) {
            sum += Q[qk_offset + i * head_dim + k] * 
                   K[qk_offset + j * head_dim + k];
        }
        
        // Scale and store
        scores[idx] = sum / sqrtf(head_dim);
    }
}

// GPU softmax - each block handles one row
__global__ void softmaxGPU(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= rows) return;
    
    float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Step 1: Find max (reduction)
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    
    // Reduce max across threads
    shared_data[tid] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];
    __syncthreads();
    
    // Step 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        if (row_output) row_output[i] = exp_val;  // Store for later
        sum += exp_val;
    }
    
    // Reduce sum
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    sum = shared_data[0];
    
    // Step 3: Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        row_output[i] /= sum;
    }
}

// GPU attention-value multiplication
__global__ void applyAttentionNaive(float* attention_weights, float* V, float* output,
                                   int batch_size, int num_heads, int seq_length, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_length * head_dim;
    
    if (idx < total_elements) {
        int b = idx / (num_heads * seq_length * head_dim);
        int h = (idx % (num_heads * seq_length * head_dim)) / (seq_length * head_dim);
        int i = (idx % (seq_length * head_dim)) / head_dim;
        int j = idx % head_dim;
        
        float sum = 0.0f;
        int head_offset = (b * num_heads + h) * seq_length;
        
        for (int k = 0; k < seq_length; k++) {
            sum += attention_weights[head_offset * seq_length + i * seq_length + k] *
                   V[head_offset * head_dim + k * head_dim + j];
        }
        
        output[idx] = sum;
    }
}

// ============================================
// PART 4: OPTIMIZED GPU - Tiled & Fused
// ============================================

// Fused attention kernel - compute everything in one pass
template<int BLOCK_SIZE>
__global__ void fusedAttentionOptimized(float* Q, float* K, float* V, float* output,
                                       int batch_size, int num_heads, int seq_length, int head_dim) {
    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_Q = shared_mem;
    float* shared_K = &shared_mem[BLOCK_SIZE * head_dim];
    float* shared_scores = &shared_mem[2 * BLOCK_SIZE * head_dim];
    
    int batch_head = blockIdx.z * num_heads + blockIdx.y;
    int tid = threadIdx.x;
    
    // Each block handles BLOCK_SIZE x seq_length attention computation
    int row_start = blockIdx.x * BLOCK_SIZE;
    
    float scale = 1.0f / sqrtf(head_dim);
    
    // Process each query position in this block
    for (int query_idx = row_start + tid; query_idx < row_start + BLOCK_SIZE && query_idx < seq_length; 
         query_idx += blockDim.x) {
        
        // Load query vector to registers
        float query[64];  // Assuming head_dim <= 64
        for (int i = 0; i < head_dim; i++) {
            query[i] = Q[batch_head * seq_length * head_dim + query_idx * head_dim + i];
        }
        
        // Compute attention scores for all keys
        float scores[256];  // Assuming seq_length <= 256
        float max_score = -INFINITY;
        
        // Tiled computation of Q @ K^T
        for (int key_tile = 0; key_tile < seq_length; key_tile += BLOCK_SIZE) {
            // Collaborative loading of keys
            __syncthreads();
            if (tid < BLOCK_SIZE && key_tile + tid < seq_length) {
                for (int i = 0; i < head_dim; i++) {
                    shared_K[tid * head_dim + i] = K[batch_head * seq_length * head_dim + 
                                                     (key_tile + tid) * head_dim + i];
                }
            }
            __syncthreads();
            
            // Compute scores for this tile
            for (int k = 0; k < BLOCK_SIZE && key_tile + k < seq_length; k++) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += query[i] * shared_K[k * head_dim + i];
                }
                scores[key_tile + k] = score * scale;
                max_score = fmaxf(max_score, scores[key_tile + k]);
            }
        }
        
        // Compute softmax
        float sum = 0.0f;
        for (int i = 0; i < seq_length; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        
        for (int i = 0; i < seq_length; i++) {
            scores[i] /= sum;
        }
        
        // Apply attention to values
        float output_vec[64] = {0};  // Assuming head_dim <= 64
        
        // Tiled computation of attention @ V
        for (int value_tile = 0; value_tile < seq_length; value_tile += BLOCK_SIZE) {
            // Collaborative loading of values
            __syncthreads();
            if (tid < BLOCK_SIZE && value_tile + tid < seq_length) {
                for (int i = 0; i < head_dim; i++) {
                    shared_K[tid * head_dim + i] = V[batch_head * seq_length * head_dim + 
                                                     (value_tile + tid) * head_dim + i];
                }
            }
            __syncthreads();
            
            // Accumulate weighted values
            for (int v = 0; v < BLOCK_SIZE && value_tile + v < seq_length; v++) {
                float weight = scores[value_tile + v];
                for (int i = 0; i < head_dim; i++) {
                    output_vec[i] += weight * shared_K[v * head_dim + i];
                }
            }
        }
        
        // Write output
        for (int i = 0; i < head_dim; i++) {
            output[batch_head * seq_length * head_dim + query_idx * head_dim + i] = output_vec[i];
        }
    }
}

// ============================================
// PART 5: FLASH ATTENTION PREVIEW
// ============================================

/*
 * Flash Attention (Dao et al., 2022) key insights:
 * 
 * 1. Standard attention is memory-bound (O(N²) memory for attention matrix)
 * 2. Flash Attention tiles computation to fit in SRAM (shared memory)
 * 3. Recomputes attention on-the-fly instead of storing
 * 4. Results in 2-4x faster training for transformers
 * 
 * Core algorithm (simplified):
 * - Split Q, K, V into blocks that fit in shared memory
 * - For each Q block:
 *   - For each K, V block:
 *     - Load to shared memory
 *     - Compute local attention
 *     - Update running statistics
 * - Never materialize full attention matrix!
 */

// Simplified Flash Attention kernel (educational version)
__global__ void flashAttentionSimplified(float* Q, float* K, float* V, float* output,
                                       int batch_size, int num_heads, int seq_length, int head_dim) {
    // This is a simplified version for learning
    // Real Flash Attention has many more optimizations
    
    const int TILE_SIZE = 32;  // Tile size for Q and K
    extern __shared__ float shared_mem[];
    
    int batch_head = blockIdx.y * num_heads + blockIdx.z;
    int q_tile = blockIdx.x;
    
    // Each thread block processes one tile of queries
    // Against all tiles of keys/values
    
    // Note: Full implementation requires:
    // - Online softmax computation
    // - Running statistics update
    // - Careful numerical stability
    // - Optimized memory access patterns
    
    // See the paper for full algorithm!
}

// ============================================
// PART 6: MASKED ATTENTION
// ============================================

// Apply causal mask for autoregressive models (GPT-style)
__global__ void applyCausalMask(float* scores, int batch_size, int num_heads, int seq_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_length * seq_length;
    
    if (idx < total_elements) {
        int seq_idx = idx % (seq_length * seq_length);
        int i = seq_idx / seq_length;
        int j = seq_idx % seq_length;
        
        // Mask future positions (j > i)
        if (j > i) {
            scores[idx] = -INFINITY;
        }
    }
}

// ============================================
// PART 7: MAIN - Comprehensive Testing
// ============================================

void initializeData(float* data, int size, float scale = 1.0f) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

// Verify attention properties
void verifyAttention(float* attention_weights, int batch_size, int num_heads, int seq_length) {
    // Check that attention weights sum to 1
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_length; i++) {
                float sum = 0.0f;
                int offset = (b * num_heads + h) * seq_length * seq_length + i * seq_length;
                for (int j = 0; j < seq_length; j++) {
                    sum += attention_weights[offset + j];
                }
                if (fabsf(sum - 1.0f) > 1e-5f) {
                    printf("Attention weights don't sum to 1! Sum = %f\n", sum);
                    return;
                }
            }
        }
    }
    printf("✓ Attention weights sum to 1\n");
}

int main() {
    printf("=======================================\n");
    printf("GPU-ACCELERATED ATTENTION MECHANISM\n");
    printf("=======================================\n\n");
    
    printf("Configuration:\n");
    printf("- Batch size: %d\n", BATCH_SIZE);
    printf("- Sequence length: %d\n", SEQ_LENGTH);
    printf("- Number of heads: %d\n", NUM_HEADS);
    printf("- Head dimension: %d\n", HEAD_DIM);
    printf("- Total dimension: %d\n", HIDDEN_DIM);
    printf("- Total parameters: %.2fM\n\n", 
           (3 * HIDDEN_DIM * HIDDEN_DIM) / 1e6f);  // Q, K, V projections
    
    // Allocate memory
    size_t qkv_size = BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * HEAD_DIM;
    size_t scores_size = BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH;
    
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_output_cpu = new float[qkv_size];
    float* h_output_gpu = new float[qkv_size];
    float* h_scores = new float[scores_size];
    
    // Initialize with random data
    initializeData(h_Q, qkv_size, 0.02f);
    initializeData(h_K, qkv_size, 0.02f);
    initializeData(h_V, qkv_size, 0.02f);
    
    // ==================== CPU IMPLEMENTATION ====================
    printf("Running CPU attention...\n");
    Timer cpu_timer;
    
    attentionCPU(h_Q, h_K, h_V, h_output_cpu, BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM);
    
    float cpu_time = cpu_timer.elapsed();
    printf("CPU time: %.2f ms\n\n", cpu_time);
    
    // ==================== GPU IMPLEMENTATION ====================
    
    // Allocate GPU memory
    float *d_Q, *d_K, *d_V, *d_output;
    float *d_scores, *d_attention_weights;
    
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_output, qkv_size * sizeof(float));
    cudaMalloc(&d_scores, scores_size * sizeof(float));
    cudaMalloc(&d_attention_weights, scores_size * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // ===== Naive GPU Implementation =====
    printf("Running naive GPU attention...\n");
    Timer gpu_naive_timer;
    
    // Compute scores
    int threads = 256;
    int blocks = (scores_size + threads - 1) / threads;
    computeScoresNaive<<<blocks, threads>>>(d_Q, d_K, d_scores, 
                                           BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM);
    
    // Apply softmax
    int softmax_blocks = BATCH_SIZE * NUM_HEADS * SEQ_LENGTH;
    int shared_size = threads * sizeof(float);
    softmaxGPU<<<softmax_blocks, threads, shared_size>>>(d_scores, d_attention_weights,
                                                         BATCH_SIZE * NUM_HEADS * SEQ_LENGTH, SEQ_LENGTH);
    
    // Apply attention to values
    blocks = (qkv_size + threads - 1) / threads;
    applyAttentionNaive<<<blocks, threads>>>(d_attention_weights, d_V, d_output,
                                            BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM);
    
    cudaDeviceSynchronize();
    float gpu_naive_time = gpu_naive_timer.elapsed();
    printf("Naive GPU time: %.2f ms\n", gpu_naive_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / gpu_naive_time);
    
    // ===== Optimized GPU Implementation =====
    printf("Running optimized GPU attention...\n");
    Timer gpu_opt_timer;
    
    const int BLOCK_SIZE = 16;
    dim3 grid(SEQ_LENGTH / BLOCK_SIZE, NUM_HEADS, BATCH_SIZE);
    dim3 block(BLOCK_SIZE * 4);  // Multiple threads per query
    
    size_t shared_mem_size = 2 * BLOCK_SIZE * HEAD_DIM * sizeof(float) + 
                            BLOCK_SIZE * SEQ_LENGTH * sizeof(float);
    
    fusedAttentionOptimized<BLOCK_SIZE><<<grid, block, shared_mem_size>>>(
        d_Q, d_K, d_V, d_output, BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM);
    
    cudaDeviceSynchronize();
    float gpu_opt_time = gpu_opt_timer.elapsed();
    printf("Optimized GPU time: %.2f ms\n", gpu_opt_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_opt_time);
    printf("Speedup vs naive: %.2fx\n\n", gpu_naive_time / gpu_opt_time);
    
    // ===== Causal Masked Attention =====
    printf("Testing causal mask...\n");
    Timer mask_timer;
    
    // Apply causal mask
    applyCausalMask<<<blocks, threads>>>(d_scores, BATCH_SIZE, NUM_HEADS, SEQ_LENGTH);
    
    // Recompute softmax with mask
    softmaxGPU<<<softmax_blocks, threads, shared_size>>>(d_scores, d_attention_weights,
                                                         BATCH_SIZE * NUM_HEADS * SEQ_LENGTH, SEQ_LENGTH);
    
    cudaDeviceSynchronize();
    float mask_time = mask_timer.elapsed();
    printf("Causal mask overhead: %.2f ms\n\n", mask_time);
    
    // Verify correctness
    cudaMemcpy(h_output_gpu, d_output, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scores, d_attention_weights, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < 1000; i++) {
        float diff = fabsf(h_output_cpu[i] - h_output_gpu[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    printf("Maximum difference CPU vs GPU: %e\n", max_diff);
    
    // Verify attention properties
    verifyAttention(h_scores, 1, 1, SEQ_LENGTH);  // Check first batch/head
    
    // ==================== PERFORMANCE ANALYSIS ====================
    printf("\n=======================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("=======================================\n");
    
    // Calculate FLOPs
    size_t score_flops = (size_t)BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH * HEAD_DIM * 2;
    size_t softmax_flops = (size_t)BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH * 5;  // exp + sum + div
    size_t value_flops = (size_t)BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH * HEAD_DIM * 2;
    size_t total_flops = score_flops + softmax_flops + value_flops;
    
    float gflops = total_flops / (gpu_opt_time * 1e6);  // ms to s, flops to gflops
    
    printf("Theoretical FLOPs: %.2f GFLOPs\n", total_flops / 1e9);
    printf("Achieved performance: %.2f GFLOPS\n", gflops);
    
    // Memory bandwidth
    size_t bytes_read = qkv_size * 3 * sizeof(float);  // Q, K, V
    size_t bytes_written = qkv_size * sizeof(float);   // Output
    size_t bytes_intermediate = scores_size * sizeof(float) * 2;  // Scores and weights
    size_t total_bytes = bytes_read + bytes_written + bytes_intermediate;
    
    float bandwidth = (total_bytes / 1e9) / (gpu_opt_time / 1e3);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    
    // Tokens per second (inference metric)
    float tokens_per_sec = (BATCH_SIZE * SEQ_LENGTH) / (gpu_opt_time / 1e3);
    printf("Inference speed: %.0f tokens/second\n", tokens_per_sec);
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_scores;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_scores);
    cudaFree(d_attention_weights);
    
    printf("\n=======================================\n");
    printf("KEY INSIGHTS\n");
    printf("=======================================\n");
    printf("1. Attention is memory-bound (O(N²) memory access)\n");
    printf("2. Fused kernels reduce memory traffic significantly\n");
    printf("3. Tiling enables larger sequences (fit in shared memory)\n");
    printf("4. Flash Attention goes further by never materializing attention matrix\n");
    printf("5. Modern LLMs spend ~30-50%% of time in attention\n");
    
    // Optional: Test with cuBLAS for comparison
    printf("\n[Optional] For production, use cuBLAS/cuDNN for matmuls\n");
    printf("Example: cublasGemmStridedBatched for batched QK^T\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why does attention have quadratic complexity O(N²)?
 * 2. Calculate memory requirements for 2048-length sequences
 * 3. Why is scaling by sqrt(d_k) necessary? (Hint: variance)
 * 4. How does causal masking enable autoregressive generation?
 * 5. Compare attention to convolution - what are the trade-offs?
 *
 * === Coding ===
 * 6. Implement multi-head attention with different head dimensions
 * 7. Add relative positional encodings (T5-style)
 * 8. Implement cross-attention (encoder-decoder)
 * 9. Create attention visualization/heatmap
 * 10. Add ALiBi positional biases
 *
 * === Optimization ===
 * 11. Implement Flash Attention v2 algorithm
 * 12. Use Tensor Cores for mixed-precision attention
 * 13. Implement sparse attention patterns (local, strided)
 * 14. Create kernel for variable-length sequences
 * 15. Optimize for different sequence lengths (padding)
 *
 * === Advanced ===
 * 16. Build a complete transformer block (attention + FFN)
 * 17. Implement gradient computation for attention
 * 18. Create KV-cache for efficient generation
 * 19. Implement Grouped Query Attention (GQA)
 * 20. Build sliding window attention (Mistral-style)
 *
 * === Research ===
 * 21. Implement Linear Attention (O(N) complexity)
 * 22. Create Performer's FAVOR+ algorithm
 * 23. Build Ring Attention for extreme lengths
 * 24. Implement PagedAttention for serving
 * 25. Create your own attention variant!
 *
 * === Production ===
 * 26. Integrate with PyTorch as custom op
 * 27. Build ONNX export for deployment
 * 28. Create benchmarking suite
 * 29. Implement int8 quantized attention
 * 30. Build attention for vision transformers (2D)
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Attention as Information Routing"
 *    - Query: What information do I need?
 *    - Key: What information is available?
 *    - Value: The actual information content
 *
 * 2. "Attention as Soft Dictionary Lookup"
 *    - Traditional: dict[key] -> value
 *    - Attention: weighted_sum(similarity(query, keys) * values)
 *
 * 3. "Why GPUs Excel at Attention"
 *    - Matrix multiplications -> Tensor Cores
 *    - Independent attention heads -> Parallel execution
 *    - Batch processing -> High throughput
 *
 * 4. "The Memory Wall"
 *    - Computation: O(N² * d)
 *    - Memory: O(N² + N * d)
 *    - As N grows, memory becomes bottleneck
 *    - Solution: Don't materialize full attention matrix!
 */
