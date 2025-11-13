/*
 * PROJECT 2: GPU-ACCELERATED TOKENIZER AND EMBEDDINGS
 * Building NLP Fundamentals from Scratch
 *
 * This project implements the foundation of any NLP/LLM system:
 * tokenization and embeddings. We'll build from first principles,
 * understanding why GPUs excel at these operations.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <random>

// ============================================
// PART 1: FIRST PRINCIPLES - Why GPUs for NLP?
// ============================================

/*
 * FUNDAMENTAL INSIGHT:
 * NLP involves massive parallelism at multiple levels:
 * 1. Token-level: Process thousands of tokens in parallel
 * 2. Batch-level: Process multiple sequences simultaneously
 * 3. Embedding-level: Matrix operations (perfect for GPUs)
 * 4. Model-level: Billions of parameters processed in parallel
 *
 * Let's build the foundation that powers ChatGPT, BERT, etc.
 */

// Configuration
const int MAX_VOCAB_SIZE = 50000;      // Typical vocab size
const int EMBEDDING_DIM = 768;         // BERT-like embedding dimension
const int MAX_SEQUENCE_LENGTH = 512;   // Maximum tokens per sequence
const int BATCH_SIZE = 32;             // Process 32 sequences at once

// Simple timer for performance measurement
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
// PART 2: CPU BASELINE - Understanding the Problem
// ============================================

// Simple tokenizer (character-level for demonstration)
class SimpleTokenizer {
public:
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    int vocab_size = 0;
    
    SimpleTokenizer() {
        // Special tokens
        add_token("<PAD>");
        add_token("<UNK>");
        add_token("<BOS>");
        add_token("<EOS>");
        
        // Add basic vocabulary (characters + common words)
        // In production, you'd load from a file
        for (char c = 'a'; c <= 'z'; c++) {
            add_token(std::string(1, c));
        }
        for (char c = 'A'; c <= 'Z'; c++) {
            add_token(std::string(1, c));
        }
        for (char c = '0'; c <= '9'; c++) {
            add_token(std::string(1, c));
        }
        add_token(" ");
        add_token(".");
        add_token(",");
        add_token("!");
        add_token("?");
        
        // Common words (simplified)
        const char* common_words[] = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are", "been"
        };
        for (const auto& word : common_words) {
            add_token(word);
        }
    }
    
    void add_token(const std::string& token) {
        if (token_to_id.find(token) == token_to_id.end()) {
            token_to_id[token] = vocab_size;
            id_to_token.push_back(token);
            vocab_size++;
        }
    }
    
    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(token_to_id["<BOS>"]);
        
        // Simple word + character tokenization
        std::string current_word;
        for (char c : text) {
            if (c == ' ') {
                if (!current_word.empty()) {
                    // Try whole word first
                    if (token_to_id.find(current_word) != token_to_id.end()) {
                        tokens.push_back(token_to_id[current_word]);
                    } else {
                        // Fall back to characters
                        for (char wc : current_word) {
                            std::string char_str(1, wc);
                            tokens.push_back(token_to_id.count(char_str) ? 
                                           token_to_id[char_str] : token_to_id["<UNK>"]);
                        }
                    }
                    current_word.clear();
                }
                tokens.push_back(token_to_id[" "]);
            } else {
                current_word += c;
            }
        }
        
        // Handle last word
        if (!current_word.empty()) {
            if (token_to_id.find(current_word) != token_to_id.end()) {
                tokens.push_back(token_to_id[current_word]);
            } else {
                for (char c : current_word) {
                    std::string char_str(1, c);
                    tokens.push_back(token_to_id.count(char_str) ? 
                                   token_to_id[char_str] : token_to_id["<UNK>"]);
                }
            }
        }
        
        tokens.push_back(token_to_id["<EOS>"]);
        return tokens;
    }
};

// CPU embedding lookup
void embeddingLookupCPU(float* embeddings, int* token_ids, float* output,
                       int batch_size, int seq_length, int embedding_dim, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            int token_id = token_ids[b * seq_length + s];
            if (token_id >= 0 && token_id < vocab_size) {
                for (int d = 0; d < embedding_dim; d++) {
                    output[b * seq_length * embedding_dim + s * embedding_dim + d] = 
                        embeddings[token_id * embedding_dim + d];
                }
            }
        }
    }
}

// CPU positional encoding (sinusoidal like in "Attention is All You Need")
void positionalEncodingCPU(float* output, int batch_size, int seq_length, int embedding_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int pos = 0; pos < seq_length; pos++) {
            for (int i = 0; i < embedding_dim; i++) {
                float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / embedding_dim);
                int idx = b * seq_length * embedding_dim + pos * embedding_dim + i;
                if (i % 2 == 0) {
                    output[idx] += sinf(angle);
                } else {
                    output[idx] += cosf(angle);
                }
            }
        }
    }
}

// CPU layer normalization (essential for transformers)
void layerNormCPU(float* input, float* output, float* gamma, float* beta,
                  int batch_size, int seq_length, int embedding_dim) {
    const float epsilon = 1e-5f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            // Compute mean
            float mean = 0.0f;
            int offset = b * seq_length * embedding_dim + s * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
                mean += input[offset + d];
            }
            mean /= embedding_dim;
            
            // Compute variance
            float variance = 0.0f;
            for (int d = 0; d < embedding_dim; d++) {
                float diff = input[offset + d] - mean;
                variance += diff * diff;
            }
            variance /= embedding_dim;
            
            // Normalize and scale
            float inv_std = 1.0f / sqrtf(variance + epsilon);
            for (int d = 0; d < embedding_dim; d++) {
                float normalized = (input[offset + d] - mean) * inv_std;
                output[offset + d] = gamma[d] * normalized + beta[d];
            }
        }
    }
}

// ============================================
// PART 3: GPU IMPLEMENTATION - Massive Parallelism
// ============================================

// GPU embedding lookup - each thread handles one embedding element
__global__ void embeddingLookupGPU(float* embeddings, int* token_ids, float* output,
                                   int batch_size, int seq_length, int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * embedding_dim;
    
    if (idx < total_elements) {
        int b = idx / (seq_length * embedding_dim);
        int s = (idx % (seq_length * embedding_dim)) / embedding_dim;
        int d = idx % embedding_dim;
        
        int token_id = token_ids[b * seq_length + s];
        if (token_id >= 0 && token_id < vocab_size) {
            output[idx] = embeddings[token_id * embedding_dim + d];
        } else {
            output[idx] = 0.0f;  // Out of bounds token
        }
    }
}

// GPU positional encoding
__global__ void positionalEncodingGPU(float* output, int batch_size, int seq_length, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * embedding_dim;
    
    if (idx < total_elements) {
        int b = idx / (seq_length * embedding_dim);
        int pos = (idx % (seq_length * embedding_dim)) / embedding_dim;
        int i = idx % embedding_dim;
        
        float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / embedding_dim);
        if (i % 2 == 0) {
            output[idx] += sinf(angle);
        } else {
            output[idx] += cosf(angle);
        }
    }
}

// GPU layer normalization - optimized with shared memory
__global__ void layerNormGPU(float* input, float* output, float* gamma, float* beta,
                             int batch_size, int seq_length, int embedding_dim) {
    extern __shared__ float shared_data[];
    
    // Each block handles one position (batch_element, sequence_position)
    int batch_idx = blockIdx.x / seq_length;
    int seq_idx = blockIdx.x % seq_length;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int offset = batch_idx * seq_length * embedding_dim + seq_idx * embedding_dim;
    
    // Step 1: Load data and compute partial sums for mean
    float sum = 0.0f;
    for (int d = tid; d < embedding_dim; d += blockDim.x) {
        sum += input[offset + d];
    }
    
    // Reduce to get mean
    shared_data[tid] = sum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / embedding_dim;
    __syncthreads();
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int d = tid; d < embedding_dim; d += blockDim.x) {
        float diff = input[offset + d] - mean;
        var_sum += diff * diff;
    }
    
    // Reduce to get variance
    shared_data[tid] = var_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_data[0] / embedding_dim;
    float inv_std = rsqrtf(variance + 1e-5f);  // rsqrtf is faster than 1/sqrtf
    
    // Step 3: Normalize and apply affine transformation
    for (int d = tid; d < embedding_dim; d += blockDim.x) {
        float normalized = (input[offset + d] - mean) * inv_std;
        output[offset + d] = gamma[d] * normalized + beta[d];
    }
}

// ============================================
// PART 4: ADVANCED - Fused Kernels
// ============================================

// Fused embedding lookup + positional encoding (reduce memory bandwidth)
__global__ void fusedEmbeddingPositionalGPU(float* embeddings, int* token_ids, float* output,
                                           int batch_size, int seq_length, int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * embedding_dim;
    
    if (idx < total_elements) {
        int b = idx / (seq_length * embedding_dim);
        int pos = (idx % (seq_length * embedding_dim)) / embedding_dim;
        int d = idx % embedding_dim;
        
        // Embedding lookup
        int token_id = token_ids[b * seq_length + pos];
        float embedding_val = 0.0f;
        if (token_id >= 0 && token_id < vocab_size) {
            embedding_val = embeddings[token_id * embedding_dim + d];
        }
        
        // Positional encoding (fused)
        float angle = pos / powf(10000.0f, (2.0f * (d / 2)) / embedding_dim);
        float pos_encoding = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        
        // Write once to global memory (saves bandwidth)
        output[idx] = embedding_val + pos_encoding;
    }
}

// ============================================
// PART 5: MAIN - Comprehensive Testing
// ============================================

void initializeEmbeddings(float* embeddings, int vocab_size, int embedding_dim) {
    // Xavier initialization
    float scale = sqrtf(6.0f / (vocab_size + embedding_dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        embeddings[i] = dist(gen);
    }
}

int main() {
    printf("====================================\n");
    printf("GPU-ACCELERATED NLP FUNDAMENTALS\n");
    printf("====================================\n\n");
    
    // Initialize tokenizer
    SimpleTokenizer tokenizer;
    printf("Vocabulary size: %d\n", tokenizer.vocab_size);
    printf("Embedding dimension: %d\n", EMBEDDING_DIM);
    printf("Max sequence length: %d\n", MAX_SEQUENCE_LENGTH);
    printf("Batch size: %d\n\n", BATCH_SIZE);
    
    // Create test data
    std::vector<std::string> test_sentences = {
        "The quick brown fox jumps over the lazy dog",
        "GPU acceleration is essential for modern NLP",
        "Transformers have revolutionized machine learning",
        "Attention is all you need for sequence modeling"
    };
    
    // Tokenize sentences
    std::vector<std::vector<int>> tokenized;
    int max_len = 0;
    for (const auto& sentence : test_sentences) {
        auto tokens = tokenizer.tokenize(sentence);
        max_len = std::max(max_len, (int)tokens.size());
        tokenized.push_back(tokens);
        
        printf("Sentence: %s\n", sentence.c_str());
        printf("Tokens: ");
        for (int t : tokens) {
            printf("%d ", t);
        }
        printf("\n\n");
    }
    
    // Pad sequences
    int* h_token_ids = new int[BATCH_SIZE * MAX_SEQUENCE_LENGTH]();
    for (int b = 0; b < std::min(BATCH_SIZE, (int)tokenized.size()); b++) {
        for (int s = 0; s < std::min(MAX_SEQUENCE_LENGTH, (int)tokenized[b].size()); s++) {
            h_token_ids[b * MAX_SEQUENCE_LENGTH + s] = tokenized[b][s];
        }
    }
    
    // Initialize embeddings
    float* h_embeddings = new float[MAX_VOCAB_SIZE * EMBEDDING_DIM];
    initializeEmbeddings(h_embeddings, tokenizer.vocab_size, EMBEDDING_DIM);
    
    // Initialize layer norm parameters
    float* h_gamma = new float[EMBEDDING_DIM];
    float* h_beta = new float[EMBEDDING_DIM];
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    // Allocate output buffers
    int output_size = BATCH_SIZE * MAX_SEQUENCE_LENGTH * EMBEDDING_DIM;
    float* h_output_cpu = new float[output_size];
    float* h_output_gpu = new float[output_size];
    float* h_output_norm = new float[output_size];
    
    // ==================== CPU BASELINE ====================
    printf("Running CPU implementation...\n");
    Timer cpu_timer;
    
    // Embedding lookup
    embeddingLookupCPU(h_embeddings, h_token_ids, h_output_cpu,
                      BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, tokenizer.vocab_size);
    
    // Positional encoding
    positionalEncodingCPU(h_output_cpu, BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    // Layer normalization
    layerNormCPU(h_output_cpu, h_output_norm, h_gamma, h_beta,
                BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    float cpu_time = cpu_timer.elapsed();
    printf("CPU time: %.2f ms\n\n", cpu_time);
    
    // ==================== GPU IMPLEMENTATION ====================
    
    // Allocate GPU memory
    int* d_token_ids;
    float *d_embeddings, *d_output, *d_gamma, *d_beta, *d_output_norm;
    
    cudaMalloc(&d_token_ids, BATCH_SIZE * MAX_SEQUENCE_LENGTH * sizeof(int));
    cudaMalloc(&d_embeddings, MAX_VOCAB_SIZE * EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_gamma, EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_beta, EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_output_norm, output_size * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_token_ids, h_token_ids, BATCH_SIZE * MAX_SEQUENCE_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embeddings, h_embeddings, MAX_VOCAB_SIZE * EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernels
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    printf("Running GPU implementation (separate kernels)...\n");
    Timer gpu_timer;
    
    // Embedding lookup
    embeddingLookupGPU<<<blocks, threads_per_block>>>(
        d_embeddings, d_token_ids, d_output,
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, tokenizer.vocab_size);
    
    // Positional encoding
    positionalEncodingGPU<<<blocks, threads_per_block>>>(
        d_output, BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    // Layer normalization
    int norm_blocks = BATCH_SIZE * MAX_SEQUENCE_LENGTH;
    int shared_mem_size = threads_per_block * sizeof(float);
    layerNormGPU<<<norm_blocks, threads_per_block, shared_mem_size>>>(
        d_output, d_output_norm, d_gamma, d_beta,
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    cudaDeviceSynchronize();
    float gpu_time = gpu_timer.elapsed();
    printf("GPU time (separate kernels): %.2f ms\n", gpu_time);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time);
    
    // ==================== FUSED KERNEL ====================
    printf("Running GPU implementation (fused kernel)...\n");
    Timer fused_timer;
    
    fusedEmbeddingPositionalGPU<<<blocks, threads_per_block>>>(
        d_embeddings, d_token_ids, d_output,
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, tokenizer.vocab_size);
    
    layerNormGPU<<<norm_blocks, threads_per_block, shared_mem_size>>>(
        d_output, d_output_norm, d_gamma, d_beta,
        BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM);
    
    cudaDeviceSynchronize();
    float fused_time = fused_timer.elapsed();
    printf("GPU time (fused kernel): %.2f ms\n", fused_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / fused_time);
    printf("Speedup vs separate: %.2fx\n\n", gpu_time / fused_time);
    
    // Verify correctness (sample check)
    cudaMemcpy(h_output_gpu, d_output_norm, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < 1000; i++) {  // Check first 1000 elements
        float diff = fabsf(h_output_norm[i] - h_output_gpu[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    printf("Maximum difference CPU vs GPU: %e\n", max_diff);
    
    // Performance analysis
    printf("\n====================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("====================================\n");
    
    size_t total_bytes = BATCH_SIZE * MAX_SEQUENCE_LENGTH * (sizeof(int) + EMBEDDING_DIM * sizeof(float));
    float bandwidth_gbps = (total_bytes / 1e9) / (fused_time / 1000.0f);
    
    printf("Total data processed: %.2f MB\n", total_bytes / 1e6);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Tokens per second: %.0f\n", (BATCH_SIZE * MAX_SEQUENCE_LENGTH) / (fused_time / 1000.0f));
    
    // Cleanup
    delete[] h_token_ids;
    delete[] h_embeddings;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_output_norm;
    
    cudaFree(d_token_ids);
    cudaFree(d_embeddings);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output_norm);
    
    printf("\n====================================\n");
    printf("KEY INSIGHTS\n");
    printf("====================================\n");
    printf("1. Embedding lookup is memory-bound (random access)\n");
    printf("2. Positional encoding is compute-bound (transcendental functions)\n");
    printf("3. Layer normalization benefits from shared memory\n");
    printf("4. Fused kernels reduce memory traffic significantly\n");
    printf("5. Real transformers process millions of tokens/sec on GPU\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why is embedding lookup memory-bound? Calculate theoretical bandwidth.
 * 2. How does positional encoding preserve word order information?
 * 3. Why is layer normalization critical for transformer stability?
 * 4. Calculate FLOPs for each operation. Which is most compute-intensive?
 *
 * === Coding ===
 * 5. Implement sub-word tokenization (BPE-style)
 * 6. Add dropout to embeddings (training mode)
 * 7. Implement learned positional embeddings
 * 8. Create a vocabulary from a text corpus
 * 9. Implement embedding matrix factorization for compression
 *
 * === Optimization ===
 * 10. Use texture memory for embedding lookup (cache optimization)
 * 11. Implement mixed-precision (FP16) embeddings
 * 12. Create a kernel for batch matrix multiplication
 * 13. Optimize for different sequence lengths (padding overhead)
 * 14. Implement gradient computation for embeddings
 *
 * === Advanced ===
 * 15. Build a simple attention mechanism on top
 * 16. Implement cross-entropy loss computation
 * 17. Create embeddings for multiple modalities (text + position + segment)
 * 18. Implement KV-cache for autoregressive generation
 * 19. Build a simple BERT-style masked language model
 * 20. Implement Flash Attention's IO-aware algorithm
 *
 * === Production ===
 * 21. Handle variable-length sequences efficiently
 * 22. Implement model parallelism for huge vocabularies
 * 23. Create ONNX export for deployment
 * 24. Build REST API for embedding service
 * 25. Implement embedding similarity search (cosine distance)
 */

/*
 * MENTAL MODELS:
 * 
 * 1. Embeddings = "Dense representation of sparse data"
 *    - Tokens are sparse (one-hot)
 *    - Embeddings are dense (learned features)
 *    
 * 2. Think of it as a massive lookup table:
 *    - 50K words × 768 dimensions = 38M parameters just for embeddings!
 *    
 * 3. GPU advantage:
 *    - Batch processing: Look up 1000s of embeddings in parallel
 *    - Matrix operations: Perfect for transforming embeddings
 *    
 * 4. Memory hierarchy matters:
 *    - Embedding table in global memory (too big for shared)
 *    - Layer norm benefits from shared memory (local statistics)
 *    - Fused kernels reduce trips to global memory
 */
