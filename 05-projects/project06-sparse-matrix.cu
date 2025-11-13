/*
 * PROJECT 6: SPARSE MATRIX OPERATIONS
 * Optimizing for Sparsity in Scientific Computing
 *
 * Most real-world matrices are sparse (>90% zeros).
 * This project teaches you to exploit sparsity for massive speedups.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Sparse Matters
// =====================================================

/*
 * SPARSE MATRICES IN THE WILD:
 * 
 * - Social networks: 1M users, ~100 friends each = 99.99% sparse
 * - Web graphs: Billions of pages, few links each
 * - Scientific simulations: PDEs create banded matrices
 * - Neural networks: Pruned models are sparse
 * 
 * DENSE APPROACH WASTES:
 * - Memory: Store millions of zeros
 * - Compute: Multiply by zero
 * - Bandwidth: Move zeros around
 * 
 * SPARSE APPROACH:
 * - Store only non-zeros
 * - Compute only what matters
 * - 10-1000x speedups possible!
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
// PART 2: SPARSE MATRIX FORMATS
// =====================================================

// COO (Coordinate) Format - simplest
struct COOMatrix {
    int rows, cols, nnz;
    int *row_indices;    // Size: nnz
    int *col_indices;    // Size: nnz  
    float *values;       // Size: nnz
    
    void allocate(int r, int c, int n) {
        rows = r; cols = c; nnz = n;
        cudaMalloc(&row_indices, nnz * sizeof(int));
        cudaMalloc(&col_indices, nnz * sizeof(int));
        cudaMalloc(&values, nnz * sizeof(float));
    }
    
    void free() {
        cudaFree(row_indices);
        cudaFree(col_indices);
        cudaFree(values);
    }
};

// CSR (Compressed Sparse Row) Format - most common
struct CSRMatrix {
    int rows, cols, nnz;
    int *row_offsets;    // Size: rows + 1
    int *col_indices;    // Size: nnz
    float *values;       // Size: nnz
    
    void allocate(int r, int c, int n) {
        rows = r; cols = c; nnz = n;
        cudaMalloc(&row_offsets, (rows + 1) * sizeof(int));
        cudaMalloc(&col_indices, nnz * sizeof(int));
        cudaMalloc(&values, nnz * sizeof(float));
    }
    
    void free() {
        cudaFree(row_offsets);
        cudaFree(col_indices);
        cudaFree(values);
    }
};

// ELL (ELLPACK) Format - GPU-friendly  
struct ELLMatrix {
    int rows, cols, nnz;
    int max_row_nnz;     // Maximum non-zeros per row
    int *col_indices;    // Size: rows * max_row_nnz
    float *values;       // Size: rows * max_row_nnz
    
    void allocate(int r, int c, int n, int max_nnz) {
        rows = r; cols = c; nnz = n; max_row_nnz = max_nnz;
        cudaMalloc(&col_indices, rows * max_row_nnz * sizeof(int));
        cudaMalloc(&values, rows * max_row_nnz * sizeof(float));
        
        // Initialize with -1 for padding
        cudaMemset(col_indices, -1, rows * max_row_nnz * sizeof(int));
    }
    
    void free() {
        cudaFree(col_indices);
        cudaFree(values);
    }
};

// =====================================================
// PART 3: SPARSE MATRIX-VECTOR MULTIPLICATION (SpMV)
// =====================================================

// Dense SpMV for comparison
__global__ void spMVDense(float *y, const float *A, const float *x, 
                         int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

// COO SpMV - simple but inefficient
__global__ void spMVCOO(float *y, const COOMatrix mat, const float *x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < mat.nnz) {
        int row = mat.row_indices[tid];
        int col = mat.col_indices[tid];
        float val = mat.values[tid];
        
        // Atomic add (race condition!)
        atomicAdd(&y[row], val * x[col]);
    }
}

// CSR SpMV - one thread per row
__global__ void spMVCSR(float *y, const CSRMatrix mat, const float *x) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < mat.rows) {
        float sum = 0.0f;
        
        // Process this row's non-zeros
        for (int idx = mat.row_offsets[row]; idx < mat.row_offsets[row + 1]; idx++) {
            int col = mat.col_indices[idx];
            float val = mat.values[idx];
            sum += val * x[col];
        }
        
        y[row] = sum;
    }
}

// CSR SpMV - vector (warp) per row for better load balancing
__global__ void spMVCSRVector(float *y, const CSRMatrix mat, const float *x) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    
    if (row < mat.rows) {
        __shared__ float shared_sum[32];
        
        float sum = 0.0f;
        int row_start = mat.row_offsets[row];
        int row_end = mat.row_offsets[row + 1];
        
        // Warp processes row in parallel
        for (int idx = row_start + lane; idx < row_end; idx += 32) {
            int col = mat.col_indices[idx];
            float val = mat.values[idx];
            sum += val * x[col];
        }
        
        // Warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

// ELL SpMV - coalesced memory access
__global__ void spMVELL(float *y, const ELLMatrix mat, const float *x) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < mat.rows) {
        float sum = 0.0f;
        
        // Process this row's non-zeros
        for (int i = 0; i < mat.max_row_nnz; i++) {
            int idx = row + i * mat.rows;  // Column-major storage
            int col = mat.col_indices[idx];
            
            if (col != -1) {  // Valid entry
                float val = mat.values[idx];
                sum += val * x[col];
            }
        }
        
        y[row] = sum;
    }
}

// =====================================================
// PART 4: FORMAT CONVERSION
// =====================================================

// Convert COO to CSR on GPU
__global__ void countRowsCOO(int *row_counts, const int *row_indices, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&row_counts[row_indices[tid]], 1);
    }
}

void convertCOOtoCSR(CSRMatrix &csr, const COOMatrix &coo) {
    // Allocate CSR
    csr.allocate(coo.rows, coo.cols, coo.nnz);
    
    // Count elements per row
    cudaMemset(csr.row_offsets, 0, (csr.rows + 1) * sizeof(int));
    
    int threads = 256;
    int blocks = (coo.nnz + threads - 1) / threads;
    countRowsCOO<<<blocks, threads>>>(csr.row_offsets + 1, coo.row_indices, coo.nnz);
    
    // Exclusive scan to get offsets
    for (int i = 1; i <= csr.rows; i++) {
        int count;
        cudaMemcpy(&count, &csr.row_offsets[i], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&csr.row_offsets[i], &csr.row_offsets[i-1], sizeof(int), cudaMemcpyDeviceToDevice);
        
        int prev;
        cudaMemcpy(&prev, &csr.row_offsets[i], sizeof(int), cudaMemcpyDeviceToHost);
        prev += count;
        cudaMemcpy(&csr.row_offsets[i], &prev, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    // Fill CSR arrays
    // (Simplified - in practice use sorting and parallel algorithms)
}

// =====================================================
// PART 5: SPARSE MATRIX-MATRIX MULTIPLICATION
// =====================================================

// Symbolic multiplication - determine result structure
__global__ void spGEMMSymbolic(int *C_row_offsets, const CSRMatrix A, 
                              const CSRMatrix B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A.rows) {
        // Use hash table or sorting to find unique column indices
        // This is simplified - real implementation is complex
        int nnz_count = 0;
        
        // For each non-zero in row of A
        for (int idx_a = A.row_offsets[row]; idx_a < A.row_offsets[row + 1]; idx_a++) {
            int col_a = A.col_indices[idx_a];
            
            // For each non-zero in corresponding row of B
            for (int idx_b = B.row_offsets[col_a]; idx_b < B.row_offsets[col_a + 1]; idx_b++) {
                nnz_count++;  // Over-counting (need unique)
            }
        }
        
        C_row_offsets[row + 1] = nnz_count;
    }
}

// =====================================================
// PART 6: SPARSE PATTERNS & OPTIMIZATIONS
// =====================================================

// Analyze sparsity pattern
__global__ void analyzePattern(int *row_lengths, int *bandwidth,
                             const CSRMatrix mat) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < mat.rows) {
        int row_nnz = mat.row_offsets[row + 1] - mat.row_offsets[row];
        row_lengths[row] = row_nnz;
        
        // Find bandwidth (max column distance from diagonal)
        int max_dist = 0;
        for (int idx = mat.row_offsets[row]; idx < mat.row_offsets[row + 1]; idx++) {
            int col = mat.col_indices[idx];
            int dist = abs(col - row);
            max_dist = max(max_dist, dist);
        }
        
        atomicMax(bandwidth, max_dist);
    }
}

// Hybrid format - use different formats for different parts
struct HybridMatrix {
    ELLMatrix ell_part;      // Regular rows
    COOMatrix coo_part;      // Irregular rows
    int ell_threshold;       // Max non-zeros for ELL
    
    void partition(const CSRMatrix &csr, int threshold) {
        ell_threshold = threshold;
        
        // Count regular vs irregular elements
        // ... implementation ...
    }
};

// =====================================================
// PART 7: CUSPARSE INTEGRATION
// =====================================================

void demonstrateCuSparse(int rows, int cols, float sparsity) {
    printf("\n=== cuSPARSE Performance ===\n");
    
    // Create random sparse matrix
    int nnz = rows * cols * (1.0f - sparsity);
    
    // Allocate host memory
    std::vector<int> h_row_offsets(rows + 1);
    std::vector<int> h_col_indices(nnz);
    std::vector<float> h_values(nnz);
    std::vector<float> h_x(cols);
    std::vector<float> h_y(rows);
    
    // Generate random sparse matrix in CSR format
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    int current_nnz = 0;
    for (int i = 0; i < rows; i++) {
        h_row_offsets[i] = current_nnz;
        
        for (int j = 0; j < cols; j++) {
            if (dist(gen) > sparsity && current_nnz < nnz) {
                h_col_indices[current_nnz] = j;
                h_values[current_nnz] = dist(gen);
                current_nnz++;
            }
        }
    }
    h_row_offsets[rows] = current_nnz;
    nnz = current_nnz;  // Actual nnz
    
    // Initialize vector
    for (int i = 0; i < cols; i++) {
        h_x[i] = 1.0f;
    }
    
    // Allocate device memory
    CSRMatrix d_csr;
    d_csr.allocate(rows, cols, nnz);
    float *d_x, *d_y;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_csr.row_offsets, h_row_offsets.data(), 
              (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_indices, h_col_indices.data(), 
              nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.values, h_values.data(), 
              nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // cuSPARSE setup
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;
    
    cusparseCreateCsr(&matA, rows, cols, nnz,
                     d_csr.row_offsets, d_csr.col_indices, d_csr.values,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);
    
    // Prepare SpMV
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                           CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    
    // Benchmark
    Timer timer;
    for (int i = 0; i < 100; i++) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();
    float time = timer.elapsed() / 100;
    
    printf("Matrix: %d x %d, sparsity: %.1f%%, nnz: %d\n", 
           rows, cols, sparsity * 100, nnz);
    printf("cuSPARSE SpMV: %.3f ms\n", time);
    printf("Effective bandwidth: %.2f GB/s\n",
           (nnz * (sizeof(int) + sizeof(float)) + 
            (rows + cols) * sizeof(float)) / (time * 1e6));
    
    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    
    d_csr.free();
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer);
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE TESTING
// =====================================================

void compareFormats(int rows, int cols, float sparsity) {
    printf("\n=== Format Comparison ===\n");
    printf("Matrix: %d x %d, sparsity: %.1f%%\n", 
           rows, cols, sparsity * 100);
    
    // Generate test data
    int nnz = rows * cols * (1.0f - sparsity);
    
    // Allocate formats
    COOMatrix coo;
    CSRMatrix csr;
    ELLMatrix ell;
    
    coo.allocate(rows, cols, nnz);
    csr.allocate(rows, cols, nnz);
    
    // Generate random sparse matrix
    std::vector<int> h_rows, h_cols;
    std::vector<float> h_vals;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    int max_row_nnz = 0;
    std::vector<int> row_nnz(rows, 0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dist(gen) > sparsity) {
                h_rows.push_back(i);
                h_cols.push_back(j);
                h_vals.push_back(dist(gen));
                row_nnz[i]++;
            }
        }
        max_row_nnz = std::max(max_row_nnz, row_nnz[i]);
    }
    
    nnz = h_rows.size();
    ell.allocate(rows, cols, nnz, max_row_nnz);
    
    printf("Actual nnz: %d, max row nnz: %d\n", nnz, max_row_nnz);
    
    // Copy COO to device
    cudaMemcpy(coo.row_indices, h_rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(coo.col_indices, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(coo.values, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate vectors
    float *d_x, *d_y;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));
    
    // Initialize input vector
    cudaMemset(d_x, 1.0f, cols * sizeof(float));
    
    // Benchmark each format
    int iterations = 100;
    dim3 block(256);
    dim3 grid((rows + block.x - 1) / block.x);
    
    // COO
    cudaMemset(d_y, 0, rows * sizeof(float));
    Timer coo_timer;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));  // Reset for atomics
        spMVCOO<<<(nnz + block.x - 1) / block.x, block>>>(d_y, coo, d_x);
    }
    cudaDeviceSynchronize();
    float coo_time = coo_timer.elapsed() / iterations;
    
    printf("\nSpMV Performance:\n");
    printf("COO format: %.3f ms\n", coo_time);
    
    // Would also test CSR, ELL, HYB formats...
    
    // Cleanup
    coo.free();
    csr.free();
    ell.free();
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    printf("==================================================\n");
    printf("SPARSE MATRIX OPERATIONS\n");
    printf("==================================================\n");
    
    // Test different scenarios
    struct TestCase {
        int rows, cols;
        float sparsity;
        const char* description;
    };
    
    TestCase cases[] = {
        {1000, 1000, 0.9f, "Small, moderately sparse"},
        {1000, 1000, 0.99f, "Small, very sparse"},
        {10000, 10000, 0.99f, "Medium, very sparse"},
        {10000, 10000, 0.999f, "Medium, extremely sparse"},
    };
    
    for (const auto& test : cases) {
        printf("\n==================================================\n");
        printf("Test: %s\n", test.description);
        
        compareFormats(test.rows, test.cols, test.sparsity);
        demonstrateCuSparse(test.rows, test.cols, test.sparsity);
    }
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Sparse formats save memory & compute\n");
    printf("2. Best format depends on sparsity pattern\n");
    printf("3. CSR good for general sparse matrices\n");
    printf("4. ELL excels with uniform sparsity\n");
    printf("5. COO simple but requires atomics\n");
    printf("6. cuSPARSE provides optimized implementations\n");
    printf("7. Real speedup comes at >90%% sparsity\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Calculate memory savings for 99% sparse matrix
 * 2. When does sparse beat dense computation?
 * 3. Compare format overhead vs benefits
 * 4. How does sparsity pattern affect performance?
 * 5. Why is SpMV memory bandwidth limited?
 *
 * === Implementation ===
 * 6. Implement CSC (column sparse) format
 * 7. Create blocked sparse format (BSR)
 * 8. Build sparse triangular solver
 * 9. Implement sparse QR factorization
 * 10. Create format auto-tuner
 *
 * === Optimization ===
 * 11. Optimize for power-law graphs
 * 12. Implement cache blocking for SpMV
 * 13. Create mixed-precision sparse ops
 * 14. Build sparse tensor operations
 * 15. Optimize for structured sparsity
 *
 * === Advanced ===
 * 16. Implement sparse neural networks
 * 17. Create sparse eigenvalue solver
 * 18. Build multigrid solver
 * 19. Implement sparse FFT
 * 20. Create graph neural network ops
 *
 * === Applications ===
 * 21. PageRank with sparse matrices
 * 22. Sparse linear system solver
 * 23. Compressed sensing recovery
 * 24. Sparse PCA/SVD
 * 25. Social network analysis
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Sparse as Compression"
 *    - Only store what matters
 *    - Trade format overhead for savings
 *    - Like video compression for matrices
 *
 * 2. "Format Zoo"
 *    - Each format has sweet spot
 *    - COO: Simple, flexible
 *    - CSR: General purpose
 *    - ELL: GPU-friendly
 *    - Choose wisely!
 *
 * 3. "Memory vs Compute"
 *    - Sparse is memory-bound
 *    - Bandwidth matters more than FLOPS
 *    - Irregular access patterns hurt
 *
 * 4. Real-World Patterns:
 *    - Banded: Physical simulations
 *    - Power-law: Social networks
 *    - Random: Machine learning
 *    - Block: Multi-physics
 */
