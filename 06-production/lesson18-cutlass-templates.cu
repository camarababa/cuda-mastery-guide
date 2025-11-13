/*
 * Lesson 18: CUTLASS & Template Metaprogramming
 * Building Reusable, Efficient CUDA Kernels
 *
 * CUTLASS = CUDA Templates for Linear Algebra Subroutines
 * Learn to build kernels that are both generic AND fast!
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <type_traits>
#include <chrono>

// Note: Full CUTLASS requires separate installation
// This lesson demonstrates the core concepts

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Templates?
// =====================================================

/*
 * THE PROBLEM:
 * 
 * You need matrix multiply for:
 * - Different types: FP32, FP16, INT8
 * - Different sizes: 128x128, 256x512, etc.
 * - Different layouts: row-major, column-major
 * 
 * Without templates: Write 100s of kernels!
 * With templates: Write once, specialize at compile time
 * 
 * CUTLASS PHILOSOPHY:
 * - Zero runtime overhead
 * - Compile-time optimization
 * - Composable building blocks
 * - Production performance
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
// PART 2: BASIC TEMPLATE CONCEPTS
// =====================================================

// Generic vector add kernel
template<typename T>
__global__ void vectorAdd(T *c, const T *a, const T *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Specialized for half precision with conversion
template<>
__global__ void vectorAdd<__half>(__half *c, const __half *a, 
                                  const __half *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Special handling for half
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// =====================================================
// PART 3: COMPILE-TIME CONFIGURATION
// =====================================================

// Configuration structure
template<int BlockM_, int BlockN_, int BlockK_, 
         int WarpM_, int WarpN_, int WarpK_,
         typename Element_>
struct GemmConfig {
    static constexpr int BlockM = BlockM_;
    static constexpr int BlockN = BlockN_;
    static constexpr int BlockK = BlockK_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int WarpK = WarpK_;
    using Element = Element_;
    
    // Derived values
    static constexpr int ThreadsPerBlock = 
        (BlockM / WarpM) * (BlockN / WarpN) * 32;
    static constexpr int SharedMemorySize = 
        (BlockM * BlockK + BlockK * BlockN) * sizeof(Element);
};

// Matrix layout traits
enum class MatrixLayout {
    RowMajor,
    ColumnMajor
};

template<MatrixLayout Layout>
struct LayoutTraits {
    template<typename T>
    __device__ static T& at(T* ptr, int row, int col, int ld);
};

template<>
struct LayoutTraits<MatrixLayout::RowMajor> {
    template<typename T>
    __device__ static T& at(T* ptr, int row, int col, int ld) {
        return ptr[row * ld + col];
    }
};

template<>
struct LayoutTraits<MatrixLayout::ColumnMajor> {
    template<typename T>
    __device__ static T& at(T* ptr, int row, int col, int ld) {
        return ptr[col * ld + row];
    }
};

// =====================================================
// PART 4: BUILDING BLOCKS - TILES
// =====================================================

// Fragment - register-level tile
template<typename Element, int M, int N>
struct Fragment {
    Element data[M][N];
    
    __device__ void fill(Element val) {
        #pragma unroll
        for (int i = 0; i < M; i++) {
            #pragma unroll
            for (int j = 0; j < N; j++) {
                data[i][j] = val;
            }
        }
    }
    
    __device__ Element& operator()(int i, int j) {
        return data[i][j];
    }
};

// Tile loader - loads from global to shared memory
template<typename Config, MatrixLayout Layout>
class TileLoader {
private:
    using Element = typename Config::Element;
    
public:
    __device__ static void load_tile(
        Element* shared_tile,
        const Element* global_ptr,
        int tile_row, int tile_col,
        int ld, int M, int N) {
        
        // Collaborative loading by all threads
        int tid = threadIdx.x;
        int stride = Config::ThreadsPerBlock;
        
        #pragma unroll
        for (int idx = tid; idx < Config::BlockM * Config::BlockK; 
             idx += stride) {
            int row = idx / Config::BlockK;
            int col = idx % Config::BlockK;
            
            int global_row = tile_row + row;
            int global_col = tile_col + col;
            
            if (global_row < M && global_col < N) {
                shared_tile[row * Config::BlockK + col] = 
                    LayoutTraits<Layout>::at(
                        const_cast<Element*>(global_ptr), 
                        global_row, global_col, ld);
            } else {
                shared_tile[row * Config::BlockK + col] = Element(0);
            }
        }
    }
};

// =====================================================
// PART 5: OPTIMIZED GEMM KERNEL
// =====================================================

template<typename Config, 
         MatrixLayout LayoutA,
         MatrixLayout LayoutB,
         MatrixLayout LayoutC>
__global__ void gemmTemplate(
    typename Config::Element* C,
    const typename Config::Element* A,
    const typename Config::Element* B,
    int M, int N, int K,
    typename Config::Element alpha,
    typename Config::Element beta) {
    
    using Element = typename Config::Element;
    
    // Shared memory tiles
    __shared__ Element As[Config::BlockM][Config::BlockK];
    __shared__ Element Bs[Config::BlockK][Config::BlockN];
    
    // Thread's position in thread block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Warp's tile position
    int warp_row = warp_id / (Config::BlockN / Config::WarpN);
    int warp_col = warp_id % (Config::BlockN / Config::WarpN);
    
    // Thread's position within warp tile
    int thread_row = lane_id / (Config::WarpN / 4);
    int thread_col = lane_id % (Config::WarpN / 4) * 4;
    
    // Register accumulator
    Fragment<Element, 4, 4> acc;
    acc.fill(Element(0));
    
    // Main loop over K dimension
    for (int k = 0; k < K; k += Config::BlockK) {
        // Load tiles to shared memory
        TileLoader<Config, LayoutA>::load_tile(
            &As[0][0], A,
            blockIdx.y * Config::BlockM, k,
            K, M, K);
            
        TileLoader<Config, LayoutB>::load_tile(
            &Bs[0][0], B,
            k, blockIdx.x * Config::BlockN,
            N, K, N);
            
        __syncthreads();
        
        // Compute on shared memory tiles
        #pragma unroll
        for (int ki = 0; ki < Config::BlockK; ki++) {
            // Load warp tile from shared to registers
            Fragment<Element, 4, 1> a_frag;
            Fragment<Element, 1, 4> b_frag;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag(i, 0) = As[warp_row * Config::WarpM + 
                                 thread_row * 4 + i][ki];
            }
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                b_frag(0, j) = Bs[ki][warp_col * Config::WarpN + 
                                     thread_col + j];
            }
            
            // Outer product
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc(i, j) += a_frag(i, 0) * b_frag(0, j);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    int c_row = blockIdx.y * Config::BlockM + warp_row * Config::WarpM + 
                thread_row * 4;
    int c_col = blockIdx.x * Config::BlockN + warp_col * Config::WarpN + 
                thread_col;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_row + i < M && c_col + j < N) {
                Element& out = LayoutTraits<LayoutC>::at(
                    C, c_row + i, c_col + j, N);
                out = alpha * acc(i, j) + beta * out;
            }
        }
    }
}

// =====================================================
// PART 6: TYPE TRAITS & SFINAE
// =====================================================

// Type traits for numeric types
template<typename T>
struct NumericTraits {
    using type = T;
    static constexpr bool is_integer = false;
    static constexpr bool is_complex = false;
    __device__ static T zero() { return T(0); }
    __device__ static T one() { return T(1); }
};

template<>
struct NumericTraits<int> {
    using type = int;
    static constexpr bool is_integer = true;
    static constexpr bool is_complex = false;
    __device__ static int zero() { return 0; }
    __device__ static int one() { return 1; }
};

// SFINAE for conditional compilation
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
printType() {
    printf("Floating point type\n");
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
printType() {
    printf("Integer type\n");
}

// =====================================================
// PART 7: EPILOGUE FUNCTORS
// =====================================================

// Base epilogue
template<typename Element>
struct LinearCombination {
    Element alpha, beta;
    
    __device__ Element operator()(Element acc, Element src) const {
        return alpha * acc + beta * src;
    }
};

// Specialized epilogue with bias and ReLU
template<typename Element>
struct BiasReLU {
    const Element* bias;
    
    __device__ Element operator()(Element acc, Element src, int idx) const {
        Element result = acc + bias[idx];
        return (result > Element(0)) ? result : Element(0);
    }
};

// =====================================================
// PART 8: CUTLASS-STYLE API
// =====================================================

// Simplified CUTLASS-style GEMM class
template<
    typename Element_,
    typename Layout_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename EpilogueOp_
>
class Gemm {
public:
    using Element = Element_;
    using Layout = Layout_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOp = EpilogueOp_;
    
    struct Arguments {
        Element const *A;
        Element const *B;
        Element *C;
        int M, N, K;
        Element alpha, beta;
        
        Arguments(Element const *A_, Element const *B_, Element *C_,
                 int M_, int N_, int K_, 
                 Element alpha_ = Element(1),
                 Element beta_ = Element(0))
            : A(A_), B(B_), C(C_), M(M_), N(N_), K(K_), 
              alpha(alpha_), beta(beta_) {}
    };
    
    // Kernel launch
    void operator()(Arguments const &args, cudaStream_t stream = nullptr) {
        // Calculate grid dimensions
        dim3 grid(
            (args.N + ThreadblockShape::kN - 1) / ThreadblockShape::kN,
            (args.M + ThreadblockShape::kM - 1) / ThreadblockShape::kM
        );
        
        dim3 block(ThreadblockShape::kThreads);
        
        // Launch kernel
        // In real CUTLASS, this would dispatch to optimized kernels
        printf("Launching GEMM: %dx%dx%d\n", args.M, args.N, args.K);
    }
};

// Shape definitions (like CUTLASS)
template<int M, int N, int K>
struct Shape {
    static constexpr int kM = M;
    static constexpr int kN = N;
    static constexpr int kK = K;
    static constexpr int kThreads = 256;  // Simplified
};

// =====================================================
// PART 9: MAIN - DEMONSTRATION
// =====================================================

template<typename T>
void testTemplateGemm(int M, int N, int K) {
    printf("\n=== Template GEMM Test (%s) ===\n", typeid(T).name());
    printf("Matrix dimensions: %dx%dx%d\n", M, N, K);
    
    // Allocate matrices
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Initialize
    cudaMemset(d_A, 1, size_A);
    cudaMemset(d_B, 2, size_B);
    cudaMemset(d_C, 0, size_C);
    
    // Configure kernel
    using Config = GemmConfig<64, 64, 8, 32, 32, 8, T>;
    
    dim3 grid((N + Config::BlockN - 1) / Config::BlockN,
              (M + Config::BlockM - 1) / Config::BlockM);
    dim3 block(Config::ThreadsPerBlock);
    
    printf("Grid: %dx%d, Block: %d threads\n", 
           grid.x, grid.y, block.x);
    printf("Shared memory: %zu bytes\n", Config::SharedMemorySize);
    
    // Warmup
    gemmTemplate<Config, MatrixLayout::RowMajor, 
                MatrixLayout::RowMajor, MatrixLayout::RowMajor>
        <<<grid, block>>>(d_C, d_A, d_B, M, N, K, T(1), T(0));
    
    // Benchmark
    Timer timer;
    int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        gemmTemplate<Config, MatrixLayout::RowMajor,
                    MatrixLayout::RowMajor, MatrixLayout::RowMajor>
            <<<grid, block>>>(d_C, d_A, d_B, M, N, K, T(1), T(0));
    }
    
    cudaDeviceSynchronize();
    float time = timer.elapsed() / iterations;
    
    float gflops = (2.0f * M * N * K) / (time * 1e6);
    printf("Time: %.2f ms\n", time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    printf("==================================================\n");
    printf("CUTLASS & TEMPLATE METAPROGRAMMING\n");
    printf("==================================================\n");
    
    // Test type traits
    printf("\n=== Type Traits Demo ===\n");
    printType<float>();
    printType<int>();
    
    // Test different types
    testTemplateGemm<float>(512, 512, 512);
    testTemplateGemm<double>(512, 512, 512);
    
    // Demonstrate CUTLASS-style API
    printf("\n=== CUTLASS-Style API ===\n");
    using MyGemm = Gemm<
        float,
        MatrixLayout::RowMajor,
        Shape<128, 128, 32>,    // Threadblock
        Shape<64, 64, 32>,      // Warp
        Shape<8, 8, 4>,         // Instruction
        LinearCombination<float>
    >;
    
    MyGemm gemm_op;
    MyGemm::Arguments args(nullptr, nullptr, nullptr, 
                          1024, 1024, 1024);
    gemm_op(args);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Templates enable code reuse without overhead\n");
    printf("2. Compile-time specialization beats runtime checks\n");
    printf("3. CUTLASS provides building blocks for kernels\n");
    printf("4. Type traits enable generic programming\n");
    printf("5. Composition > inheritance for GPU code\n");
    printf("6. Modern C++ features improve GPU code\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why compile-time vs runtime configuration?
 * 2. Calculate register usage for different configs
 * 3. How do templates affect binary size?
 * 4. Compare template vs virtual function overhead
 * 5. When to use CUTLASS vs cuBLAS?
 *
 * === Implementation ===
 * 6. Add FP16 specialization with tensor cores
 * 7. Create epilogue with GELU activation
 * 8. Implement strided batched GEMM
 * 9. Add support for complex numbers
 * 10. Create auto-tuning framework
 *
 * === Optimization ===
 * 11. Implement double buffering in shared memory
 * 12. Add software pipelining
 * 13. Optimize for different tile sizes
 * 14. Implement split-K GEMM
 * 15. Create architecture-specific variants
 *
 * === Advanced ===
 * 16. Build convolution using CUTLASS
 * 17. Implement attention with templates
 * 18. Create custom collective operations
 * 19. Build sparse GEMM templates
 * 20. Design new epilogue functors
 *
 * === Production ===
 * 21. Integrate with PyTorch/TensorFlow
 * 22. Create JIT compilation system
 * 23. Build profiling decorators
 * 24. Implement kernel fusion
 * 25. Create DSL for kernel generation
 */

/*
 * MENTAL MODELS:
 *
 * 1. "LEGO Blocks"
 *    - Each template is a building block
 *    - Snap together at compile time
 *    - No runtime overhead
 *
 * 2. "Recipe vs Cooking"
 *    - Template: Recipe (instructions)
 *    - Instantiation: Cooking (actual kernel)
 *    - Specialization: Recipe variations
 *
 * 3. "Compile-Time Factory"
 *    - Types flow through templates
 *    - Compiler generates optimal code
 *    - Zero abstraction penalty
 *
 * 4. CUTLASS Philosophy:
 *    - Performance of assembly
 *    - Productivity of high-level code
 *    - Composition over configuration
 */
