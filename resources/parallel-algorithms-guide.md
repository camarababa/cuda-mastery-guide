# CUDA Parallel Algorithms - Comprehensive Guide

## Table of Contents
1. [Reduction Algorithms](#reduction)
2. [Scan/Prefix Sum](#scan)
3. [Sorting Algorithms](#sorting)
4. [Matrix Operations](#matrix)
5. [Graph Algorithms](#graph)
6. [Image Processing](#image)
7. [Numerical Methods](#numerical)

---

## 1. Reduction Algorithms <a name="reduction"></a>

### 1.1 Parallel Sum Reduction

**Problem**: Sum N elements in parallel

**Sequential Complexity**: O(N)
**Parallel Complexity**: O(log N)

**Evolution of Implementation**:

#### Version 1: Naive (Interleaved Addressing)
```cuda
__global__ void reduce_v1(int *g_data, int *g_out, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();

    // Interleaved addressing (SLOW - divergent warps)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```
**Issues**: Divergent warps, bank conflicts

#### Version 2: Sequential Addressing
```cuda
__global__ void reduce_v2(int *g_data, int *g_out, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();

    // Sequential addressing (BETTER - no divergence in first warp)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```
**Performance**: ~2x faster than v1

#### Version 3: First Add During Load
```cuda
__global__ void reduce_v3(int *g_data, int *g_out, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load
    sdata[tid] = 0;
    if (i < n) sdata[tid] = g_data[i];
    if (i + blockDim.x < n) sdata[tid] += g_data[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```
**Performance**: Processes 2x elements per block

#### Version 4: Unroll Last Warp
```cuda
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4(int *g_data, int *g_out, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = 0;
    if (i < n) sdata[tid] = g_data[i];
    if (i + blockDim.x < n) sdata[tid] += g_data[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Unroll last warp (no syncthreads needed)
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```
**Performance**: ~40% faster by eliminating last 6 __syncthreads()

#### Version 5: Completely Unrolled
```cuda
template <unsigned int blockSize>
__global__ void reduce_v5(int *g_data, int *g_out, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    sdata[tid] = 0;
    if (i < n) sdata[tid] = g_data[i];
    if (i + blockSize < n) sdata[tid] += g_data[i + blockSize];
    __syncthreads();

    // Completely unrolled for blockSize
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```
**Performance**: Best performance, 8x faster than naive

### 1.2 Other Reduction Operations

**Min/Max Reduction**:
```cuda
__device__ void warpReduceMin(volatile float* sdata, int tid) {
    sdata[tid] = fminf(sdata[tid], sdata[tid + 32]);
    sdata[tid] = fminf(sdata[tid], sdata[tid + 16]);
    sdata[tid] = fminf(sdata[tid], sdata[tid + 8]);
    sdata[tid] = fminf(sdata[tid], sdata[tid + 4]);
    sdata[tid] = fminf(sdata[tid], sdata[tid + 2]);
    sdata[tid] = fminf(sdata[tid], sdata[tid + 1]);
}
```

**Custom Binary Operation**:
```cuda
template <typename T, typename BinaryOp>
__global__ void reduce_generic(T *g_data, T *g_out, int n, BinaryOp op) {
    extern __shared__ T sdata[];
    // ... reduction logic using op(a, b) instead of +
}
```

---

## 2. Scan/Prefix Sum <a name="scan"></a>

### 2.1 Inclusive Scan

**Problem**: Given array [a₀, a₁, a₂, ...], compute [a₀, a₀+a₁, a₀+a₁+a₂, ...]

**Applications**:
- Stream compaction
- Radix sort
- Quicksort partitioning
- Polynomial evaluation
- Resource allocation

### Naive Parallel Scan (Kogge-Stone)
```cuda
__global__ void scan_naive(int *g_data, int *g_out, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    temp[tid] = (tid < n) ? g_data[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pin;

        if (tid >= offset)
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
        else
            temp[pout * n + tid] = temp[pin * n + tid];

        __syncthreads();
    }

    g_out[tid] = temp[pout * n + tid];
}
```
**Complexity**: O(N log N) operations (not work-efficient!)

### 2.2 Work-Efficient Scan (Blelloch)

**Up-Sweep Phase** (Reduce):
```cuda
for (int d = 0; d < log2(n); d++) {
    for all k = 0 to n-1 by 2^(d+1) in parallel:
        x[k + 2^(d+1) - 1] += x[k + 2^d - 1]
}
```

**Down-Sweep Phase** (Distribute):
```cuda
x[n-1] = 0  // Set last element to 0
for (int d = log2(n) - 1; d >= 0; d--) {
    for all k = 0 to n-1 by 2^(d+1) in parallel:
        t = x[k + 2^d - 1]
        x[k + 2^d - 1] = x[k + 2^(d+1) - 1]
        x[k + 2^(d+1) - 1] += t
}
```

**Implementation**:
```cuda
__global__ void scan_blelloch(int *g_data, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int offset = 1;

    temp[2*tid] = g_data[2*tid];
    temp[2*tid+1] = g_data[2*tid+1];

    // Up-sweep (reduction) phase
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) temp[n - 1] = 0; // Clear last element

    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    g_data[2*tid] = temp[2*tid];
    g_data[2*tid+1] = temp[2*tid+1];
}
```
**Complexity**: O(N) operations (work-efficient!)

### 2.3 Multi-Block Scan

**Segmented Scan Strategy**:
1. Scan each block independently
2. Scan block sums
3. Add block sums to all elements

```cuda
// Phase 1: Scan each block
scan_block<<<blocks, threads>>>(data, block_sums, n);

// Phase 2: Scan the block sums
scan_block<<<1, blocks>>>(block_sums, NULL, blocks);

// Phase 3: Add scanned sums to each block
add_scanned_sums<<<blocks, threads>>>(data, block_sums, n);
```

---

## 3. Sorting Algorithms <a name="sorting"></a>

### 3.1 Bitonic Sort

**Best for**: Small-medium arrays, power-of-2 sizes
**Complexity**: O(log²N)

```cuda
__global__ void bitonic_sort_step(int *data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i^j;

    if (ixj > i) {
        int vi = data[i];
        int vixj = data[ixj];

        if ((i&k)==0) {
            if (vi > vixj) {
                data[i] = vixj;
                data[ixj] = vi;
            }
        } else {
            if (vi < vixj) {
                data[i] = vixj;
                data[ixj] = vi;
            }
        }
    }
}

void bitonic_sort(int *data, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k>>1; j > 0; j >>= 1) {
            bitonic_sort_step<<<n/256, 256>>>(data, j, k);
        }
    }
}
```

### 3.2 Radix Sort

**Best for**: Integer sorting, large arrays
**Complexity**: O(kN) where k = number of bits

**Algorithm**:
1. For each bit position (LSB to MSB):
   2. Count 0s and 1s
   3. Compute prefix sum of counts
   4. Scatter elements to new positions

```cuda
__global__ void radix_sort_pass(int *input, int *output, int bit, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_false[BLOCK_SIZE];
    __shared__ int s_true[BLOCK_SIZE];

    // Load and classify
    int value = (i < n) ? input[i] : 0;
    int bin = (value >> bit) & 1;

    // Scan for false (0s)
    s_false[tid] = !bin;
    __syncthreads();
    scan_block(s_false, BLOCK_SIZE);

    // Scan for true (1s)
    s_true[tid] = bin;
    __syncthreads();
    scan_block(s_true, BLOCK_SIZE);

    // Scatter
    int pos;
    if (bin == 0) {
        pos = s_false[tid];
    } else {
        int total_false = s_false[BLOCK_SIZE-1];
        pos = total_false + s_true[tid];
    }

    if (i < n) output[pos] = value;
}
```

### 3.3 Merge Sort

**Best for**: Stable sorting, linked structures
**Complexity**: O(N log N)

```cuda
__global__ void merge_sort(int *data, int *temp, int size, int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = 2 * width * tid;

    if (start >= size) return;

    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);

    merge(data, temp, start, mid, end);
}
```

---

## 4. Matrix Operations <a name="matrix"></a>

### 4.1 Matrix Multiplication (GEMM)

**Evolution of Optimizations**:

#### Version 1: Naive
```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
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
```
**Performance**: ~50 GFLOPS on modern GPU

#### Version 2: Shared Memory Tiling
```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < N && tile * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```
**Performance**: ~500 GFLOPS (10x improvement)

#### Version 3: Advanced Optimizations
- Register blocking
- Vectorized loads (float4)
- Loop unrolling
- Double buffering
- Warp-level tiling

**Performance**: ~2000+ GFLOPS (approaching cuBLAS)

### 4.2 Matrix Transpose

**Coalesced Transpose with Shared Memory**:
```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float *in, float *out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }

    __syncthreads();

    // Coalesced write to global memory (transposed)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width)
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
```

---

## 5. Graph Algorithms <a name="graph"></a>

### 5.1 Breadth-First Search (BFS)

```cuda
__global__ void bfs_kernel(int *nodes, int *edges, int *cost, bool *frontier,
                           bool *visited, bool *done, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes && frontier[tid]) {
        frontier[tid] = false;
        visited[tid] = true;

        int start = nodes[tid];
        int end = nodes[tid + 1];

        for (int i = start; i < end; i++) {
            int neighbor = edges[i];
            if (!visited[neighbor]) {
                cost[neighbor] = cost[tid] + 1;
                frontier[neighbor] = true;
                *done = false;
            }
        }
    }
}
```

### 5.2 Single-Source Shortest Path (Bellman-Ford)

```cuda
__global__ void bellman_ford(int *edges, int *weights, int *dist, bool *changed, int E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < E) {
        int u = edges[tid * 2];
        int v = edges[tid * 2 + 1];
        int w = weights[tid];

        int new_dist = dist[u] + w;
        if (new_dist < dist[v]) {
            atomicMin(&dist[v], new_dist);
            *changed = true;
        }
    }
}
```

---

## 6. Image Processing <a name="image"></a>

### 6.1 2D Convolution

```cuda
__global__ void convolution_2d(float *input, float *output, float *kernel,
                                int width, int height, int ksize) {
    __shared__ float tile[TILE_SIZE + KSIZE - 1][TILE_SIZE + KSIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int halo = ksize / 2;

    // Load tile with halo
    for (int i = ty; i < TILE_SIZE + 2*halo; i += blockDim.y) {
        for (int j = tx; j < TILE_SIZE + 2*halo; j += blockDim.x) {
            int gRow = blockIdx.y * TILE_SIZE + i - halo;
            int gCol = blockIdx.x * TILE_SIZE + j - halo;

            if (gRow >= 0 && gRow < height && gCol >= 0 && gCol < width)
                tile[i][j] = input[gRow * width + gCol];
            else
                tile[i][j] = 0.0f;
        }
    }

    __syncthreads();

    // Perform convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                sum += tile[ty + i][tx + j] * kernel[i * ksize + j];
            }
        }
        output[row * width + col] = sum;
    }
}
```

### 6.2 Histogram Equalization

```cuda
__global__ void histogram_kernel(unsigned char *image, int *hist, int size) {
    __shared__ int local_hist[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();

    // Compute local histogram
    if (gid < size) {
        atomicAdd(&local_hist[image[gid]], 1);
    }
    __syncthreads();

    // Accumulate to global histogram
    if (tid < 256) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}
```

---

## 7. Numerical Methods <a name="numerical"></a>

### 7.1 Finite Difference (Heat Equation)

```cuda
__global__ void heat_2d(float *u, float *u_new, int nx, int ny, float dt, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx-1 && j < ny-1) {
        int idx = i + j * nx;
        float alpha = dt / (dx * dx);

        u_new[idx] = u[idx] + alpha * (
            u[idx-1] + u[idx+1] + u[idx-nx] + u[idx+nx] - 4.0f * u[idx]
        );
    }
}
```

### 7.2 Monte Carlo Integration

```cuda
__global__ void monte_carlo_pi(float *results, curandState *states, int n_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState local_state = states[tid];
    int count = 0;

    for (int i = 0; i < n_samples; i++) {
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);

        if (x*x + y*y <= 1.0f) count++;
    }

    states[tid] = local_state;
    results[tid] = (float)count / n_samples * 4.0f;
}
```

---

## Performance Comparison Table

| Algorithm | Naive | Optimized | Speedup |
|-----------|-------|-----------|---------|
| Reduction | 100ms | 1.2ms | 83x |
| Scan | 80ms | 5ms | 16x |
| Sort (1M) | 150ms | 8ms | 19x |
| MatMul (1024²) | 5000ms | 12ms | 417x |
| Convolution | 200ms | 6ms | 33x |

**Note**: Times are approximate for modern high-end GPUs

---

## Key Takeaways

1. **Start Simple**: Always implement naive version first
2. **Use Shared Memory**: Critical for performance
3. **Avoid Divergence**: Branch uniformity matters
4. **Coalesce Access**: Memory patterns are crucial
5. **Profile Everything**: Measure, don't guess
6. **Use Libraries**: cuBLAS, Thrust often beat hand-coded
7. **Understand Patterns**: Many algorithms share common patterns

---

## Further Reading

- "Parallel Prefix Sum (Scan) with CUDA" - Mark Harris
- "Optimizing Parallel Reduction in CUDA" - Mark Harris
- "Thinking Parallel" series - NVIDIA
- GPU Gems series - NVIDIA

**Practice**: Implement each algorithm, profile it, optimize it!
