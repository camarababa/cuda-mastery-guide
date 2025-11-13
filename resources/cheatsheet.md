# CUDA Programming Cheat Sheet

## Function Qualifiers

```cuda
__global__    // Kernel: host→device, runs on device
__device__    // Device function: device→device
__host__      // Host function: host→host (default)
__host__ __device__  // Can run on both
```

## Memory Management

```cuda
// Allocate
cudaMalloc(&d_ptr, size);
cudaMallocManaged(&ptr, size);  // Unified memory

// Copy
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);

// Set
cudaMemset(d_ptr, value, size);

// Free
cudaFree(d_ptr);
```

## Kernel Launch

```cuda
// Basic
kernel<<<numBlocks, threadsPerBlock>>>(args);

// With shared memory
kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(args);

// With stream
kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(args);
```

## Thread Indexing

```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

## Built-in Variables

```cuda
gridDim       // Grid dimensions (blocks)
blockDim      // Block dimensions (threads)
blockIdx      // Block index within grid
threadIdx     // Thread index within block
warpSize      // Warp size (32 on all current GPUs)
```

## Synchronization

```cuda
__syncthreads();          // Block-level barrier
cudaDeviceSynchronize();  // Host waits for device
cudaStreamSynchronize(stream);  // Wait for stream
```

## Error Checking

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## Shared Memory

```cuda
// Static
__shared__ float cache[256];

// Dynamic
extern __shared__ float cache[];
// Launch: kernel<<<blocks, threads, sharedSize>>>()
```

## Atomic Operations

```cuda
atomicAdd(&addr, val);
atomicSub(&addr, val);
atomicExch(&addr, val);
atomicMin(&addr, val);
atomicMax(&addr, val);
atomicInc(&addr, val);
atomicDec(&addr, val);
atomicCAS(&addr, compare, val);
```

## Device Properties

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
printf("Warp Size: %d\n", prop.warpSize);
```

## Common Patterns

### Reduction (Sum)
```cuda
__shared__ float sdata[256];
sdata[tid] = data[idx];
__syncthreads();

for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

### Matrix Multiply (Naive)
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float sum = 0.0f;
for (int k = 0; k < width; k++) {
    sum += A[row * width + k] * B[k * width + col];
}
C[row * width + col] = sum;
```

## Compilation Flags

```bash
-O3                # Optimization level 3
-arch=sm_86        # RTX 2050 architecture
-g -G              # Debug info for cuda-gdb
-lineinfo          # Line info for profiling
-Xcompiler -Wall   # Enable all warnings
-use_fast_math     # Fast math (less accurate)
```

## Common Block/Thread Sizes

```cuda
// 1D
dim3 threads(256);
dim3 blocks((N + 255) / 256);

// 2D
dim3 threads(16, 16);  // 256 threads
dim3 blocks((width + 15) / 16, (height + 15) / 16);

// 3D
dim3 threads(8, 8, 8);  // 512 threads
dim3 blocks(...);
```

## Performance Tips

- Use shared memory for frequently accessed data
- Coalesce global memory accesses
- Avoid divergent warps (if/else)
- Maximize occupancy
- Minimize host↔device transfers
- Use streams for concurrency
- Profile with NSight
