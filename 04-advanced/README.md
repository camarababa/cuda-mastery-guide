# Week 4: Advanced CUDA Features

## Module Overview

Master advanced GPU programming techniques that unlock the full potential of modern NVIDIA GPUs.

---

## Lesson 14: Dynamic Parallelism ⭐⭐⭐⭐
**File:** `lesson14-dynamic-parallelism.cu`
**Duration:** 2-3 hours

**What you'll learn:**
- Launch kernels from kernels
- Recursive GPU algorithms
- Adaptive computation
- Device-side memory allocation
- Work generation patterns

**Key concepts:**
- Parent-child kernels
- Device synchronization
- Stack limitations
- When to use (and not use)

**Run it:**
```bash
nvcc -arch=sm_35 -rdc=true -lcudadevrt -o dynamic lesson14-dynamic-parallelism.cu
./dynamic
```

**Note:** Requires GPU with CC 3.5+ (Kepler or newer)

---

## Lesson 15: CUDA Graphs ⭐⭐⭐⭐
**File:** `lesson15-cuda-graphs.cu`
**Duration:** 3-4 hours

**What you'll learn:**
- Eliminate kernel launch overhead
- Record and replay GPU work
- Stream capture API
- Graph updates and optimizations
- Perfect for iterative algorithms

**Key concepts:**
- Graph creation and instantiation
- Manual vs stream capture
- Conditional execution
- Performance analysis

**Run it:**
```bash
nvcc -o graphs lesson15-cuda-graphs.cu
./graphs
```

---

## Lesson 16: Multi-GPU Programming ⭐⭐⭐⭐⭐
**File:** `lesson16-multi-gpu.cu`
**Duration:** 4-5 hours

**What you'll learn:**
- Scale beyond single GPU
- Peer-to-peer communication
- Unified Memory multi-GPU
- Data and model parallelism
- NCCL collective operations

**Key concepts:**
- GPU discovery and management
- P2P memory access
- Communication patterns
- Load balancing
- Synchronization strategies

**Run it:**
```bash
nvcc -o multigpu lesson16-multi-gpu.cu
./multigpu
```

**Note:** Best with 2+ GPUs, but demos work with single GPU

---

## Advanced Features Summary

### When to Use Each Feature

**Dynamic Parallelism:**
- Recursive algorithms (QuickSort, tree traversal)
- Adaptive mesh refinement
- Irregular parallelism
- Reducing CPU-GPU synchronization

**CUDA Graphs:**
- Iterative algorithms (solvers, training loops)
- Many small kernels
- Low-latency requirements
- Static workflow patterns

**Multi-GPU:**
- Large datasets (won't fit on one GPU)
- Need more compute power
- Distributed algorithms
- Production deployments

---

## Performance Considerations

### Dynamic Parallelism
- Launch overhead exists
- Limited recursion depth
- Best for coarse-grained parallelism
- Consider alternatives first

### CUDA Graphs
- Huge wins for kernel launch overhead
- Initial recording cost
- Memory for graph storage
- Not suitable for dynamic workflows

### Multi-GPU
- Communication is expensive
- Minimize data movement
- Overlap compute and communication
- Consider topology (PCIe vs NVLink)

---

## Common Patterns

### Adaptive Algorithms
```cpp
if (should_subdivide(region)) {
    // Launch child kernels
    childKernel<<<...>>>(...);
    cudaDeviceSynchronize();
}
```

### Iterative Solvers
```cpp
// Record once
cudaStreamBeginCapture(stream);
for (int i = 0; i < iterations_per_graph; i++) {
    kernel1<<<...>>>();
    kernel2<<<...>>>();
}
cudaStreamEndCapture(stream, &graph);

// Replay many times
for (int i = 0; i < num_replays; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

### Data Distribution
```cpp
for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);
    size_t offset = gpu * chunk_size;
    cudaMemcpyAsync(d_data[gpu], h_data + offset, ...);
    kernel<<<...>>>(d_data[gpu], ...);
}
```

---

## Debugging Tips

1. **Dynamic Parallelism**
   - Check compute capability
   - Monitor stack usage
   - Verify parent-child synchronization

2. **CUDA Graphs**
   - Validate graph before instantiation
   - Check for unsupported operations
   - Profile graph vs traditional

3. **Multi-GPU**
   - Check peer access capability
   - Monitor PCIe bandwidth
   - Verify data distribution

---

## Compilation Flags

```bash
# Dynamic Parallelism
nvcc -arch=sm_35 -rdc=true -lcudadevrt

# CUDA Graphs (standard compilation)
nvcc -O3

# Multi-GPU with NCCL
nvcc -lnccl

# All features with debugging
nvcc -g -G -arch=sm_70 -rdc=true -lcudadevrt -lnccl
```

---

## Resources

- [CUDA Programming Guide - Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-parallelism)
- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [Multi-GPU Programming](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#multiple-gpus)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

---

## What's Next?

After mastering these advanced features:
- Build complex GPU-only applications
- Optimize for specific hardware
- Scale to GPU clusters
- Contribute to HPC/AI frameworks

You're now ready for cutting-edge GPU computing! 🚀