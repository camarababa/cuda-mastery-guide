# Week 6: Production & Deployment

## Module Overview

Master the skills needed to build production-ready GPU applications, from profiling to deployment.

---

## Lesson 17: Profiling & Optimization ⭐⭐⭐⭐
**File:** `lesson17-profiling-optimization.cu`
**Duration:** 3-4 hours

**What you'll learn:**
- Systematic optimization workflow
- Using NVIDIA Nsight tools
- Identifying bottlenecks
- Roofline model analysis
- Performance metrics that matter

**Key concepts:**
- CUDA events for timing
- NVTX markers
- Occupancy calculation
- Memory vs compute bound
- Profile-guided optimization

**Run it:**
```bash
nvcc -o profiling lesson17-profiling-optimization.cu -lnvToolsExt
./profiling

# Profile with Nsight
nsys profile -o report ./profiling
ncu --set full ./profiling
```

---

## Lesson 18: CUTLASS & Templates (Coming Soon)
**Topics:**
- C++ template metaprogramming
- Building efficient GEMM kernels
- CUTLASS architecture
- Compile-time optimization

---

## Lesson 19: Error Handling & Debugging (Coming Soon)
**Topics:**
- Proper CUDA error checking
- cuda-gdb debugging
- Memory leak detection
- Race condition analysis
- Recovery strategies

---

## Lesson 20: Deployment & Integration (Coming Soon)
**Topics:**
- CUDA + Python (pybind11)
- Building libraries
- Docker + CUDA
- Cloud deployment
- CI/CD for GPU code

---

## Production Best Practices

### 1. Always Profile First
- Don't guess where the bottleneck is
- Use appropriate tools for the job
- Measure, don't assume

### 2. Error Handling
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```

### 3. Memory Management
- Always free what you allocate
- Use RAII patterns in C++
- Consider memory pools
- Profile memory usage

### 4. Optimization Workflow
1. Profile to find bottleneck
2. Analyze metrics
3. Apply targeted optimization
4. Verify improvement
5. Repeat until fast enough

### 5. Testing
- Unit test kernels
- Verify correctness before optimizing
- Test edge cases
- Benchmark regularly

---

## Tools You Must Know

### Profiling
- **nsys**: System-wide timeline
- **ncu**: Kernel analysis
- **Nsight Compute**: Interactive profiling
- **nvidia-smi**: GPU monitoring

### Debugging
- **cuda-gdb**: CUDA debugger
- **cuda-memcheck**: Memory error detector
- **compute-sanitizer**: Race detection

### Development
- **CUDA Toolkit**: Compiler and libraries
- **cuDNN**: Deep learning primitives
- **CUTLASS**: Template library
- **Thrust**: STL-like library

---

## Performance Checklist

- [ ] Coalesced memory access
- [ ] High occupancy (but not always)
- [ ] Minimal divergence
- [ ] Efficient use of shared memory
- [ ] Optimal block size
- [ ] Minimized synchronization
- [ ] Fused kernels where possible
- [ ] Appropriate precision (FP32/FP16/INT8)

---

## Common Pitfalls

1. **Premature Optimization**
   - Profile first!
   - Optimize the right thing

2. **Ignoring Error Checking**
   - Silent failures
   - Debugging nightmares

3. **Memory Leaks**
   - Always match malloc/free
   - Use tools to detect

4. **Wrong Metrics**
   - Occupancy isn't everything
   - Focus on actual performance

5. **Not Testing Correctness**
   - Fast but wrong is useless
   - Always verify results

---

## Resources

- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [GTC Talks](https://www.nvidia.com/gtc/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Nsight Tools](https://developer.nvidia.com/nsight-tools)

---

## Next Steps

After mastering production skills:
- Build and deploy real applications
- Contribute to open source
- Share your optimizations
- Keep learning new techniques

Remember: The best GPU code is not just fast, but also correct, maintainable, and deployable!
