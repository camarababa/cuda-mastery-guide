# Week 3: Performance Optimization

## Module Overview

Now that you understand memory, we'll focus on algorithmic optimization and achieving maximum performance.

---

## Lessons

### Lesson 7: Parallel Reduction ⭐⭐⭐
**File:** `lesson07-reduction.cu`
**Duration:** 3-4 hours

**What you'll learn:**
- The reduction pattern (sum, max, min)
- Building from naive to optimized
- Warp-level optimizations
- Avoiding thread divergence
- Sequential vs interleaved addressing

**Key concepts:**
- Warp-level primitives
- Reduction trees
- Two-phase reduction
- Performance evolution

**Run it:**
```bash
cd ~/cuda-learning/03-optimization
nvcc -o lesson07 lesson07-reduction.cu
./lesson07
```

**Expected speedup:** 10-50x over CPU

---

## Core Optimization Techniques

### 1. Avoid Warp Divergence
```cuda
// Bad: Threads diverge
if (tid % 2 == 0) {
    // Half the warp does this
}

// Good: Threads converge
if (tid < s) {
    // All threads in warp do same thing
}
```

### 2. Sequential Addressing
```cuda
// Bad: Interleaved (divergence)
if (tid % (2*s) == 0) {
    sdata[tid] += sdata[tid + s];
}

// Good: Sequential (no divergence)
if (tid < s) {
    sdata[tid] += sdata[tid + s];
}
```

### 3. Warp Unrolling
```cuda
// Last 32 threads are in same warp
// No __syncthreads() needed!
if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    // ... unroll all the way to 1
}
```

---

## Performance Analysis

### Measuring Performance

```cuda
// Use CUDA events for accurate timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<...>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Key Metrics

- **Throughput:** GB/s or Gflops
- **Occupancy:** % of GPU utilized
- **Bandwidth utilization:** % of peak bandwidth
- **Speedup:** Time_baseline / Time_optimized

---

## Week 3 Goals

By end of this week:
- ✓ Implement efficient reduction
- ✓ Understand warp-level programming
- ✓ Achieve 50-100x CPU speedup
- ✓ Profile with NSight tools

---

## Practice Projects

1. **Dot Product:** a · b = Σ(a[i] * b[i])
2. **Find Maximum:** Parallel max reduction
3. **Histogram:** Count frequencies in parallel

---

## Profiling Tools

```bash
# NSight Systems (timeline)
nsys profile ./program

# NSight Compute (kernel analysis)
ncu ./program

# Basic profiling
nvprof ./program
```

---

## Resources

- Mark Harris: "Optimizing Parallel Reduction in CUDA"
- CUDA Best Practices: Chapter 10 (Occupancy)
- NSight documentation

---

## What's Next

**Week 4:** Advanced CUDA features
- Streams and concurrency
- Dynamic parallelism
- CUDA libraries
