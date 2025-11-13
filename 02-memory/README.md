# Week 2: Memory Optimization

## Module Overview

Memory is the #1 performance bottleneck in GPU programming. This week focuses on understanding and optimizing memory access patterns.

---

## Lessons

### Lesson 6: Shared Memory ⭐⭐⭐
**File:** `lesson06-shared-memory.cu`
**Duration:** 2-3 hours

**What you'll learn:**
- What shared memory is (on-chip cache)
- How to use it (`__shared__` keyword)
- Why it's 100x faster than global memory
- Synchronization with `__syncthreads()`
- Matrix transpose optimization

**Key concepts:**
- Shared memory hierarchy
- Thread cooperation
- Memory tiling
- Performance comparison

**Run it:**
```bash
cd ~/cuda-learning/02-memory
nvcc -o lesson06 lesson06-shared-memory.cu
./lesson06
```

---

## Memory Hierarchy Quick Reference

```
Fastest  ┌──────────────────┐
    ↑    │   Registers      │  1 cycle latency
    │    ├──────────────────┤
    │    │ Shared Memory    │  2-4 cycles, 48 KB per block
    │    ├──────────────────┤
    │    │ L1/L2 Cache      │  Automatic, ~100-200 cycles
    │    ├──────────────────┤
Slowest  │ Global Memory    │  200-800 cycles, 4 GB total
    ↓    └──────────────────┘
```

---

## Key Optimization Principles

1. **Use shared memory** for frequently accessed data
2. **Coalesce** global memory accesses
3. **Avoid bank conflicts** in shared memory
4. **Minimize** global memory transfers
5. **Reuse** data in registers when possible

---

## Week 2 Goals

By end of this week:
- ✓ Understand memory hierarchy completely
- ✓ Use shared memory effectively
- ✓ Achieve 2-5x speedup through memory optimization
- ✓ Recognize memory-bound vs compute-bound code

---

## Practice Project

**Matrix Transpose Optimization:**
1. Implement naive version
2. Add shared memory tiling
3. Fix bank conflicts (padding)
4. Measure speedup at each step

**Target:** 3-5x speedup over naive version

---

## Resources

- `cheatsheet.md` - Memory syntax reference
- `parallel-algorithms-guide.md` - Memory patterns
- NVIDIA Best Practices Guide - Chapter 9 (Memory Optimization)

---

## What's Next

**Week 3:** Advanced optimization techniques
- Reduction algorithms
- Matrix multiplication
- Warp-level primitives
